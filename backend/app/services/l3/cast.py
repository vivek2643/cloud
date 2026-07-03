"""
The cast / speaker map -- who is in a clip, and which voice is whose.

WHY this exists. A cut needs to say WHO is talking (the right on-screen person),
whether that person is ON CAMERA while they speak, and where they sit in frame
(for reframing). Three independent signals carry pieces of that answer and none
carries all of it:

  * L1 diarization tags every word with a VOICE id (``S0``, ``S1``, ...) -- audio
    only, no idea who is visible.
  * The VLM tags visible PEOPLE (``p1``, ``p2``, ... with role + frame region)
    and, separately, when each is *visibly speaking* (``speaking`` spans) -- video
    only, never told the diarization answer.
  * Neither knows the other's labels.

This module fuses them WITHOUT a model call: it intersects each visible person's
``speaking`` spans with the diarized words underneath, and the voice that
dominates a person's on-camera speech IS that person's voice. The result is a
``ClipCast`` -- a per-clip map ``voice id -> {person, role, on/off camera,
frame region, confidence}`` -- that the speech path uses for correct labels and
the take picker uses to prefer the take where the speaker is on camera.

NEVER DISCARD. A voice that links to no visible person is not dropped: it becomes
an OFF-CAMERA cast member (when the clip otherwise has visible speakers, so we
can tell) or an UNKNOWN one (when there is no visible-speaking signal at all, so
we genuinely can't tell). The brain/picker is handed every option, never fewer.

Pure: plain dicts in (one clip's ``perception`` + its diarized ``words``),
dataclasses out. No DB, no VLM, so it is trivially testable and cheap enough to
compute inline in the hero path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# A person must dominate a voice's on-camera speech by at least this share before
# we call the A/V link confident; below it the link still stands (best guess) but
# the confidence rides along so consumers can stay cautious.
_LINK_CONFIDENT = 0.6


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

@dataclass
class CastMember:
    """One voice in a clip + everything we could resolve about who owns it."""
    voice_speaker_id: Optional[str]        # diarization id (S0, S1, ...) or None
    person_id: Optional[str] = None         # VLM local_id (p1, ...) or None
    role: Optional[str] = None              # VLM role ('interviewer', 'main subject', ...)
    region: Optional[dict] = None           # coarse frame box for reframing
    # Tri-state: True = seen speaking on camera, False = voice present but no
    # visible speaker (off-camera), None = no visible-speaking signal to judge.
    on_camera: Optional[bool] = None
    av_link_confidence: float = 0.0         # 0..1 strength of the voice<->person link
    speaking_spans: List[Tuple[int, int]] = field(default_factory=list)
    durable: Optional[dict] = None          # VLM durable traits (cross-clip re-id, later)
    appearance: Optional[str] = None        # VLM canonical description of the linked face

    def label(self) -> str:
        """Best human-facing name for this member: role, else person, else voice."""
        if self.role:
            return self.role
        if self.person_id:
            return self.person_id
        return self.voice_speaker_id or "unknown"

    def on_camera_at(self, ms: int) -> Optional[bool]:
        """Whether this voice's owner is visibly speaking at ``ms`` (None when we
        have no visible-speaking signal at all to judge by)."""
        if self.on_camera is None:
            return None
        return any(a <= ms <= b for a, b in self.speaking_spans)

    def on_camera_ratio(self, start_ms: int, end_ms: int) -> Optional[float]:
        """Fraction of [start, end] this voice's owner is visibly speaking (None
        when there is no visible-speaking signal). Used to prefer the take where
        the speaker is actually on screen."""
        if self.on_camera is None:
            return None
        dur = max(1, end_ms - start_ms)
        covered = sum(_overlap_ms(a, b, start_ms, end_ms) for a, b in self.speaking_spans)
        return min(1.0, covered / dur)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_speaker_id": self.voice_speaker_id,
            "person_id": self.person_id,
            "role": self.role,
            "region": self.region,
            "on_camera": self.on_camera,
            "av_link_confidence": round(self.av_link_confidence, 3),
        }


@dataclass
class ClipCast:
    """The per-clip voice -> person map + lookups."""
    file_id: Optional[str]
    members: List[CastMember] = field(default_factory=list)
    has_visible_speakers: bool = False      # did the VLM log any speaking spans?

    def resolve(self, voice_speaker_id: Optional[str]) -> Optional[CastMember]:
        """The cast member for a diarized voice id (None when unmapped)."""
        if voice_speaker_id is None:
            return None
        for m in self.members:
            if m.voice_speaker_id == voice_speaker_id:
                return m
        return None

    def display_label(self, voice_speaker_id: Optional[str]) -> Optional[str]:
        """Who to show as the speaker for a diarized voice -- the resolved person/
        role when we have one, else the raw voice id (never nothing)."""
        m = self.resolve(voice_speaker_id)
        if m is not None:
            return m.label()
        return voice_speaker_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "has_visible_speakers": self.has_visible_speakers,
            "members": [m.to_dict() for m in self.members],
        }


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _voice_overlap_in_span(words: List[dict], a: int, b: int) -> Dict[str, int]:
    """Per-voice spoken duration inside [a, b] -- the audio's verdict on who is
    talking while a visible person's mouth moves."""
    tally: Dict[str, int] = {}
    for w in words:
        spk = w.get("speaker")
        if spk is None:
            continue
        ov = _overlap_ms(int(w.get("start_ms", 0)), int(w.get("end_ms", 0)), a, b)
        if ov > 0:
            tally[spk] = tally.get(spk, 0) + ov
    return tally


def _speaking_by_person(speaking: List[dict]) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for s in speaking:
        subj = s.get("subject")
        a, b = int(s.get("start_ms", 0)), int(s.get("end_ms", 0))
        if subj is None or b <= a:
            continue
        out.setdefault(subj, []).append((a, b))
    for subj in out:
        out[subj].sort()
    return out


def _all_voices(words: List[dict]) -> List[str]:
    """Diarized voice ids in first-appearance order (stable, dedup)."""
    seen: List[str] = []
    for w in words:
        spk = w.get("speaker")
        if spk is not None and spk not in seen:
            seen.append(spk)
    return seen


def build_cast(perception: Optional[dict], words: List[dict]) -> ClipCast:
    """Fuse the VLM's visible people with L1 diarization into a per-clip cast.

    Links each visible person to the voice that dominates their on-camera speech,
    then folds in every remaining voice as an off-camera / unknown member so
    nothing is discarded. Pure and best-effort: missing tracks simply yield a
    thinner (but always valid) cast.
    """
    perception = perception or {}
    persons = list(perception.get("persons") or [])
    speaking = list(perception.get("speaking") or [])
    has_visible = bool(speaking)

    voices = _all_voices(words)
    spk_by_person = _speaking_by_person(speaking)
    person_by_id = {p.get("local_id"): p for p in persons if p.get("local_id")}

    # 1) For each visible person, find the voice that dominates their on-camera
    #    speech and how strongly (share of their on-camera spoken time).
    #    claims[voice] = (person_id, confidence, spans) -- greedily kept best.
    claims: Dict[str, Tuple[str, float, List[Tuple[int, int]]]] = {}
    for pid, spans in spk_by_person.items():
        tally: Dict[str, int] = {}
        for a, b in spans:
            for spk, ov in _voice_overlap_in_span(words, a, b).items():
                tally[spk] = tally.get(spk, 0) + ov
        total = sum(tally.values())
        if total <= 0:
            continue
        voice, best = max(tally.items(), key=lambda kv: kv[1])
        conf = best / total
        prev = claims.get(voice)
        if prev is None or conf > prev[1]:
            claims[voice] = (pid, conf, spans)

    # 2) Materialize one member per diarized voice (never drop a voice).
    members: List[CastMember] = []
    linked_persons: set = set()
    for voice in voices:
        claim = claims.get(voice)
        if claim is not None:
            pid, conf, spans = claim
            p = person_by_id.get(pid) or {}
            region = p.get("frame_region")
            if region is None and spans:
                region = next((s.get("region") for s in speaking
                               if s.get("subject") == pid and s.get("region")), None)
            members.append(CastMember(
                voice_speaker_id=voice, person_id=pid, role=p.get("role"),
                region=region, on_camera=True, av_link_confidence=conf,
                speaking_spans=spans, durable=p.get("durable"),
                appearance=p.get("canonical_description"),
            ))
            linked_persons.add(pid)
        else:
            # A voice no visible person claims: off-camera when the clip HAS
            # visible speakers (so we can tell), else genuinely unknown.
            members.append(CastMember(
                voice_speaker_id=voice, on_camera=(False if has_visible else None),
            ))

    # 3) Visible people who never linked to a voice (silent on camera, or speech
    #    the diarizer missed) are still part of the cast -- carry them as
    #    voiceless members so reframing/labels can still reach them.
    for pid, p in person_by_id.items():
        if pid in linked_persons:
            continue
        spans = spk_by_person.get(pid, [])
        members.append(CastMember(
            voice_speaker_id=None, person_id=pid, role=p.get("role"),
            region=p.get("frame_region"),
            on_camera=True if spans else None,
            speaking_spans=spans, durable=p.get("durable"),
            appearance=p.get("canonical_description"),
        ))

    return ClipCast(file_id=perception.get("file_id"), members=members,
                    has_visible_speakers=has_visible)
