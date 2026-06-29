"""
v2 capture ATOMS -- the honest substrate the whole cut pipeline now stands on.

An ATOM is one timestamped *thing the camera/mic captured*, on exactly one of
four CHANNELS (see vocab.py):

    SAID   -- a spoken line / voiceover   (from L1 transcript + diarization)
    DONE   -- a physical action / change  (from the VLM, peak snapped to motion)
    SHOWN  -- a held subject to be seen    (from the VLM, peak = representative frame)
    HEARD  -- non-speech sound             (from the RMS envelope; built, suppressed)

Atoms are DETECTION, not judgment: no editorial buckets (reaction/b-roll/insert),
no roles, no relations. Each carries a SUBJECT tag (person/place/object/graphic)
on the video channels and a CONFIDENCE used as the keep gate. A `peak_ms` (the
impact / reveal / punch instant) is the anchor the energy combiner zooms toward
and the fused field cuts onto.

This module is pure (clip artifacts in, Atom list out) and has no DB/VLM call, so
it is trivially testable. The downstream combiner (l3.combine) turns surviving
atoms into hero cuts + capture-moments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.l3 import vocab

# Per-channel confidence floor: keep an atom only when the source is at least
# this sure it is a real, usable beat. Recall-first (low) for everything we
# trust; strict for Heard so only genuinely salient sound survives (not breaths
# / mouth-clicks / ambient spikes -- the noise that used to leak in as "person").
_CHANNEL_FLOOR: Dict[str, float] = {
    vocab.CHANNEL_SAID: 0.0,     # transcript-derived; trust it
    vocab.CHANNEL_DONE: 0.30,
    vocab.CHANNEL_SHOWN: 0.30,
    vocab.CHANNEL_HEARD: 0.70,   # strict: only noticeable sounds
}

# A weaker (held) atom co-extensive with a stronger event atom of the SAME actor
# is that cut's framing, not its own card. Fold it in when their overlap covers
# at least this fraction of the shorter span.
_FOLD_OVERLAP_FRAC = 0.7

# Off-camera / non-cut dialogue lexicon flags (mirror anchors.py).
_OFFCAMERA_FLAGS = ("offscreen", "production_cue", "backchannel")

# Heard detection (strict): only sound clearly above the clip's own speech-vs-
# floor midpoint, sustained, counts. Tighter than the old audio-event anchor.
_HEARD_THR_FRAC = 0.70         # threshold = floor + frac*(speech_level - floor)
_HEARD_MIN_MS = 700            # shorter bursts are clicks/noise
_HEARD_MERGE_MS = 250
_SPEECH_PAD_MS = 200


@dataclass
class Atom:
    channel: str                      # said | done | shown | heard
    start_ms: int
    end_ms: int
    peak_ms: int                      # impact / reveal / punch instant
    confidence: float = 0.5
    subject: Optional[str] = None     # person | place | object | graphic (video channels)
    actor: Optional[str] = None       # person local_id when known (instance, not type)
    label: str = ""
    text: Optional[str] = None        # said: the spoken words
    speaker: Optional[str] = None     # said: diarized speaker label
    on_camera: Optional[bool] = None  # said / folded attribute
    region: Optional[dict] = None     # coarse frame box for reframing
    content_key: Optional[str] = None # take-grouping identity (speech line, etc.)
    summary: Optional[str] = None     # graphic gist (what it conveys)
    source_id: Optional[str] = None   # originating artifact id
    flags: List[str] = field(default_factory=list)  # folded attributes: on_camera, gesturing, ...

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel, "start_ms": self.start_ms, "end_ms": self.end_ms,
            "peak_ms": self.peak_ms, "confidence": round(self.confidence, 3),
            "subject": self.subject, "actor": self.actor, "label": self.label,
            "text": self.text, "speaker": self.speaker, "on_camera": self.on_camera,
            "region": self.region, "content_key": self.content_key,
            "summary": self.summary, "source_id": self.source_id, "flags": self.flags,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _norm_subject(v) -> Optional[str]:
    s = str(getattr(v, "value", v) or "").strip().lower()
    return s if s in vocab.SUBJECT_SET else None


def _norm_channel(v) -> Optional[str]:
    s = str(getattr(v, "value", v) or "").strip().lower()
    return s if s in vocab.CHANNEL_SET else None


def _is_offcamera(seg: dict) -> bool:
    flags = seg.get("flags") or []
    return any(f in flags for f in _OFFCAMERA_FLAGS)


def _mean(xs: List[float], lo: int, hi: int) -> float:
    seg = xs[max(0, lo):max(lo + 1, hi)]
    return sum(seg) / len(seg) if seg else 0.0


# --------------------------------------------------------------------------
# Channel builders
# --------------------------------------------------------------------------

def _said_atoms(clip) -> List[Atom]:
    """One Said atom per on-camera sentence, from the dialogue lens. Speaker /
    on-camera / actor are resolved from the VLM `speaking` spans (the person
    visibly talking over the line), so a Said atom shares an actor id with a
    Shown.person atom of the same human -- which is what the fold rule needs."""
    dialogue = clip.dialogue or {}
    sentences = dialogue.get("sentence") or dialogue.get("topic") or []
    perception = clip.perception or {}
    speaking = [(s.get("subject"), int(s.get("start_ms", 0)), int(s.get("end_ms", 0)))
                for s in (perception.get("speaking") or [])
                if s.get("subject") is not None]

    def actor_for(a: int, b: int) -> Optional[str]:
        best, best_ov = None, 0
        for subj, sa, sb in speaking:
            ov = _overlap(a, b, sa, sb)
            if ov > best_ov:
                best, best_ov = subj, ov
        return best

    out: List[Atom] = []
    for s in sentences:
        if _is_offcamera(s):
            continue
        a = int(s.get("src_in_ms", s.get("raw_in_ms", 0)))
        b = int(s.get("src_out_ms", s.get("raw_out_ms", a)))
        if b <= a:
            continue
        text = (s.get("text") or "").strip()
        actor = actor_for(a, b)
        out.append(Atom(
            channel=vocab.CHANNEL_SAID, start_ms=a, end_ms=b, peak_ms=(a + b) // 2,
            confidence=1.0, subject=vocab.SUBJECT_PERSON, actor=actor,
            label=text[:200], text=text, speaker=s.get("speaker"),
            on_camera=actor is not None, content_key=text.lower() or None,
            source_id=str(s.get("seg_id", "")),
        ))
    return out


def _snap_peak(start: int, end: int, motion: Optional[dict]) -> int:
    """Peak = the strongest motion impact inside the span, else its midpoint."""
    mid = (start + end) // 2
    if not motion:
        return mid
    pts = [(int(p.get("ts_ms", 0)), float(p.get("score", 0.0)))
           for p in (motion.get("action_points") or [])
           if isinstance(p, dict) and p.get("ts_ms") is not None]
    inside = [(t, sc) for t, sc in pts if start <= t <= end]
    return max(inside, key=lambda t: t[1])[0] if inside else mid


def _vlm_atoms(perception: dict, motion: Optional[dict]) -> List[Atom]:
    """Done / Shown atoms from the VLM's `atoms` track (v2 detection output)."""
    out: List[Atom] = []
    for a in (perception.get("atoms") or []):
        ch = _norm_channel(a.get("channel"))
        if ch not in (vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN):
            continue
        s, e = int(a.get("start_ms", 0)), int(a.get("end_ms", 0))
        if e <= s:
            continue
        pk = a.get("peak_ms")
        if pk is None:
            pk = _snap_peak(s, e, motion) if ch == vocab.CHANNEL_DONE else (s + e) // 2
        pk = max(s, min(int(pk), e))
        conf = a.get("confidence")
        out.append(Atom(
            channel=ch, start_ms=s, end_ms=e, peak_ms=pk,
            confidence=_clamp01(float(conf)) if conf is not None else 0.5,
            subject=_norm_subject(a.get("subject")),
            actor=a.get("actor") or a.get("subject_id"),
            label=str(a.get("label") or "")[:200],
            region=a.get("region"), content_key=a.get("content_key"),
            summary=(str(a.get("summary")).strip()[:400] or None) if a.get("summary") else None,
            source_id=str(a.get("id", "")),
        ))
    return out


# --- v1 -> atoms fallback ------------------------------------------------------
# Until L2 is re-run under SCHEMA_VERSION 6, clips carry the OLD content_units /
# cutaways tracks instead of `atoms`. Map them onto v2 atoms so the v2 pipeline
# produces cuts on the existing corpus immediately (validate before re-running).
_V1_KIND_CHANNEL = {
    "action": vocab.CHANNEL_DONE,
    "performance": vocab.CHANNEL_DONE,
}
_V1_PRIMITIVE_SUBJECT = {
    "person": vocab.SUBJECT_PERSON,
    "place": vocab.SUBJECT_PLACE,
    "object": vocab.SUBJECT_OBJECT,
    "graphic": vocab.SUBJECT_GRAPHIC,
    "action": None,
}


def _atoms_from_v1(perception: dict, motion: Optional[dict]) -> List[Atom]:
    out: List[Atom] = []
    for u in (perception.get("content_units") or []):
        kind = (u.get("kind") or "").lower()
        s, e = int(u.get("start_ms", 0)), int(u.get("end_ms", 0))
        if e <= s or kind == "speech":  # speech comes from the dialogue lens
            continue
        ch = _V1_KIND_CHANNEL.get(kind, vocab.CHANNEL_SHOWN)
        conf = u.get("confidence")
        out.append(Atom(
            channel=ch, start_ms=s, end_ms=e,
            peak_ms=_snap_peak(s, e, motion) if ch == vocab.CHANNEL_DONE else (s + e) // 2,
            confidence=_clamp01(float(conf)) if conf is not None else 0.5,
            subject=_V1_PRIMITIVE_SUBJECT.get((u.get("primitive") or "").lower()),
            actor=u.get("subject") or u.get("actor"),
            label=str(u.get("label") or u.get("content_key") or kind)[:200],
            content_key=u.get("content_key"), source_id=str(u.get("unit_id", "")),
        ))
    for c in (perception.get("cutaways") or []):
        s, e = int(c.get("start_ms", 0)), int(c.get("end_ms", 0))
        if e <= s:
            continue
        prim = (c.get("primitive") or "").lower()
        subject = _V1_PRIMITIVE_SUBJECT.get(prim)
        aff = (c.get("affordance") or "").lower()
        if subject is None:
            subject = {"reaction": vocab.SUBJECT_PERSON, "insert": vocab.SUBJECT_GRAPHIC}.get(
                aff, vocab.SUBJECT_PLACE)
        # A changing graphic (screen-rec) is Done; everything else held = Shown.
        ch = vocab.CHANNEL_DONE if (subject == vocab.SUBJECT_GRAPHIC and prim == "action") \
            else vocab.CHANNEL_SHOWN
        sal = c.get("salience_hint")
        conf = c.get("confidence")
        peak = c.get("peak_ms")
        out.append(Atom(
            channel=ch, start_ms=s, end_ms=e,
            peak_ms=max(s, min(int(peak), e)) if peak is not None else (s + e) // 2,
            confidence=_clamp01(float(conf)) if conf is not None
            else (_clamp01(float(sal)) if sal is not None else 0.5),
            subject=subject, actor=c.get("subject"),
            label=str(c.get("label") or "")[:200],
            summary=(str(c.get("summary")).strip()[:400] or None) if c.get("summary") else None,
            source_id=str(c.get("id", "")),
        ))
    return out


def _heard_atoms(audio: Optional[dict], sentences: List[dict]) -> List[Atom]:
    """Strict non-speech sound from the RMS envelope minus the speech mask. Built
    for completeness (and to keep noise OFF the person channel) but suppressed
    downstream -- only genuinely loud, sustained sound clears the threshold."""
    audio = audio or {}
    rms = audio.get("rms_db") or []
    hop = int(audio.get("prosody_hop_ms") or 0)
    if not rms or hop <= 0:
        return []
    n = len(rms)
    speech = [False] * n
    for s in sentences:
        a = int(s.get("src_in_ms", s.get("raw_in_ms", 0))) - _SPEECH_PAD_MS
        b = int(s.get("src_out_ms", s.get("raw_out_ms", 0))) + _SPEECH_PAD_MS
        for i in range(max(0, a // hop), min(n, b // hop + 1)):
            speech[i] = True
    srt = sorted(rms)
    floor = srt[int(0.20 * (n - 1))]
    sp_vals = [rms[i] for i in range(n) if speech[i]]
    speech_level = (sorted(sp_vals)[len(sp_vals) // 2] if sp_vals else srt[int(0.90 * (n - 1))])
    if speech_level <= floor:
        return []
    thr = floor + _HEARD_THR_FRAC * (speech_level - floor)
    runs: List[List[int]] = []
    i = 0
    while i < n:
        if rms[i] >= thr and not speech[i]:
            j, gap, k = i, 0, i
            while k < n:
                if rms[k] >= thr and not speech[k]:
                    j, gap = k, 0
                elif speech[k]:
                    break
                else:
                    gap += hop
                    if gap > _HEARD_MERGE_MS:
                        break
                k += 1
            runs.append([i, j])
            i = k
        else:
            i += 1
    out: List[Atom] = []
    span = max(1.0, speech_level - floor)
    for a_i, b_i in runs:
        a, b = a_i * hop, (b_i + 1) * hop
        if b - a < _HEARD_MIN_MS:
            continue
        loud = _mean(rms, a_i, b_i + 1)
        conf = _clamp01(0.5 + 0.5 * ((loud - thr) / span))
        out.append(Atom(
            channel=vocab.CHANNEL_HEARD, start_ms=a, end_ms=b, peak_ms=(a + b) // 2,
            confidence=conf, label="non-speech sound",
        ))
    return out


# --------------------------------------------------------------------------
# Gate + fold
# --------------------------------------------------------------------------

def _gate(atoms: List[Atom]) -> List[Atom]:
    """Drop atoms below their channel's confidence floor (recall-first)."""
    return [a for a in atoms if a.confidence >= _CHANNEL_FLOOR.get(a.channel, 0.3)]


def _fold_redundant(atoms: List[Atom]) -> List[Atom]:
    """Dominant-channel anti-flood. A held/weaker atom (Shown, or a lower-conf
    Done) that is co-extensive with a STRONGER event atom of the SAME actor is
    that cut's framing, not its own card: fold it in as an attribute (the speech
    cut becomes 'on_camera'/'gesturing') and drop the duplicate. A talking head
    thus yields ONE Said cut, not Said + a redundant Shown.person + a Done gesture.

    Only same-actor pairs fold, so a genuine cutaway to a DIFFERENT subject (a
    listener, the product, the scenery) always survives as its own cut."""
    # Event atoms (Said/Done) are the potential dominants; rank by confidence.
    dominants = sorted(
        (a for a in atoms if a.channel in (vocab.CHANNEL_SAID, vocab.CHANNEL_DONE)),
        key=lambda a: a.confidence, reverse=True)
    kept: List[Atom] = []
    for a in atoms:
        # Only weaker, held-or-equal atoms are fold candidates.
        if a.channel == vocab.CHANNEL_SAID:
            kept.append(a)
            continue
        folded = False
        for d in dominants:
            if d is a or d.confidence < a.confidence:
                continue
            if a.actor is None or d.actor is None or a.actor != d.actor:
                continue
            shorter = max(1, min(a.end_ms - a.start_ms, d.end_ms - d.start_ms))
            if _overlap(a.start_ms, a.end_ms, d.start_ms, d.end_ms) >= _FOLD_OVERLAP_FRAC * shorter:
                tag = "on_camera" if a.channel == vocab.CHANNEL_SHOWN else "gesturing"
                if tag not in d.flags:
                    d.flags.append(tag)
                folded = True
                break
        if not folded:
            kept.append(a)
    return kept


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def build_atoms(clip) -> List[Atom]:
    """Every captured atom for one clip: Said (transcript), Done/Shown (VLM
    `atoms`, or a v1 fallback from content_units/cutaways), Heard (RMS, strict).
    Gated by per-channel confidence, then the dominant-channel fold removes
    redundant framing. Time-sorted."""
    perception = clip.perception or {}
    motion = clip.motion
    dialogue = clip.dialogue or {}
    sentences = dialogue.get("sentence") or dialogue.get("topic") or []

    atoms: List[Atom] = []
    atoms += _said_atoms(clip)
    if perception.get("atoms"):
        atoms += _vlm_atoms(perception, motion)
    else:
        atoms += _atoms_from_v1(perception, motion)   # legacy corpus, pre re-run
    atoms += _heard_atoms(clip.audio, sentences)

    # Clamp to the clip.
    dur = clip.duration_ms or 0
    for a in atoms:
        a.start_ms = max(0, a.start_ms)
        if dur:
            a.end_ms = min(dur, a.end_ms)
        a.peak_ms = max(a.start_ms, min(a.peak_ms, a.end_ms))
    atoms = [a for a in atoms if a.end_ms > a.start_ms]

    atoms = _gate(atoms)
    atoms = _fold_redundant(atoms)
    atoms.sort(key=lambda a: (a.start_ms, a.peak_ms))
    return atoms
