"""
Voice-ID pass (voice_id_pass.plan.md): cast-blind audio-visual lip-sync
binding, replacing the still-frame mouth-motion guess (identity/speaker_
frames.py + identity/speaker_pass.py, deleted). One Gemini call PER VOICE,
watching short video+audio clips and answering a native lip-sync question:
is the person on screen the one actually speaking THIS audio? A listener's
clip self-rejects (lips don't match the sound) -- the exact failure that
left ambiguous voices unbound under the silent-still design (multicam
podcast footage: each camera frames one person, so a window only ever has
ONE visible candidate, and with no audio the model has nothing to reject a
listener with).

Two-part split (the plan's centerpiece):
  - Part A (`select_clips` / `run_voice_id_pass`): cast-BLIND, depends only
    on pre-Pass-2 data (the voice map, diarization turns, sync groups) --
    runs in PARALLEL with Pass 2 in ingest.py. The model never sees person
    descriptions or cast info; it only judges lip-sync against heard audio.
  - Part B (`bind_from_verdicts`): instant, code-only, runs AFTER
    reconcile. Maps each "speaking" clip to the global person `reconcile`
    says is visible in that cut, tallies votes per voice, majority+margin
    binds the winner -- "model perceives, code decides" applies here too:
    identity resolution is entirely code, never asked of the model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.services.l3.identity.windows import (
    _cut_span_ms,
    _loudness_peak_ms,
    _other_voice_turns_by_file,
    clean_windows,
    voice_turns,
)
from app.services.llm.base import media_block, text_block

logger = logging.getLogger(__name__)

# Clean turns sampled per voice (top-K longest -- most articulation to
# read), and the clip length centered on each turn's loudness peak.
K = 4
CLIP_MS = 3000
# Ceiling on total clips emitted for one voice, after camera fan-out (a
# moment covered by several outlook-group cameras emits one clip per
# camera) -- keeps a chatty multicam voice from exploding the plan.
MAX_CLIPS_PER_VOICE = 8
CLIP_WIDTH_PX = 512

# Binding requires the winning person to take a clear MAJORITY of the
# OPINIONATED "speaking" clips (those that resolved to a visible person at
# all), with a real margin over the runner-up -- same contract identity/
# speaker_pass.aggregate_votes proved, ported here over CLIP votes instead
# of burst-window votes. "not_speaking"/"no_face" clips are abstentions:
# they never dilute the denominator, only genuine conflict (two different
# people each drawing "speaking" votes) blocks a bind.
MIN_VOTE_SHARE = 0.5
MIN_VOTE_MARGIN = 1


@dataclass
class ClipRequest:
    voice: str
    clip_id: str
    file_id: str
    start_ms: int
    end_ms: int


@dataclass
class ClipVerdict:
    voice: str
    clip_id: str
    file_id: str
    verdict: str   # "speaking" | "not_speaking" | "no_face"
    center_ms: int


class VerdictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    verdict: str = "no_face"


class VoiceIdOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: List[VerdictItem] = Field(default_factory=list)


_SYSTEM = (
    "You are shown short CLIPS. In each clip you can HEAR one person speaking "
    "and you SEE one person on camera. For each clip, judge from LIP-SYNC -- do "
    "the lips/jaw articulate in time with the speech you hear -- and decide:\n"
    "  - \"speaking\"      the on-screen person is the one speaking this audio\n"
    "  - \"not_speaking\"  the on-screen person is silent, listening, or reacting\n"
    "  - \"no_face\"       no mouth is clearly visible/legible to judge\n\n"
    "Return exactly one verdict per clip, using the clip_id given in its label. "
    "Never guess; if you cannot read the mouth against the audio, use no_face."
)


def _clip_window(turn_s: int, turn_e: int, t_star: int, clip_ms: int = CLIP_MS) -> Tuple[int, int]:
    """A `clip_ms`-wide window centered on `t_star` (the turn's loudness
    peak), clamped inside [turn_s, turn_e) -- shrinks rather than reaching
    outside the clean turn it came from."""
    half = clip_ms // 2
    lo, hi = t_star - half, t_star + half
    if lo < turn_s:
        lo, hi = turn_s, min(turn_e, turn_s + clip_ms)
    if hi > turn_e:
        hi, lo = turn_e, max(turn_s, turn_e - clip_ms)
    return lo, hi


def _group_key_for_file(fid: str, groups: Dict[str, Dict[str, Any]]) -> str:
    """The outlook-group id `fid` belongs to, or `fid` itself when it's not
    in any declared group (its own singleton "group") -- used to decide
    which clean windows are really the SAME moment seen from different
    cameras, rather than an incidental millisecond collision."""
    for gid, grp in groups.items():
        if fid in (grp.get("members") or ()):
            return gid
    return fid


def select_clips(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
    groups: Dict[str, Dict[str, Any]],
    audio_by_file: Optional[Dict[str, dict]] = None,
    *, k: int = K, clip_ms: int = CLIP_MS, max_clips: int = MAX_CLIPS_PER_VOICE,
) -> List[ClipRequest]:
    """The whole Part A planning step -- pure code, no model call, depends
    only on pre-Pass-2 data. For each voice: its clean solo turns (identity/
    windows.clean_windows, unchanged from the still-frame design), grouped
    into distinct MOMENTS by (outlook_group, start_ms, end_ms) -- an outlook
    group's member lattices are already re-based onto the group's
    authoritative clock (sync.lattice_merge.replicate_outlook_speech/
    authoritative_view), so the SAME semantic moment carries the SAME span
    on every camera in the group, and grouping first means top-K keeps K
    DISTINCT moments rather than K copies of the loudest one or two. The
    top-K longest moments are kept (most articulation to read); each fans
    out to one ClipRequest per covering camera, centered on the moment's
    loudness peak, capped at `max_clips` total."""
    audio_by_file = audio_by_file or {}
    out: List[ClipRequest] = []

    for voice, turns in sorted(voice_turns(turns_by_file, voice_of).items()):
        other_turns = _other_voice_turns_by_file(turns_by_file, voice_of, voice)
        windows = clean_windows(turns, other_turns)
        if not windows:
            continue

        by_span: Dict[Tuple[str, int, int], List[str]] = {}
        for fid, s, e in windows:
            key = (_group_key_for_file(fid, groups), s, e)
            by_span.setdefault(key, []).append(fid)

        ranked_spans = sorted(by_span.keys(), key=lambda key: key[2] - key[1], reverse=True)[:k]

        n = 0
        for i, key in enumerate(ranked_spans):
            if n >= max_clips:
                break
            _gid, s, e = key
            for fid in sorted(by_span[key]):
                if n >= max_clips:
                    break
                audio = audio_by_file.get(fid) or {}
                t_star = _loudness_peak_ms(audio.get("rms_db") or [], int(audio.get("hop_ms") or 0), s, e)
                clip_s, clip_e = _clip_window(s, e, t_star, clip_ms)
                out.append(ClipRequest(
                    voice=voice, clip_id=f"{voice}:c{i}:{fid[:8]}",
                    file_id=fid, start_ms=clip_s, end_ms=clip_e,
                ))
                n += 1
    return out


def run_voice_id_pass(
    clip_requests: List[ClipRequest], clips_b64: Dict[str, str],
) -> List[ClipVerdict]:
    """One Gemini call per voice, cast-blind (the prompt never names or
    describes a person -- only clip pixels + audio). Returns the raw
    per-clip verdicts; a clip with no extracted bytes contributes nothing
    (never a fabricated verdict), and a clip the model didn't return a
    verdict for defaults to "no_face" (never silently dropped from the
    tally)."""
    settings = get_settings()
    by_voice: Dict[str, List[ClipRequest]] = {}
    for r in clip_requests:
        by_voice.setdefault(r.voice, []).append(r)

    out: List[ClipVerdict] = []
    for voice, reqs in sorted(by_voice.items()):
        blocks: List[Any] = []
        kept: List[ClipRequest] = []
        for r in reqs:
            b64 = clips_b64.get(r.clip_id)
            if b64 is None:
                continue
            blocks.append(text_block(f"CLIP {r.clip_id}:"))
            blocks.append(media_block(b64, "video/mp4"))
            kept.append(r)
        if not blocks:
            continue

        from app.services.llm import ingest_gemini as ig
        completion = ig.complete_gemini(
            _SYSTEM, blocks, VoiceIdOutput, max_tokens=2048,
            model=settings.identity_voice_id_model, thinking=settings.identity_voice_id_thinking,
        )
        parsed = VoiceIdOutput.model_validate(completion.data)
        verdict_by_clip = {v.clip_id: v.verdict for v in parsed.verdicts}
        for r in kept:
            verdict = verdict_by_clip.get(r.clip_id, "no_face")
            out.append(ClipVerdict(voice=r.voice, clip_id=r.clip_id, file_id=r.file_id,
                                   verdict=verdict, center_ms=(r.start_ms + r.end_ms) // 2))
    return out


def _cut_at(file_id: str, center_ms: int, cuts_by_file: Dict[str, List[Any]], lattice: Any) -> Optional[Any]:
    """The cut in `file_id` whose resolved span covers `center_ms` -- a
    clip's midpoint, used to look up which cut (and thus which visible
    persons) a "speaking" verdict actually refers to."""
    for cut in cuts_by_file.get(file_id, []):
        span = _cut_span_ms(cut, lattice)
        if span is not None and span[0] <= center_ms < span[1]:
            return cut
    return None


def bind_from_verdicts(
    verdicts: List[ClipVerdict],
    visible_persons: Dict[Tuple[str, str], List[str]],
    cuts_by_file: Dict[str, List[Any]],
    lattices: Dict[str, Any],
) -> Tuple[Dict[str, Optional[str]], set]:
    """Part B -- runs after reconcile, instant and code-only. For each
    voice: keep the clips the model called "speaking", resolve each to the
    cut covering its center_ms and that cut's visible_persons (usually one
    person; a multi-person cut adds a vote per visible person), then
    majority+margin bind the winner. `not_speaking`/`no_face` clips are
    abstentions -- they never count against a winner and never contribute a
    vote either.

    Returns `(owner_by_voice, off_camera_voices)`. `owner_by_voice` covers
    every voice that had at least one verdict (bound -> person id, unbound
    -> None); a voice entirely absent from `verdicts` (Track A never got a
    clip to show for it) is NOT in this dict -- the caller reconciles that
    against the full voice roster. `off_camera_voices` is every voice here
    whose votes didn't clear the majority+margin bar."""
    by_voice: Dict[str, List[ClipVerdict]] = {}
    for v in verdicts:
        by_voice.setdefault(v.voice, []).append(v)

    owner_by_voice: Dict[str, Optional[str]] = {}
    off_camera: set = set()
    for voice, vlist in sorted(by_voice.items()):
        counts: Dict[str, int] = {}
        for v in vlist:
            if v.verdict != "speaking":
                continue
            cut = _cut_at(v.file_id, v.center_ms, cuts_by_file, lattices.get(v.file_id))
            if cut is None:
                continue
            for pid in visible_persons.get((v.file_id, cut.source_ref), []):
                counts[pid] = counts.get(pid, 0) + 1

        if not counts:
            owner_by_voice[voice] = None
            off_camera.add(voice)
            continue

        opinionated = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        winner, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if top / opinionated < MIN_VOTE_SHARE or top - second < MIN_VOTE_MARGIN:
            owner_by_voice[voice] = None
            off_camera.add(voice)
        else:
            owner_by_voice[voice] = winner

        import os as _os
        if _os.environ.get("IDENTITY_DEBUG"):
            named = [v.verdict for v in vlist]
            print(f"[IDDEBUG] {voice}: {len(vlist)} clip(s), verdicts={named}, "
                 f"votes={counts} -> {owner_by_voice[voice]}", flush=True)
        if owner_by_voice[voice] is None:
            logger.info("voice_id: voice %s unbound (%d clip(s), no confident majority)",
                       voice, len(vlist))
    return owner_by_voice, off_camera
