"""
Shared voice-window utilities used by the voice-ID pass (identity/voice_id.py,
voice_id_pass.plan.md) -- carried over unchanged from the still-frame design
(identity/speaker_frames.py, deleted) since finding a voice's CLEAN turns
(no other voice overlapping nearby) is the same problem whether the payload
sent downstream is a burst of stills or a short video+audio clip.

  1. `voice_turns` -- V's turns across every file, via the global voice map.
  2. `clean_windows` -- the maximal sub-span of each turn where no OTHER
     voice's turn comes within GUARD_MS (removes overlapping-speech
     contamination).
  3. `_loudness_peak_ms` -- the loudest voiced instant in a window, used to
     center a burst/clip on peak articulation.
  4. `_cut_span_ms` -- resolved (start_ms, end_ms) for one Pass2Cut-shaped
     object, used to map a clip's instant back to the cut (and thus the
     visible persons) it falls inside.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import resolve_speech_span_ms

# Chosen defaults (voice_first_identity.plan.md section 7) -- tunable,
# clip-relative where possible.
GUARD_MS = 150
MIN_WIN_MS = 400


def voice_turns(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
) -> Dict[str, List[Tuple[str, int, int]]]:
    """voice -> [(file_id, start_ms, end_ms), ...] -- every turn across
    every file whose local speaker maps to that global voice (identity/
    voices.assign_voices). A (file_id, local_speaker) with no voice mapping
    contributes nothing -- defensive; the full-roster `assign_voices` call
    should never actually leave one uncovered."""
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    for fid, turns in turns_by_file.items():
        for start_ms, end_ms, local in turns:
            if not local:
                continue
            voice = voice_of.get((fid, local))
            if voice is None:
                continue
            out.setdefault(voice, []).append((fid, int(start_ms), int(end_ms)))
    return out


def _other_voice_turns_by_file(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
    exclude_voice: str,
) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for fid, turns in turns_by_file.items():
        for s, e, local in turns:
            v = voice_of.get((fid, local))
            if v is not None and v != exclude_voice:
                out.setdefault(fid, []).append((int(s), int(e)))
    return out


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _largest_clean_subspan(s: int, e: int, forbidden: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """The largest sub-interval of [s, e) that avoids every forbidden
    interval -- None when [s, e) is fully covered by forbidden zones."""
    merged = _merge_intervals(forbidden)
    pieces: List[Tuple[int, int]] = []
    cur = s
    for fs, fe in merged:
        if fe <= cur or fs >= e:
            continue
        if fs > cur:
            pieces.append((cur, min(fs, e)))
        cur = max(cur, fe)
        if cur >= e:
            break
    if cur < e:
        pieces.append((cur, e))
    if not pieces:
        return None
    return max(pieces, key=lambda p: p[1] - p[0])


def clean_windows(
    turns: List[Tuple[str, int, int]],
    other_turns_by_file: Dict[str, List[Tuple[int, int]]],
    guard_ms: int = GUARD_MS, min_win_ms: int = MIN_WIN_MS,
) -> List[Tuple[str, int, int]]:
    """For each of V's turns, the maximal sub-span with no OTHER voice's
    turn within `guard_ms` on either side, kept only when >= `min_win_ms`
    (overlapping/nearby speech contaminates "whose mouth moves"). One
    window per turn (the plan's "carve the maximal sub-span", singular) --
    [(file_id, start_ms, end_ms), ...]."""
    out: List[Tuple[str, int, int]] = []
    for fid, s, e in turns:
        forbidden = [(os_ - guard_ms, oe + guard_ms) for os_, oe in other_turns_by_file.get(fid, [])]
        span = _largest_clean_subspan(s, e, forbidden)
        if span is not None and span[1] - span[0] >= min_win_ms:
            out.append((fid, span[0], span[1]))
    return out


def _cut_span_ms(cut: Any, lattice: Any) -> Optional[Tuple[int, int]]:
    """Best-effort (s, e) ms for one Pass2Cut-shaped object (duck-typed --
    file_id/kind/word_span/atom_ids -- this module stays decoupled from the
    pass2 schema, same pattern the rest of identity/ already uses)."""
    if lattice is None:
        return None
    if cut.kind == "speech" and cut.word_span:
        return resolve_speech_span_ms(lattice.words, lattice.atoms, tuple(cut.word_span), [])
    if cut.kind == "video" and cut.atom_ids:
        atoms_by_id = {a.atom_id: a for a in lattice.atoms}
        members = [atoms_by_id[i] for i in cut.atom_ids if i in atoms_by_id]
        if not members:
            return None
        return min(a.start_ms for a in members), max(a.end_ms for a in members)
    return None


def _value_at(arr: List[float], hop_ms: int, ts_ms: int) -> Optional[float]:
    if not arr or hop_ms <= 0:
        return None
    i = ts_ms // hop_ms
    return arr[i] if 0 <= i < len(arr) else None


def _loudness_peak_ms(rms_db: List[float], hop_ms: int, s: int, e: int) -> int:
    """argmax rms_db over [s, e) -- the loudest voiced instant: louder =
    mouth most open. Midpoint fallback with no usable audio signal."""
    mid = (s + e) // 2
    if not rms_db or hop_ms <= 0:
        return mid
    lo, hi = max(0, s // hop_ms), min(len(rms_db) - 1, (e - 1) // hop_ms)
    if hi < lo:
        return mid
    return max(range(lo, hi + 1), key=lambda i: rms_db[i]) * hop_ms
