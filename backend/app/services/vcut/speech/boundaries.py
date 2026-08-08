"""
speech_cuts_pipeline.plan.md section 9 -- Stage 4: word-index range -> exact
ms. PURE (no I/O) so a beat's boundaries are reproducible and re-derivable
without re-calling the LLM.

Conceptually mirrors app.services.l3.lattice._snap_word_edge (find the
overlapping silence in the gap, snap to its midpoint) but reimplemented
locally (vcut stays isolated from L3 business logic) with ONE addition the
old function doesn't have: the extension is bounded to BREATH_PAD_MS, never
unbounded -- so a very long real silence still yields a snug (not sprawling)
cut, and an edge with no BREATH_PAD_MS-reachable silence at all is left at
its raw word boundary rather than reaching arbitrarily far.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from app.services.vcut.speech.inputs import Word
from app.services.vcut.speech.params import BREATH_PAD_MS


def _overlapping_silence(
    lo_ms: int, hi_ms: int, silences: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """The first silence interval overlapping [lo_ms, hi_ms), clamped to
    that window -- None if no silence reaches into the window at all."""
    for s, e in silences:
        if e > lo_ms and s < hi_ms:
            return (max(s, lo_ms), min(e, hi_ms))
    return None


def _snap_edge_left(raw_in: int, floor_ms: int, silences: List[Tuple[int, int]]) -> int:
    """Extend the IN edge earlier into silence, up to BREATH_PAD_MS, never
    past ``floor_ms`` (the preceding word's own end -- never clips it)."""
    window_lo = max(floor_ms, raw_in - BREATH_PAD_MS)
    sil = _overlapping_silence(window_lo, raw_in, silences)
    if sil is None:
        return raw_in
    s, e = sil
    mid = (s + e) // 2
    return max(window_lo, min(raw_in, mid))


def _snap_edge_right(raw_out: int, ceiling_ms: int, silences: List[Tuple[int, int]]) -> int:
    """Mirror of _snap_edge_left: extend later, up to BREATH_PAD_MS, never
    past ``ceiling_ms`` (the following word's own start)."""
    window_hi = min(ceiling_ms, raw_out + BREATH_PAD_MS)
    sil = _overlapping_silence(raw_out, window_hi, silences)
    if sil is None:
        return raw_out
    s, e = sil
    mid = (s + e) // 2
    return min(window_hi, max(raw_out, mid))


def compute_boundaries_ms(
    word_span: Tuple[int, int], words: List[Word], silences: List[Tuple[int, int]],
    duration_ms: int,
) -> Tuple[int, int]:
    """[i, j] (inclusive word indices) -> (in_ms, out_ms), breath-padded
    into adjacent silence, never crossing into a neighboring word."""
    i, j = word_span
    i = max(0, min(i, len(words) - 1))
    j = max(i, min(j, len(words) - 1))

    raw_in = words[i].start_ms
    raw_out = words[j].end_ms
    prev_end = words[i - 1].end_ms if i > 0 else 0
    next_start = words[j + 1].start_ms if j + 1 < len(words) else max(raw_out, duration_ms)

    in_ms = _snap_edge_left(raw_in, prev_end, silences)
    out_ms = _snap_edge_right(raw_out, next_start, silences)
    return in_ms, out_ms
