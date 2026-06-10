"""
L1 derived signal: dialogue cut-cost grid.

A cheap, deterministic, CPU-only derivation over signals L1 already persists
(Whisper word timings + fillers + diarization speakers + the audio pause map /
RMS envelope). No model, no GPU -- pure arithmetic over arrays.

What it produces
----------------
1. ``cut_cost``  -- a dense per-hop curve (default hop = 100 ms, the minimum
   expected cut granularity). Semantics:

       0.0  = ideal seam (cheap, clean place to cut)
       1.0  = forbidden  (mid-word -- cutting here clips speech)

   "Safe to cut" in the user's framing is simply ``1 - cut_cost``. We store the
   cost form because the editor MINIMIZES it when placing cuts, and costs
   compose (total edit cost = sum of seam costs).

   The curve is NOT flat-zero during speech: the valuable dialogue seams live in
   the gaps *between* words. So cost ~= 1 inside a word, then dips toward 0 in
   each inter-word gap -- deeper for longer gaps and for stronger boundaries
   (sentence ends, speaker changes, fillers). A small protected "handle" near
   each word edge keeps cuts from landing flush against speech (breathing room).

2. ``cut_points`` -- discrete, exact-timestamp seam candidates (not quantized to
   the grid; Whisper word edges are ~20-50 ms precise). Each carries the gap it
   sits in so a later stage can place asymmetric in/out points or J/L cuts
   (audio leading or trailing the picture).

This is a dialogue signal. Silent / music-only files yield an empty grid
(``has_dialogue = False``); beat- and motion-driven cut grids are a separate,
later generalization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Tunables live in one place so we can re-tune at the end of the project
# against real clips. See cut_grid_params.py.
from app.services.l1.cut_grid_params import (  # noqa: E402
    FILLER_EDGE_MULT,
    HANDLE_MS,
    HOP_MS,
    MIN_SEAM_MS,
    SENTENCE_END_MULT,
    SENTENCE_GAP_MS,
    SILENCE_FULL_MS,
    SPEAKER_CHANGE_MULT,
    TERMINAL_PUNCT,
    WORD_COST,
)


# --- Data structures ------------------------------------------------------

@dataclass
class CutPoint:
    ts_ms: int            # best instant to cut within the seam (energy trough)
    gap_start_ms: int     # seam region start (end of the previous word)
    gap_end_ms: int       # seam region end (start of the next word)
    kind: str             # word_gap|sentence_end|speaker_change|pause|filler_edge
    score: float          # 0..1, higher = cheaper/cleaner seam (= 1 - floor cost)

    def to_dict(self) -> Dict:
        return {
            "ts_ms": self.ts_ms,
            "gap_start_ms": self.gap_start_ms,
            "gap_end_ms": self.gap_end_ms,
            "kind": self.kind,
            "score": round(self.score, 3),
        }


@dataclass
class DialogueCutGrid:
    hop_ms: int
    cut_cost: List[float] = field(default_factory=list)
    cut_points: List[CutPoint] = field(default_factory=list)
    has_dialogue: bool = False

    def cost_payload(self) -> List[float]:
        """Rounded curve for compact JSONB storage."""
        return [round(c, 3) for c in self.cut_cost]

    def points_payload(self) -> List[Dict]:
        return [p.to_dict() for p in self.cut_points]


# --- Internal helpers -----------------------------------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _sorted_words(words: List[dict]) -> List[dict]:
    """Keep only real, positive-duration words, sorted by start time."""
    out = []
    for w in words or []:
        try:
            s = int(w.get("start_ms", 0))
            e = int(w.get("end_ms", 0))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        out.append({
            "start_ms": s,
            "end_ms": e,
            "text": str(w.get("text", "")),
            "is_filler": bool(w.get("is_filler", False)),
            "speaker": w.get("speaker"),
        })
    out.sort(key=lambda w: (w["start_ms"], w["end_ms"]))
    return out


def _min_energy_ts(
    rms_db: List[float],
    hop_ms: int,
    lo_ms: int,
    hi_ms: int,
    fallback_ms: int,
) -> int:
    """Quietest instant in [lo_ms, hi_ms] per the RMS envelope; the natural
    place to land a cut. Falls back to the interval midpoint when the envelope
    is unavailable."""
    if not rms_db or hop_ms <= 0 or hi_ms <= lo_ms:
        return fallback_ms
    i0 = max(0, lo_ms // hop_ms)
    i1 = min(len(rms_db) - 1, hi_ms // hop_ms)
    if i1 < i0:
        return fallback_ms
    best_i = i0
    best_v = rms_db[i0]
    for i in range(i0 + 1, i1 + 1):
        v = rms_db[i]
        if v < best_v:
            best_v = v
            best_i = i
    return best_i * hop_ms


def _is_sentence_end(text: str, gap_ms: int) -> bool:
    t = (text or "").rstrip()
    return (bool(t) and t[-1] in TERMINAL_PUNCT) or gap_ms >= SENTENCE_GAP_MS


def _seam_floor_and_kind(prev: dict, nxt: dict, gap_ms: int) -> tuple[float, str]:
    """Length-based cost floor for a gap, reduced by boundary strength, plus the
    dominant boundary kind for labelling the discrete cut point."""
    # Longer gap -> lower floor (more room to cut cleanly).
    floor = _clamp01(1.0 - gap_ms / SILENCE_FULL_MS)

    speaker_change = (
        prev.get("speaker") is not None
        and nxt.get("speaker") is not None
        and prev.get("speaker") != nxt.get("speaker")
    )
    filler_edge = prev.get("is_filler") or nxt.get("is_filler")
    sentence_end = _is_sentence_end(prev.get("text", ""), gap_ms)

    if speaker_change:
        floor *= SPEAKER_CHANGE_MULT
    if sentence_end:
        floor *= SENTENCE_END_MULT
    if filler_edge:
        floor *= FILLER_EDGE_MULT

    # Kind precedence: most editorially meaningful label wins.
    if speaker_change:
        kind = "speaker_change"
    elif filler_edge:
        kind = "filler_edge"
    elif gap_ms >= SILENCE_FULL_MS:
        kind = "pause"
    elif sentence_end:
        kind = "sentence_end"
    else:
        kind = "word_gap"

    return _clamp01(floor), kind


# --- Public API -----------------------------------------------------------

def compute_dialogue_cut_grid(
    words: List[dict],
    rms_db: Optional[List[float]] = None,
    prosody_hop_ms: int = 0,
    duration_ms: int = 0,
    hop_ms: int = HOP_MS,
) -> DialogueCutGrid:
    """
    Build the dialogue cut-cost grid.

    Parameters
    ----------
    words : flattened transcript words, each ``{start_ms, end_ms, text,
            is_filler, speaker}`` (``speaker`` optional; present after L1
            diarization).
    rms_db : coarse energy envelope (dB) sampled every ``prosody_hop_ms``; used
            only to snap each discrete cut point to the quietest instant in its
            gap. Optional.
    prosody_hop_ms : sample hop for ``rms_db``.
    duration_ms : total media duration; bounds the curve length. When 0 we fall
            back to the last word's end time.
    hop_ms : grid resolution (default 100 ms).
    """
    words = _sorted_words(words)
    if not words:
        return DialogueCutGrid(hop_ms=hop_ms, has_dialogue=False)

    if duration_ms <= 0:
        duration_ms = words[-1]["end_ms"]
    n = max(1, math.ceil(duration_ms / hop_ms))

    # Silence is freely cuttable -> baseline cost 0. Words overwrite with 1.0.
    cost = [0.0] * n

    def _set(idx: int, value: float) -> None:
        if 0 <= idx < n:
            cost[idx] = value

    # 1) Forbid mid-word hops.
    for w in words:
        i0 = w["start_ms"] // hop_ms
        i1 = w["end_ms"] // hop_ms
        for i in range(i0, i1 + 1):
            _set(i, WORD_COST)

    cut_points: List[CutPoint] = []

    # 2) Carve a U-shaped dip into every inter-word gap and emit a cut point.
    for prev, nxt in zip(words[:-1], words[1:]):
        a = prev["end_ms"]
        b = nxt["start_ms"]
        gap = b - a
        if gap <= 0:
            continue

        floor, kind = _seam_floor_and_kind(prev, nxt, gap)

        # Shape the gap: cost rises from `floor` at the centre back up toward
        # WORD_COST within HANDLE_MS of either word edge (protect breathing room).
        i0 = a // hop_ms
        i1 = b // hop_ms
        for i in range(i0, i1 + 1):
            t = i * hop_ms
            if t <= a or t >= b:
                continue  # leave word hops at WORD_COST
            d = min(t - a, b - t)
            if d <= HANDLE_MS:
                c = floor + (WORD_COST - floor) * (1.0 - d / HANDLE_MS)
            else:
                c = floor
            _set(i, _clamp01(c))

        if gap >= MIN_SEAM_MS:
            lo = a + HANDLE_MS
            hi = b - HANDLE_MS
            mid = (a + b) // 2
            if hi <= lo:
                lo, hi = a, b
            ts = _min_energy_ts(rms_db or [], prosody_hop_ms, lo, hi, mid)
            cut_points.append(CutPoint(
                ts_ms=ts, gap_start_ms=a, gap_end_ms=b,
                kind=kind, score=_clamp01(1.0 - floor),
            ))

    # 3) Lead-in / lead-out silence are free seams too (clip start / end).
    first_start = words[0]["start_ms"]
    if first_start >= MIN_SEAM_MS:
        cut_points.insert(0, CutPoint(
            ts_ms=max(0, first_start - HANDLE_MS), gap_start_ms=0,
            gap_end_ms=first_start, kind="pause", score=1.0,
        ))
    last_end = words[-1]["end_ms"]
    if duration_ms - last_end >= MIN_SEAM_MS:
        cut_points.append(CutPoint(
            ts_ms=min(duration_ms, last_end + HANDLE_MS), gap_start_ms=last_end,
            gap_end_ms=duration_ms, kind="pause", score=1.0,
        ))

    cut_points.sort(key=lambda p: p.ts_ms)
    return DialogueCutGrid(
        hop_ms=hop_ms,
        cut_cost=cost,
        cut_points=cut_points,
        has_dialogue=True,
    )
