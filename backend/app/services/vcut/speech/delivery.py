"""
speech_cuts_pipeline.plan.md section 10 -- Stage 5 (part 1): per-take
delivery metrics from audio (+ optional visual), group-normalized within
each take group ("best of these, not an absolute bar"). PURE (no I/O).

pace is the one term scored against a FIXED natural band (PACE_LO..
PACE_HI) rather than group-normalized -- section 10's own wording singles
it out ("scored against a natural band") separately from the others.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.services.vcut.speech.inputs import Word
from app.services.vcut.speech.params import (
    HESITATION_GAP_MS, MIN_VOICED_FRAC, MIN_VOICED_SPREAD_DB, PACE_HI, PACE_LO,
    W_DYNAMICS, W_ENERGY, W_HESITATION, W_PACE, W_VISUAL,
)


def is_voiced_beat(energy_db: float, voiced_ref_db: float, floor_ref_db: float) -> bool:
    """True iff a beat's median energy reads as real speech, not a
    hallucination over ambient noise. Self-calibrating against the file's own
    loud/quiet references (params.RMS_*_PCTL): a beat clears the gate when its
    median sits at least MIN_VOICED_FRAC of the way up from the noise floor to
    the voiced level. Fail-open when the file's loud-vs-quiet spread is too
    small to discriminate (uniform level, or no rms data -> refs 0.0/0.0)."""
    spread = voiced_ref_db - floor_ref_db
    if spread < MIN_VOICED_SPREAD_DB:
        return True
    return energy_db >= floor_ref_db + MIN_VOICED_FRAC * spread


@dataclass
class TakeMetrics:
    energy: float                          # raw median rms_db over the span
    dynamics: float                        # raw stdev of rms_db over the span
    pace_wps: float                        # raw non-filler words/sec
    hesitation_ms: float                   # raw total intra-span gap time > HESITATION_GAP_MS
    visual_delivery: Optional[float] = None  # 0..1 from frames.py, or None (unavailable)


def _median(values: List[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _stdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _rms_slice(rms_db: List[float], hop_ms: int, in_ms: int, out_ms: int) -> List[float]:
    if not rms_db or hop_ms <= 0:
        return []
    lo_i = max(0, in_ms // hop_ms)
    hi_i = min(len(rms_db) - 1, out_ms // hop_ms)
    if lo_i > hi_i:
        return []
    return rms_db[lo_i:hi_i + 1]


def compute_take_metrics(
    words: List[Word], in_ms: int, out_ms: int, rms_db: List[float], rms_hop_ms: int,
    visual_delivery: Optional[float] = None,
) -> TakeMetrics:
    """``words``: every word in the beat's span (already sliced by the
    caller -- boundaries.py's word_span, not re-derived here)."""
    span_words = [w for w in words if w.start_ms >= in_ms and w.end_ms <= out_ms]
    speaking_words = [w for w in span_words if not w.is_filler]

    rms_slice = _rms_slice(rms_db, rms_hop_ms, in_ms, out_ms)
    energy = _median(rms_slice)
    dynamics = _stdev(rms_slice)

    duration_s = max(0.001, (out_ms - in_ms) / 1000.0)
    pace_wps = len(speaking_words) / duration_s

    hesitation_ms = 0.0
    for a, b in zip(span_words, span_words[1:]):
        gap = b.start_ms - a.end_ms
        if gap > HESITATION_GAP_MS:
            hesitation_ms += gap

    return TakeMetrics(
        energy=energy, dynamics=dynamics, pace_wps=pace_wps, hesitation_ms=hesitation_ms,
        visual_delivery=visual_delivery,
    )


def _pace_score(wps: float) -> float:
    """1.0 inside [PACE_LO, PACE_HI], linear falloff outside, floored at 0."""
    if PACE_LO <= wps <= PACE_HI:
        return 1.0
    if wps < PACE_LO:
        return max(0.0, 1.0 - (PACE_LO - wps) / PACE_LO)
    return max(0.0, 1.0 - (wps - PACE_HI) / PACE_HI)


def _minmax_normalize(values: List[float]) -> List[float]:
    """0..1 within the group; a degenerate (all-equal) group scores everyone
    1.0 -- no discriminating signal means no one is worse than anyone else."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def group_delivery_scores(metrics: List[TakeMetrics]) -> List[float]:
    """One fused delivery score per take (same order as ``metrics``)."""
    if not metrics:
        return []

    energy_n = _minmax_normalize([m.energy for m in metrics])
    dynamics_n = _minmax_normalize([m.dynamics for m in metrics])
    pace_n = [_pace_score(m.pace_wps) for m in metrics]
    # fewer/shorter hesitations = better -> negate before normalizing so a
    # HIGHER normalized value always means better, matching every other term.
    hesitation_n = _minmax_normalize([-m.hesitation_ms for m in metrics])

    visual_raw = [m.visual_delivery for m in metrics]
    if all(v is None for v in visual_raw):
        visual_n = [0.0] * len(metrics)  # no visual signal at all -> the term contributes nothing
    else:
        visual_n = _minmax_normalize([v if v is not None else 0.0 for v in visual_raw])

    scores = []
    for e, d, p, h, v in zip(energy_n, dynamics_n, pace_n, hesitation_n, visual_n):
        scores.append(
            W_ENERGY * e + W_DYNAMICS * d + W_PACE * p + W_HESITATION * h + W_VISUAL * v
        )
    return scores
