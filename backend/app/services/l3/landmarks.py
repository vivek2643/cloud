"""
brain_perception_upgrade.plan.md Change 1, Mechanism A -- the four interior-
structure landmark channels (act / adx / sil / shot), extracted here as a
pure, dependency-free module so BOTH cut producers compute them from ONE
source of truth:

- the old L3/v4 pipeline (l3.post.assemble_cut_records), and
- the new vcut pipeline's salience bridge (vcut.salience.build_landmarks),

per brain_cut_salience_parity.plan.md's full-landmark-parity extension. This
kills the copy-paste drift the earlier act-only vcut bridge risked: the
channel math lives here once, and each pipeline supplies its own persisted L1
signals (rms_db / silence_intervals / shot_points+composition_points) to it.

No I/O, no model, stdlib only -- so vcut can import it without breaking its
isolation contract (same spirit as vcut already importing v4_segment_params).

A compact, code-owned distillation of INTERIOR structure within a cut's own
span -- action hits, audio rise/fall change points, silence gaps, and internal
shot/composition cuts. Any channel with zero interior events is OMITTED
entirely, so a fully static/silent cut lands as {} -- no breadcrumb, no wasted
bytes. This is the SAME contract footage_map._landmarks_tag reads
(landmarks[ch]["n"]) and observe.inspect_cut warms.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_LANDMARK_ACT_CAP = 5
_LANDMARK_ADX_CAP = 5
_LANDMARK_SIL_CAP = 5
_LANDMARK_SHOT_CAP = 8
# Data-derived (not an absolute dB constant): a normalized-domain floor so a
# perfectly flat clip (mean_abs_diff == 0) doesn't flag every sampling
# wiggle as a "change" -- see _adx_changes.
_ADX_MIN_DELTA = 0.12


# --------------------------------------------------------------------------
# Small generic helpers -- self-contained here (not imported from l3.post)
# so this module has no l3.post dependency and stays cheaply importable by
# vcut. These are 1-3 line clip-relative normalizers, duplicated across the
# codebase already (feel.py/framing.py/layers.py each carry their own
# _clamp01); it's the CHANNEL logic below that must stay single-sourced.
# --------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def series_lohi(arr: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Min/max of a signal series (ignoring None), for clip-relative
    normalisation. (None, None) when there's nothing usable. Public because
    both pipelines need it to derive the rms lo/hi the adx channel takes."""
    vals = [float(v) for v in (arr or []) if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def _norm_in_clip(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    if value is None or lo is None or hi is None or hi <= lo:
        return None
    return _clamp01((value - lo) / (hi - lo))


def _interior_edge_guard(span_ms: int) -> int:
    """Mirrors footage_map._peak_tag's edge band: ~1 hop (L1's own
    granularity, 100ms) or 10% of the span, whichever is larger -- a
    landmark pinned to the cut's own boundary is just that edge restated,
    not real interior structure."""
    return max(100, span_ms // 10)


def _is_interior(off_ms: int, span_ms: int, edge_guard: int) -> bool:
    return edge_guard <= off_ms <= span_ms - edge_guard


def _act_hits(
    action_energy: List[float], action_points: List[Dict[str, Any]], hop_ms: int,
    s: int, e: int, edge_guard: int,
) -> Optional[Dict[str, Any]]:
    """Interior local maxima of `action_energy` AND/OR `action_points`
    (subject-motion impacts) that fall inside the span, deduped to one
    candidate per hop bin (an action_point landing on a local max shouldn't
    double-count), ranked by the raw action_energy value at that bin (a
    within-span ranking -- scores are never surfaced, only used to pick the
    top-K), then reported in TIME order."""
    span_ms = e - s
    if hop_ms <= 0 or span_ms <= 0:
        return None
    candidates: Dict[int, float] = {}  # bin index -> score

    if action_energy and len(action_energy) >= 3:
        lo_i, hi_i = s // hop_ms, max(s // hop_ms, (e - 1) // hop_ms)
        for i in range(max(1, lo_i), min(len(action_energy) - 1, hi_i + 1)):
            # Strict on both sides -- a flat/plateau run (>= would flag EVERY
            # point on a perfectly flat curve as its own "local max") has no
            # real interior structure at all.
            if action_energy[i] > action_energy[i - 1] and action_energy[i] > action_energy[i + 1]:
                off = i * hop_ms - s
                if _is_interior(off, span_ms, edge_guard):
                    candidates[i] = max(candidates.get(i, 0.0), action_energy[i])

    for pt in (action_points or []):
        try:
            ts = int(pt.get("ts_ms", 0))
        except (TypeError, ValueError):
            continue
        if not (s <= ts < e):
            continue
        off = ts - s
        if not _is_interior(off, span_ms, edge_guard):
            continue
        i = ts // hop_ms
        fallback = action_energy[i] if 0 <= i < len(action_energy) else 1.0
        candidates[i] = max(candidates.get(i, 0.0), fallback)

    if not candidates:
        return None
    n = len(candidates)
    top = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:_LANDMARK_ACT_CAP]
    offsets = sorted(i * hop_ms - s for i, _score in top)
    return {"n": n, "hits": offsets}


def _adx_changes(
    rms_db: List[float], rms_hop_ms: int, rms_lo: Optional[float], rms_hi: Optional[float],
    s: int, e: int, edge_guard: int,
) -> Optional[Dict[str, Any]]:
    """Significant rises/falls of the clip-relative-normalized rms envelope
    (SAME `_norm_in_clip`/`series_lohi` normalization the quality scores
    use -- no absolute-dB constants). A change point is a consecutive-bin
    delta that exceeds a threshold derived from THIS span's own diff
    distribution (floored at `_ADX_MIN_DELTA` so a flat span can't flag
    noise), ranked by magnitude for the top-K, reported in TIME order."""
    span_ms = e - s
    if not rms_db or rms_hop_ms <= 0 or span_ms <= 0:
        return None
    lo_i, hi_i = s // rms_hop_ms, max(s // rms_hop_ms, (e - 1) // rms_hop_ms)
    idxs = [i for i in range(lo_i, hi_i + 1) if 0 <= i < len(rms_db)]
    if len(idxs) < 2:
        return None
    curve = [_norm_in_clip(rms_db[i], rms_lo, rms_hi) for i in idxs]
    if any(v is None for v in curve):
        return None
    diffs = [curve[k + 1] - curve[k] for k in range(len(curve) - 1)]
    mean_abs = _mean([abs(d) for d in diffs]) or 0.0
    threshold = max(mean_abs * 1.5, _ADX_MIN_DELTA)
    candidates: List[Tuple[float, int, str]] = []
    for k, d in enumerate(diffs):
        if abs(d) < threshold:
            continue
        off = idxs[k + 1] * rms_hop_ms - s
        if not _is_interior(off, span_ms, edge_guard):
            continue
        candidates.append((abs(d), off, "up" if d > 0 else "down"))
    if not candidates:
        return None
    n = len(candidates)
    top = sorted(candidates, key=lambda c: c[0], reverse=True)[:_LANDMARK_ADX_CAP]
    top.sort(key=lambda c: c[1])
    return {"n": n, "changes": [{"off": off, "dir": d} for _score, off, d in top]}


def _sil_gaps(
    silence_intervals: List[dict], rms_hop_ms: int, s: int, e: int, edge_guard: int,
) -> Optional[Dict[str, Any]]:
    """`silence_intervals` clipped to the span, ranked by clipped duration
    (a longer gap is more worth surfacing) for the top-K, reported in TIME
    order. Ignores anything shorter than one `prosody_hop_ms` (L1's own
    sampling grain -- a shorter "gap" isn't a real measured silence)."""
    span_ms = e - s
    if not silence_intervals or span_ms <= 0:
        return None
    min_dur = max(1, rms_hop_ms)
    candidates: List[Tuple[int, int]] = []
    for gap in silence_intervals:
        try:
            g0, g1 = int(gap["start_ms"]), int(gap["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        c0, c1 = max(g0, s), min(g1, e)
        dur = c1 - c0
        if dur < min_dur:
            continue
        off = c0 - s
        if not _is_interior(off, span_ms, edge_guard):
            continue
        candidates.append((dur, off))
    if not candidates:
        return None
    n = len(candidates)
    top = sorted(candidates, key=lambda c: c[0], reverse=True)[:_LANDMARK_SIL_CAP]
    top.sort(key=lambda c: c[1])
    return {"n": n, "gaps": [{"off": off, "dur": dur} for dur, off in top]}


def _shot_cuts(
    shot_points: List[Dict[str, Any]], composition_points: List[Dict[str, Any]],
    s: int, e: int, edge_guard: int,
) -> Optional[Dict[str, Any]]:
    """`shot_points` (hard=true) + `composition_points` (hard=false)
    strictly inside `(s, e)` and past the edge guard -- an edge "shot cut"
    is just this cut's own boundary. No magnitude to rank by (every internal
    cut is structurally equal); kept in TIME order, capped at K."""
    span_ms = e - s
    if span_ms <= 0:
        return None
    candidates: List[Tuple[int, bool]] = []
    for pt in (shot_points or []):
        try:
            ts = int(pt.get("ts_ms", 0))
        except (TypeError, ValueError):
            continue
        off = ts - s
        if s < ts < e and _is_interior(off, span_ms, edge_guard):
            candidates.append((off, True))
    for pt in (composition_points or []):
        try:
            ts = int(pt.get("ts_ms", 0))
        except (TypeError, ValueError):
            continue
        off = ts - s
        if s < ts < e and _is_interior(off, span_ms, edge_guard):
            candidates.append((off, False))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    n = len(candidates)
    kept = candidates[:_LANDMARK_SHOT_CAP]
    return {"n": n, "cuts": [{"off": off, "hard": hard} for off, hard in kept]}


def _landmarks(
    action_energy: List[float], action_points: List[Dict[str, Any]], hop_ms: int,
    rms_db: List[float], rms_hop_ms: int, rms_lo: Optional[float], rms_hi: Optional[float],
    silence_intervals: List[dict], shot_points: List[Dict[str, Any]],
    composition_points: List[Dict[str, Any]], s: int, e: int,
) -> Dict[str, Any]:
    """Compute all four landmark channels for one cut's span. Pure,
    deterministic -- see the module docstring above for the shape and
    rationale. {} when the cut has no interior structure on any channel
    (or a degenerate/zero-length span)."""
    span_ms = e - s
    if span_ms <= 0:
        return {}
    edge_guard = _interior_edge_guard(span_ms)
    out: Dict[str, Any] = {}
    act = _act_hits(action_energy, action_points, hop_ms, s, e, edge_guard)
    if act:
        out["act"] = act
    adx = _adx_changes(rms_db, rms_hop_ms, rms_lo, rms_hi, s, e, edge_guard)
    if adx:
        out["adx"] = adx
    sil = _sil_gaps(silence_intervals, rms_hop_ms, s, e, edge_guard)
    if sil:
        out["sil"] = sil
    shot = _shot_cuts(shot_points, composition_points, s, e, edge_guard)
    if shot:
        out["shot"] = shot
    return out
