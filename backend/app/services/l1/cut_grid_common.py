"""
Shared, dependency-free helpers for the L1 cut-grid channels.

Two recurring shapes:

  * "hit" channels (BEAT, ACTION) -- a set of discrete instants you want to cut
    ON (beat onsets, motion impacts). ``hit_cost_curve`` turns those instants
    into a dense cost curve that dips to 0 exactly on each hit and ramps back to
    1 within a tolerance window (triangular well). Off-hit => avoid (cost 1).

  * "avoid" channels (DIALOGUE, CAMERA) -- a dense magnitude you want to be LOW
    when you cut. ``normalize_pctl`` robustly maps raw magnitudes to 0..1 using a
    percentile (resistant to a few huge spikes).

Pure Python lists in / out so this stays importable without numpy on the
caller side (motion_dynamics still uses numpy internally for the flow math).
"""
from __future__ import annotations

from typing import List


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def n_hops(duration_ms: int, hop_ms: int) -> int:
    if duration_ms <= 0 or hop_ms <= 0:
        return 0
    return int(duration_ms // hop_ms) + 1


def hit_cost_curve(
    hits_ms: List[int],
    duration_ms: int,
    hop_ms: int,
    tol_ms: int,
) -> List[float]:
    """Dense cost curve for a "cut ON these instants" channel.

    cost = 1 everywhere, dipping linearly to 0 at each hit and back up to 1 over
    +/- ``tol_ms`` (a triangular well). Overlapping wells take the minimum
    (deepest) cost. Returns one value per hop, rounded to 3 dp.
    """
    n = n_hops(duration_ms, hop_ms)
    if n == 0:
        return []
    cost = [1.0] * n
    if not hits_ms or tol_ms <= 0:
        return cost
    span = float(tol_ms)
    for h in hits_ms:
        lo = int((h - tol_ms) // hop_ms)
        hi = int((h + tol_ms) // hop_ms)
        for i in range(max(0, lo), min(n - 1, hi) + 1):
            t = i * hop_ms
            well = abs(t - h) / span        # 0 at the hit, 1 at the edge
            if well < cost[i]:
                cost[i] = well
    return [round(clamp01(c), 3) for c in cost]


def percentile(values: List[float], pctl: float) -> float:
    """Linear-interpolated percentile (pctl in 0..100). Empty -> 0."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    rank = (pctl / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def normalize_pctl(values: List[float], pctl: float) -> List[float]:
    """Map magnitudes to 0..1 by dividing by their ``pctl`` percentile.

    Percentile (not max) so a handful of outliers don't flatten everything.
    Values above the percentile clamp to 1. Rounded to 3 dp.
    """
    if not values:
        return []
    ref = percentile(values, pctl)
    if ref <= 0:
        return [0.0] * len(values)
    return [round(clamp01(v / ref), 3) for v in values]


def local_maxima(
    values: List[float],
    hop_ms: int,
    floor: float,
    min_gap_ms: int,
) -> List[int]:
    """Indices->timestamps of local maxima above ``floor``, min ``min_gap_ms``
    apart (keeping the stronger peak when two are too close)."""
    n = len(values)
    if n == 0:
        return []
    cand: List[tuple] = []  # (value, ts_ms)
    for i in range(n):
        v = values[i]
        if v < floor:
            continue
        left = values[i - 1] if i > 0 else -1.0
        right = values[i + 1] if i < n - 1 else -1.0
        if v >= left and v >= right:
            cand.append((v, i * hop_ms))
    cand.sort(key=lambda x: -x[0])  # strongest first
    chosen: List[int] = []
    for _, ts in cand:
        if all(abs(ts - c) >= min_gap_ms for c in chosen):
            chosen.append(ts)
    chosen.sort()
    return chosen
