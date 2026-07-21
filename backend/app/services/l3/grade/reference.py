"""Robust per-group color reference (color_shot_matching.plan.md Phase 2a):
the MEDIAN member stats of a scene-group, used as the single target both
Balance (Phase 2b) and Match (Phase 4) pull every group member toward. A
median (not a picked 'anchor' shot, not a mean) is robust to one outlier
shot in the group and needs no maybe-empty quality signal.

Inputs are WORKING-SPACE scalars (already projected by the caller in
job.py); this module stays pure numeric with no tone.py dependency, same
convention as leveling.py."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional


@dataclass
class GroupReference:
    mid_gray: float
    black_point: float
    white_point: float
    rgb_mean: List[float]   # per-channel median [r, g, b]


def _median(values: List[float], default: float) -> float:
    vals = [v for v in values if v is not None]
    return float(median(vals)) if vals else default


def compute_group_reference(member_stats: List[Dict]) -> Optional[GroupReference]:
    """Median-member reference over a group's stats dicts (either
    display-space or working-space, depending on the caller -- Match wants
    display-space so its own `_proj` can project it; Balance wants
    working-space directly, see job.py). None for a <2-member group
    (nothing to converge)."""
    if len(member_stats) < 2:
        return None
    mids = [m.get("mid_gray") for m in member_stats]
    blacks = [m.get("black_point") for m in member_stats]
    whites = [m.get("white_point") for m in member_stats]
    rgbs = [m.get("rgb_mean") or [0.5, 0.5, 0.5] for m in member_stats]
    rgb_med = [_median([r[c] for r in rgbs], 0.5) for c in range(3)]
    return GroupReference(
        mid_gray=_median(mids, 0.5),
        black_point=_median(blacks, 0.0),
        white_point=_median(whites, 1.0),
        rgb_mean=rgb_med,
    )
