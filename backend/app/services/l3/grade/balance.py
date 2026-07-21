"""Balance layer (color_shot_matching.plan.md Phase 2b): the missing
'shot match' step. Per scene-group, pull each member's exposure, white
balance, and contrast toward a ROBUST group reference (reference.py) so a
multi-file reel converges shot-to-shot. Runs once per document in
run_grade_job (v1 only), composed BEFORE Match in resolver.py.

Works entirely on WORKING-SPACE scalars (projected by the caller). The
exposure move is a PIVOT GAIN + OFFSET (not slope-only): the composite
slope ceiling (resolver.COMPOSITE_SLOPE_MAX) would otherwise cap a very
dull shot's lift; a pivot-at-black gain adds a positive offset that lifts
midtones and survives the slope clamp, without crushing shadows (the
positive offset never trips COMPOSITE_MID_FLOOR, which only floors negative
offsets)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.l3.grade.cdl import Grade, compose
from app.services.l3.grade.reference import GroupReference

# How hard Balance converges each axis toward the group reference. Higher =
# tighter shot-to-shot consistency, at the cost of erasing real intra-scene
# variation. Exposure is the dominant inconsistency axis, so it converges
# hardest; WB/contrast are gentler because they compose on top.
BALANCE_EXPOSURE_STRENGTH = 0.8
BALANCE_WB_STRENGTH = 0.6
BALANCE_CONTRAST_STRENGTH = 0.5

# Never-worse ceilings (a single shot can't be gained/contrasted past these
# multipliers toward the reference, so one wild member can't drag the math).
BALANCE_WB_CLAMP = 1.4
BALANCE_CONTRAST_CLAMP = 1.5


def _pivot_gain_offset(value: float, target: float, black: float,
                       strength: float) -> Tuple[float, float]:
    """slope/offset (pivot at `black`) moving `value` toward `target` by
    `strength`. out = in*slope + offset, with offset = black*(1 - slope) so
    the black point is preserved and midtones lift via the offset term
    (survives the composite SLOPE ceiling)."""
    if value <= 1e-6:
        return 1.0, 0.0
    goal = value + (target - value) * strength
    slope = goal / value
    offset = black * (1.0 - slope)
    return slope, offset


def solve_balance(
    ordered_stats: List[Optional[Dict]],
    groups: List[List[int]],
    references: Dict[int, GroupReference],
    keys: List[str],
) -> Dict[str, Grade]:
    """shot_key -> a Balance delta toward its group's reference. `groups` is
    the SAME grouping match uses (list of index lists into ordered_stats);
    `references[gi]` is that group's GroupReference (None-groups skipped);
    `keys[i]` is shot i's key. Members of singleton/no-reference groups get
    no delta."""
    out: Dict[str, Grade] = {}
    for gi, idxs in enumerate(groups):
        ref = references.get(gi)
        if ref is None or len(idxs) < 2:
            continue
        for i in idxs:
            stats = ordered_stats[i] or {}
            mid = stats.get("mid_gray")
            black = float(stats.get("black_point") or 0.0)
            white = float(stats.get("white_point") or 1.0)
            rgb = stats.get("rgb_mean") or [0.5, 0.5, 0.5]

            # 1) exposure (pivot at black)
            if mid is not None:
                es, eo = _pivot_gain_offset(float(mid), ref.mid_gray, black,
                                            BALANCE_EXPOSURE_STRENGTH)
            else:
                es, eo = 1.0, 0.0
            exposure = Grade(slope=(es, es, es), offset=(eo, eo, eo))

            # 2) white balance (per-channel gain toward the reference cast)
            eps = 1e-6
            wb = []
            for c in range(3):
                full = ref.rgb_mean[c] / max(eps, rgb[c])
                g = 1.0 + (full - 1.0) * BALANCE_WB_STRENGTH
                wb.append(max(1.0 / BALANCE_WB_CLAMP, min(BALANCE_WB_CLAMP, g)))
            # normalize so WB doesn't change overall exposure (green channel = 1)
            wb = [w / wb[1] for w in wb]
            white_balance = Grade(slope=(wb[0], wb[1], wb[2]))

            # 3) contrast (range toward reference range, black held)
            shot_range = max(1e-4, white - black)
            ref_range = max(1e-4, ref.white_point - ref.black_point)
            full_cs = ref_range / shot_range
            cs = 1.0 + (full_cs - 1.0) * BALANCE_CONTRAST_STRENGTH
            cs = max(1.0 / BALANCE_CONTRAST_CLAMP, min(BALANCE_CONTRAST_CLAMP, cs))
            co = black * (1.0 - cs)
            contrast = Grade(slope=(cs, cs, cs), offset=(co, co, co))

            delta = compose(compose(exposure, white_balance, 1.0), contrast, 1.0)
            out[keys[i]] = delta
    return out
