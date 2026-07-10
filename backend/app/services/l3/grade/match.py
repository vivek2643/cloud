"""
Match layer (color_grading.plan.md SS6, Fork D "grade-groups"): deterministic
clustering of SOURCE FILES by measured color similarity, so footage from the
same scene/lighting setup grades consistently instead of each file drifting
independently. Anchor = highest quality in the group; other members get a
CONSERVATIVE delta nudging their measured color toward the anchor's, never a
full match (a full match would erase real differences between genuinely
different shots that just happen to cluster).

Deliberately file-level, not cut-level: `color_stats` (the only numeric,
non-free-text similarity signal available) is measured per FILE (SS2.2), and
a continuous single-camera shoot already wants consistent grading across all
its cuts by construction. This also sidesteps needing a scene-continuity
signal whose exact shape/reliability this pass hasn't verified -- clustering
on a signal we fully control and understand beats guessing at one we don't.
`total_quality` (per-cut, from `cut_records`) is optional and only used to
break ties in anchor selection when available; clustering itself never needs
it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.grade.cdl import Grade

RGB_DIST_MAX = 0.12       # mean-RGB Euclidean distance (0..1 scale) to group two files
MATCH_STRENGTH = 0.4      # conservative: nudge 40% of the way toward the anchor, never all the way


def _rgb_dist(a: List[float], b: List[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def cluster_grade_groups(
    color_stats_by_file: Dict[str, Dict[str, Any]],
) -> List[List[str]]:
    """Greedy single-link clustering of file_ids by `rgb_mean` distance.
    Deterministic given a stable iteration order (sorted file_ids)."""
    ids = sorted(f for f, cs in color_stats_by_file.items() if cs and cs.get("rgb_mean"))
    groups: List[List[str]] = []
    assigned: Dict[str, int] = {}
    for fid in ids:
        rgb = color_stats_by_file[fid]["rgb_mean"]
        best_group: Optional[int] = None
        best_dist = RGB_DIST_MAX
        for gi, members in enumerate(groups):
            # Compare against the group's centroid (mean of members so far).
            centroid = [
                sum(color_stats_by_file[m]["rgb_mean"][c] for m in members) / len(members)
                for c in range(3)
            ]
            d = _rgb_dist(rgb, centroid)
            if d < best_dist:
                best_dist = d
                best_group = gi
        if best_group is None:
            groups.append([fid])
            assigned[fid] = len(groups) - 1
        else:
            groups[best_group].append(fid)
            assigned[fid] = best_group
    return groups


def _quality_for(file_id: str, total_quality_by_file: Dict[str, float]) -> float:
    return total_quality_by_file.get(file_id, 0.0)


def solve_match_deltas(
    color_stats_by_file: Dict[str, Dict[str, Any]],
    total_quality_by_file: Optional[Dict[str, float]] = None,
) -> Dict[str, Grade]:
    """file_id -> a conservative CDL delta nudging that file's measured color
    toward its group's anchor (the highest-quality file in a group with 2+
    members; ties broken by file_id for determinism). The anchor itself
    always gets identity -- it's what the rest of the group matches TO."""
    total_quality_by_file = total_quality_by_file or {}
    groups = cluster_grade_groups(color_stats_by_file)
    out: Dict[str, Grade] = {}

    for members in groups:
        if len(members) < 2:
            continue
        anchor = max(members, key=lambda f: (_quality_for(f, total_quality_by_file), f))
        anchor_rgb = color_stats_by_file[anchor]["rgb_mean"]
        for fid in members:
            if fid == anchor:
                continue
            member_rgb = color_stats_by_file[fid]["rgb_mean"]
            eps = 1e-6
            # Per-channel multiplier that would fully match anchor's mean,
            # damped by MATCH_STRENGTH so it's a nudge, not a replacement.
            full_slope = [anchor_rgb[c] / max(eps, member_rgb[c]) for c in range(3)]
            slope = tuple(1.0 + (s - 1.0) * MATCH_STRENGTH for s in full_slope)
            out[fid] = Grade(slope=slope, offset=(0.0, 0.0, 0.0), power=(1.0, 1.0, 1.0), sat=1.0)
    return out
