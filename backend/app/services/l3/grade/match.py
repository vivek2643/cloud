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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.grade.cdl import Grade, compose
from app.services.l3.grade.reference import GroupReference
from app.services.l3.grade.tone import WORKING_SPACE_V1, to_working_scalar

RGB_DIST_MAX = 0.12       # mean-RGB Euclidean distance (0..1 scale) to group two files
MATCH_STRENGTH = 0.4      # conservative: nudge 40% of the way toward the anchor, never all the way


def _rgb_dist(a: List[float], b: List[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _proj(value: Optional[float], working_space: str) -> Optional[float]:
    """Project a DISPLAY scalar into the space the match delta is APPLIED.
    Under v1 the composed CDL runs between `to_working` and `from_working`
    (lut_bake.py), so a levels/cast delta solved on raw display span stats
    would be applied to LINEARIZED values and mis-anchor (the same
    display-vs-linear mismatch that crushed the correct layer). Legacy is
    identity (no float32 round-trip -> byte-for-byte unchanged). `None` passes
    through so the mid-gray nudge's optional-input handling is preserved."""
    if value is None or working_space != WORKING_SPACE_V1:
        return value
    return to_working_scalar(value, None, WORKING_SPACE_V1)


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


# --------------------------------------------------------------------------
# color_grading_upgrade.plan.md Step 1.4: TIMELINE-AWARE sequence matching.
#
# Replaces the whole-document, whole-file clustering above (kept for
# `legacy`) with a NEIGHBOR-based match over each shot's own SPAN stats
# (grade.measure_span, Step 1.2): two shots only ever match when they're
# ADJACENT in program order (or share a file_id) AND their span-level color
# is close -- never a global "these two files happen to look similar"
# cluster, which is exactly the failure mode this step fixes (it could drag
# together two unrelated scenes that just happen to share a palette).
# --------------------------------------------------------------------------

# Same distance family as RGB_DIST_MAX, applied to per-SPAN (not whole-file)
# rgb_mean -- span stats are the more precise signal (Step 1.2), so the same
# threshold now catches matches whole-file measurement diluted away.
SPAN_RGB_DIST_MAX = 0.12
# color_shot_matching.plan.md Phase 4a: with Balance (job.py) now owning the
# bulk of exposure convergence toward the same group reference, Match's
# strengths were raised so the residual placement/cast spread actually
# closes instead of leaving ~60%/70% of it in place. Safe because the
# composite guardrails (resolver.COMPOSITE_SLOPE_MAX/MID_FLOOR) still bound
# the final stacked CDL regardless. Only consumed by `solve_sequence_match`
# (the v1-only path) -- legacy's `solve_match_deltas` uses the separate
# `MATCH_STRENGTH` constant above and is unaffected.
SPAN_MATCH_STRENGTH = 0.85    # was 0.4 -- converge placement hard toward the group reference
CAST_MATCH_STRENGTH = 0.6     # was 0.3 -- lift cast convergence for genuine same-scene groups
CAST_CLAMP = 1.3              # never-worse ceiling on the per-channel cast multiplier
MID_MATCH_CLAMP = 1.5         # was 1.2 -- allow a real exposure convergence, still bounded


@dataclass
class ShotStats:
    """One timeline shot's identity + its span (or file-fallback) color
    measurement, in PROGRAM order -- what `solve_sequence_match` groups and
    matches. `quality` is optional (defaults to 0 -- ties then break on
    `key` alone, same determinism `cluster_grade_groups`'s anchor pick uses)."""
    key: str            # seg_id or op_id
    file_id: str
    stats: Optional[Dict[str, Any]] = field(default=None)
    quality: float = 0.0


def group_neighbors(ordered_shots: List[ShotStats]) -> List[List[int]]:
    """Chain-link ADJACENT shots into groups: shot i joins the running group
    if it shares a `file_id` with the immediately preceding shot (same
    continuous source -- always groups) OR its span `rgb_mean` sits within
    `SPAN_RGB_DIST_MAX` of the preceding shot's (a genuine same-scene/setup
    cut, e.g. a multicam angle change). A shot with no qualifying neighbor
    starts its own group of 1 (skipped downstream -- nothing to match it
    to). Deliberately chain-linked to the PREVIOUS shot, not a fixed group
    anchor, so gradual continuity (a slow lighting drift across many cuts of
    one continuous scene) doesn't fracture the group, while two DISTANT,
    merely similar-looking shots (never adjacent) can never be pulled
    together -- the failure mode whole-file clustering had."""
    groups: List[List[int]] = []
    for i, shot in enumerate(ordered_shots):
        rgb = (shot.stats or {}).get("rgb_mean")
        if groups:
            prev = ordered_shots[groups[-1][-1]]
            prev_rgb = (prev.stats or {}).get("rgb_mean")
            same_file = bool(shot.file_id) and shot.file_id == prev.file_id
            close = bool(rgb and prev_rgb) and _rgb_dist(rgb, prev_rgb) < SPAN_RGB_DIST_MAX
            if same_file or close:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def _levels_delta_toward(
    m_black: float, m_white: float, a_black: float, a_white: float,
    m_mid: Optional[float], a_mid: Optional[float], strength: float,
) -> Tuple[float, float]:
    """Percentile-based slope/offset nudging a member's black/white/mid-gray
    placement toward the anchor's, damped by `strength`. Re-anchors on the
    (damped) target black point every time a slope adjustment changes it --
    same re-anchoring discipline `correct.py`'s levels solver uses -- so the
    black point always lands exactly where intended regardless of how many
    nudges compose."""
    target_black = m_black + (a_black - m_black) * strength
    if m_white <= m_black:
        return 1.0, target_black - m_black
    full_slope = (a_white - a_black) / max(1e-4, (m_white - m_black))
    slope = 1.0 + (full_slope - 1.0) * strength
    offset = target_black - m_black * slope
    if m_mid is not None and a_mid is not None:
        projected_mid = m_mid * slope + offset
        target_mid = m_mid + (a_mid - m_mid) * strength
        if projected_mid > 1e-6:
            extra = max(1.0 / MID_MATCH_CLAMP, min(MID_MATCH_CLAMP, target_mid / projected_mid))
            slope *= extra
            offset = target_black - m_black * slope
    return slope, offset


def solve_sequence_match(
    ordered_shots: List[ShotStats], groups: Optional[List[List[int]]] = None,
    working_space: str = "rec709", references: Optional[Dict[int, GroupReference]] = None,
) -> Dict[str, Grade]:
    """shot_key -> a conservative CDL delta nudging that shot's SPAN-measured
    color toward its neighbor-group's anchor (the highest-quality span in a
    run of 2+ adjacent/same-file shots; ties broken by `key`). The anchor
    itself gets no delta. Percentile-based (black/white/mid-gray placement)
    PLUS a damped per-channel cast nudge, composed -- see module docstring
    for why grouping is neighbor-only, not global clustering.

    `groups` (color_grading_upgrade.plan.md Step 3.2, optional): a
    pre-computed grouping (`grade.scene_group.group_shots_semantically`) to
    use INSTEAD of the default RGB-based `group_neighbors` -- lets matching
    align shots that are the same scene BY MEANING even when a transient
    (a bright object entering) skews their RGB. None (default) keeps Step
    1.4's behavior exactly.

    `working_space` (v1): the space the resulting delta is APPLIED in at bake
    time. GROUPING stays in display space (a perceptual "same scene?" gate,
    unchanged by this arg), but the slope/offset/cast DELTAS are solved on
    working-space-projected span stats so they anchor correctly through the
    v1 bake wrapper (see `_proj`). Legacy default solves in display space.

    `references` (color_shot_matching.plan.md Phase 3, optional): group-index
    -> a robust `GroupReference` (DISPLAY-space, same as `_proj` expects --
    see job.py's comment on building two references per group to avoid
    double-projection) to match EVERY member toward, replacing the
    single-shot `max(quality, key)` anchor -- and replacing "the anchor is
    exempt" with "every member gets a delta," since there's no longer a
    distinguished member. `None` (default) reproduces today's anchor-based
    behavior exactly, byte-for-byte -- only `run_grade_job` (v1, when
    `settings.grade_shot_match_v2`) passes `references`."""
    out: Dict[str, Grade] = {}
    for gi, idxs in enumerate(groups if groups is not None else group_neighbors(ordered_shots)):
        if len(idxs) < 2:
            continue
        members = [ordered_shots[i] for i in idxs]
        ref = (references or {}).get(gi)
        if ref is not None:
            a_black = _proj(float(ref.black_point), working_space)
            a_white = _proj(float(ref.white_point), working_space)
            a_mid = _proj(float(ref.mid_gray), working_space)
            a_rgb = [_proj(float(c), working_space) for c in ref.rgb_mean]
            anchor_key = None   # no member is "the anchor" -- every member matches the reference
        else:
            anchor = max(members, key=lambda s: (s.quality, s.key))
            a = anchor.stats or {}
            a_black = _proj(float(a.get("black_point") if a.get("black_point") is not None else 0.0), working_space)
            a_white = _proj(float(a.get("white_point") if a.get("white_point") is not None else 1.0), working_space)
            a_mid = a.get("mid_gray")
            a_mid = _proj(float(a_mid), working_space) if a_mid is not None else None
            a_rgb = [_proj(float(c), working_space) for c in (a.get("rgb_mean") or [0.5, 0.5, 0.5])]
            anchor_key = anchor.key

        for s in members:
            if anchor_key is not None and s.key == anchor_key:
                continue
            m = s.stats or {}
            m_black = _proj(float(m.get("black_point") if m.get("black_point") is not None else 0.0), working_space)
            m_white = _proj(float(m.get("white_point") if m.get("white_point") is not None else 1.0), working_space)
            m_mid = m.get("mid_gray")
            m_mid = _proj(float(m_mid), working_space) if m_mid is not None else None
            m_rgb = [_proj(float(c), working_space) for c in (m.get("rgb_mean") or [0.5, 0.5, 0.5])]

            luma_slope, luma_offset = _levels_delta_toward(
                m_black, m_white, a_black, a_white, m_mid, a_mid, SPAN_MATCH_STRENGTH,
            )
            luma = Grade(slope=(luma_slope,) * 3, offset=(luma_offset,) * 3)

            eps = 1e-6
            cast_slope = tuple(
                max(1.0 / CAST_CLAMP, min(CAST_CLAMP,
                    1.0 + ((a_rgb[c] / max(eps, m_rgb[c])) - 1.0) * CAST_MATCH_STRENGTH))
                for c in range(3)
            )
            cast = Grade(slope=cast_slope)

            out[s.key] = compose(luma, cast, 1.0)
    return out
