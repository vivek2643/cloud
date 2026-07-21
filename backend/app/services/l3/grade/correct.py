"""
Correct layer (color_grading.plan.md SS5): the first real intelligence in
the grade stack (SS3: Measure -> CORRECT -> Match -> Look -> Arc ->
Soft-local -> bake). Auto exposure/contrast via a never-worse levels
stretch, white balance via gray-world refined by a sanity-checked
white-patch candidate, and a mild log/flat pre-lift. Semantic-gated: skips
already-graded footage and anything with significant existing clipping
(nothing safe to correct there).

Skin-anchored WB (the plan's first-choice WB source when available) is
deliberately NOT implemented here: doing it well requires a skin-tone
reference that doesn't privilege one skin tone over others, and getting
that wrong is a real fairness problem, not an engineering shortcut to defer
casually. Gray-world + a sanity-checked white-patch candidate is the
unbiased, purely-mathematical default the plan's own risk notes call the
workhorse ("skin + gray-world + never-worse is the workhorse" -- SS16).

`white_reference` (SS2.3): `solve_correct_grade` accepts an optional
`white_reference_rgb` -- the MEAN sampled RGB of the region pass2b proposed,
already re-verified as neutral (`colorspace.is_neutral`) by the caller. When
given, it wins over gray-world/white-patch (the plan's stated priority: a
real neutral surface beats two statistical guesses). NOTE: this module does
NOT sample video pixels itself -- pass2b only proposes a region (normalized
x,y,w,h) + `hero_ts_ms` on `cut_records`; turning that into an actual mean
RGB requires decoding a specific frame at a specific crop, which is its own
follow-up pipeline piece (an L3 verification pass once cuts exist), not
something to do synchronously inside a document resolve. This function's
job is the (smaller, still real) other half: given a sampled color, decide
whether to trust it and how to use it.

Never-worse derivation (the one-directional levels stretch): map
[black_point, white_point] (measured) -> [target_low, target_high] where
target_low is TARGET_BLACK only if the measured black is ALREADY above it
(there's real headroom to safely tighten into it -- flat/log footage);
otherwise target_low = black_point itself, i.e. don't touch it (lifting a
near-true-black point risks revealing sensor noise, not detail). target_high
is symmetric. This means the stretch can only ever gain contrast from a
flatter-than-target source, never manufacture headroom that isn't there.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from app.services.l3.grade.cdl import Grade
from app.services.l3.grade.colorspace import is_neutral, lab_to_srgb
from app.services.l3.grade.tone import WORKING_SPACE_V1, to_working_scalar

TARGET_BLACK = 0.02
TARGET_WHITE = 0.97
MID_GRAY_PIVOT = 0.5
MAX_CLIP_PCT_FOR_STRETCH = 0.08   # skip the levels stretch above this much existing clipping
WB_MULTIPLIER_TRUST_MAX = 2.0     # a white-patch candidate this non-uniform isn't neutral
WB_MULTIPLIER_CLAMP = 1.5         # cap any single-channel WB gain (never-worse: don't overcorrect)
LOG_FLAT_PRE_LIFT = 1.06          # gentle extra contrast on top of the levels stretch for flat input
# Hard ceiling on the levels-stretch contrast. Low-contrast-but-CORRECT footage
# (e.g. a dim indoor podcast: black~0.15, white~0.65) is statistically almost
# indistinguishable from true log, so `is_log_flat` will misfire on it. An
# uncapped stretch to full [0.02, 0.97] range then produces a ~2.2x slope with a
# big negative offset -> crushed shadows / blown highlights on footage that was
# fine. Capping the slope (blacks stay anchored to target, highlights just relax
# toward full range instead of snapping to it) makes the correction never-worse
# regardless of whether the log/flat guess was right.
LEVELS_SLOPE_MAX = 1.5
# color_grading_upgrade.plan.md Step 1.3: a natural on-screen mid-gray target
# (a touch below the classic 18%-card 0.46 sRGB convention, so the nudge never
# reads as washed-out/over-brightened).
TARGET_MID_GRAY = 0.42
# The mid-gray retarget is a SMALL extra nudge on top of the already-anchored
# black/white stretch, not a second independent correction -- bounded tight
# so it can't fight the levels stretch or blow past LEVELS_SLOPE_MAX on its own.
MID_GRAY_EXTRA_CLAMP = 1.2

# color_skin_vibrance.plan.md S4.3: skin-anchored tint correction. Human skin
# across all tones clusters along a line through the origin in the Lab a*/b*
# plane at this hue angle (+a* red, +b* yellow); what varies legitimately
# between people/lighting is position ALONG that line (warmth) and L*
# (lightness) -- never touched. What's never natural on skin is displacement
# PERPENDICULAR to it, the green<->magenta cast a bad/mixed white balance
# adds -- that's the only thing corrected, so no skin tone is privileged.
SKIN_LOCUS_DEG = 50.0
SKIN_TINT_STRENGTH = 0.7     # remove this fraction of the perpendicular (tint) residual
SKIN_WB_WEIGHT = 0.5         # skin gets a VOTE in WB, not a veto (blended with gray-world)
SKIN_L_MIN, SKIN_L_MAX = 20.0, 92.0   # plausible skin lightness; outside -> not skin, skip
SKIN_MIN_CHROMA = 3.0        # near-neutral sample -> not a confident skin read, skip
SKIN_MAX_PERP = 25.0         # residual bigger than this -> not skin (colored object), skip

# color_skin_vibrance.plan.md S4.4: vibrance normalization. A bounded global
# saturation lift toward a target mean Lab chroma -- raises the floor on
# lifeless log/flat clips, never touches already-vivid footage.
TARGET_CHROMA = 22.0     # target mean Lab chroma; below -> boost, at/above -> leave
SAT_BOOST_MAX = 1.25     # hard cap: a global sat lift past this over-saturates skin/reds


def _compose(s1: float, o1: float, s2: float, o2: float) -> Tuple[float, float]:
    """Chain two linear ops (apply op1 then op2): out = (in*s1+o1)*s2+o2."""
    return s1 * s2, o1 * s2 + o2


def _project(value: float, working_space: str) -> float:
    """Project a DISPLAY scalar into the space the CDL is actually applied.
    v1 bakes the CDL BETWEEN `to_working` (display->linear) and `from_working`
    (lut_bake.py), so a levels solve targeting display anchors must be solved
    on the LINEARIZED points to round-trip correctly (otherwise a
    display-space negative offset gets subtracted from a linearized midtone and
    crushes it -- the "everything too dark" bug). Legacy solves in display
    space exactly as before -- identity here (no float32 round-trip, so
    byte-for-byte unchanged)."""
    if working_space != WORKING_SPACE_V1:
        return value
    return to_working_scalar(value, None, WORKING_SPACE_V1)


def _skin_multiplier(skin_lab: Optional[Any]) -> Optional[Tuple[float, float, float]]:
    """color_skin_vibrance.plan.md S4.3: given a skin `[L*, a*, b*]` sample,
    decide whether it's a confident skin read and, if so, return the
    per-channel RGB multiplier that pulls its off-locus (perpendicular) tint
    residual toward the universal skin locus -- WITHOUT touching along-locus
    warmth or L* (the fairness-safe half: never a skin-tone target). None
    when the sample doesn't gate as skin (implausible lightness, too neutral
    to read confidently, or too far off-locus to be skin rather than a
    colored object) -- the caller then casts no skin vote at all."""
    if not skin_lab:
        return None
    try:
        L, a, b = (float(v) for v in skin_lab)
    except (TypeError, ValueError):
        return None
    if not (SKIN_L_MIN <= L <= SKIN_L_MAX):
        return None
    if math.hypot(a, b) < SKIN_MIN_CHROMA:
        return None
    theta = math.radians(SKIN_LOCUS_DEG)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    r_par = a * cos_t + b * sin_t
    d_perp = -a * sin_t + b * cos_t
    if abs(d_perp) >= SKIN_MAX_PERP:
        return None
    d_perp_target = d_perp * (1.0 - SKIN_TINT_STRENGTH)
    a_target = r_par * cos_t - d_perp_target * sin_t
    b_target = r_par * sin_t + d_perp_target * cos_t

    eps = 1e-6
    measured_rgb = lab_to_srgb((L, a, b))
    target_rgb = lab_to_srgb((L, a_target, b_target))
    m = [target_rgb[c] / max(eps, measured_rgb[c]) for c in range(3)]
    clamped = tuple(max(1.0 / WB_MULTIPLIER_CLAMP, min(WB_MULTIPLIER_CLAMP, float(v))) for v in m)
    return clamped  # type: ignore[return-value]


def _solve_wb(
    color_stats: Dict[str, Any], white_reference_rgb: Optional[Tuple[float, float, float]],
    skin_lab: Optional[Any] = None,
) -> Tuple[float, float, float]:
    if white_reference_rgb is not None and is_neutral(white_reference_rgb):
        # A verified real neutral surface beats two statistical guesses --
        # same white-patch formula (this IS a white patch, just one a human/
        # VLM pointed at instead of "brightest pixels"). Wins over the skin
        # vote too, same priority as gray-world/white-patch.
        eps = 1e-6
        peak = max(white_reference_rgb)
        wb = [peak / max(eps, c) for c in white_reference_rgb]
        clamped = [max(1.0 / WB_MULTIPLIER_CLAMP, min(WB_MULTIPLIER_CLAMP, float(v))) for v in wb]
        return clamped[0], clamped[1], clamped[2]

    gray_world = color_stats.get("wb_gray_world") or [1.0, 1.0, 1.0]
    patch = color_stats.get("wb_white_patch") or [1.0, 1.0, 1.0]
    wb = list(gray_world)
    if patch and min(patch) > 1e-6 and max(patch) / min(patch) < WB_MULTIPLIER_TRUST_MAX:
        # Two independent estimates agreeing raises confidence; averaging
        # damps either one's individual error rather than trusting one alone.
        wb = [(gray_world[i] + patch[i]) / 2.0 for i in range(3)]
    skin_m = _skin_multiplier(skin_lab)
    if skin_m is not None:
        # A VOTE, not a veto: blended in at SKIN_WB_WEIGHT so gray-world stays
        # the temperature workhorse and skin only nudges the tint it's blind to.
        wb = [wb[i] * (1.0 + (skin_m[i] - 1.0) * SKIN_WB_WEIGHT) for i in range(3)]
    clamped = [max(1.0 / WB_MULTIPLIER_CLAMP, min(WB_MULTIPLIER_CLAMP, float(v))) for v in wb]
    return clamped[0], clamped[1], clamped[2]


def _solve_levels(
    color_stats: Dict[str, Any], clip_shadow: float, clip_highlight: float,
    working_space: str = "rec709",
) -> Tuple[float, float]:
    black = float(color_stats.get("black_point") if color_stats.get("black_point") is not None else 0.0)
    white = float(color_stats.get("white_point") if color_stats.get("white_point") is not None else 1.0)
    if clip_shadow > MAX_CLIP_PCT_FOR_STRETCH or clip_highlight > MAX_CLIP_PCT_FOR_STRETCH or white <= black:
        return 1.0, 0.0
    # Pick the never-worse display-space anchors first (the >/< comparisons are
    # monotone under to_working, so the CHOICE is space-independent), then solve
    # slope/offset in whichever space the CDL is applied (see `_project`).
    target_low = TARGET_BLACK if black > TARGET_BLACK else black
    target_high = TARGET_WHITE if white < TARGET_WHITE else white
    black = _project(black, working_space)
    white = _project(white, working_space)
    target_low = _project(target_low, working_space)
    target_high = _project(target_high, working_space)
    slope = (target_high - target_low) / max(1e-4, (white - black))
    slope = min(slope, LEVELS_SLOPE_MAX)
    # Re-anchor with the (possibly capped) slope so blacks still land on
    # target_low; the highlight target simply relaxes below target_high rather
    # than the stretch snapping the whole range open.
    offset = target_low - black * slope
    return slope, offset


def _solve_levels_v1(
    color_stats: Dict[str, Any], clip_shadow: float, clip_highlight: float,
    working_space: str = WORKING_SPACE_V1,
) -> Tuple[float, float]:
    """Percentile-based exposure (Step 1.3): start from the SAME never-worse
    black/white anchoring `_solve_levels` already does, then ALSO nudge the
    resulting `mid_gray` toward `TARGET_MID_GRAY` -- a small, tightly-clamped
    extra multiplier, re-anchored the same way (blacks stay pinned to
    target, `LEVELS_SLOPE_MAX` re-applied if the composed slope would exceed
    it). This catches a shot whose black/white points are already fine (so
    the legacy stretch alone is a no-op) but whose overall exposure still
    reads dim/bright -- the case percentile-only correction can't reach."""
    slope, offset = _solve_levels(color_stats, clip_shadow, clip_highlight, working_space=working_space)
    mid = color_stats.get("mid_gray")
    if mid is None:
        return slope, offset
    # slope/offset already live in `working_space`; project the mid measurement
    # and its target into the SAME space so the nudge is solved consistently.
    mid = _project(float(mid), working_space)
    target_mid = _project(TARGET_MID_GRAY, working_space)
    projected_mid = mid * slope + offset
    if projected_mid <= 1e-6:
        return slope, offset
    extra = max(1.0 / MID_GRAY_EXTRA_CLAMP, min(MID_GRAY_EXTRA_CLAMP, target_mid / projected_mid))
    new_slope = slope * extra
    black = float(color_stats.get("black_point") if color_stats.get("black_point") is not None else 0.0)
    target_low = TARGET_BLACK if black > TARGET_BLACK else black
    black = _project(black, working_space)
    target_low = _project(target_low, working_space)
    if new_slope > LEVELS_SLOPE_MAX:
        new_slope = LEVELS_SLOPE_MAX
    new_offset = target_low - black * new_slope
    return new_slope, new_offset


def solve_correct_grade(
    color_stats: Optional[Dict[str, Any]],
    *,
    already_graded: bool = False,
    white_reference_rgb: Optional[Tuple[float, float, float]] = None,
    pipeline: str = "legacy",
    skin_vibrance: bool = False,
) -> Grade:
    """Deterministic correction from ONE file's `color_stats` row (see
    `grade.measure.fetch_color_stats`), optionally refined by a verified
    `white_reference_rgb` sample (SS2.3 -- see module docstring for what
    "verified" means and what's not built yet). Identity when there's
    nothing safe to do: no measurement, already-graded footage (semantic
    gate), or footage with significant existing clipping (never-worse gate).
    `pipeline=="v1"` (color_grading_upgrade.plan.md Step 1.3) additionally
    targets a mid-gray placement via `_solve_levels_v1`; `legacy` is the
    untouched black/white-only stretch.

    `skin_vibrance` (color_skin_vibrance.plan.md, v1-only): also folds a
    skin-anchored tint vote into the WB solve (`subject_lab` -- the
    face-region sample from `measure_span`, when a subject box was resolved
    -- preferred over `skin_lab`'s center-weighted proxy; neither present ->
    no skin vote) and lifts `sat` toward `TARGET_CHROMA` when `chroma_mean`
    reads low (missing/high chroma -> `sat=1.0`, never-worse). Off (default)
    -> byte-identical to today: no skin vote, `sat=1.0`."""
    if not color_stats or already_graded:
        return Grade()

    clip_shadow = float(color_stats.get("clip_shadow_pct") or 0.0)
    clip_highlight = float(color_stats.get("clip_highlight_pct") or 0.0)
    black = float(color_stats.get("black_point") if color_stats.get("black_point") is not None else 0.0)

    working_space = WORKING_SPACE_V1 if pipeline == "v1" else "rec709"
    skin_active = skin_vibrance and pipeline == "v1"
    skin_lab = (color_stats.get("subject_lab") or color_stats.get("skin_lab")) if skin_active else None
    wb_r, wb_g, wb_b = _solve_wb(color_stats, white_reference_rgb, skin_lab=skin_lab)
    levels_fn = _solve_levels_v1 if pipeline == "v1" else _solve_levels
    luma_slope, luma_offset = levels_fn(color_stats, clip_shadow, clip_highlight, working_space=working_space)

    if color_stats.get("is_log_flat") and clip_shadow <= MAX_CLIP_PCT_FOR_STRETCH:
        # Pivot-preserving around REC709's standard mid-gray target so the
        # extra lift doesn't drag overall brightness with it. Under v1 the
        # levels op above is in working (linear) space, so the pivot must be
        # too -- otherwise the lift composes across two different spaces.
        lift_slope = LOG_FLAT_PRE_LIFT
        lift_offset = _project(MID_GRAY_PIVOT, working_space) * (1.0 - LOG_FLAT_PRE_LIFT)
        luma_slope, luma_offset = _compose(luma_slope, luma_offset, lift_slope, lift_offset)
        # The lift composes ON TOP of the already-capped stretch, so the product
        # can exceed the ceiling again -- re-clamp, re-anchoring blacks to target.
        if luma_slope > LEVELS_SLOPE_MAX:
            target_low = TARGET_BLACK if black > TARGET_BLACK else black
            luma_slope = LEVELS_SLOPE_MAX
            luma_offset = _project(target_low, working_space) - _project(black, working_space) * luma_slope

    sat = 1.0
    if skin_active:
        chroma = color_stats.get("chroma_mean")
        if chroma is not None:
            chroma = float(chroma)
            if chroma > 1e-6:
                # Only ever >= 1.0 -- never desaturate; already-vivid footage
                # (chroma >= TARGET_CHROMA) leaves sat at 1.0 unchanged.
                sat = max(1.0, min(SAT_BOOST_MAX, TARGET_CHROMA / chroma))

    slope = (wb_r * luma_slope, wb_g * luma_slope, wb_b * luma_slope)
    offset = (luma_offset, luma_offset, luma_offset)
    return Grade(slope=slope, offset=offset, power=(1.0, 1.0, 1.0), sat=sat)
