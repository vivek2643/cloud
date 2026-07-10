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

from typing import Any, Dict, Optional, Tuple

from app.services.l3.grade.cdl import Grade
from app.services.l3.grade.colorspace import is_neutral

TARGET_BLACK = 0.02
TARGET_WHITE = 0.97
MID_GRAY_PIVOT = 0.5
MAX_CLIP_PCT_FOR_STRETCH = 0.08   # skip the levels stretch above this much existing clipping
WB_MULTIPLIER_TRUST_MAX = 2.0     # a white-patch candidate this non-uniform isn't neutral
WB_MULTIPLIER_CLAMP = 1.5         # cap any single-channel WB gain (never-worse: don't overcorrect)
LOG_FLAT_PRE_LIFT = 1.10          # mild extra contrast on top of the levels stretch for flat input


def _compose(s1: float, o1: float, s2: float, o2: float) -> Tuple[float, float]:
    """Chain two linear ops (apply op1 then op2): out = (in*s1+o1)*s2+o2."""
    return s1 * s2, o1 * s2 + o2


def _solve_wb(
    color_stats: Dict[str, Any], white_reference_rgb: Optional[Tuple[float, float, float]]
) -> Tuple[float, float, float]:
    if white_reference_rgb is not None and is_neutral(white_reference_rgb):
        # A verified real neutral surface beats two statistical guesses --
        # same white-patch formula (this IS a white patch, just one a human/
        # VLM pointed at instead of "brightest pixels").
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
    clamped = [max(1.0 / WB_MULTIPLIER_CLAMP, min(WB_MULTIPLIER_CLAMP, float(v))) for v in wb]
    return clamped[0], clamped[1], clamped[2]


def _solve_levels(color_stats: Dict[str, Any], clip_shadow: float, clip_highlight: float) -> Tuple[float, float]:
    black = float(color_stats.get("black_point") if color_stats.get("black_point") is not None else 0.0)
    white = float(color_stats.get("white_point") if color_stats.get("white_point") is not None else 1.0)
    if clip_shadow > MAX_CLIP_PCT_FOR_STRETCH or clip_highlight > MAX_CLIP_PCT_FOR_STRETCH or white <= black:
        return 1.0, 0.0
    target_low = TARGET_BLACK if black > TARGET_BLACK else black
    target_high = TARGET_WHITE if white < TARGET_WHITE else white
    slope = (target_high - target_low) / max(1e-4, (white - black))
    offset = target_low - black * slope
    return slope, offset


def solve_correct_grade(
    color_stats: Optional[Dict[str, Any]],
    *,
    already_graded: bool = False,
    white_reference_rgb: Optional[Tuple[float, float, float]] = None,
) -> Grade:
    """Deterministic correction from ONE file's `color_stats` row (see
    `grade.measure.fetch_color_stats`), optionally refined by a verified
    `white_reference_rgb` sample (SS2.3 -- see module docstring for what
    "verified" means and what's not built yet). Identity when there's
    nothing safe to do: no measurement, already-graded footage (semantic
    gate), or footage with significant existing clipping (never-worse gate)."""
    if not color_stats or already_graded:
        return Grade()

    clip_shadow = float(color_stats.get("clip_shadow_pct") or 0.0)
    clip_highlight = float(color_stats.get("clip_highlight_pct") or 0.0)

    wb_r, wb_g, wb_b = _solve_wb(color_stats, white_reference_rgb)
    luma_slope, luma_offset = _solve_levels(color_stats, clip_shadow, clip_highlight)

    if color_stats.get("is_log_flat") and clip_shadow <= MAX_CLIP_PCT_FOR_STRETCH:
        # Pivot-preserving around REC709's standard mid-gray target so the
        # extra lift doesn't drag overall brightness with it.
        lift_slope = LOG_FLAT_PRE_LIFT
        lift_offset = MID_GRAY_PIVOT * (1.0 - LOG_FLAT_PRE_LIFT)
        luma_slope, luma_offset = _compose(luma_slope, luma_offset, lift_slope, lift_offset)

    slope = (wb_r * luma_slope, wb_g * luma_slope, wb_b * luma_slope)
    offset = (luma_offset, luma_offset, luma_offset)
    return Grade(slope=slope, offset=offset, power=(1.0, 1.0, 1.0), sat=1.0)
