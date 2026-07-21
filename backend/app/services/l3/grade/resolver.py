"""
The deterministic grade resolver (color_grading.plan.md SS3): turns a
timeline segment/operation's explicit `grade` override (if any) plus
whatever measurement/look/arc context is available into the ONE small,
JSON-safe descriptor (`{cdl, creative_lut_ref, working_space, grade_hash}`)
that gets baked into `document["resolved"]`, same place `layers.py`
already bakes geometric `transform`.

Stack order (SS3): Measure -> Correct -> Balance -> Match -> Leveling ->
Look -> Arc -> Soft-local -> NL trims -> BAKE. This module is the seam
every later build step plugs into:
  - SS5 (correct) is wired: `grade.correct.solve_correct_grade` composes
    onto the stack first, using the file's `color_stats` row.
  - Balance (color_shot_matching.plan.md Phase 2b, v1-only) composes next:
    a pre-computed per-shot delta pulling exposure/white-balance/contrast
    toward its scene-group's robust reference (`grade.balance.solve_balance`,
    `grade.reference.compute_group_reference`) -- the missing "shot match"
    step that Match alone (a placement + cast nudge, not a full exposure
    convergence) never fully closed. Same "computed once per document,
    passed in as a delta" pattern as match/leveling.
  - SS6 (match) is wired: a pre-computed per-file delta (see
    `grade.match.solve_match_deltas`, computed ONCE per document resolve in
    `layers.resolve` since grade-groups are a whole-document clustering, not
    a per-clip decision) composes next.
  - Leveling (color_grading_upgrade.plan.md Phase 2, gated on `settings.
    grade_even_lighting`) composes next: a pre-computed per-shot bounded
    exposure/tonal-placement nudge toward the sequence's smooth target (see
    `grade.leveling.solve_leveling`), same "computed once per document,
    passed in as a delta" pattern as match.
  - SS7 (look) is wired: `sequence_look.mode` selects ONE of the three input
    modes (SS7) -- a preset recipe, a reference-image transfer (needs
    pre-computed `reference_stats`, see `reference_transfer.py`), or a
    `.cube` upload (passed straight through as `creative_lut_ref`, never
    baked into the CDL delta -- it composes at bake time in `lut_bake.py`
    instead). No auto-pick: a document with no `look.mode` set gets no Look
    contribution at all, per the plan's explicit "no auto look-selection."
  - SS8 (arc) is wired: `item["arc_intent"]` (set by the `tag_arc_intent`
    verb, categorical only -- see `l3/act.py`) selects a delta from
    `arc.py`'s deterministic table, scaled by `sequence_look.arc_intensity`
    (0 = flat/invisible, the default, per the plan's "invisible by default").
  - The explicit per-clip override (`item["grade"]`, an NL trim or manual
    dial) is a DELTA composed on top of the whole stack via `cdl.compose`
    (SS8's amplitude-scaling semantics), not a replacement -- nudging
    "warmer" on one clip should adjust whatever auto-correction already
    computed, not erase it.

Never computes pixels itself -- produces a `Grade` (SS2.1's `cdl`) that
`lut_bake.py` turns into bytes only when something actually asks for them
(the cube endpoint, or the render compositor), so a timeline edit that
touches ten clips is ten cheap hash computations, not ten LUT bakes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.l3.grade.arc import solve_arc_grade
from app.services.l3.grade.cdl import Grade, compose, grade_hash
from app.services.l3.grade.correct import solve_correct_grade
from app.services.l3.grade.presets import get_preset
from app.services.l3.grade.reference_transfer import solve_reference_transfer
from app.services.l3.grade.softlocal import solve_vignette
from app.services.l3.grade.tone import WORKING_SPACE_V1, from_working, to_working

DEFAULT_WORKING_SPACE = "rec709"

# --------------------------------------------------------------------------
# Composite guardrails (color_grading_upgrade.plan.md -- v1 only).
#
# Every upstream layer bounds ITSELF (correct's LEVELS_SLOPE_MAX, match's
# CAST_CLAMP, leveling's stop caps), but Correct + Match + Leveling + the
# log-flat pre-lift COMPOSE MULTIPLICATIVELY, so the stacked CDL can exceed
# any single layer's cap (an observed composite slope ~2.65). These clamp the
# FINAL composed CDL so no combination of layers can over-contrast or crush
# shadows. They operate in the v1 working (linear) space -- the space the CDL
# actually runs in at bake time (tone.to_working -> apply_cdl -> from_working).
COMPOSITE_SLOPE_MAX = 2.0
# A nominal linear mid-gray = display 0.5 projected into the v1 working space
# (~0.214). The composed per-channel offset is floored so this mid-gray can
# never be pushed below COMPOSITE_MID_FLOOR (linear) -- i.e. the stack may
# darken, but can't collapse midtones/shadows to black, which was the original
# "everything too dark" failure (a display-space negative offset applied to a
# linearized midtone). Deliberately a low floor (well under a corrected mid's
# ~0.146 linear target), so it's an outlier backstop, not a per-clip tuning.
COMPOSITE_MID_FLOOR = 0.02


def _clamp_composite_v1(grade: Grade) -> Grade:
    """Apply the composite slope ceiling + negative-offset floor to a fully
    composed v1 CDL (see the module constants above for the reasoning)."""
    import numpy as np

    mid_lin = float(to_working(np.array([0.5], dtype=np.float32), WORKING_SPACE_V1)[0])
    slope = tuple(min(float(s), COMPOSITE_SLOPE_MAX) for s in grade.slope)
    # offset floored per channel so mid_lin*slope + offset >= COMPOSITE_MID_FLOOR
    offset = tuple(
        max(float(grade.offset[c]), COMPOSITE_MID_FLOOR - mid_lin * slope[c])
        for c in range(3)
    )
    return Grade(slope=slope, offset=offset, power=grade.power, sat=grade.sat)


def _corrected_source_stats(
    color_stats: Optional[Dict[str, Any]], stack: Grade, *, pipeline: str = "legacy"
) -> Optional[Dict[str, Any]]:
    """Project a file's measured `rgb_mean`/`rgb_std` through the correct+match
    `stack` so the Look layer solves against the image AS CORRECTED, not the raw
    source. Without this, a reference transfer computes its slope/offset from the
    raw means while the correct layer has already stretched exposure -- the two
    stack and DOUBLE-apply, crushing shadows / blowing highlights.

    correct+match only ever produce slope/offset (power=1, sat=1 by
    construction), so mean' = clamp(mean*slope+offset) and std' = std*slope is
    exact for that stack, not an approximation.

    `pipeline=="v1"` (color_grading_upgrade.plan.md Step 1.5): the CDL the
    stack composes into runs INSIDE the working-space wrapper at bake time
    (Step 1.1), so the mean projection is done there too -- linearize,
    apply slope/offset, re-encode -- keeping the corrected stats in the SAME
    space the reference image's own stats were measured in (display-encoded,
    `reference_transfer.compute_image_stats`), so the two sides of the
    transfer stay comparable. `std` stays a plain slope scale either way
    (the working-space encode/decode is nonlinear, so std doesn't project
    through it exactly -- an approximation already accepted for the legacy
    path too, not a new one)."""
    if not color_stats:
        return None
    mean = color_stats.get("rgb_mean") or [0.5, 0.5, 0.5]
    std = color_stats.get("rgb_std") or [0.2, 0.2, 0.2]
    if pipeline == "v1":
        import numpy as np

        working_mean = to_working(np.array(mean, dtype=np.float32), WORKING_SPACE_V1)
        corr_working = np.clip(
            working_mean * np.array(stack.slope, dtype=np.float32)
            + np.array(stack.offset, dtype=np.float32), 0.0, 1.0,
        )
        corr_mean = from_working(corr_working, WORKING_SPACE_V1).tolist()
    else:
        corr_mean = [min(1.0, max(0.0, mean[c] * stack.slope[c] + stack.offset[c])) for c in range(3)]
    corr_std = [max(0.0, std[c] * stack.slope[c]) for c in range(3)]
    return {**color_stats, "rgb_mean": corr_mean, "rgb_std": corr_std}


def _solve_look(sequence_look: Optional[Dict[str, Any]], color_stats: Optional[Dict[str, Any]]) -> Grade:
    if not sequence_look:
        return Grade()
    mode = sequence_look.get("mode")
    if mode == "preset":
        preset = get_preset(sequence_look.get("preset_id") or "")
        return preset.grade if preset else Grade()
    if mode == "reference":
        ref_stats = sequence_look.get("reference_stats")
        if not ref_stats or not color_stats:
            return Grade()
        strength = sequence_look.get("match_strength")
        kwargs = {"match_strength": float(strength)} if strength is not None else {}
        return solve_reference_transfer(color_stats, ref_stats, **kwargs)
    # mode == "lut" (or unset): the .cube itself composes at bake time via
    # creative_lut_ref, not as a CDL delta here.
    return Grade()


def resolve_clip_grade(
    item: Dict[str, Any],
    *,
    color_stats: Optional[Dict[str, Any]] = None,
    sequence_look: Optional[Dict[str, Any]] = None,
    already_graded: bool = False,
    match_delta: Optional[Grade] = None,
    balance_delta: Optional[Grade] = None,
    leveling_delta: Optional[Grade] = None,
    pipeline: str = "legacy",
) -> Dict[str, Any]:
    """Resolve ONE clip's (spine segment or op) final grade descriptor.

    `item` is the raw `EditSegment`/`EditOperation` dict; `item.get("grade")`
    is the per-clip explicit override (SS2.4) -- a delta nudge composed onto
    the stack, never a whole new creative LUT (that's the sequence-level
    Look, SS7). `already_graded` is the SS5 semantic gate ("skip
    already-graded footage") -- not yet wired to a real per-segment
    cut_records lookup (mapping an arbitrary trimmed EditSegment span back
    to its source cut is its own piece of work), so it defaults to False;
    callers that have it available should pass it through. `match_delta` is
    this clip's SS6 grade-groups delta (this file's nudge toward its group's
    anchor), already resolved once for the whole document by the caller.
    `balance_delta` (color_shot_matching.plan.md Phase 2b, v1-only): this
    shot's exposure/white-balance/contrast nudge toward its scene-group's
    robust median reference (`grade.balance.solve_balance`), already
    resolved once for the whole document -- composed BETWEEN Correct and
    Match so the two stages converge on the same target instead of fighting.
    `pipeline` (color_grading_upgrade.plan.md Step 1.1/1.3/1.5): "legacy"
    (default) is today's exact stack -- callers that never pass it get
    byte-identical output. "v1" selects the percentile-based correct layer,
    the `rec709_v1` working space (baked into the CDL's tone response at
    bake time), and projects the Look layer's corrected-source stats through
    that same working space. `leveling_delta` (Phase 2, gated by the caller
    on `settings.grade_even_lighting`) is this shot's bounded exposure/
    tonal-placement nudge toward the sequence's smooth target (`grade.
    leveling.solve_leveling`), already resolved once for the whole document
    -- composed between Match and Look, same pattern as `match_delta`, so
    the Look layer still solves against the fully-corrected-so-far image.
    """
    stack = solve_correct_grade(color_stats, already_graded=already_graded, pipeline=pipeline)
    if balance_delta is not None:
        stack = compose(stack, balance_delta, 1.0)
    if match_delta is not None:
        stack = compose(stack, match_delta, 1.0)
    if leveling_delta is not None:
        stack = compose(stack, leveling_delta, 1.0)
    # The Look layer sits on top of correct+match+leveling, so it must be solved
    # against the ALREADY-CORRECTED image (see _corrected_source_stats). Passing
    # the raw color_stats here is what made a reference drop double-stretch exposure.
    corrected_stats = _corrected_source_stats(color_stats, stack, pipeline=pipeline)
    stack = compose(stack, _solve_look(sequence_look, corrected_stats), 1.0)
    arc_intensity = (sequence_look or {}).get("arc_intensity")
    stack = compose(stack, solve_arc_grade(item.get("arc_intent"), arc_intensity), 1.0)

    override = Grade.from_dict(item.get("grade")) if item.get("grade") else None
    resolved = compose(stack, override, 1.0) if override is not None else stack

    # v1 only: bound the FINAL composed CDL so the multiplicatively-stacked
    # layers (correct+match+leveling+lift+override) can't over-contrast or
    # crush shadows regardless of how they combine (Fixes 2 & 3).
    if pipeline == "v1":
        resolved = _clamp_composite_v1(resolved)

    creative_lut_ref = (sequence_look or {}).get("lut_ref") if sequence_look else None
    if sequence_look and sequence_look.get("mode") != "lut":
        creative_lut_ref = None
    default_ws = WORKING_SPACE_V1 if pipeline == "v1" else DEFAULT_WORKING_SPACE
    working_space = item.get("working_space") or default_ws

    # SS9 soft-local: opt-in only (never a surprise vignette on untouched
    # footage) via sequence_look.vignette_strength. `item.get("subject_box")`
    # (color_grading_upgrade.plan.md Step 1.7: the masking-foundation seam)
    # is the normalized (x,y,w,h) box a caller that's already done the
    # segment->cut_records.framing.subject_box mapping can pass through --
    # no caller does that mapping yet in Phase 1, so this stays center-
    # anchored/absent in practice (no visual change), but resolve -> hash ->
    # bake all already carry it end-to-end for Phase 3 to wire for real.
    vignette_strength = (sequence_look or {}).get("vignette_strength")
    subject_box = item.get("subject_box")
    soft_local = None
    if vignette_strength:
        soft_local = {"vignette": solve_vignette(subject_box, strength=float(vignette_strength))}
        if subject_box:
            soft_local["subject_box"] = list(subject_box)

    h = grade_hash(
        resolved,
        creative_lut_ref=creative_lut_ref,
        working_space=working_space,
        soft_local=soft_local,
    )
    return {
        "cdl": resolved.to_dict(),
        "creative_lut_ref": creative_lut_ref,
        "working_space": working_space,
        "soft_local": soft_local,
        "grade_hash": h,
    }
