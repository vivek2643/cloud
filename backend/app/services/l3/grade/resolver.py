"""
The deterministic grade resolver (color_grading.plan.md SS3): turns a
timeline segment/operation's explicit `grade` override (if any) plus
whatever measurement/look/arc context is available into the ONE small,
JSON-safe descriptor (`{cdl, creative_lut_ref, working_space, grade_hash}`)
that gets baked into `document["resolved"]`, same place `layers.py`
already bakes geometric `transform`.

Stack order (SS3): Measure -> Correct -> Match -> Look -> Arc -> Soft-local
-> NL trims -> BAKE. This module is the seam every later build step plugs
into:
  - SS5 (correct) is wired: `grade.correct.solve_correct_grade` composes
    onto the stack first, using the file's `color_stats` row.
  - SS6 (match) is wired: a pre-computed per-file delta (see
    `grade.match.solve_match_deltas`, computed ONCE per document resolve in
    `layers.resolve` since grade-groups are a whole-document clustering, not
    a per-clip decision) composes next.
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

DEFAULT_WORKING_SPACE = "rec709"


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
    """
    stack = solve_correct_grade(color_stats, already_graded=already_graded)
    if match_delta is not None:
        stack = compose(stack, match_delta, 1.0)
    stack = compose(stack, _solve_look(sequence_look, color_stats), 1.0)
    arc_intensity = (sequence_look or {}).get("arc_intensity")
    stack = compose(stack, solve_arc_grade(item.get("arc_intent"), arc_intensity), 1.0)

    override = Grade.from_dict(item.get("grade")) if item.get("grade") else None
    resolved = compose(stack, override, 1.0) if override is not None else stack

    creative_lut_ref = (sequence_look or {}).get("lut_ref") if sequence_look else None
    if sequence_look and sequence_look.get("mode") != "lut":
        creative_lut_ref = None
    working_space = item.get("working_space") or DEFAULT_WORKING_SPACE

    h = grade_hash(
        resolved,
        creative_lut_ref=creative_lut_ref,
        working_space=working_space,
    )
    return {
        "cdl": resolved.to_dict(),
        "creative_lut_ref": creative_lut_ref,
        "working_space": working_space,
        "grade_hash": h,
    }
