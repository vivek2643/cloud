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
  - SS7 (look) / SS8 (arc) still no-op (identity) until their own build
    steps land -- each will compose here in the same fixed order, ahead of
    the explicit override.
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

from app.services.l3.grade.cdl import Grade, compose, grade_hash
from app.services.l3.grade.correct import solve_correct_grade

DEFAULT_WORKING_SPACE = "rec709"


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
    # SS7 (look) / SS8 (arc) compose here next, in that order, once their
    # own build steps land.

    override = Grade.from_dict(item.get("grade")) if item.get("grade") else None
    resolved = compose(stack, override, 1.0) if override is not None else stack

    creative_lut_ref = (sequence_look or {}).get("lut_ref") if sequence_look else None
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
