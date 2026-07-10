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
  - SS5 (correct) / SS6 (match) / SS7 (look) / SS8 (arc) each contribute a
    `Grade` that gets composed here, in that fixed order, before the
    explicit per-clip override (an NL trim, or a manual dial) is applied
    last so it always wins.
  - For now (parity-engine step -- color_grading.plan.md build order #2),
    only "explicit override, else identity" is wired up; the other stack
    stages are no-ops until their own build step lands. This keeps the
    render/preview parity plumbing provably correct before any grade
    *decision* logic exists on top of it.

Never computes pixels itself -- produces a `Grade` (SS2.1's `cdl`) that
`lut_bake.py` turns into bytes only when something actually asks for them
(the cube endpoint, or the render compositor), so a timeline edit that
touches ten clips is ten cheap hash computations, not ten LUT bakes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.l3.grade.cdl import Grade, grade_hash, identity_grade

DEFAULT_WORKING_SPACE = "rec709"


def resolve_clip_grade(
    item: Dict[str, Any],
    *,
    color_stats: Optional[Dict[str, Any]] = None,
    sequence_look: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve ONE clip's (spine segment or op) final grade descriptor.

    `item` is the raw `EditSegment`/`EditOperation` dict; `item.get("grade")`
    is the per-clip explicit override (SS2.4) -- CDL fields only, since a
    per-clip override is a nudge on top of the stack, never a whole new
    creative LUT (that's the sequence-level Look, SS7). `color_stats` and
    `sequence_look` are threaded through now so later build steps (correct/
    match/look/arc) can extend this function's body without changing its
    call sites in layers.py/resolve-timeline.ts.
    """
    override = Grade.from_dict(item.get("grade"))

    # SS5 correct / SS6 match / SS7 look / SS8 arc all compose here, in
    # that order, ahead of the explicit override -- each currently a no-op
    # (identity) until its own build step lands.
    base = identity_grade()
    resolved = base if override is None else _compose_override(base, override, item.get("grade"))

    creative_lut_ref = (sequence_look or {}).get("lut_ref") if sequence_look else None
    working_space = item.get("working_space") or DEFAULT_WORKING_SPACE

    h = grade_hash(
        resolved,
        creative_lut_ref=creative_lut_ref,
        working_space=working_space,
    )
    out: Dict[str, Any] = {
        "cdl": resolved.to_dict(),
        "creative_lut_ref": creative_lut_ref,
        "working_space": working_space,
        "grade_hash": h,
    }
    return out


def _compose_override(base: Grade, override: Grade, raw: Optional[Dict[str, Any]]) -> Grade:
    """An explicit `item["grade"]` IS the resolved CDL for now (there's no
    upstream stack contribution yet to layer it on top of) -- once correct/
    match/arc land, this becomes `cdl.compose(stack_result, override, 1.0)`
    so the override always wins as the last word, per the module docstring's
    stack order."""
    if raw is None:
        return base
    return override
