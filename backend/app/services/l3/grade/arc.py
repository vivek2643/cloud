"""
Arc layer (color_grading.plan.md SS8, Fork C): an invisible-by-default
per-beat grade nudge over the ASSEMBLED TIMELINE. EDSO tags each segment
with one categorical intent (calm | build | peak | resolve) -- output a
CATEGORY only, the same "model owns categories, code owns numbers" split
used throughout this codebase (`shot_size`, `energy_grade`,
`classify_camera_move`). This module owns the deterministic category->CDL
delta table; the user's single intensity dial scales every delta's
amplitude uniformly via the SAME `cdl.compose` semantics the whole stack
already uses (0 = flat/no arc, 1 = full arc as authored).

The delta table below is a first-pass, reasonable-but-not-colorist-reviewed
set (same honesty caveat as `presets.py`): calm settles cool and soft, build
warms and tightens toward tension, peak is the most saturated/contrasty/warm
moment, resolve eases back down. Small, bounded deltas by design -- this is
meant to read as a subconscious pulse across a cut, not a set of jarring
per-beat color jumps.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.services.l3.grade.cdl import Grade, compose, identity_grade

ARC_INTENTS = ("calm", "build", "peak", "resolve")

_DELTAS: Dict[str, Grade] = {
    "calm": Grade(slope=(0.98, 1.0, 1.02), offset=(0.0, 0.0, 0.0), power=(1.0, 1.0, 1.0), sat=0.92),
    "build": Grade(slope=(1.02, 1.0, 0.99), offset=(0.005, 0.0, 0.0), power=(1.0, 1.0, 1.0), sat=1.05),
    "peak": Grade(slope=(1.06, 1.02, 0.95), offset=(0.01, 0.005, 0.0), power=(1.0, 1.0, 1.0), sat=1.15),
    "resolve": Grade(slope=(0.99, 1.0, 1.01), offset=(0.0, 0.0, 0.0), power=(1.0, 1.0, 1.0), sat=0.90),
}


def solve_arc_grade(intent: Optional[str], intensity: Optional[float]) -> Grade:
    """`intent` not one of ARC_INTENTS (including None/untagged) -> identity,
    regardless of intensity: an untagged segment has no arc position to
    express. `intensity` defaults to 0 (flat/invisible) when unset -- the
    plan's own "invisible by default," so a document that never sets the
    dial never sees the arc even if segments happen to be tagged."""
    delta = _DELTAS.get(intent or "")
    if delta is None:
        return identity_grade()
    amount = 0.0 if intensity is None else max(0.0, min(1.0, float(intensity)))
    return compose(identity_grade(), delta, amount)
