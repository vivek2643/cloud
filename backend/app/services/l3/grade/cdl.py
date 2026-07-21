"""
ASC CDL math -- the steerable, round-trippable spine of every grade
(color_grading.plan.md SS2.1). Pure, no I/O.

The CDL (Color Decision List) is deliberately simple: per-channel
slope/offset/power plus one saturation scalar, applied directly to whatever
code values arrive (video-range or log -- CDL was designed to be space-
agnostic; that's why the professional pipeline round-trips it as `.cdl`/
`.ccc` rather than a baked LUT). Everything upstream of this module (correct/
match/look/arc/soft-local) only ever produces a `Grade`, never touches
pixels directly; everything downstream (lut_bake.py) only ever asks this
module "what does this Grade do to an RGB array," so preview and export can
never see two different answers to that question.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Rec.709 luma coefficients -- matches l1/color_stats.py's luma weighting and
# the render pipeline's assumed working color space.
LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722


@dataclass(frozen=True)
class Grade:
    """One resolved CDL. `slope`/`offset`/`power` are per-channel [r, g, b];
    `sat` is a single scalar (1.0 = no change, 0.0 = grayscale)."""
    slope: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    power: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    sat: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope": list(self.slope),
            "offset": list(self.offset),
            "power": list(self.power),
            "sat": self.sat,
        }

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Grade":
        if not d:
            return Grade()
        slope = d.get("slope") or [1.0, 1.0, 1.0]
        offset = d.get("offset") or [0.0, 0.0, 0.0]
        power = d.get("power") or [1.0, 1.0, 1.0]
        sat = d.get("sat", 1.0)
        return Grade(
            slope=(float(slope[0]), float(slope[1]), float(slope[2])),
            offset=(float(offset[0]), float(offset[1]), float(offset[2])),
            power=(float(power[0]), float(power[1]), float(power[2])),
            sat=float(sat),
        )


def identity_grade() -> Grade:
    return Grade()


def identity_grade_json(working_space: str = "rec709") -> Dict[str, Any]:
    """The `resolve_clip_grade`-shaped dict for a flat/no-op grade -- the
    graceful-fallback value `layers.resolve` uses under `v1` when a shot has
    no persisted `resolved_grades` row yet (color_grading_upgrade.plan.md
    Step 1.0 §5: preview must always render, never block on the job)."""
    g = Grade()
    return {
        "cdl": g.to_dict(),
        "creative_lut_ref": None,
        "working_space": working_space,
        "soft_local": None,
        "grade_hash": grade_hash(g, working_space=working_space),
    }


def is_identity(grade: Grade, eps: float = 1e-9) -> bool:
    return (
        all(abs(s - 1.0) < eps for s in grade.slope)
        and all(abs(o) < eps for o in grade.offset)
        and all(abs(p - 1.0) < eps for p in grade.power)
        and abs(grade.sat - 1.0) < eps
    )


def compose(base: Grade, delta: Grade, amount: float = 1.0) -> Grade:
    """Blend `delta` onto `base` at `amount` (0=base, 1=full delta applied on
    top of base). Used by the arc layer to scale a per-beat CDL delta by the
    user's single intensity dial (color_grading.plan.md SS8) -- slope/power
    scale multiplicatively (1.0 = no-op), offset/sat-delta scale additively
    (0.0 = no-op), all lerped by `amount` so 0 = flat/base, 1 = full delta."""
    amount = max(0.0, min(1.0, amount))
    if amount <= 0.0:
        return base

    def lerp_mult(b: float, d: float) -> float:
        # d is a multiplier around 1.0; lerp the multiplier itself toward 1.
        return b * (1.0 + (d - 1.0) * amount)

    def lerp_add(b: float, d: float) -> float:
        return b + d * amount

    slope = tuple(lerp_mult(base.slope[i], delta.slope[i]) for i in range(3))
    offset = tuple(lerp_add(base.offset[i], delta.offset[i]) for i in range(3))
    power = tuple(lerp_mult(base.power[i], delta.power[i]) for i in range(3))
    sat = lerp_mult(base.sat, delta.sat)
    return Grade(slope=slope, offset=offset, power=power, sat=sat)  # type: ignore[arg-type]


def apply_cdl(rgb, grade: Grade):
    """Apply the CDL to an (..., 3) float32 array of RGB values in 0..1.
    `out = clamp(in * slope + offset, 0, 1) ^ power`, then desaturate toward
    luma by `sat`. Never-worse guardrail is the CALLER's job (the correct
    layer decides slope/offset/power in the first place); this function is a
    dumb, faithful CDL evaluator -- it must behave identically here and in
    lut_bake.py's cube sampling, since those two call sites are what preview
    and export both key off of."""
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32)
    slope = np.array(grade.slope, dtype=np.float32)
    offset = np.array(grade.offset, dtype=np.float32)
    power = np.array(grade.power, dtype=np.float32)

    out = np.clip(arr * slope + offset, 0.0, 1.0)
    # power=0 would be degenerate (undefined at 0**0 for some inputs); guard
    # against a bad upstream value rather than let it produce NaN.
    safe_power = np.where(power <= 1e-6, 1.0, power)
    out = np.power(out, safe_power)
    out = np.clip(out, 0.0, 1.0)

    if abs(grade.sat - 1.0) > 1e-9:
        luma = (out[..., 0] * LUMA_R + out[..., 1] * LUMA_G + out[..., 2] * LUMA_B)
        luma = luma[..., None]
        out = np.clip(luma + grade.sat * (out - luma), 0.0, 1.0)

    return out


def grade_hash(
    grade: Grade,
    *,
    creative_lut_ref: Optional[str] = None,
    working_space: str = "rec709",
    soft_local: Optional[Dict[str, Any]] = None,
    lut_size: int = 33,
    schema_version: int = 1,
    tone_contrast: float = 0.0,
) -> str:
    """Stable content hash for a fully-resolved grade -- the cache key every
    baked .cube is stored/served under (color_grading.plan.md SS4/SS11:
    "cache by grade_hash(clip)"). Anything that can change the baked cube's
    bytes MUST be part of this payload -- `tone_contrast`
    (color_tone_contrast.plan.md) included, since it changes `from_working`'s
    output during the bake."""
    payload = {
        "v": schema_version,
        "cdl": grade.to_dict(),
        "creative_lut_ref": creative_lut_ref,
        "working_space": working_space,
        "soft_local": soft_local or {},
        "lut_size": lut_size,
        "tone_contrast": tone_contrast,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
