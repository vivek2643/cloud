"""
Soft-local layer (color_grading.plan.md SS9, Fork B): feathered, subject-
anchored spatial adjustments. Deliberately NOT baked into the CDL/LUT -- a
3D color LUT is a pointwise value->value map with no notion of pixel
position, so the plan doc's "bakes into the per-clip LUT" is a shorthand,
not literal (see `lut_bake.py`'s module docstring for the full reasoning).
This module instead produces a small, JSON-safe SPATIAL descriptor, applied
as an ADDITIONAL deterministic pass alongside (not inside) the color LUT.

Scope: only the attention vignette is implemented -- a soft, subject-
anchored radial darkening toward the frame edges, the most universal and
well-defined of the three effects SS9 lists. Sky-gradient (horizon_y-aware)
and directional side-lift are the same MECHANISM (a feathered spatial
multiplier) but need signals (horizon detection) this pass doesn't produce;
left as a documented follow-up, not silently dropped.

Honest parity note (unlike the CDL/LUT engine, which is byte-identical both
sides by construction): the export side applies this via ffmpeg's own
`vignette` filter (a well-tested standard primitive) while preview applies
an independently-written WebGL radial falloff. Both anchor on the same
subject point and scale with the same strength, so they read as the same
soft effect, but the exact falloff CURVE is not guaranteed pixel-identical
the way the CDL engine is. That's an intentional, bounded trade for a
"soft, feathered, never a hard mask" cosmetic effect, not a corner cut on
the parity contract the plan calls make-or-break for actual color values --
see SS16's own framing of soft-local as approximate by nature.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DEFAULT_STRENGTH = 0.25  # gentle by default -- "all feathered, no hard masks"
MAX_ANGLE_RAD = 1.0472   # PI/3 -- ffmpeg vignette's angle at strength=1.0


def solve_vignette(
    subject_box: Optional[Tuple[float, float, float, float]] = None,
    *,
    strength: float = DEFAULT_STRENGTH,
) -> Dict[str, Any]:
    """A center (cx, cy, normalized 0..1) + strength descriptor for a soft
    radial darkening, anchored on the subject when known (else frame
    center). `subject_box` is the normalized (x, y, w, h) box from
    `cut_records.framing.subject_box` (SS1's re-use note: reframing/subject
    detection already exists, this only reuses it for metering)."""
    if subject_box:
        x, y, w, h = subject_box
        cx = max(0.0, min(1.0, x + w / 2.0))
        cy = max(0.0, min(1.0, y + h / 2.0))
    else:
        cx, cy = 0.5, 0.5
    return {"cx": cx, "cy": cy, "strength": max(0.0, min(1.0, strength))}


def vignette_ffmpeg_filter(vignette: Optional[Dict[str, Any]]) -> Optional[str]:
    """`vignette` descriptor -> an ffmpeg `vignette` filter clause, or None
    for a no-op (absent/zero strength). x0/y0 are pixel EXPRESSIONS (ffmpeg
    evaluates `w`/`h` at filter time to the frame size)."""
    if not vignette:
        return None
    strength = float(vignette.get("strength") or 0.0)
    if strength <= 0.001:
        return None
    cx, cy = float(vignette.get("cx", 0.5)), float(vignette.get("cy", 0.5))
    angle = strength * MAX_ANGLE_RAD
    return f"vignette=angle={angle:.4f}:x0=w*{cx:.4f}:y0=h*{cy:.4f}"
