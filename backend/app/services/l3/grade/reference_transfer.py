"""
Look layer, mode 2 -- reference-image drop (color_grading.plan.md SS7.2): a
Reinhard-style color transfer (match both mean AND spread per RGB channel,
not just a flat cast shift) fitted into the CDL spine, with a match-strength
dial. Stays steerable + arc-able since the output is just another `Grade`.

"Skin-aware" here means the transfer is DAMPED by the match-strength dial
rather than built around one hardcoded skin-tone reference -- see
`grade/correct.py`'s docstring for why this pass doesn't attempt a
skin-privileging correction. A full-strength Reinhard transfer can push
skin tones toward the reference image's own cast in ways a user pointing at
a mood/palette reference didn't intend; defaulting conservative and letting
the caller dial up is the safer default than guessing at "protect skin"
math without real cross-skin-tone validation data.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from app.services.l3.grade.cdl import Grade

DEFAULT_MATCH_STRENGTH = 0.6
WB_MULTIPLIER_CLAMP = 1.6
OFFSET_CLAMP = 0.3


def compute_image_stats(rgb_pixels: Any) -> Dict[str, Any]:
    """Mean + std per RGB channel for an arbitrary decoded image ((N,3) or
    (H,W,3) array, 0..1 float or 0..255 uint8) -- same shape as a
    `color_stats` row's rgb_mean/rgb_std, so both sides of the transfer
    speak the same language. This is the ONE spot outside L1 that measures
    pixels directly; call it once when a reference image is uploaded and
    cache the result on `EditDocument.look.reference_stats` (SS2.4) rather
    than re-decoding the image on every resolve."""
    import numpy as np

    arr = np.asarray(rgb_pixels, dtype=np.float32)
    if arr.max() > 1.5:  # heuristic: looks like 0..255, normalize
        arr = arr / 255.0
    flat = arr.reshape(-1, 3)
    return {"rgb_mean": flat.mean(axis=0).tolist(), "rgb_std": flat.std(axis=0).tolist()}


def solve_reference_transfer(
    source_stats: Dict[str, Any],
    reference_stats: Dict[str, Any],
    *,
    match_strength: float = DEFAULT_MATCH_STRENGTH,
) -> Grade:
    """Per-channel Reinhard-style transfer: nudge SOURCE's mean+std toward
    REFERENCE's, damped by `match_strength` (0=no change, 1=full transfer).
    Slope/offset are bounded regardless of strength so a wildly different
    reference image can't blow out the result."""
    if not source_stats or not reference_stats:
        return Grade()
    src_mean = source_stats.get("rgb_mean") or [0.5, 0.5, 0.5]
    src_std = source_stats.get("rgb_std") or [0.2, 0.2, 0.2]
    ref_mean = reference_stats.get("rgb_mean") or [0.5, 0.5, 0.5]
    ref_std = reference_stats.get("rgb_std") or [0.2, 0.2, 0.2]
    if not src_mean or not src_std or not ref_mean or not ref_std:
        return Grade()

    amount = max(0.0, min(1.0, match_strength))
    eps = 1e-4
    slope: list = []
    offset: list = []
    for c in range(3):
        full_slope = ref_std[c] / max(eps, src_std[c])
        full_offset = ref_mean[c] - src_mean[c] * full_slope
        s = 1.0 + (full_slope - 1.0) * amount
        o = full_offset * amount
        slope.append(max(1.0 / WB_MULTIPLIER_CLAMP, min(WB_MULTIPLIER_CLAMP, s)))
        offset.append(max(-OFFSET_CLAMP, min(OFFSET_CLAMP, o)))
    return Grade(slope=tuple(slope), offset=tuple(offset), power=(1.0, 1.0, 1.0), sat=1.0)
