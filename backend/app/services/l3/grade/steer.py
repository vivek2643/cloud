"""
NL steering (color_grading.plan.md SS10): EDSO turns "warmer", "less teal",
"more cinematic" into a small set of NAMED, BOUNDED dials -- never raw CDL
numbers. Same "model emits intent, code emits numbers" split `framing.py`
uses for reframing (the model never picks a crop rectangle by hand either).

Each dial is -1..1 (0 = no change, unset = 0); deterministic code maps them
to a CDL nudge. `explain_grade` is the read-only inverse -- a short,
human-readable summary of a resolved CDL, for the "explain the grade"
capability (trust + steerability, SS10).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.l3.grade.cdl import Grade

DIAL_NAMES = ("warmth", "tint", "brightness", "contrast", "saturation")


def _clamp(v: Optional[float]) -> float:
    return 0.0 if v is None else max(-1.0, min(1.0, float(v)))


def solve_steer_grade(
    *,
    warmth: Optional[float] = None,
    tint: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
) -> Grade:
    """warmth: +1 warmer (more red / less blue) .. -1 cooler.
    tint: +1 more magenta (red+blue up, green down) .. -1 more green.
    brightness: +1 brighter (lifted overall) .. -1 darker.
    contrast: +1 punchier (steeper around mid-gray) .. -1 flatter.
    saturation: +1 more saturated .. -1 more desaturated."""
    w, t, b, c, s = (
        _clamp(warmth), _clamp(tint), _clamp(brightness), _clamp(contrast), _clamp(saturation),
    )

    slope = [
        1.0 + w * 0.12 + t * 0.06,
        1.0 - t * 0.08,
        1.0 - w * 0.12 + t * 0.06,
    ]
    offset = [b * 0.08, b * 0.08, b * 0.08]

    # Contrast: a slope stretch pivoted on mid-gray so brightness stays put.
    contrast_mult = 1.0 + c * 0.25
    pivot = 0.5
    slope = [sl * contrast_mult for sl in slope]
    offset = [off + pivot * (1.0 - contrast_mult) for off in offset]

    sat = 1.0 + s * 0.35
    return Grade(slope=tuple(slope), offset=tuple(offset), power=(1.0, 1.0, 1.0), sat=sat)


def explain_grade(cdl: Dict[str, Any]) -> str:
    """A short, human-readable gloss of a resolved CDL -- e.g. 'warmer,
    more saturated, slightly higher contrast than baseline'. Purely
    descriptive (read-only); never used to decide anything."""
    slope = cdl.get("slope") or [1.0, 1.0, 1.0]
    offset = cdl.get("offset") or [0.0, 0.0, 0.0]
    sat = cdl.get("sat", 1.0)
    eps = 0.01

    bits = []
    warmth = slope[0] - slope[2]
    if warmth > eps:
        bits.append("warmer" if warmth < 0.15 else "much warmer")
    elif warmth < -eps:
        bits.append("cooler" if warmth > -0.15 else "much cooler")

    mean_offset = sum(offset) / 3.0
    if mean_offset > eps:
        bits.append("brighter")
    elif mean_offset < -eps:
        bits.append("darker")

    if sat > 1.0 + eps:
        bits.append("more saturated" if sat < 1.2 else "much more saturated")
    elif sat < 1.0 - eps:
        bits.append("desaturated" if sat > 0.8 else "heavily desaturated")

    mean_slope = sum(slope) / 3.0
    if mean_slope > 1.0 + eps:
        bits.append("higher contrast")
    elif mean_slope < 1.0 - eps:
        bits.append("flatter/lower contrast")

    if not bits:
        return "No change from baseline (identity grade)."
    return ", ".join(bits) + " than baseline."
