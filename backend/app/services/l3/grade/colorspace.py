"""Standard sRGB -> CIE Lab (D65) conversion -- exact, not an approximation.
Used to judge whether a sampled `white_reference` region (color_grading.plan.md
SS2.3) actually reads as neutral before the correct layer trusts it."""
from __future__ import annotations

from typing import Tuple

_D65 = (0.95047, 1.0, 1.08883)


def _inv_gamma(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)


def srgb_to_lab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """`rgb` each in 0..1 (sRGB, gamma-encoded). Returns (L*, a*, b*)."""
    r, g, b = (_inv_gamma(max(0.0, min(1.0, c))) for c in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = _D65
    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return L, a, b_


def is_neutral(rgb: Tuple[float, float, float], *, tolerance: float = 6.0) -> bool:
    """True when `rgb`'s Lab a*/b* cast is small enough to trust as a
    genuine neutral surface (a white_reference candidate), not a colored
    object the model mistook for one."""
    _, a, b = srgb_to_lab(rgb)
    return abs(a) < tolerance and abs(b) < tolerance
