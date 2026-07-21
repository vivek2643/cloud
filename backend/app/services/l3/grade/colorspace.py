"""Standard sRGB <-> CIE Lab (D65) conversion -- exact, not an approximation.
`srgb_to_lab` is used to judge whether a sampled `white_reference` region
(color_grading.plan.md SS2.3) actually reads as neutral before the correct
layer trusts it. `lab_to_srgb` (color_skin_vibrance.plan.md S4.5) is its exact
inverse, needed to turn a corrected skin Lab target back into an RGB
multiplier for the WB solve."""
from __future__ import annotations

from typing import Tuple

_D65 = (0.95047, 1.0, 1.08883)


def _inv_gamma(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    """Forward sRGB OETF (linear -> gamma-encoded) -- inverse of `_inv_gamma`."""
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


def _f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)


def _f_inv(t: float) -> float:
    """Inverse of `_f` -- the CIE breakpoint on `t` (fy) is `t**3 > 0.008856`
    (equivalently `t > 6/29`), matching `_f`'s breakpoint on its input."""
    t3 = t ** 3
    return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787


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


def lab_to_srgb(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Exact inverse of `srgb_to_lab`. `lab` is (L*, a*, b*); returns
    (r, g, b) each clamped to 0..1 (sRGB, gamma-encoded)."""
    L, a, b_ = lab
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b_ / 200.0
    xn, yn, zn = _D65
    x, y, z = _f_inv(fx) * xn, _f_inv(fy) * yn, _f_inv(fz) * zn
    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    return tuple(
        max(0.0, min(1.0, _gamma(max(0.0, min(1.0, c)))))
        for c in (r_lin, g_lin, b_lin)
    )


def is_neutral(rgb: Tuple[float, float, float], *, tolerance: float = 6.0) -> bool:
    """True when `rgb`'s Lab a*/b* cast is small enough to trust as a
    genuine neutral surface (a white_reference candidate), not a colored
    object the model mistook for one."""
    _, a, b = srgb_to_lab(rgb)
    return abs(a) < tolerance and abs(b) < tolerance
