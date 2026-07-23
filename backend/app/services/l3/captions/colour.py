"""
Caption colour resolution (caption_style_mvp.plan.md #3): a style's
`colour_id` + `outline_enabled`/`shadow_enabled` -> concrete hex colours for
one caption event. Fully deterministic -- each of the 4 fixed swatches
(styles.COLOURS) already knows its own outline/shadow ink (dark ink for
white/yellow/cyan, light ink for charcoal), so there is no runtime
footage/grade sampling, no contrast floor, no "strong outline" escalation
left in the public MVP path.

`vibrant_accent`/`_footage_luma`/`contrast_ratio` are kept and exported --
not for resolving a style's OWN colour any more, but because suggest.py's
RANKING still needs footage brightness/palette as an edit SIGNAL (e.g. the
hard constraint "do not return Charcoal for predominantly dark footage").

Pure function, no I/O.
"""
from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.l3.captions.styles import COLOURS, DEFAULT_COLOUR_ID

# Fixed, restrained treatment -- see ass_export.py (ASS units) and
# caption-overlay.tsx (CSS px) for where these actually get applied; this
# module only decides ink COLOUR, not width/opacity.
_WHITE = (1.0, 1.0, 1.0)
_NEAR_BLACK = (0.06, 0.06, 0.06)


def _hex_to_rgb01(hexcolor: str) -> Tuple[float, float, float]:
    h = (hexcolor or "").lstrip("#")
    if len(h) != 6:
        return _WHITE
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    except ValueError:
        return _WHITE


def _srgb_channel_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = (_srgb_channel_to_linear(max(0.0, min(1.0, c))) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    la, lb = _relative_luminance(a) + 0.05, _relative_luminance(b) + 0.05
    return max(la, lb) / min(la, lb)


def footage_luma(color_stats: Optional[Dict[str, Any]]) -> float:
    """A cheap "how bright does this clip typically read" proxy: the file's
    overall `rgb_mean` luma. Used by suggest.py's hard constraints, not by
    colour resolution itself any more."""
    if not color_stats:
        return 0.4  # a mid-dark default: bias toward the safer general case
    mean = color_stats.get("rgb_mean") or [0.4, 0.4, 0.4]
    return _relative_luminance(tuple(mean[:3]))


def _is_skinlike(rgb: Tuple[float, float, float]) -> bool:
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return (0.0 <= h <= 0.11 or h >= 0.97) and 0.15 <= s <= 0.6 and v >= 0.35


def vibrant_accent(palette: Optional[List[Sequence[float]]]) -> Optional[Tuple[float, float, float]]:
    """The most saturated, non-skin-toned, non-neutral entry in a file's
    `color_stats.palette`. None when there's nothing usable. Used by
    suggest.py as an edit SIGNAL (e.g. "does this footage already have a
    vibrant accent close to Yellow/Cyan"), not to pick a style's colour."""
    if not palette:
        return None
    candidates: List[Tuple[float, Tuple[float, float, float]]] = []
    for entry in palette:
        if len(entry) < 3:
            continue
        rgb = (float(entry[0]), float(entry[1]), float(entry[2]))
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        if s < 0.35 or v < 0.25:
            continue
        if _is_skinlike(rgb):
            continue
        candidates.append((s, rgb))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def resolve_colour(colour_id: str, *, outline_enabled: bool = False, shadow_enabled: bool = False) -> Dict[str, Any]:
    """A style's `colour_id` + toggles -> concrete hex colours for one
    caption event. Deterministic: no footage/grade signal is consulted.
    Returns `{fill, outline_enabled, shadow_enabled, outline, shadow}`
    (`outline`/`shadow` are always populated with the swatch's own ink hex,
    even when disabled, so a caller can turn the toggle on/off without a
    second lookup)."""
    swatch = COLOURS.get(colour_id, COLOURS[DEFAULT_COLOUR_ID])
    return {
        "fill": swatch["hex"],
        "outline_enabled": bool(outline_enabled),
        "shadow_enabled": bool(shadow_enabled),
        "outline": swatch["ink"],
        "shadow": swatch["ink"],
    }
