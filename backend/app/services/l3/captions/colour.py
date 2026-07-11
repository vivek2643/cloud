"""
Caption colour resolution (captions.plan.md SS11): turns a style's abstract
`colour` spec (`source` + base hex hints) into a concrete, LEGIBLE colour set
for one caption event, given the signals available at that point in the
timeline -- the file's `color_stats` (palette + rgb_mean, SS2's "L1
color_stats.palette") and the clip's resolved grade CDL (SS3's already-
composed `resolve_clip_grade` output, reused rather than recomputed -- see
resolver.py).

Pure function, no I/O: every signal is passed in already-fetched, same
"measure once, resolve many times" contract as `grade.resolver`.
"""
from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# WCAG-style contrast floor. 3.0 is lenient (WCAG AA large-text is 3:1) --
# captions always carry an outline/shadow too, so the floor only has to
# catch genuinely illegible pairings (e.g. white-on-white highlights), not
# do all the legibility work alone.
MIN_CONTRAST = 3.0

_WHITE = (1.0, 1.0, 1.0)
_NEAR_BLACK = (0.06, 0.06, 0.06)


def _hex_to_rgb01(hexcolor: str) -> Tuple[float, float, float]:
    h = hexcolor.lstrip("#")
    if len(h) != 6:
        return _WHITE
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    except ValueError:
        return _WHITE


def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = (max(0.0, min(1.0, c)) for c in rgb)
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def _srgb_channel_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = (_srgb_channel_to_linear(max(0.0, min(1.0, c))) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    la, lb = _relative_luminance(a) + 0.05, _relative_luminance(b) + 0.05
    return max(la, lb) / min(la, lb)


def _footage_luma(color_stats: Optional[Dict[str, Any]]) -> float:
    """A cheap legibility proxy for "how bright does this clip typically
    read": the file's overall `rgb_mean` luma. Not a per-zone pixel sample
    (that would need a frame decode -- out of scope for a pure resolver
    step; see captions.plan.md SS2's frame-extraction note, which is a
    preview-tile concern, not a per-event colour concern)."""
    if not color_stats:
        return 0.4  # a mid-dark default: bias toward white+outline, the safest general case
    mean = color_stats.get("rgb_mean") or [0.4, 0.4, 0.4]
    return _relative_luminance(tuple(mean[:3]))  # type: ignore[arg-type]


def _is_skinlike(rgb: Tuple[float, float, float]) -> bool:
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return (0.0 <= h <= 0.11 or h >= 0.97) and 0.15 <= s <= 0.6 and v >= 0.35


def vibrant_accent(palette: Optional[List[Sequence[float]]]) -> Optional[Tuple[float, float, float]]:
    """The most saturated, non-skin-toned, non-neutral entry in a file's
    `color_stats.palette` (SS11 "palette_accent emphasis colour pulled from
    a vibrant, non-skin palette entry"). None when there's nothing usable
    (falls back to a fixed accent -- never a crash / blank colour)."""
    if not palette:
        return None
    candidates: List[Tuple[float, Tuple[float, float, float]]] = []
    for entry in palette:
        if len(entry) < 3:
            continue
        rgb = (float(entry[0]), float(entry[1]), float(entry[2]))
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        if s < 0.35 or v < 0.25:  # too grey/dark to read as an "accent"
            continue
        if _is_skinlike(rgb):
            continue
        candidates.append((s, rgb))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _grade_tint(grade: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
    """"What does white look like under this clip's resolved grade" -- the
    CDL applied to a neutral white swatch, i.e. `match_grade`'s tint
    (SS11's "match_grade ties caption colour to the chosen grade")."""
    if not grade:
        return None
    cdl = grade.get("cdl") or {}
    slope = cdl.get("slope") or [1.0, 1.0, 1.0]
    offset = cdl.get("offset") or [0.0, 0.0, 0.0]
    try:
        return tuple(  # type: ignore[return-value]
            max(0.0, min(1.0, float(slope[c]) * 1.0 + float(offset[c]))) for c in range(3)
        )
    except (TypeError, ValueError, IndexError):
        return None


def _ensure_legible(fill: Tuple[float, float, float], bg_luma: float) -> Tuple[Tuple[float, float, float], bool]:
    """Returns (fill, needs_strong_outline). If `fill` doesn't clear
    MIN_CONTRAST against the estimated background, flip toward the opposite
    pole (white<->near-black) rather than silently shipping an illegible
    caption (SS11 "always compute a legibility floor")."""
    bg = (bg_luma, bg_luma, bg_luma)
    if contrast_ratio(fill, bg) >= MIN_CONTRAST:
        return fill, False
    flipped = _NEAR_BLACK if _relative_luminance(fill) > 0.5 else _WHITE
    if contrast_ratio(flipped, bg) >= MIN_CONTRAST:
        return flipped, False
    # Neither pole clears the floor against this footage (e.g. mid-grey,
    # heavily textured) -- keep the original fill but flag it for a strong
    # outline/shadow/box, which is what actually saves legibility here.
    return fill, True


def resolve_colour(
    colour_spec: Dict[str, Any],
    *,
    color_stats: Optional[Dict[str, Any]] = None,
    grade: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A style's abstract `colour` spec -> concrete hex colours, contrast-
    checked against this clip's footage. Returns
    `{fill, emphasis_fill, outline, shadow, box, strong_outline}` (all hex
    strings except the trailing bool)."""
    source = colour_spec.get("source") or "white"
    fill_rgb = _hex_to_rgb01(colour_spec.get("fill") or "#ffffff")
    emphasis_rgb = _hex_to_rgb01(colour_spec.get("emphasis_fill") or colour_spec.get("fill") or "#ffffff")
    box = colour_spec.get("box")

    if source == "match_grade":
        tint = _grade_tint(grade)
        if tint is not None:
            fill_rgb = tint
            emphasis_rgb = tint
    elif source == "palette_accent":
        accent = vibrant_accent((color_stats or {}).get("palette") if color_stats else None)
        if accent is not None:
            emphasis_rgb = accent
    elif source == "black_box":
        box = box or "#000000"
    elif source == "high_contrast":
        # Deliberately loud: pick whichever pole (white/near-black) contrasts
        # HARDEST against this footage, not just "clears the floor".
        bg_luma = _footage_luma(color_stats)
        fill_rgb = _WHITE if bg_luma < 0.5 else _NEAR_BLACK
        emphasis_rgb = vibrant_accent((color_stats or {}).get("palette") if color_stats else None) or fill_rgb

    bg_luma = _footage_luma(color_stats)
    fill_rgb, strong_fill = _ensure_legible(fill_rgb, bg_luma)
    emphasis_rgb, strong_emph = _ensure_legible(emphasis_rgb, bg_luma)

    return {
        "fill": _rgb01_to_hex(fill_rgb),
        "emphasis_fill": _rgb01_to_hex(emphasis_rgb),
        "outline": colour_spec.get("outline") or "#000000",
        "shadow": colour_spec.get("shadow") or "#000000",
        "box": box,
        "strong_outline": bool(strong_fill or strong_emph),
    }
