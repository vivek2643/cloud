"""
Caption style bundles (caption_style_mvp.plan.md): the `CaptionStyle` schema
plus the curated catalogs (fonts/colours/positions/animations/cases/sizes) it's
built from. Pure data -- no I/O, no signals -- everything downstream
(resolver, suggest, ass_export) composes these dicts.

MVP catalog is DELIBERATELY four values per category, no exceptions -- every
knob that used to be a continuous dial (animation intensity, letter tracking,
outline width, shadow parameters, line-wrap limits, beat sync, an open colour
picker) is now either gone or a fixed internal constant. This is a full
replacement of the prior "Suggested + Standards, 5+7 tiles, 5 colour sources,
4 animation presets with tunable intensity" system with a much smaller
Standard + 4 AI-picks catalog. `CaptionStyle.from_dict` still parses the OLD
(pre-MVP) shape so previously saved documents keep rendering -- see
`_from_legacy_dict` -- but nothing new is ever written in that shape again.

Font set: 4 curated families, all SIL Open Font License, self-hosted under
`frontend/public/fonts/` (WOFF2, preview) and
`backend/app/services/render/caption_fonts/` (TTF/OTF, the ffmpeg/libass
burn) -- see those directories' NOTICE files for exact source/license per
family.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Fonts -- family, weight, a plausible system fallback stack. Family names
# here must match frontend globals.css's `@font-face` declarations AND the
# self-hosted binaries' embedded name exactly, or preview/export diverge.
# --------------------------------------------------------------------------

FONTS: Dict[str, Dict[str, Any]] = {
    "montserrat": {
        "family": "Montserrat", "weight": 800, "archetype": "modern_workhorse",
        "fallback_stack": "'Montserrat', Arial, sans-serif", "license": "OFL",
    },
    "anton": {
        "family": "Anton", "weight": 400, "archetype": "punchy_creator",
        "fallback_stack": "'Anton', Impact, 'Arial Narrow Bold', sans-serif", "license": "OFL",
    },
    "jost": {
        "family": "Jost", "weight": 800, "archetype": "futura_geometric",
        "fallback_stack": "'Jost', Futura, 'Century Gothic', sans-serif", "license": "OFL",
    },
    "inter": {
        "family": "Inter", "weight": 600, "archetype": "neutral_standard",
        "fallback_stack": "'Inter', -apple-system, 'Segoe UI', sans-serif", "license": "OFL",
    },
}
DEFAULT_FONT_ID = "inter"

# --------------------------------------------------------------------------
# Colours -- 4 fixed swatches, no dynamic/grade-matched/palette source. Each
# swatch declares its own OWN outline/shadow ink: white/yellow/cyan (light
# text) get a dark outline+shadow when enabled; charcoal (dark text) gets a
# light one -- so legibility never depends on a runtime footage sample.
# --------------------------------------------------------------------------

_DARK_INK = "#000000"
_LIGHT_INK = "#FFFFFF"

COLOURS: Dict[str, Dict[str, str]] = {
    "white": {"label": "White", "hex": "#FFFFFF", "ink": _DARK_INK},
    "yellow": {"label": "Vibrant Yellow", "hex": "#FFEB3B", "ink": _DARK_INK},
    "cyan": {"label": "Cyan", "hex": "#00E5FF", "ink": _DARK_INK},
    "charcoal": {"label": "Charcoal", "hex": "#1A1A1A", "ink": _LIGHT_INK},
}
DEFAULT_COLOUR_ID = "white"

# --------------------------------------------------------------------------
# Outline / shadow -- booleans only; width/opacity are fixed constants owned
# by ass_export.py (ASS units) and caption-overlay.tsx (CSS px), not exposed
# here as tunables.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Position -- 4 fixed vertical placements, no dynamic/speaker/per-cut
# caption_zones routing for MVP styles (placement.py falls back to
# lower_third when analysis is unavailable).
# --------------------------------------------------------------------------

POSITIONS = ("lower_third", "center", "top", "bottom_dynamic")
DEFAULT_POSITION = "lower_third"

# --------------------------------------------------------------------------
# Animation -- 4 fixed presets, no per-style intensity/beat_sync/emphasis
# dial. Internal timing constants live in ass_export.py (export) and
# resolve-captions.ts (preview), kept in lockstep by convention + tests.
# --------------------------------------------------------------------------

ANIMATIONS = ("active_reader", "pop", "fade_up", "sequential_reveal")
DEFAULT_ANIMATION = "fade_up"

# --------------------------------------------------------------------------
# Case -- 4 values. "sentence" capitalizes the first word of each caption
# event and lowercases the rest (a caption event is the closest available
# unit to "a sentence" -- there's no punctuation-based sentence splitter in
# the transcript word stream).
# --------------------------------------------------------------------------

CASES = ("original", "sentence", "upper", "lower")
DEFAULT_CASE = "original"

# --------------------------------------------------------------------------
# Size -- 4 values, mapped to a fixed frame-height percentage used
# identically by preview (caption-overlay.tsx) and export (ass_export.py).
# --------------------------------------------------------------------------

SIZES = ("small", "regular", "large", "xl")
DEFAULT_SIZE = "regular"
SIZE_FRAME_PCT: Dict[str, float] = {"small": 0.036, "regular": 0.045, "large": 0.055, "xl": 0.065}
# Max characters/line is NOT a user-facing control, but it must shrink as
# text gets bigger or a large/XL caption overflows its safe box -- purely a
# function of `size`, same rationale `LEVELS_SLOPE_MAX`-style caps elsewhere
# in this codebase use (a derived safety bound, not a dial).
MAX_CHARS_PER_LINE: Dict[str, int] = {"small": 42, "regular": 34, "large": 28, "xl": 22}
MAX_LINES = 2


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------

@dataclass
class CaptionStyle:
    style_id: str
    label: str
    tier: str = "standard"          # "suggested" | "standard"
    font_id: str = DEFAULT_FONT_ID
    colour_id: str = DEFAULT_COLOUR_ID
    outline_enabled: bool = False
    shadow_enabled: bool = False
    position: str = DEFAULT_POSITION
    animation: str = DEFAULT_ANIMATION
    case: str = DEFAULT_CASE
    size: str = DEFAULT_SIZE
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        font = FONTS.get(self.font_id, FONTS[DEFAULT_FONT_ID])
        colour = COLOURS.get(self.colour_id, COLOURS[DEFAULT_COLOUR_ID])
        return {
            "style_id": self.style_id,
            "label": self.label,
            "tier": self.tier,
            "font": {
                "font_id": self.font_id, "family": font["family"], "weight": font["weight"],
                "fallback_stack": font["fallback_stack"],
            },
            "colour": {
                "colour_id": self.colour_id, "fill": colour["hex"],
                "outline_enabled": self.outline_enabled, "shadow_enabled": self.shadow_enabled,
                "outline": colour["ink"], "shadow": colour["ink"],
            },
            "position": self.position,
            "animation": self.animation,
            "case": self.case,
            "size": self.size,
            "size_pct": SIZE_FRAME_PCT.get(self.size, SIZE_FRAME_PCT[DEFAULT_SIZE]),
            "max_lines": MAX_LINES,
            "max_chars_per_line": MAX_CHARS_PER_LINE.get(self.size, MAX_CHARS_PER_LINE[DEFAULT_SIZE]),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptionStyle":
        """Reconstruct from a `to_dict()`-shaped snapshot. Detects and
        normalizes the OLD (pre-MVP) shape -- nested `font.font_id`/
        `animation.preset`/`placement.anchor`/`colour.source` -- so a
        document saved before this rewrite still resolves to a valid, close
        MVP-catalog style instead of crashing or silently going blank. Never
        rewrites the stored snapshot; this is a read-time normalization
        only (the plan's "preserve old saved documents")."""
        if _looks_legacy(d):
            return _from_legacy_dict(d)
        font_id = d.get("font", {}).get("font_id") if isinstance(d.get("font"), dict) else None
        colour_block = d.get("colour") if isinstance(d.get("colour"), dict) else {}
        return cls(
            style_id=d.get("style_id") or "custom",
            label=d.get("label") or "Custom",
            tier=d.get("tier") or "standard",
            font_id=font_id if font_id in FONTS else DEFAULT_FONT_ID,
            colour_id=colour_block.get("colour_id") if colour_block.get("colour_id") in COLOURS else DEFAULT_COLOUR_ID,
            outline_enabled=bool(colour_block.get("outline_enabled", False)),
            shadow_enabled=bool(colour_block.get("shadow_enabled", False)),
            position=d.get("position") if d.get("position") in POSITIONS else DEFAULT_POSITION,
            animation=d.get("animation") if d.get("animation") in ANIMATIONS else DEFAULT_ANIMATION,
            case=d.get("case") if d.get("case") in CASES else DEFAULT_CASE,
            size=d.get("size") if d.get("size") in SIZES else DEFAULT_SIZE,
            rationale=d.get("rationale"),
        )


def apply_overrides(style: CaptionStyle, overrides: Optional[Dict[str, Any]]) -> CaptionStyle:
    """A shallow, field-level patch: overrides never compose/accumulate like
    a grade delta -- each field is either the base style's value or the
    override's, nothing in between. Only the MVP-public fields are
    patchable (font/colour/position/animation/case/size/outline/shadow)."""
    if not overrides:
        return style
    import copy
    s = copy.deepcopy(style)
    if overrides.get("font_id") in FONTS:
        s.font_id = overrides["font_id"]
    if overrides.get("colour_id") in COLOURS:
        s.colour_id = overrides["colour_id"]
    if "outline_enabled" in overrides:
        s.outline_enabled = bool(overrides["outline_enabled"])
    if "shadow_enabled" in overrides:
        s.shadow_enabled = bool(overrides["shadow_enabled"])
    if overrides.get("position") in POSITIONS:
        s.position = overrides["position"]
    if overrides.get("animation") in ANIMATIONS:
        s.animation = overrides["animation"]
    if overrides.get("case") in CASES:
        s.case = overrides["case"]
    if overrides.get("size") in SIZES:
        s.size = overrides["size"]
    return s


# --------------------------------------------------------------------------
# Legacy (pre-MVP) snapshot normalization
# --------------------------------------------------------------------------

_LEGACY_FONT_MAP = {
    "inter_tight": "inter", "poppins_extrabold": "montserrat", "anton": "anton",
    "nunito": "inter", "fraunces": "inter", "permanent_marker": "inter",
}
_LEGACY_ANIMATION_MAP = {"fade": "fade_up", "karaoke": "active_reader", "slide": "fade_up", "pop": "pop"}
_LEGACY_POSITION_MAP = {
    "lower_third": "lower_third", "center": "center", "top": "top",
    "dynamic": "lower_third", "speaker": "lower_third",
}
_LEGACY_CASE_MAP = {"as-is": "original", "upper": "upper"}


def _looks_legacy(d: Dict[str, Any]) -> bool:
    """The MVP shape's `colour` is a dict with `colour_id`; the legacy
    shape's `colour` is a dict with `source`. `position`/`animation` are
    flat strings in the MVP shape, nested dicts (`placement.anchor`,
    `animation.preset`) in the legacy one -- any of these is enough to tell
    them apart."""
    colour = d.get("colour")
    if isinstance(colour, dict) and "source" in colour:
        return True
    if isinstance(d.get("animation"), dict):
        return True
    if isinstance(d.get("placement"), dict):
        return True
    return False


def _nearest_colour_id(hexcolor: Optional[str]) -> str:
    """Nearest of the 4 fixed swatches by RGB Euclidean distance -- the
    plan's "dynamic colour source -> nearest fixed palette colour" mapping."""
    if not hexcolor:
        return DEFAULT_COLOUR_ID
    h = hexcolor.lstrip("#")
    if len(h) != 6:
        return DEFAULT_COLOUR_ID
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return DEFAULT_COLOUR_ID
    best_id, best_dist = DEFAULT_COLOUR_ID, float("inf")
    for cid, spec in COLOURS.items():
        ch = spec["hex"].lstrip("#")
        cr, cg, cb = (int(ch[i:i + 2], 16) for i in (0, 2, 4))
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_id, best_dist = cid, dist
    return best_id


def _from_legacy_dict(d: Dict[str, Any]) -> CaptionStyle:
    font_block = d.get("font") if isinstance(d.get("font"), dict) else {}
    anim_block = d.get("animation") if isinstance(d.get("animation"), dict) else {}
    place_block = d.get("placement") if isinstance(d.get("placement"), dict) else {}
    colour_block = d.get("colour") if isinstance(d.get("colour"), dict) else {}

    legacy_font_id = font_block.get("font_id")
    legacy_case = font_block.get("case") or d.get("case")
    legacy_preset = anim_block.get("preset")
    legacy_anchor = place_block.get("anchor")
    legacy_fill = colour_block.get("fill")

    # The legacy system always rendered outline + shadow (no toggle existed)
    # -- preserve that look for old documents rather than silently stripping
    # it, which would visually change every saved caption at once.
    return CaptionStyle(
        style_id=d.get("style_id") or "custom",
        label=d.get("label") or "Custom",
        tier=d.get("tier") or "standard",
        font_id=_LEGACY_FONT_MAP.get(legacy_font_id, DEFAULT_FONT_ID),
        colour_id=_nearest_colour_id(legacy_fill),
        outline_enabled=True,
        shadow_enabled=True,
        position=_LEGACY_POSITION_MAP.get(legacy_anchor, DEFAULT_POSITION),
        animation=_LEGACY_ANIMATION_MAP.get(legacy_preset, DEFAULT_ANIMATION),
        case=_LEGACY_CASE_MAP.get(legacy_case, DEFAULT_CASE),
        size=DEFAULT_SIZE,
        rationale=d.get("rationale"),
    )


# --------------------------------------------------------------------------
# The permanent Standard (never regenerated -- see caption_style_mvp.plan.md
# "Permanent Standard"). Reset restores exactly these values.
# --------------------------------------------------------------------------

STANDARD = CaptionStyle(
    style_id="std_standard", label="Standard", tier="standard",
    font_id="inter", colour_id="white",
    outline_enabled=False, shadow_enabled=True,
    position="lower_third", animation="fade_up",
    case="original", size="regular",
)


def get_standard(style_id: Optional[str] = None) -> Optional[CaptionStyle]:
    """`style_id` is accepted (not required) for backward call-site
    compatibility -- there is exactly one Standard now, so any id (or None)
    resolves to it. A caller passing a stale Standards-catalog id from
    before this rewrite (e.g. "std_bold_yellow") still gets a valid style
    rather than None."""
    return STANDARD


def list_standards() -> List[Dict[str, Any]]:
    return [STANDARD.to_dict()]


def list_fonts() -> List[Dict[str, Any]]:
    return [{"font_id": k, **v} for k, v in FONTS.items()]


def list_colours() -> List[Dict[str, Any]]:
    return [{"colour_id": k, **v} for k, v in COLOURS.items()]


_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")


def slugify_style_id(prefix: str, *parts: str) -> str:
    """A stable, readable style_id for a generated suggestion, e.g.
    `slugify_style_id("sugg", "montserrat", "yellow", "pop")` ->
    `"sugg_montserrat_yellow_pop"`. Used by suggest.py so a given
    combination always gets the same id across regenerations (only the
    SET of 4 offered ids changes, not what a fixed id means)."""
    bits = [_IDENTIFIER_RE.sub("_", p.lower()).strip("_") for p in parts if p]
    return "_".join([prefix, *bits]) if bits else prefix
