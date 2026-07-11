"""
Caption style bundles (captions.plan.md SS3/SS5): the `CaptionStyle` schema
plus the curated catalogs (fonts/animations/placements/colours) it's built
from. Pure data -- no I/O, no signals -- everything downstream (resolver,
suggest, ass_export) composes these dicts.

v1 animation vocabulary is DELIBERATELY restricted (SS16 resolved #2) to the
four effects that render *identically* in the DOM/canvas preview and in
ASS/libass: fade, pop (scale/colour emphasis), karaoke (word-fill), slide.
Typewriter / word-bounce / highlight-box (SS5's fuller list) need true
per-frame kinetic typography for a faithful ASS parity -- deferred to phase 2
(SS15) rather than shipped as a lookalike approximation that would quietly
diverge between preview and export.

Font set: 6 curated families, all SIL Open Font License (so they can be
self-hosted + embedded in an ffmpeg burn freely -- SS16 "lock ~6 SIL OFL
fonts"). Only FAMILY METADATA lives here; no binaries are bundled by this
change (see `frontend/public/fonts/` -- the existing Telegraf entry is the
same "path exists, files dropped in later" precedent, `globals.css` already
falls back to the system stack when a font file is absent). Both the CSS
`@font-face` the frontend registers and the `fontfile`/`fontsdir` ffmpeg burn
need real files before a family renders as anything but its system fallback;
until then every family still degrades to a legible system font, never a
blank/broken caption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Fonts (SS5) -- family, weight, a plausible system fallback stack, and the
# self-hosted file stem this family will resolve to once real font binaries
# land under frontend/public/fonts/ + a backend fonts dir (SS12/SS13).
# --------------------------------------------------------------------------

FONTS: Dict[str, Dict[str, Any]] = {
    "anton": {
        "family": "Anton", "weight": 400, "archetype": "condensed_impact",
        "fallback_stack": "'Anton', Impact, 'Arial Narrow Bold', sans-serif",
        "license": "OFL",
    },
    "poppins_extrabold": {
        "family": "Poppins", "weight": 800, "archetype": "bold_geometric",
        "fallback_stack": "'Poppins', 'Montserrat', Arial, sans-serif",
        "license": "OFL",
    },
    "inter_tight": {
        "family": "Inter Tight", "weight": 600, "archetype": "neutral_workhorse",
        "fallback_stack": "'Inter Tight', Inter, -apple-system, sans-serif",
        "license": "OFL",
    },
    "nunito": {
        "family": "Nunito", "weight": 700, "archetype": "rounded_friendly",
        "fallback_stack": "'Nunito', 'Comic Sans MS', 'Trebuchet MS', sans-serif",
        "license": "OFL",
    },
    "fraunces": {
        "family": "Fraunces", "weight": 600, "archetype": "editorial",
        "fallback_stack": "'Fraunces', Georgia, 'Times New Roman', serif",
        "license": "OFL",
    },
    "permanent_marker": {
        "family": "Permanent Marker", "weight": 400, "archetype": "marker_handwritten",
        "fallback_stack": "'Permanent Marker', 'Bradley Hand', cursive",
        "license": "OFL",
    },
}

# --------------------------------------------------------------------------
# Animations (SS5, restricted to the v1 parity-safe set -- SS16 #2)
# --------------------------------------------------------------------------

ANIMATION_PRESETS = ("fade", "pop", "karaoke", "slide")
EMPHASIS_MODES = ("semantic", "loudness", "none")


@dataclass
class AnimationSpec:
    preset: str = "fade"           # one of ANIMATION_PRESETS
    intensity: float = 0.6         # 0..1, scales overshoot/emphasis amplitude
    beat_sync: bool = False
    emphasis: str = "loudness"     # one of EMPHASIS_MODES


# --------------------------------------------------------------------------
# Placement (SS5/SS9)
# --------------------------------------------------------------------------

PLACEMENT_ANCHORS = ("lower_third", "center", "top", "dynamic", "speaker")


@dataclass
class PlacementSpec:
    anchor: str = "dynamic"
    safe_area: bool = True
    # Minimum time (ms) a placement is held before it's allowed to move again
    # (SS9 hysteresis) -- computed per weld-run by placement.py, this is just
    # the style's stated preference for how "sticky" placement should feel.
    stability_ms: int = 1200


# --------------------------------------------------------------------------
# Colour (SS5/SS11)
# --------------------------------------------------------------------------

COLOUR_SOURCES = ("white", "black_box", "match_grade", "palette_accent", "high_contrast")


@dataclass
class ColourSpec:
    source: str = "white"
    fill: str = "#ffffff"
    emphasis_fill: str = "#ffffff"
    outline: str = "#000000"
    shadow: str = "#000000"
    box: Optional[str] = None      # box fill colour, only used by black_box


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------

@dataclass
class CaptionStyle:
    style_id: str
    label: str
    tier: str = "standard"         # "suggested" | "standard"
    font_id: str = "inter_tight"
    case: str = "as-is"            # "as-is" | "upper"
    tracking: float = 0.0          # letter-spacing, em
    max_lines: int = 2
    max_chars_per_line: int = 32
    animation: AnimationSpec = field(default_factory=AnimationSpec)
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    colour: ColourSpec = field(default_factory=ColourSpec)
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        font = FONTS.get(self.font_id, FONTS["inter_tight"])
        return {
            "style_id": self.style_id,
            "label": self.label,
            "tier": self.tier,
            "font": {
                "font_id": self.font_id,
                "family": font["family"], "weight": font["weight"],
                "fallback_stack": font["fallback_stack"],
                "case": self.case, "tracking": self.tracking,
                "max_lines": self.max_lines, "max_chars_per_line": self.max_chars_per_line,
            },
            "animation": {
                "preset": self.animation.preset, "intensity": self.animation.intensity,
                "beat_sync": self.animation.beat_sync, "emphasis": self.animation.emphasis,
            },
            "placement": {
                "anchor": self.placement.anchor, "safe_area": self.placement.safe_area,
                "stability_ms": self.placement.stability_ms,
            },
            "colour": {
                "source": self.colour.source, "fill": self.colour.fill,
                "emphasis_fill": self.colour.emphasis_fill, "outline": self.colour.outline,
                "shadow": self.colour.shadow, "box": self.colour.box,
            },
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptionStyle":
        """Reconstruct from a `to_dict()`-shaped snapshot (SS3 "resolved
        style properties inlined too" -- a document's persisted
        `captions.base_style` is exactly this shape, so a Suggested pick
        stays resolvable even after its ephemeral suggest.py cache entry is
        gone / regenerated differently)."""
        font = d.get("font") or {}
        anim = d.get("animation") or {}
        place = d.get("placement") or {}
        colour = d.get("colour") or {}
        return cls(
            style_id=d.get("style_id") or "custom",
            label=d.get("label") or "Custom",
            tier=d.get("tier") or "standard",
            font_id=font.get("font_id") or "inter_tight",
            case=font.get("case") or "as-is",
            tracking=float(font.get("tracking") or 0.0),
            max_lines=int(font.get("max_lines") or 2),
            max_chars_per_line=int(font.get("max_chars_per_line") or 32),
            animation=AnimationSpec(
                preset=anim.get("preset") or "fade",
                intensity=float(anim.get("intensity") if anim.get("intensity") is not None else 0.6),
                beat_sync=bool(anim.get("beat_sync") or False),
                emphasis=anim.get("emphasis") or "loudness",
            ),
            placement=PlacementSpec(
                anchor=place.get("anchor") or "dynamic",
                safe_area=bool(place.get("safe_area") if place.get("safe_area") is not None else True),
                stability_ms=int(place.get("stability_ms") or 1200),
            ),
            colour=ColourSpec(
                source=colour.get("source") or "white",
                fill=colour.get("fill") or "#ffffff",
                emphasis_fill=colour.get("emphasis_fill") or "#ffffff",
                outline=colour.get("outline") or "#000000",
                shadow=colour.get("shadow") or "#000000",
                box=colour.get("box"),
            ),
            rationale=d.get("rationale"),
        )


def apply_overrides(style: CaptionStyle, overrides: Optional[Dict[str, Any]]) -> CaptionStyle:
    """A shallow, field-level patch (SS7 "Standards" refine-from-suggestion):
    overrides never compose/accumulate like a grade delta -- each field is
    either the base style's value or the override's, nothing in between."""
    if not overrides:
        return style
    import copy
    s = copy.deepcopy(style)
    if "font_id" in overrides and overrides["font_id"] in FONTS:
        s.font_id = overrides["font_id"]
    if "case" in overrides and overrides["case"] in ("as-is", "upper"):
        s.case = overrides["case"]
    if "tracking" in overrides:
        try:
            s.tracking = float(overrides["tracking"])
        except (TypeError, ValueError):
            pass
    if "max_lines" in overrides:
        try:
            s.max_lines = max(1, min(3, int(overrides["max_lines"])))
        except (TypeError, ValueError):
            pass
    if "max_chars_per_line" in overrides:
        try:
            s.max_chars_per_line = max(10, min(60, int(overrides["max_chars_per_line"])))
        except (TypeError, ValueError):
            pass
    anim = overrides.get("animation") or {}
    if anim.get("preset") in ANIMATION_PRESETS:
        s.animation.preset = anim["preset"]
    if "intensity" in anim:
        try:
            s.animation.intensity = max(0.0, min(1.0, float(anim["intensity"])))
        except (TypeError, ValueError):
            pass
    if "beat_sync" in anim:
        s.animation.beat_sync = bool(anim["beat_sync"])
    if anim.get("emphasis") in EMPHASIS_MODES:
        s.animation.emphasis = anim["emphasis"]
    place = overrides.get("placement") or {}
    if place.get("anchor") in PLACEMENT_ANCHORS:
        s.placement.anchor = place["anchor"]
    if "safe_area" in place:
        s.placement.safe_area = bool(place["safe_area"])
    colour = overrides.get("colour") or {}
    if colour.get("source") in COLOUR_SOURCES:
        s.colour.source = colour["source"]
    return s


# --------------------------------------------------------------------------
# Standards catalog (SS7): the universal, hand-authored building blocks.
# --------------------------------------------------------------------------

def _standard(style_id: str, label: str, **kw: Any) -> CaptionStyle:
    anim = kw.pop("animation", {})
    place = kw.pop("placement", {})
    colour = kw.pop("colour", {})
    return CaptionStyle(
        style_id=style_id, label=label, tier="standard",
        animation=AnimationSpec(**anim), placement=PlacementSpec(**place),
        colour=ColourSpec(**colour), **kw,
    )


STANDARDS: List[CaptionStyle] = [
    _standard(
        "std_clean_white", "Clean White", font_id="inter_tight", max_chars_per_line=32,
        animation={"preset": "fade", "intensity": 0.4, "emphasis": "none"},
        placement={"anchor": "dynamic"},
        colour={"source": "white", "fill": "#ffffff", "emphasis_fill": "#ffffff",
                "outline": "#000000", "shadow": "#000000"},
    ),
    _standard(
        "std_bold_caps", "Bold Caps", font_id="anton", case="upper", tracking=0.02,
        max_chars_per_line=24,
        animation={"preset": "pop", "intensity": 0.8, "beat_sync": True, "emphasis": "loudness"},
        placement={"anchor": "dynamic"},
        colour={"source": "high_contrast", "fill": "#ffffff", "emphasis_fill": "#ffe14d",
                "outline": "#000000", "shadow": "#000000"},
    ),
    _standard(
        "std_karaoke_box", "Karaoke Box", font_id="poppins_extrabold", max_chars_per_line=28,
        animation={"preset": "karaoke", "intensity": 0.6, "emphasis": "loudness"},
        placement={"anchor": "lower_third"},
        colour={"source": "black_box", "fill": "#ffffff", "emphasis_fill": "#ffffff",
                "outline": "#000000", "shadow": "#000000", "box": "#000000"},
    ),
    _standard(
        "std_editorial_serif", "Editorial Serif", font_id="fraunces", max_chars_per_line=36,
        animation={"preset": "fade", "intensity": 0.3, "emphasis": "semantic"},
        placement={"anchor": "center"},
        colour={"source": "match_grade", "fill": "#ffffff", "emphasis_fill": "#ffffff",
                "outline": "#000000", "shadow": "#000000"},
    ),
    _standard(
        "std_playful_pop", "Playful Pop", font_id="nunito", max_chars_per_line=26,
        animation={"preset": "pop", "intensity": 0.7, "beat_sync": True, "emphasis": "loudness"},
        placement={"anchor": "dynamic"},
        colour={"source": "palette_accent", "fill": "#ffffff", "emphasis_fill": "#ffffff",
                "outline": "#000000", "shadow": "#000000"},
    ),
    _standard(
        "std_marker_note", "Marker Note", font_id="permanent_marker", max_chars_per_line=22,
        animation={"preset": "slide", "intensity": 0.5, "emphasis": "none"},
        placement={"anchor": "top"},
        colour={"source": "white", "fill": "#ffffff", "emphasis_fill": "#ffffff",
                "outline": "#000000", "shadow": "#000000"},
    ),
]

_STANDARDS_BY_ID: Dict[str, CaptionStyle] = {s.style_id: s for s in STANDARDS}


def get_standard(style_id: str) -> Optional[CaptionStyle]:
    return _STANDARDS_BY_ID.get(style_id)


def list_standards() -> List[Dict[str, Any]]:
    return [s.to_dict() for s in STANDARDS]


def list_fonts() -> List[Dict[str, Any]]:
    return [{"font_id": k, **v} for k, v in FONTS.items()]
