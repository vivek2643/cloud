"""
Look layer, mode 1 -- authored parametric presets (color_grading.plan.md
SS7). Each preset is a CDL DELTA (composed on top of the correct/match
stack, same `cdl.compose` amplitude semantics as the arc layer -- never a
raw replacement), not a whole new creative LUT: the plan's own framing is
"modes 1-2 collapse into our parametric CDL spine," so a preset lives
entirely in `Grade` fields, no `.cube` needed.

Honest scope note: these ~12 recipes are a reasonable FIRST PASS at common
grading archetypes, hand-tuned by CDL math (teal/orange split-toning
approximated via asymmetric offset-vs-slope weighting -- offset biases
shadows, slope biases highlights, the standard CDL trick), not the product
of real footage + colorist review. The plan calls for "author ~10-12 canon
looks; taste-validate" -- the taste-validation half needs human eyes on
real footage, which this pass can't do. Treat these as a solid starting
catalog to refine, not a finished one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.services.l3.grade.cdl import Grade


@dataclass(frozen=True)
class Preset:
    preset_id: str
    label: str
    description: str
    grade: Grade


PRESETS: List[Preset] = [
    Preset(
        "natural", "Natural",
        "No stylization -- exactly the correct/match result, a clean baseline.",
        Grade(),
    ),
    Preset(
        "cinematic_teal_orange", "Cinematic Teal & Orange",
        "Cool teal shadows, warm orange highlights/skin, punchier contrast.",
        Grade(slope=(1.14, 1.0, 0.87), offset=(-0.025, 0.006, 0.035), power=(1.0, 1.0, 1.0), sat=1.18),
    ),
    Preset(
        "warm_film", "Warm Film",
        "Lifted, warm blacks and gently rolled-off highlights -- a soft film feel.",
        Grade(slope=(1.07, 1.0, 0.9), offset=(0.03, 0.016, -0.004), power=(0.96, 1.0, 1.04), sat=0.92),
    ),
    Preset(
        "cool_modern", "Cool Modern",
        "Clean, slightly cool, punchy contrast -- a contemporary commercial look.",
        Grade(slope=(0.93, 1.0, 1.1), offset=(-0.014, 0.0, 0.022), power=(1.05, 1.02, 1.0), sat=1.1),
    ),
    Preset(
        "moody_desaturated", "Moody Desaturated",
        "Cool, low-saturation, deeper shadows -- a somber/tense mood.",
        Grade(slope=(0.96, 0.99, 1.03), offset=(-0.02, -0.01, 0.0), power=(1.05, 1.03, 1.0), sat=0.72),
    ),
    Preset(
        "vintage_fade", "Vintage Fade",
        "Faded, lifted blacks with a warm midtone cast and softened contrast.",
        Grade(slope=(1.0, 0.98, 0.92), offset=(0.045, 0.03, 0.02), power=(0.96, 0.97, 1.0), sat=0.82),
    ),
    Preset(
        "high_contrast_mono", "High-Contrast Monochrome",
        "Near-desaturated with strong contrast -- a graphic, punchy black & white feel.",
        Grade(slope=(1.12, 1.12, 1.12), offset=(-0.03, -0.03, -0.03), power=(1.0, 1.0, 1.0), sat=0.06),
    ),
    Preset(
        "golden_hour", "Golden Hour",
        "Warm gold highlights and a gentle overall glow -- late-afternoon light.",
        Grade(slope=(1.11, 1.02, 0.83), offset=(0.025, 0.01, -0.012), power=(0.97, 1.0, 1.0), sat=1.13),
    ),
    Preset(
        "clean_corporate", "Clean Corporate",
        "Neutral with a light contrast lift -- crisp, professional, unobtrusive.",
        Grade(slope=(1.05, 1.05, 1.05), offset=(-0.012, -0.012, -0.012), power=(1.0, 1.0, 1.0), sat=1.06),
    ),
    Preset(
        "vibrant_vlog", "Vibrant Vlog",
        "Punchy saturation and clean whites -- energetic travel/lifestyle content.",
        Grade(slope=(1.03, 1.03, 1.0), offset=(0.0, 0.0, 0.0), power=(1.0, 1.0, 1.0), sat=1.22),
    ),
    Preset(
        "muted_editorial", "Muted Editorial",
        "Soft contrast, cool-neutral, restrained saturation -- an editorial/documentary feel.",
        Grade(slope=(0.98, 0.99, 1.01), offset=(0.01, 0.008, 0.0), power=(0.98, 0.98, 0.98), sat=0.85),
    ),
    Preset(
        "blue_hour", "Blue Hour",
        "Cool blue shadows and a moody, low-key overall cast -- night/dusk content.",
        Grade(slope=(0.87, 0.95, 1.17), offset=(-0.016, -0.008, 0.032), power=(1.04, 1.02, 1.0), sat=0.88),
    ),
]

_BY_ID: Dict[str, Preset] = {p.preset_id: p for p in PRESETS}


def get_preset(preset_id: str) -> Preset | None:
    return _BY_ID.get(preset_id)


def list_presets() -> List[Dict[str, str]]:
    """Gallery listing (SS12): id/label/description -- the frontend renders
    a live thumbnail per preset itself (baking a preview frame through each
    preset's cube is a UI-layer concern, not this catalog's). `mode` tags
    every entry "preset" (color_response_engine.plan.md's combined gallery
    also lists "engine" entries from `look_engine.list_engine_looks` --
    the frontend needs this to know which `look.mode`/id field to set)."""
    return [
        {"preset_id": p.preset_id, "label": p.label, "description": p.description, "mode": "preset"}
        for p in PRESETS
    ]
