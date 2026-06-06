"""Recipe registry: style key -> Recipe instance."""
from __future__ import annotations

from typing import Dict, List

from app.services.l3.recipes.base import Recipe
from app.services.l3.recipes.styles import (
    BeatSyncMusicMontage,
    CinematicBRoll,
    HighlightMontage,
    SocialShort,
    TalkingHead,
    Trailer,
    TutorialExplainer,
    VlogWalkthrough,
)

_INSTANCES: List[Recipe] = [
    HighlightMontage(),
    TalkingHead(),
    Trailer(),
    BeatSyncMusicMontage(),
    VlogWalkthrough(),
    SocialShort(),
    TutorialExplainer(),
    CinematicBRoll(),
]

RECIPES: Dict[str, Recipe] = {r.key: r for r in _INSTANCES}

# Default fallback style when the planner names an unknown style.
DEFAULT_STYLE = "vlog"


def get_recipe(style: str) -> Recipe:
    return RECIPES.get(style) or RECIPES[DEFAULT_STYLE]


def list_styles() -> List[dict]:
    return [{"key": r.key, "label": r.label} for r in _INSTANCES]
