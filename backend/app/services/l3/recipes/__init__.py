"""
Recipes: deterministic "cooking" of editorial units into an A/V timeline.

Each recipe is a standard editing style. The LLM picks WHICH recipe(s) and
supplies symbolic guidance (target length, footage scope, intent); the recipe
does the precise math -- selecting units, snapping cuts to real boundaries,
shaping pace -- and emits an AVTimeline (independent video + audio tracks).

The composer runs one recipe per section and stitches the results, so a single
edit can mix styles.
"""
from __future__ import annotations

from app.services.l3.recipes.base import (
    AVTimeline,
    PlacedClip,
    Recipe,
    RecipeContext,
    SectionPlan,
)
from app.services.l3.recipes.registry import RECIPES, get_recipe, list_styles

__all__ = [
    "AVTimeline",
    "PlacedClip",
    "Recipe",
    "RecipeContext",
    "SectionPlan",
    "RECIPES",
    "get_recipe",
    "list_styles",
]
