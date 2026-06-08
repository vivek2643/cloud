"""Recipe registry: style key -> Recipe instance.

Every style is a declarative StylePolicy (see policy.py) interpreted by the one
general PolicyRecipe. Adding a style = adding a policy, not a class.
"""
from __future__ import annotations

from typing import Dict, List

from app.services.l3.recipes.base import Recipe
from app.services.l3.recipes.policy import STYLE_POLICIES, PolicyRecipe

_INSTANCES: List[Recipe] = [PolicyRecipe(p) for p in STYLE_POLICIES.values()]

RECIPES: Dict[str, Recipe] = {r.key: r for r in _INSTANCES}

# Default fallback style when the planner names an unknown style.
DEFAULT_STYLE = "vlog"


def get_recipe(style: str) -> Recipe:
    return RECIPES.get(style) or RECIPES[DEFAULT_STYLE]


def list_styles() -> List[dict]:
    return [{"key": r.key, "label": r.label} for r in _INSTANCES]
