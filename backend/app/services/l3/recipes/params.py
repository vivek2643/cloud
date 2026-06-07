"""
Recipe parameters: the knobs the planner can turn per section.

Each recipe used to hard-code its pace and clip lengths. RecipeParams lifts
those into section.params so the LLM planner (and, crucially, the critic->replan
loop) can tune a section without forking a new recipe: slow a frantic montage,
shorten an over-long cut, push or pull the music bed, bias toward motion. All
knobs are optional -- absent params fall back to each recipe's original default,
so behavior is unchanged unless the planner asks for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# pace -> clip-length multiplier. "fast" holds clips ~half as long; "slow" ~1.7x.
PACE_MULT = {"slow": 1.7, "medium": 1.0, "fast": 0.55}


@dataclass
class RecipeParams:
    raw: Dict[str, Any]

    @classmethod
    def from_section(cls, section) -> "RecipeParams":
        return cls(dict(getattr(section, "params", None) or {}))

    # -- primitives --
    def _num(self, key: str) -> Optional[float]:
        v = self.raw.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def pace(self) -> str:
        p = str(self.raw.get("pace", "medium")).lower()
        return p if p in PACE_MULT else "medium"

    @property
    def pace_mult(self) -> float:
        return PACE_MULT[self.pace]

    # -- derived knobs (default = the recipe's original literal) --
    def max_clip_ms(self, base_ms: int) -> int:
        """Per-clip length cap. `max_clip_s` overrides outright; otherwise the
        recipe's base is scaled by pace."""
        ov = self._num("max_clip_s")
        if ov and ov > 0:
            return int(ov * 1000)
        return max(1, int(base_ms * self.pace_mult))

    def music_gain_db(self, default: float) -> float:
        ov = self._num("music_gain_db")
        return ov if ov is not None else default

    def energy_weight(self, default: float) -> float:
        """Weight on motion/energy when ranking visual units. Higher = punchier,
        action-forward selection; lower (or negative) = calmer, scenic."""
        ov = self._num("energy_weight")
        return ov if ov is not None else default

    def broll_ratio(self, default: float = 0.0) -> float:
        """0..1: fraction of a speech-led section to interleave with b-roll
        cutaways. The lightweight 'blend' knob -- lets a talking_head borrow
        montage texture without switching recipes."""
        ov = self._num("broll_ratio")
        if ov is None:
            return default
        return max(0.0, min(1.0, ov))
