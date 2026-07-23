"""
Per-genre default for the cuts energy dial (the client-side view-math
tightness axis -- see cuts-view.tsx). The rest of this module (the cuts-v2
five-band EnergyParams mapping) was retired in cleanup.plan.md B3.
"""
from __future__ import annotations

from typing import Optional

# Genre -> default energy CENTER (the slider's starting point, not a cap; the
# editor still dials the full range). Long-form/observational sits low so shots
# breathe (a podcast, an interview, scenery); short-form/punchy sits high (a
# product/action reel cuts tight). Instructional content sits just below
# Balanced (steady, follow the steps). Tunable; unknown genres -> Balanced.
_GENRE_DEFAULT_ENERGY = {
    "interview": 0.3,
    "talking_head": 0.3,
    "scenic": 0.3,
    "broll": 0.3,
    "tutorial": 0.4,
    "demo": 0.4,
    "screen_recording": 0.4,
    "vlog": 0.5,
    "event": 0.5,
    "other": 0.5,
    "performance": 0.6,
    "product": 0.7,
    "action": 0.7,
}
DEFAULT_ENERGY = 0.5


def default_energy_for(content_type: Optional[str]) -> float:
    """The slider's starting energy for a detected genre (see the table). The
    editor can still move anywhere; this only sets where the dial opens."""
    return _GENRE_DEFAULT_ENERGY.get((content_type or "").strip().lower(), DEFAULT_ENERGY)
