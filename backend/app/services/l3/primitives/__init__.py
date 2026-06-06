"""
Shared deterministic primitives for the AI editor.

This package is the single foundation every recipe/style builds on. The
guiding principle: the LLM makes symbolic choices (style, beats, footage
regions), while everything in here computes precise, math-based results from
the L1/L2 analysis we already store -- cut boundaries, quality scores, and the
editorial units the editor actually assembles.

Modules:
  loader      - load_file_analyses(): pull L1/L2 data for a set of files.
  boundaries  - detect real cut points (speech, silence, beats, motion).
  quality     - per-modality 0..1 quality + combined take_quality.
  units       - build SpeechUnit / VisualUnit (snapped to real boundaries).
  takes       - cluster near-duplicate takes; pick the best.
"""
from __future__ import annotations

from app.services.l3.primitives.boundaries import (
    Boundary,
    beat_grid,
    motion_boundaries,
    silence_boundaries,
    snap_to_boundary,
    speech_boundaries,
)
from app.services.l3.primitives.loader import (
    AudioData,
    FileAnalysis,
    ShotRow,
    TranscriptData,
    WordTok,
    load_file_analyses,
)
from app.services.l3.primitives.quality import (
    energy_score,
    speech_quality,
    take_quality,
    visual_quality,
)
from app.services.l3.primitives.takes import dedup_speech_units, dedup_visual_units
from app.services.l3.primitives.units import EditUnit, build_units

__all__ = [
    "Boundary",
    "beat_grid",
    "motion_boundaries",
    "silence_boundaries",
    "snap_to_boundary",
    "speech_boundaries",
    "AudioData",
    "FileAnalysis",
    "ShotRow",
    "TranscriptData",
    "WordTok",
    "load_file_analyses",
    "energy_score",
    "speech_quality",
    "take_quality",
    "visual_quality",
    "dedup_speech_units",
    "dedup_visual_units",
    "EditUnit",
    "build_units",
]
