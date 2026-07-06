"""
Tuning knobs for the cuts-v3 lattice builder (``lattice.py``). See
cuts_v3.plan.md, section 2.
"""
from __future__ import annotations

# A video atom shorter than this isn't its own atom -- merged forward so the
# non-speech remainder still has total coverage. Finer than base_cuts'
# MIN_CUT_MS (400ms): atoms are meant to over-segment (the LLM merges them
# back in pass 1/2; under-split is the only fatal error).
MIN_ATOM_MS = 200

# Coarse camera descriptor thresholds for the atom-table text (mean over the
# atom's own span): a MOVE atom reads "pan" (a clean, deliberate move) only
# when both coherence and stability are high; otherwise "handheld" (shaky /
# incoherent). A HOLD atom (mean camera_motion below this) always reads "hold".
CAMERA_HOLD_MOTION_MAX = 0.15
CAMERA_PAN_COHERENCE_MIN = 0.7
CAMERA_PAN_STABILITY_MIN = 0.6
