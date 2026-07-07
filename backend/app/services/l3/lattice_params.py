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

# Camera descriptor for the atom-table text. An atom is now a WHOLE coherent
# span (a pan is one atom, not [hold][move][settle] -- camera moves stopped
# being atom boundaries in the editorial pass, see cuts_v3_editorial.plan.md
# section B), so the descriptor is classified from the DOMINANT confirmed
# camera state over the span, not the diluted mean:
#   * < CAMERA_MOVE_FRAC_MIN of the span in confirmed MOVE  -> "hold"
#   * otherwise a MOVE atom reads "pan" (a clean, deliberate move) when both
#     coherence and stability are high, else "handheld" (shaky / incoherent).
CAMERA_MOVE_FRAC_MIN = 0.35
CAMERA_HOLD_MOTION_MAX = 0.15
CAMERA_PAN_COHERENCE_MIN = 0.7
CAMERA_PAN_STABILITY_MIN = 0.6

# Action promotion (editorial pass, section C). An action span -- a cluster of
# L1-detected action_points (discrete impacts: a hit, catch, jump) inside the
# non-speech remainder -- is carved into its OWN atom and typed "action", so a
# subject-motion payoff is a first-class candidate cut, never buried in a big
# "settle" atom. Anchors are the trustworthy, genre-robust signal here (L1
# detects them with its own normalization), so promotion keys off them, NOT an
# absolute energy threshold that wouldn't travel across footage types.
#   * anchors within ACTION_ANCHOR_MERGE_MS of each other are ONE action span,
#   * padded by an ANCHOR-RELATIVE wind-up/follow-through (see below).
ACTION_ANCHOR_MERGE_MS = 700

# Boundaries-v2 (cuts_v3_boundaries_v2.plan.md): the action pad is no longer a
# flat magic-ms. It scales with the cluster's OWN rhythm -- a fast flurry of
# impacts gets a tight pad, a lone slow swing breathes more -- so it travels
# across footage without per-clip tuning. pad = ACTION_PAD_FRAC * median gap
# between the cluster's anchors, floored at PERCEPTUAL_FLOOR_MS. Naturally
# bounded: a cluster only merges anchors within ACTION_ANCHOR_MERGE_MS, so the
# median gap (hence the pad) can't run away. A single-anchor cluster has no gap
# and falls back to the floor.
ACTION_PAD_FRAC = 0.5

# The one honest constant: a floor rooted in perception, not footage. A span
# shorter than this reads as a flash, never a "shot" -- it's the minimum pad and
# the minimum a cut can ever be tightened to by the dial. Deliberately NOT
# clip-relative (a usable shot length does not depend on the source file length).
PERCEPTUAL_FLOOR_MS = 200
