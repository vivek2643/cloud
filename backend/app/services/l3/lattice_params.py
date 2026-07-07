"""
Tuning knobs for the cuts-v3 lattice builder (``lattice.py``). See
cuts_v3.plan.md, section 2, and cuts_v3_deterministic_keep.plan.md.

Deterministic-keep philosophy: this file holds ONLY perceptual constants
(rooted in human vision, not tunable band-aids). Every keep/drop/action
threshold and every camera-classification cutoff that used to live here is
gone -- where motion rises and falls is now read off each clip's OWN energy
histogram (``lattice._otsu``), and camera behaviour is categorized by the LLM
from the raw ``mot``/``coh`` numbers, never from a hand-set constant like
0.35 / 0.5 / 0.7.
"""
from __future__ import annotations

# A video atom shorter than this isn't its own atom -- merged forward so the
# non-speech remainder still has total coverage, and the same floor a regime
# flicker must clear to count as a real regime. Finer than base_cuts'
# MIN_CUT_MS (400ms): atoms are meant to over-segment (the LLM merges them
# back in pass 1/2; under-split is the only fatal error). Perceptual, not
# footage-relative -- a shard this short reads as a flash on any clip.
MIN_ATOM_MS = 200
