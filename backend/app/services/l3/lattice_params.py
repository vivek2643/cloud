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
# flicker must clear to count as a real regime. Finer than the old base-cuts
# MIN_CUT_MS (400ms): atoms are meant to over-segment (the LLM merges them
# back in pass 1/2; under-split is the only fatal error). Perceptual, not
# footage-relative -- a shard this short reads as a flash on any clip.
MIN_ATOM_MS = 200

# Two boundaries closer than this are the SAME cut point (one event seen by two
# signals) -- collapse them so we don't emit a hairline sliver between.
SNAP_MS = 150

# Words are merged into one speech TURN until the silence between them exceeds
# this -- so a gap longer than this is an INTENTIONAL pause (a real breath/stop)
# and becomes a boundary; anything shorter is just the rhythm of talking and is
# kept inside one turn. (A speaker CHANGE always breaks a turn regardless.)
LONG_PAUSE_MS = 1200
