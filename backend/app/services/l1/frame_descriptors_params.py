"""
Tuning knobs for the per-frame perceptual descriptor (pHash), in one place.

Mirrors ``scene_cuts_params.py``'s convention: heuristic defaults parked here so
the sampling rate and hash geometry can be re-tuned in one spot. The pHash *match
threshold* (how many bits of Hamming distance still counts as "same picture")
lives with the read-side consumer, not here -- this module only decides how the
signal is produced and stored.
"""
from __future__ import annotations

# Decode/sampling: one frame every 500ms (2 fps). Justified in brain_mirror.plan
# Phase 1 -- matches feel._JOIN_CONTIG_MS / footage_map._RUN_GAP_MS (500ms) so the
# nearest sampled frame to any seam instant is always within half a hop, while
# keeping storage trivial (~2 hashes per source second).
SAMPLE_FPS = 2

# pHash geometry: standard DCT perceptual hash. Decode a small luma frame at
# PHASH_SIDE x PHASH_SIDE, take the 2D DCT, keep the top-left low-frequency
# PHASH_LOWFREQ x PHASH_LOWFREQ block, and threshold each coefficient against the
# block's median (excluding the DC term) -> PHASH_LOWFREQ**2 = 64 bits.
PHASH_SIDE = 32
PHASH_LOWFREQ = 8
