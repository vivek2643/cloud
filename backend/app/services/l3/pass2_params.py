"""
Tuning knobs for the cuts-v3 pass-2 vision call (``pass2.py``). See
cuts_v3.plan.md, section 5: "sharded ONLY by context budget (<= ~120k tokens
of images per call, whole clips per shard)".
"""
from __future__ import annotations

# Claude's image token cost is roughly (width_px * height_px) / 750. At the
# plan's ~768px-wide stills (768x432 for 16:9 source, a bit more for taller
# aspect ratios), that's ~500 tokens/image with headroom. Conservative on
# purpose: better to shard one call early than overflow one late.
EST_TOKENS_PER_IMAGE = 500
MAX_IMAGE_TOKENS_PER_SHARD = 120_000
MAX_IMAGES_PER_SHARD = MAX_IMAGE_TOKENS_PER_SHARD // EST_TOKENS_PER_IMAGE  # 240

STILL_WIDTH_PX = 768

# Additional shard cap beyond the plan's image-token budget, added after
# real ingest runs: a shard's OUTPUT (one full Pass2Cut record per
# speech_cut/video_tentative_group, each a verbose nested object) grows
# with cut count independent of image count, and at ~40-80 cuts in one
# call the model measurably starts stringifying/dropping fields under its
# own complexity -- not a truncation (plenty of token budget left), a
# reliability cliff in generating one very large nested structure. Capping
# cuts per shard, not just images, keeps each call's output smaller.
# Lowered 25 -> 15 after observing the missing-field failure recur even on
# a 17-cut shard -- this isn't purely a size cliff, so this is a mitigation
# (smaller blast radius per call) more than a guaranteed fix.
MAX_CUTS_PER_SHARD = 15
