"""
Tuning knobs for cuts-v2 speech granularity + prosody grading
(``partition.py``'s ``_said_candidates``, Phase C2 of
``cuts_v2_boundaries.plan.md``), in one place. Mirrors
``video_segment_params.py``'s convention: heuristic defaults, parked here for
re-tuning via the validation harness (``scripts/viz_cuts.py``) against real
clips.
"""
from __future__ import annotations

# --- Turn merge (Broad band only) -------------------------------------------
# Two consecutive same-speaker thoughts closer than this NEVER merge into one
# turn regardless of what prosody says -- an absolute ceiling so a truly long
# silence can't bridge just because pitch happened to look "sustained" right
# before it trails off into noise.
MAX_BRIDGE_GAP_MS = 6000

# --- Prosody grading ---------------------------------------------------------
# Read the trailing PITCH + ENERGY trend in this window just before a gap
# starts (the tail of the preceding thought) -- not the gap itself, which is
# silence and carries no pitch. A falling contour here is the classic
# declarative-statement-ending shape; a flat or rising one means the speaker
# isn't finished (a dramatic pause).
PROSODY_TAIL_MS = 500

# A pitch drop across the tail window of at least this many Hz reads as
# "falling" (terminal intonation). Below this, pitch counts as sustained.
PITCH_FALL_HZ = 15.0

# An energy drop across the tail window of at least this many dB reads as
# "dropping" (trailing off). Both this AND the pitch fall must hold for a gap
# to grade as a real break; either alone bridges (an intentional pause can
# still trail off in energy while holding pitch, or vice versa).
ENERGY_DROP_DB = 3.0
