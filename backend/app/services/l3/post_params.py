"""
Tuning knobs for the cuts-v3 post-compute assembly (``post.py``). See
cuts_v3.plan.md, section 6.
"""
from __future__ import annotations

# Breathing room kept on each side of a video cut's anchor envelope (so a hit
# isn't clipped flush at frame zero -- keeps the wind-up / follow-through
# readable). Extracted from the retired cuts-v2 partition_params.py
# (cleanup.plan.md B3); a shared v2/v3 concept.
ANCHOR_PAD_MS = 250

# How much an action_energy sample must move from a cut's own end-of-span
# value to count as "something new is happening" -- below this, extending
# max_ms further reveals nothing (cuts_v3.plan.md: "action_energy static =>
# it won't get better").
FLATLINE_BAND = 0.05

# Slow-mo / fast-mo sanity limits on a cut's pace multiplier, independent of
# any per-cut taste fence: below SPEED_FLOOR a "slow" cut reads as a freeze,
# above SPEED_CEIL a "fast" cut no longer reads as motion at all.
SPEED_FLOOR = 0.25
SPEED_CEIL = 4.0

# Fixed, product-wide target relative visual velocities for pace levels
# L1 (slowest) .. L5 (fastest). Monotonic increasing by construction --
# see compute_pace_envelope's saturation behavior.
PACE_LEVEL_TARGETS = (0.5, 0.8, 1.0, 1.3, 1.8)

# energy_grade band upper bounds (mean action_energy over the cut's span);
# anything above the last band's bound reads as "high".
ENERGY_GRADE_BANDS = (("calm", 0.2), ("active", 0.5))
