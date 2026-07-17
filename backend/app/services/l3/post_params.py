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

# --------------------------------------------------------------------------
# Camera-move label (post._classify_camera_move). These are ABSOLUTE, physical
# rates against the SIGNED camera-velocity series (camera_dx/dy in fractions of
# the frame per second, camera_zoom in scale-change per second) -- a pan is a
# pan regardless of a clip's own spread, so unlike the cut-structure signals
# these are fixed, interpretable thresholds, not clip-relative.
# --------------------------------------------------------------------------
# A translation (pan/tilt) counts as a real move once the NET displacement over
# the cut sweeps at least this fraction of the frame per second.
CAMERA_PAN_RATE = 0.06
# A zoom counts once the NET scale change reaches this much per second.
CAMERA_ZOOM_RATE = 0.04
# Below the move thresholds, the shot is "static" unless there's appreciable
# per-hop jitter with a near-zero net path (hand-held wobble) AND the global
# model doesn't hold together -- then it's "shaky".
CAMERA_SHAKE_RATE = 0.10        # summed |per-hop| travel / sec that reads as agitated
CAMERA_SHAKE_COHERENCE = 0.5    # mean coherence below this = not one rigid move
# A translation-dominant move where the subject is also busy and the frame
# holds together reads as the camera FOLLOWING the subject, not a free pan.
CAMERA_FOLLOW_ACTION = 0.35     # mean action_energy (file-normalized) over the span
CAMERA_FOLLOW_COHERENCE = 0.6

# --------------------------------------------------------------------------
# cuts_v4_segmentation.plan.md section 6: min_ms becomes CONTENT-aware for a
# V4 video cut instead of anchor-derived -- a sparse/monotonous span collapses
# hard at high energy (small floor), a dense one holds more room so real
# events aren't clipped. density (v4_segment.VideoCut.density) is 0..1;
# min_ms floors at V4_MIN_MS_FLOOR (sparse) and rises toward
# V4_MIN_MS_FLOOR + V4_MIN_MS_DENSE_BONUS (fully dense).
# --------------------------------------------------------------------------
V4_MIN_MS_FLOOR = 400
V4_MIN_MS_DENSE_BONUS = 1200
