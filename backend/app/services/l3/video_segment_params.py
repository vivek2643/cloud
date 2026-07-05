"""
Tuning knobs for cuts-v2 camera-move-state video segmentation
(``video_segments.py``, Phase C1 of ``cuts_v2_boundaries.plan.md``), in one
place. Mirrors ``partition_params.py``'s convention: heuristic defaults, parked
here for re-tuning via the validation harness (``scripts/viz_cuts.py``)
against real clips.
"""
from __future__ import annotations

# --- Hold / move classification (energy-independent -- real detection) -----
# camera_stability is ABSOLUTE (not per-clip normalized, unlike camera_motion
# and action_energy) -- the reliable signal the plan calls out. A hop reads as
# a HOLD when the camera is both steady (no jerk) and not moving much;
# anything else is a candidate MOVE hop. The camera_motion threshold is
# known-imperfect (file-relative, like partition_params.ACTION_CALM_PCTL was
# for the retired impact-window detector).
STABILITY_HOLD_MIN = 0.75
MOTION_HOLD_MAX = 0.15

# Hysteresis: a hop's CONFIRMED state only flips after this many CONSECUTIVE
# hops of the new raw state, so one noisy hop can't flap the segmentation.
HYSTERESIS_HOPS = 3

# A hold shorter than this is a brief steadying mid-move, not a real settle --
# folded into its surrounding segment instead of becoming its own boundary.
HOLD_MIN_MS = 500

# --- Broad-band merge (fewer, longer segments) ------------------------------
# At Broad, a settle only counts as a real boundary if the hold it starts runs
# at least this long -- shorter holds (a brief steadying mid-pan) merge away.
BROAD_HOLD_MIN_MS = 2500

# --- Subject-motion-beat sub-split (Balanced and above) ---------------------
# Within one camera-based segment, a subject-motion beat becomes an
# additional split point when it's a strong, isolated local maximum of the
# SEGMENT'S OWN action energy. The floor loosens (more beats admitted) as the
# dial rises Balanced -> Sharp; Broad/Calm never sub-split at all -- this is
# the dial's GRANULARITY axis for video (a deliberate departure from
# cuts_v2.plan.md's "detect once, energy-independent" North Star; see
# cuts_v2_boundaries.plan.md's "Honest risks" #1).
BEAT_FLOOR_PCTL_BALANCED = 85.0
BEAT_FLOOR_PCTL_TIGHT = 70.0
BEAT_FLOOR_PCTL_SHARP = 55.0
BEAT_MIN_GAP_MS = 500
# A window whose action energy barely varies has no real "beat" to split on --
# a percentile floor over a near-flat window collapses to the flat value
# itself, which would otherwise let local-maxima fire on every hop (verified
# against a synthetic flat-baseline clip, which over-split into a cut every
# 500ms). Require genuine dynamic range before considering any split point.
BEAT_MIN_DYNAMIC_RANGE = 0.15

# --- Tagging (PROVISIONAL) ---------------------------------------------------
# Real done-vs-shown LABELING waits on the image pass (Phase C3); this is only
# enough of a heuristic to keep the two tags meaningful in the meantime. A
# segment tags `done` when its own peak action energy clears this percentile
# of the WHOLE CLIP's action energy; otherwise `shown`. A clip whose action
# energy barely varies has nothing to distinguish -- default everything
# `shown` rather than let a flat baseline's own percentile trivially "clear"
# itself (the same flat-signal trap BEAT_MIN_DYNAMIC_RANGE guards above).
DONE_TAG_PCTL = 60.0
DONE_TAG_MIN_DYNAMIC_RANGE = 0.15
