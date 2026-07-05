"""
Tuning knobs for scene/shot detection (cuts v2), in one place.

Mirrors ``cut_grid_params.py``'s convention: heuristic defaults, not yet
empirically tuned against real footage -- parked here so they can be re-tuned
in one place once the validation harness (``scripts/viz_cuts.py``) has been run
against real clips, per the plan's "detection robustness" section.
"""
from __future__ import annotations

# Decode: tiny color frames are enough to catch a shot change; shot detection
# doesn't need motion_dynamics' 10fps -- composition doesn't change that fast.
SCENE_FPS = 5
SCENE_W = 64
SCENE_H = 36

# Per-frame histogram: Hue x Saturation only (ignore Value/brightness so a
# flash, an exposure ramp, or a light flicker doesn't read as a shot change).
SCENE_HUE_BINS = 16
SCENE_SAT_BINS = 8

# Frame-to-frame DRIFT = 1 - histogram correlation. UNLIKE motion_dynamics'
# action/camera channels (raw optical-flow magnitude has no fixed scale, so
# those normalize by percentile), correlation is ALREADY an absolute,
# meaningfully-scaled quantity (1 = identical .. 0/negative = uncorrelated) --
# so the floors below are absolute thresholds, not percentile-relative to the
# clip's own noise. A percentile floor would misread a stable, grainy clip's
# own frame-to-frame sensor noise as a "spike" once normalized up to its own
# (tiny) dynamic range -- verified against a synthetic static+noise clip, which
# false-positived on every hop under a percentile floor.
SHOT_DRIFT_FLOOR = 0.45     # correlation must drop below ~0.55 -- a real cut
SHOT_MIN_GAP_MS = 500        # min spacing between two accepted shot cuts

# A softer, within-shot COMPOSITION change (reframe, subject enters/exits) --
# same signal, lower bar.
COMPOSITION_DRIFT_FLOOR = 0.15
COMPOSITION_MIN_GAP_MS = 800
# A composition candidate this close to an accepted shot cut IS that shot cut,
# not a separate sub-boundary -- drop it rather than double-report the same
# instant under two kinds.
COMPOSITION_SHOT_MERGE_MS = 300
