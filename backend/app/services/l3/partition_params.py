"""
Tuning knobs for the cuts-v2 unified partition (``partition.py``), in one place.

Mirrors ``l1/cut_grid_params.py``'s convention: heuristic defaults, parked here
so they can be re-tuned in one place once the validation harness
(``scripts/viz_cuts.py``) has been run against real clips -- see
cuts_v2.plan.md's "Detection robustness" section.
"""
from __future__ import annotations

# --- Priority (North star #3: priority, deterministic) ----------------------
# Higher wins a contested span; the loser is demoted to a TAG on the winner
# (overlap >= OVERLAP_TAG_FRAC of the candidate) or trimmed to its free
# remainder. Values are the fused-field `energy`-independent CLAIM priority,
# not a display weight.
PRIORITY_SAID = 1.0
PRIORITY_DONE = 0.6
PRIORITY_SHOWN = 0.3

# A contested candidate is absorbed as a TAG (not its own cut) once this much
# of it is already claimed by a higher-priority cut; otherwise the free
# remainder becomes its own cut.
OVERLAP_TAG_FRAC = 0.60

# A free remainder shorter than this isn't a meaningful sub-unit -- it is
# silently absorbed (no cut, no tag: too small to be either) rather than
# emitted as a sliver cut. Keeps the bias toward over-split (many clean
# sub-units) without producing unusable micro-cuts.
MIN_SUBUNIT_MS = 300

# --- Video segment sub-split floor ------------------------------------------
# A camera-based video segment (see video_segments.py, cuts_v2_boundaries.plan
# Phase C1) must be at least this long before the dial's granularity axis will
# even consider sub-splitting it at a subject-motion beat -- too short to be
# worth it otherwise. Retired the old impact-WINDOW `done` detector
# (`ACTION_CALM_PCTL`/`DONE_MIN_MS`) it used to belong to; kept the same
# constant/value since the "how long before we bother subdividing further"
# meaning carried over cleanly.
GRAN_SPLIT_MIN_MS = 1600

# The tightest a video beat ever insets to (the peak-inset floor for done/shown)
# -- a FLOOR (minimum), never a ceiling; low energy keeps cuts full-length.
# v2-specific -- overrides energy.CORE_FLOOR_MS (600). Retuned per
# cuts_v2_boundaries.plan's "Tightness floor retune": a ~700-800ms safety net,
# not a full 1s -- let the per-band fraction do the work so cuts *generally*
# land ~1s but can dip a little under for a genuinely short beat. A beat whose
# proportional core is larger still keeps the larger core (the inset is a
# fraction of the beat's own span).
VIDEO_CORE_FLOOR_MS = 750

# --- Anchor-aware video tightness (cuts_v2_boundaries.plan follow-on) --------
# Tightness shrinks a video cut to its important core, but must NEVER trim off
# a real "payoff" instant (a ball-hit, an impact, an audio crack). So the inset
# is ANCHORED: the kept core is guaranteed to contain every anchor found inside
# the cut. Anchors are the SPARSE, genuinely-important instants only -- L1
# subject-motion impacts (action_points) + sharp audio-energy onsets -- NOT
# every motion wiggle (that would keep everything and defeat tightness). A cut
# with no anchors falls back to the plain peak-inset. This is the concrete
# encoding of "nothing important is ever missed" while still trimming dead time.
#
# Breathing room kept on each side of the anchor envelope (so a hit isn't
# clipped flush at frame zero -- keeps the wind-up / follow-through readable).
ANCHOR_PAD_MS = 250

# An audio ONSET anchor: a rise of at least this many dB over one prosody hop
# in the rms_db envelope -- a percussive transient (impact/crack/clap) that
# optical flow alone tends to miss (flow peaks on the swing, not the contact).
AUDIO_ANCHOR_RISE_DB = 6.0
# Minimum spacing between audio-onset anchors, so one loud event isn't counted
# as a dozen adjacent anchors.
AUDIO_ANCHOR_MIN_GAP_MS = 300

# --- Boundary snapping / merge -----------------------------------------------
# Two claimed cuts touching within this gap (after snapping) are the same
# continuous run, not two separate cuts with a hairline seam between them.
MERGE_GAP_MS = 60

# Fixed default tightness for B2's boundary SNAP only (candidate spans
# themselves are energy-independent by design -- see North Star #1/#5). B3
# layers the real tightness dial on top of the claimed cuts; this constant is
# just the field's attractor-weight lambda while detecting/claiming.
DEFAULT_SNAP_ENERGY = 0.5
