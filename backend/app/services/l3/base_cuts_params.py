"""
Tuning knobs for the cuts-v2 BASE partition (``base_cuts.py``).

The base layer is the robust, deterministic skeleton the whole cuts pipeline
sits on: a clip sliced ONLY at cut points we trust unconditionally -- no dial,
no actions, no tags, no junk filtering. These thresholds define "trust". Parked
here so they can be tuned in one place against real clips (see
``scripts/viz_base_cuts.py``).
"""
from __future__ import annotations

# Two boundaries closer than this are the SAME cut point (one event seen by two
# signals, e.g. a speaker handover that's also a camera settle) -- collapse them
# so we don't emit a hairline sliver between.
SNAP_MS = 150

# A slice shorter than this isn't its own base cut -- it's merged into its
# neighbour so coverage stays total (nothing is ever dropped at the base).
MIN_CUT_MS = 400

# --- Speech (diarization) ----------------------------------------------------
# Words are merged into one speech TURN until the silence between them exceeds
# this -- so a gap longer than this is an INTENTIONAL pause (a real breath/stop)
# and becomes a boundary; anything shorter is just the rhythm of talking and is
# kept inside one turn. (A speaker CHANGE always breaks a turn regardless.)
LONG_PAUSE_MS = 1200

# --- Camera state (reused thresholds live in video_segment_params) -----------
# The HOLD/MOVE machine (steady + not-moving = HOLD, else MOVE) is imported from
# video_segments; every confirmed HOLD<->MOVE transition is a boundary
# ("cut when the camera starts moving, and again where it settles").

# --- Disturbance (bad-camera spans to cut around) ----------------------------
# A DISTURBANCE is a stretch of genuinely bad camera with NOTHING worth
# watching: shake / whip / focus-hunt AND no subject payoff. The two-part test
# is what keeps fast INTENTIONAL action (a follow-pan, a ball-hit) -- which also
# shakes the camera -- from being mislabeled as junk. No single signal is
# reliable and per-hop thresholding flip-flaps, so we score, smooth, then
# trigger with hysteresis. Detect the BAD span; its edges are boundaries
# (dropping it is a separate junk-removal job).
#
# Per-hop badness reuses L1's engineered signals (see motion_dynamics.py):
#   badness = camera_cut_cost * (1 - action_energy)
# camera_cut_cost = "how bad is the camera" -- already motion-gated (chaos +
#   transient + blur), so a STEADY camera watching moving subjects costs ~0.
# action_energy   = RANSAC-residual subject motion (camera-compensated), so it
#   is HIGH exactly when there's real content -> vetoes disturbance there.
DIST_ACTION_VETO = 1.0      # weight of the (1 - action_energy) veto term

# Smooth the badness score over this window before thresholding -- turns the
# spiky per-hop signal into a stable envelope so it can't oscillate.
DIST_SMOOTH_MS = 350

# Hysteresis (Schmitt trigger): ENTER a disturbance only when smoothed badness
# clears ON; stay in it until it drops below OFF. The gap between them is what
# stops a value hovering near one threshold toggling every hop.
DIST_ON = 0.35
DIST_OFF = 0.20

# A disturbance shorter than this is a blip, not a span worth a boundary.
DIST_MIN_MS = 300
# Two bad-camera runs separated by a gap this short are ONE disturbance (the
# shake dipped under threshold for an instant) -- bridge them so a pervasively
# shaky clip reads as a single disturbance span, not a string of fragments.
DIST_BRIDGE_MS = 800
