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
# A hop is a DISTURBANCE when the camera is genuinely unstable or its motion is
# incoherent (a whip/jerk/focus-hunt, not a clean intentional move). Detect the
# BAD span; its edges are boundaries (so the disturbance can be dropped later as
# junk -- that's a separate job). Absolute thresholds (camera_stability and
# camera_coherence are 0..1, not per-clip normalized), tuned against real clips.
DIST_STABILITY_MAX = 0.45
DIST_COHERENCE_MAX = 0.35
# A disturbance shorter than this is a blip, not a span worth a boundary.
DIST_MIN_MS = 250
# Two bad-camera runs separated by a gap this short are ONE disturbance (the
# shake dipped under threshold for an instant) -- bridge them so a pervasively
# shaky clip reads as a single disturbance span, not a string of fragments.
DIST_BRIDGE_MS = 800
