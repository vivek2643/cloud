"""
Tuning knobs for ALL L1 cut-grid channels, in one place.

These are HEURISTIC defaults, deliberately NOT empirically tuned yet. They were
chosen for interpretability (linear ramps, round numbers, percentile
normalization) rather than because human "cuttability" is actually linear. They
are parked here so we can re-tune them at the END of the project against real
clips -- not now. Nothing is read from the environment on purpose: keep one
reviewable source of truth until we have a validation harness.

Channels (all share the convention: cost 0 = ideal seam .. 1 = avoid):
  - DIALOGUE : cut in speech gaps          (avoid-curve; dips in word gaps)
  - BEAT     : cut on musical beats        (hit-curve; dips at each onset)
  - ACTION   : cut on motion impacts       (hit-curve; dips at each motion peak)
  - CAMERA   : cut when the camera is calm (avoid-curve; high during moves/blur)
"""
from __future__ import annotations

# =========================================================================
# DIALOGUE (cut_cost.py)
# =========================================================================
HOP_MS = 100              # grid resolution = minimum expected cut granularity
HANDLE_MS = 80            # protected breathing room next to each word edge
SILENCE_FULL_MS = 700     # gap >= this bottoms the seam cost out to ~0
SENTENCE_GAP_MS = 600     # a gap this long reads as a structural/sentence break
MIN_SEAM_MS = 120         # don't emit discrete points for sub-this micro-gaps
WORD_COST = 1.0           # cost inside a word (forbidden)

# Boundary-type multipliers (stack multiplicatively; lower = cheaper cut).
SPEAKER_CHANGE_MULT = 0.35   # cutting on a speaker handoff is a strong seam
SENTENCE_END_MULT = 0.5      # cutting at a sentence boundary is clean
FILLER_EDGE_MULT = 0.3       # we WANT to cut around "um"/"uh"

TERMINAL_PUNCT = ".?!…"

# =========================================================================
# BEAT (beat_cost.py)
# =========================================================================
BEAT_HOP_MS = 100         # grid resolution for the beat cost curve
BEAT_TOL_MS = 70          # +/- window around a beat that reads as "on the beat"
# Beats come straight from audio_features.onsets_ms (librosa) -- no new compute.

# =========================================================================
# MOTION: ACTION + CAMERA/DISTORTION (motion_dynamics.py)
# =========================================================================
MOTION_FPS = 10           # optical-flow sample rate -> hop = 1000/MOTION_FPS ms
MOTION_W = 160            # tiny gray frame width for flow (cheap, ratio-forced)
MOTION_H = 90             # tiny gray frame height
MOTION_NORM_PCTL = 90     # percentile used to normalize raw magnitudes -> 0..1

# Global camera-motion model: a similarity (translation + zoom + roll) transform
# fitted to the dense flow each hop. The fit cleanly separates camera motion from
# subject motion (residual) and tells us HOW the camera moved (pan/zoom/roll).
MOTION_GRID_STEP = 8      # subsample the flow field every Nth px to fit the model
MOTION_RANSAC_PX = 2.0    # RANSAC inlier threshold (px at MOTION_W x MOTION_H)
CAMERA_STABILITY_WIN_MS = 300  # smoothing window for the temporal-stability signal
# Stability is measured as RELATIVE jerk = |change in camera velocity| / current
# speed, so a constant-velocity move (any speed) reads as steady and only
# accelerations (move onsets, whips, bumps) read as transient. Absolute, not
# file-relative, so a clip that is entirely one smooth move isn't mis-scored.
CAMERA_REL_JERK_FULL = 1.0     # relative velocity change that reads as fully unstable
CAMERA_MIN_SPEED_PX = 0.5      # speed floor (px/hop) so still frames don't divide by ~0
CAMERA_JERK_DEADBAND_PX = 0.5  # ignore sub-px fit jitter; a steady move shouldn't read as jerky

# Action (hit-curve): impacts = local maxima of subject motion.
ACTION_PEAK_PCTL = 75     # a peak must exceed this percentile of action energy
ACTION_TOL_MS = 120       # +/- window around an impact that reads as "on the hit"
ACTION_MIN_PEAK_GAP_MS = 250  # min spacing between accepted impacts

# Camera/Distortion (avoid-curve). A SMOOTH, COHERENT, SUSTAINED move (dolly,
# steady pan/zoom) is cheap to cut; only INCOHERENT motion (shake, subject
# thrash), JERKY TRANSIENTS (whip-pan, bump) and BLUR are expensive. So the cost
# is gated by motion *quality*, not raw magnitude:
#   cost = W_chaos*motion*(1-coherence) + W_transient*motion*(1-stability)
#        + W_blur*blur
CAMERA_CHAOS_WEIGHT = 0.7      # incoherent motion the global model can't explain
CAMERA_TRANSIENT_WEIGHT = 0.7  # jerky, non-sustained camera velocity changes
CAMERA_BLUR_WEIGHT = 0.5       # motion-blur / distortion (uncuttable frame)
