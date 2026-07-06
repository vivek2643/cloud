"""
Tuning knobs for cuts-v3 transition-point detection
(``motion_dynamics.py``'s ``transition_points`` -- premium natural cut
instants: occlusion wipes and degenerate/unusable frame spans). See
``cuts_v3.plan.md``, section 1a. Heuristic defaults, parked here for
re-tuning against real clips, same convention as ``cut_grid_params.py``.
"""
from __future__ import annotations

# --- Occlusion wipe: a large near-field blob sweeps the frame ---------------
# A classic pass-by transition (someone/something crosses right in front of
# the lens) -- editors hunt for these as natural, motivated cut points.

# A grid point's flow magnitude (px/hop, in the MOTION_W x MOTION_H downsample
# space -- see cut_grid_params.py) above this counts as "sweeping": fast,
# large, near-field motion a clean pan/zoom doesn't produce at this scale.
WIPE_MAG_PX = 8.0
# A wipe needs at least this FRACTION of the sampled grid sweeping at once.
# Verified against a synthetic frame-filling sweep (a textured blob crossing
# the whole frame in ~0.4s): peak measured fraction ~0.4 at 10fps sampling
# (pyramidal Farneback smooths a very fast sweep's peak instant) -- 0.5 never
# fired on a genuine full-frame sweep, so this is calibrated to that ceiling,
# not guessed.
WIPE_AREA_FRAC = 0.30
# ...and the camera model's coherence must ALSO collapse: a wipe is chaotic
# near-field motion the global similarity fit can't explain, whereas a clean
# fast pan keeps high coherence even at high magnitude -- so high-coherence
# fast motion is never mistaken for a wipe.
WIPE_COHERENCE_MAX = 0.35
# Two wipe candidates closer than this are one event.
WIPE_MIN_GAP_MS = 500
# A true wipe RECOVERS: the swept-area fraction must drop back under half its
# own floor within this window after the peak, or it's a sustained disturbance
# (already covered elsewhere), not a quick occlusion sweep.
WIPE_RECOVERY_MS = 600

# --- Degeneracy: the frame collapses to one texture (over-zoom, lens blocked)
# Reuses motion_dynamics' own `blur` signal (1 - sharpness/file-reference,
# already computed in the same optical-flow pass) -- degenerate needs it
# maxed out AND sustained, not just one soft frame from fast motion. Marks
# "must cut by here": past this point there is nothing left to watch.
DEGENERATE_BLUR_MIN = 0.92
DEGENERATE_MIN_MS = 500
