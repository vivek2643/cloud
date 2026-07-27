"""
Tuning knobs for the V4 deterministic video segmenter (``v4_segment.py``).
See cuts_v4_segmentation.plan.md. Perceptual/structural floors only -- every
threshold that could instead be read off a clip's OWN signal range already is
(the novelty curve is clip-relative via post._series_lohi/_norm_in_clip); what
remains here are the small number of genuinely absolute choices: how wide a
"local neighborhood" is, how long a bump has to persist to count as a
deliberate move, and the floors that keep a punchy cut from clipping its own
payload.
"""
from __future__ import annotations

# Rolling-baseline radius for the novelty curve (Step 2): how far on each side
# of an instant its "local neighborhood" extends when asking "does this stand
# out". ~1-2s total window per the plan; expressed as a radius.
NOVELTY_BASELINE_RADIUS_MS = 800

# A novelty peak must clear this fraction of the working span's OWN
# (curve_max - curve_min) range, added to curve_min, to count as a real event
# (Step 3.2's "prominence threshold relative to the span's own curve").
PEAK_PROMINENCE_RATIO = 0.35

# ...AND clear this ABSOLUTE floor on the (already clip-normalized 0..1)
# novelty scale. Relative prominence alone is scale-invariant, so a heavily
# periodicity-discounted curve (every value shrunk toward 0, but the SHAPE
# unchanged) would still always have SOME "most prominent" point by pure
# relative comparison -- this absolute floor is what actually lets the
# periodicity discount suppress a genuinely periodic span down to kind="none".
NOVELTY_ABSOLUTE_FLOOR = 0.15

# Two peaks closer than this are one event, not two (non-max suppression
# radius for _find_peaks / the "two near bursts -> consolidated to one" case).
PEAK_MIN_GAP_MS = 500

# Periodicity discount (Step 2): a working span's signal is "periodic" when
# the best normalized autocorrelation (excluding trivial adjacent lags), or
# the discrete evenly-spaced-events test, meets this bar -- a blinking light /
# wave / timelapse is highly self-similar at some repeat lag; a one-off burst
# is not. At/above the bar, novelty is scaled by (1 - periodicity_score) --
# continuous, not a fixed haircut, so a near-perfect repeat suppresses almost
# entirely (a blink is all "change" but no "event") while a borderline score
# only dents it. Below the bar, no discount at all.
PERIODICITY_SCORE_THRESHOLD = 0.55

# Camera-move payload (Step 3.3): a hop counts as "moving" once the combined
# |dx|+|dy|+|zoom| clears this per-hop magnitude, AND camera_coherence at that
# hop clears CAMERA_MOVE_COHERENCE_MIN (deliberate, not shake/handheld). The
# move's core must sustain for at least CAMERA_MOVE_MIN_MS to count as a real
# payload rather than a flick.
CAMERA_MOVE_MAGNITUDE_MIN = 0.03
CAMERA_MOVE_COHERENCE_MIN = 0.6
CAMERA_MOVE_MIN_MS = 500

# Point-anchor edges (Step 4): how far the novelty curve must decay from its
# own peak, on each side, before the edge stops chasing it -- floored/ceilinged
# so a flat-topped or noisy curve never produces a degenerate or runaway pad.
# Asymmetric by construction: FOLLOW_THROUGH > RUN_UP, so a punchy cut favors
# playing a beat past the peak over dwelling before it (Step 4's "the natural
# span still includes a comfortable build + settle", weighted toward settle).
DECAY_FRACTION = 0.3
RUN_UP_FLOOR_MS = 300
FOLLOW_THROUGH_FLOOR_MS = 500
MAX_PAD_MS = 3000

# Camera-quality edge snap floor: a hop at/below this camera_stability reads
# as a whip/bump (a clean place to cut) rather than a smooth in-progress
# move. Shared with the Step-1 structural regime classification below
# (REGIME_STABILITY_TRANSIENT_MAX) -- same physical meaning, one name would
# be redundant to invent twice.
EDGE_SNAP_STABILITY_MAX = 0.4

# --------------------------------------------------------------------------
# cut_structure_and_scene_specificity.plan.md Part 1: structure-first camera-
# regime classification -- WHERE it is visually clean to cut, independent of
# content. Reuses the SAME physical thresholds already established just
# above (EDGE_SNAP_STABILITY_MAX) and for deliberate-move classification
# (CAMERA_MOVE_COHERENCE_MIN / CAMERA_MOVE_MAGNITUDE_MIN) rather than
# inventing new numbers with the same meaning.
# --------------------------------------------------------------------------

# A hop's camera_stability at/below this reads as "transient" (whip/bump) --
# the same bar the old edge-quality snap used.
REGIME_STABILITY_TRANSIENT_MAX = EDGE_SNAP_STABILITY_MAX
# A hop's |dx|+|dy|+|zoom| at/above this AND camera_coherence at/above this
# reads as a deliberate "coherent-move"; below the magnitude floor it's a
# "static-hold"; above the magnitude floor but below the coherence floor
# it's "shake" (moving, but not as one rigid gesture).
REGIME_MAGNITUDE_MOVE_MIN = CAMERA_MOVE_MAGNITUDE_MIN
REGIME_COHERENCE_MOVE_MIN = CAMERA_MOVE_COHERENCE_MIN
# blur is already clip-relative at the source (motion_dynamics: 1 - sharp /
# the clip's own reference-percentile sharpness), so a fixed 0..1 threshold
# is meaningful the same way EDGE_SNAP_STABILITY_MAX is: a hop at/above this
# reads as visibly softer than the clip's own typical sharpness.
REGIME_BLUR_MAX = 0.5

# cuts_content_first_segmentation.plan.md Part 1: clip-relative camera-move
# gate. REGIME_MAGNITUDE_MOVE_MIN (0.03, absolute) never clears for aerial/
# drone flow (~50x smaller) -- _clip_move_threshold instead reads a hop as
# "moving" once it clears a robust HIGH percentile of THIS clip's own
# |dx|+|dy|+|zoom| magnitude series, scaled by MOVE_RELATIVE_FRACTION.
MOVE_MAGNITUDE_PERCENTILE_LO = 0.25
MOVE_MAGNITUDE_PERCENTILE_HI = 0.75
MOVE_RELATIVE_FRACTION = 0.5
# Minimum absolute magnitude a hop must ALSO clear regardless of the clip-
# relative math -- far below REGIME_MAGNITUDE_MOVE_MIN, but still well above
# pure optical-flow roundoff/sensor noise.
MOVE_ABSOLUTE_FLOOR = 0.008
# The clip's own high percentile must clear this many multiples of its low
# percentile (or the absolute floor, whichever's larger) to count as genuine
# motion SPREAD -- otherwise (a uniformly near-zero, truly locked/still
# clip) the relative gate is skipped entirely and REGIME_MAGNITUDE_MOVE_MIN
# is used unchanged, so sensor noise is never promoted to "a move."
MOVE_SPREAD_RATIO_MIN = 2.5

# cuts_content_first_segmentation.plan.md Part 3: action-energy STRUCTURE
# (peaks -> peaks + runs/lulls) -- a separate, coarser anchor from the
# novelty curve's local-contrast peaks, for constant-motion content with no
# isolated peak to stand out against (nothing "surprises" a flat,
# continuously-elevated signal). Deliberately conservative -- start with
# fewer, cleaner edges; calibrate against the harness, not by eye (the
# plan's own instruction).
# A hop counts as "in an active run" once clip-normalized action clears this
# fraction of the clip's own range.
ACTION_RUN_BASELINE_FRACTION = 0.3
# A run shorter than this is noise, never its own "moment."
ACTION_RUN_MIN_MS = 800
# Within a run, a drop below this fraction of the RUN's OWN mean level (not
# the clip baseline) is a candidate lull -- splits the run into separate
# moments.
LULL_LEVEL_FRACTION = 0.4
# A dip shorter than this is wobble, not a real lull.
LULL_MIN_MS = 600
# After subtracting whatever the novelty-curve/camera-move events already
# cover (this mechanism only fills gaps they leave -- never double-counts a
# burst/pan those already anchor), a surviving fragment shorter than this
# isn't its own moment either.
ACTION_RUN_LEFTOVER_MIN_MS = 500

# A working span is "dead" (nothing salient anywhere, not even a quiet
# blink) when its peak clip-normalized action_energy AND peak clip-
# normalized rms both sit below this floor, AND no hop even reaches
# REGIME_MAGNITUDE_MOVE_MIN of camera motion. Distinct from low NOVELTY (a
# periodic/blinking signal has real amplitude but scores low on contrast --
# see _novelty_curve's periodicity discount; that still gets a "none"-kind
# representative cut). A genuinely dead span now produces NO event at all
# (the camera-start-still fix) rather than a fabricated "steadiest instant."
DEAD_ENERGY_FLOOR = 0.15

# Step 3 reconcile: a content event's padded edge only snaps to a nearby
# structural seam when the seam sits within this fraction of the edge's own
# padded window duration (a genuine sliver, not real content) AND the
# resulting trim is under SNAP_MS_FLOOR ms -- both must hold, so a short
# window isn't gutted just because a seam happens to sit proportionally
# close. Otherwise the seam sits inside real content and is left alone.
SNAP_FRAC = 0.25
SNAP_MS_FLOOR = 400

# Consolidation floor (Step 5): two anchors' cuts closer than this merge into
# one (content-aware only in the loose sense that it's the same perceptual
# scale as the point-anchor floors above, not a per-clip statistic).
MIN_CUT_GAP_MS = 400

# Fallback representative window (Step 3.4 / no anchor anywhere): a modest,
# steadiest-instant-centered window, never the whole span.
REPRESENTATIVE_WINDOW_MS = 1500

# Density (post.compute_pace_envelope's content-aware min_ms, plan section 6):
# novel-peak rate (peaks/sec) at/above this reads as "fully dense" (density=1);
# a sparse/monotonous span reads near 0.
DENSITY_PEAKS_PER_SEC_CAP = 1.0

# v4_cuts_as_primitive.plan.md section 6: a finished cut shorter than this
# isn't a distinct usable moment on its own (most likely a sliver left by the
# cross-working-span overlap clamp) -- merge it into whichever neighbor it
# sits closer to. Same perceptual scale as MIN_CUT_GAP_MS; duration-based,
# never atom-ownership-based (atoms are no longer part of this module's loop).
MIN_CUT_DURATION_MS = 400

# --------------------------------------------------------------------------
# v4_cluster_tree_cuts.plan.md: a moment is now a CLUSTER of events, not one
# flat span. Events within a cluster are close enough to fuse at the
# broadest energy; a big dead gap starts a new cluster (a new VideoCut).
# --------------------------------------------------------------------------

# A gap between two consecutive events (by window edge, not peak) starts a
# NEW cluster once it exceeds this many times the working span's OWN median
# inter-event gap -- content-derived, not a fixed number: a burst of hits
# 300ms apart reads a 900ms gap as "the same rally"; a burst 2s apart reads
# the same 900ms gap as tight, not a break. Clamped to
# [MIN_CUT_GAP_MS, MAX_CLUSTER_SEPARATION_MS] below so one huge outlier gap
# in a tiny sample can't blow the threshold out, and a very tight span still
# gets SOME separation floor.
CLUSTER_SEPARATION_MULTIPLIER = 2.0
MAX_CLUSTER_SEPARATION_MS = 3000

# The tightest a single EVENT's own window is ever allowed to collapse to
# inside the per-level cluster resolver (resolve_cluster) -- the per-piece
# analogue of RUN_UP_FLOOR_MS/FOLLOW_THROUGH_FLOOR_MS combined. Never below
# readability: an event narrower than this at max energy would flash by
# unreadably rather than read as a hit.
MIN_EVENT_PIECE_MS = RUN_UP_FLOOR_MS + FOLLOW_THROUGH_FLOOR_MS

# image_plan.build_image_plan: at most this many of a cluster's own event
# peaks get their own straddle frame pair (evenly sampled across the event
# list when there are more). Bounds the frame cost of a large, busy cluster
# flat regardless of how many events it holds.
MAX_CLUSTER_EVENT_FRAMES = 4

# resolve_cluster's rising salience gate (the core "extract the usable, discard
# the scrap" lever along the energy dial). As energy rises, an event survives
# only if its (clip-normalized) salience clears energy * GATE * (cluster's OWN
# max event score) -- so weak/connective/noise events fall away first and the
# survivors trim tight and separate into distinct pieces. Relative to the
# cluster's own peak (never an absolute score), so it's generic: a lone strong
# peak keeps one piece, several genuinely-strong events keep several, a
# monotonous span (all-low, periodicity-discounted) collapses toward its single
# best window. The single strongest event is always kept (a cut is never
# empty). At energy 0 the gate is 0 -> everything survives (broad = whole
# moment). ~0.8 clears the noise floor at the sharp band (e=0.9 -> ~0.72*max)
# while still keeping a cluster of genuinely comparable hits intact.
CLUSTER_PRUNE_GATE = 0.8
