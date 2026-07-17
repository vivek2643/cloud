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

# Camera-quality edge snap (Step 4): search this far around a computed edge
# for a cleaner instant (a whip/bump -- low camera_stability -- or a blurred
# frame) to land on instead, so an edge never freezes mid-smooth-move.
EDGE_SNAP_SEARCH_MS = 300
EDGE_SNAP_STABILITY_MAX = 0.4

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
