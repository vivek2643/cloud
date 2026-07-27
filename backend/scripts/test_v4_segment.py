"""
Pure unit tests for the V4 deterministic video segmenter
(``app.services.l3.v4_segment``) -- no DB, no model call. The V4 cut IS the
primitive (v4_cuts_as_primitive.plan.md): no atoms in this module's loop at
all, so these tests never construct a Lattice/Atom fixture.

Run:  .venv/bin/python scripts/test_v4_segment.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import v4_segment as v4  # noqa: E402
from app.services.l3.v4_segment_params import MIN_CUT_DURATION_MS  # noqa: E402


def _flat_motion(n, hop=100, action=0.05, stability=0.95, coh=0.9):
    return {
        "hop_ms": hop, "action_energy": [action] * n, "camera_stability": [stability] * n,
        "camera_coherence": [coh] * n, "camera_motion": [0.0] * n, "blur": [0.1] * n,
        "camera_dx": [0.0] * n, "camera_dy": [0.0] * n, "camera_zoom": [0.0] * n,
        "action_points": [], "transition_points": [],
    }


def _segment(motion, audio=None, scene=None, speech_spans=None, duration_ms=10_000):
    return v4.segment_video(
        file_id="f1", duration_ms=duration_ms, speech_spans=speech_spans or [],
        motion=motion, audio=audio or {}, scene=scene or {},
    )


# --------------------------------------------------------------------------
# The plan's own table cases (cuts_v4_segmentation.plan.md section 10)
# --------------------------------------------------------------------------

def test_burst_out_of_calm_yields_one_tight_point_cut_after_peak():
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = ([0.05] * 40
                                + [0.05, 0.1, 0.3, 0.7, 0.95, 0.9, 0.6, 0.3, 0.15, 0.08]
                                + [0.05] * 50)
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "point", c.salience
    # Tight: nowhere near the whole 10s span, and it ends AFTER the peak
    # (peak is at hop 44 -> 4400ms).
    assert c.src_out_ms - c.src_in_ms < 2500, c
    assert c.src_in_ms <= 4400 <= c.src_out_ms, c
    assert c.src_out_ms > 4400, "a point cut must play a beat past its own peak"
    print("ok  test_burst_out_of_calm_yields_one_tight_point_cut_after_peak")


def test_blinking_periodic_energy_yields_none_not_split():
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = [0.1 if (i % 10) < 3 else 0.8 for i in range(n)]
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    assert cuts[0].salience["kind"] == "none", cuts[0].salience
    assert cuts[0].src_out_ms - cuts[0].src_in_ms < 10_000, "must not keep the whole span"
    print("ok  test_blinking_periodic_energy_yields_none_not_split")


def test_uniform_static_yields_no_cut_at_all():
    """cut_structure_and_scene_specificity.plan.md Part 1: a genuinely dead
    span (no action/rms variation, no camera motion -- exactly the
    camera-start-still's own working span) now produces NO event and so no
    cut, instead of the old steadiest-instant fallback (which is precisely
    what used to fabricate a cut on the static setup hold)."""
    cuts = _segment(_flat_motion(100))
    assert cuts == [], cuts
    print("ok  test_uniform_static_yields_no_cut_at_all")


def test_smooth_pan_yields_span_cut_at_move_start_and_settle():
    """Also the plan's own "static-head-then-move shot -> cut excludes the
    static head" calibration case: 0-2000ms is a dead static head, the move
    starts at 2000ms, and the cut starts exactly there -- the head
    contributes nothing (no special-case dead-edge trim needed)."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.0] * 20 + [0.08] * 40 + [0.0] * 40
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "span", c.salience
    assert c.src_in_ms == 2000 and c.src_out_ms == 6000, c
    assert c.salience["span_ms"] == [2000, 6000], c.salience
    print("ok  test_smooth_pan_yields_span_cut_at_move_start_and_settle")


def test_two_separated_bursts_yield_two_cuts():
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(10, 15):
        ae[i] = 0.9
    for i in range(70, 75):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion)
    assert len(cuts) == 2, cuts
    assert all(c.salience["kind"] == "point" for c in cuts), cuts
    assert cuts[0].src_out_ms <= cuts[1].src_in_ms, "must not overlap"
    print("ok  test_two_separated_bursts_yield_two_cuts")


def test_two_near_bursts_consolidate_to_one():
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(40, 44):
        ae[i] = 0.9
    for i in range(46, 50):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    print("ok  test_two_near_bursts_consolidate_to_one")


# --------------------------------------------------------------------------
# Salience: contrast beats absolute level
# --------------------------------------------------------------------------

def test_contrast_based_peak_beats_absolute_level_on_ramp_then_plateau():
    """A ramp into a sustained high plateau: the plateau is the highest
    ABSOLUTE level in the clip, but it has zero novelty once it's the new
    normal -- the transition itself (contrast) must be what wins."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = [0.05] * 20 + [0.05 + (0.9 - 0.05) * (i / 10) for i in range(10)] + [0.9] * 70
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "point", c.salience
    # The peak must land near the RAMP (hops 20-30 -> 2000-3000ms), not deep
    # in the flat high plateau (e.g. ts=9000ms, the absolute-max instant).
    assert 1500 <= c.salience["peak_ms"] <= 3500, c.salience
    print("ok  test_contrast_based_peak_beats_absolute_level_on_ramp_then_plateau")


# --------------------------------------------------------------------------
# Speech subtraction
# --------------------------------------------------------------------------

def test_speech_spans_are_subtracted_from_working_spans():
    motion = _flat_motion(100)
    ae = [0.05] * 100
    for i in range(15, 20):    # 1500-2000ms, INSIDE the speech span -> must vanish
        ae[i] = 0.9
    for i in range(70, 75):    # 7000-7500ms, outside speech -> must survive
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion, speech_spans=[(1000, 3000)])
    assert all(c.src_in_ms >= 3000 or c.src_out_ms <= 1000 for c in cuts), cuts
    assert any(c.src_in_ms >= 6000 for c in cuts), "the surviving burst must still produce a cut"
    print("ok  test_speech_spans_are_subtracted_from_working_spans")


def test_no_cut_ever_crosses_into_a_speech_span():
    """Every emitted cut must be fully outside every declared speech span --
    the load-bearing invariant that lets post.py's own zero-overlap check
    between video and speech cuts pass without ever needing to know V4's
    internals."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(0, n):   # action everywhere, including inside speech
        ae[i] = 0.9 if i % 7 == 0 else 0.05
    motion["action_energy"] = ae
    speech_spans = [(1000, 2500), (5000, 5800), (8000, 8600)]
    cuts = _segment(motion, speech_spans=speech_spans)
    for c in cuts:
        for s, e in speech_spans:
            assert c.src_out_ms <= s or c.src_in_ms >= e, (c, (s, e))
    print("ok  test_no_cut_ever_crosses_into_a_speech_span")


# --------------------------------------------------------------------------
# v4_cuts_as_primitive.plan.md section 6/9: geometry-only finalize --
# disjoint + clamped to working span, sub-min_ms sliver merges into neighbor.
# --------------------------------------------------------------------------

def test_finalize_cuts_are_always_disjoint_and_sorted():
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(10, 15):
        ae[i] = 0.9
    for i in range(30, 35):
        ae[i] = 0.9
    for i in range(70, 75):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion)
    ordered = sorted(cuts, key=lambda c: c.src_in_ms)
    assert ordered == cuts, "segment_video must already return cuts in order"
    for a, b in zip(ordered, ordered[1:]):
        assert a.src_out_ms <= b.src_in_ms, f"overlap: {a.src_out_ms} > {b.src_in_ms}"
    print("ok  test_finalize_cuts_are_always_disjoint_and_sorted")


def test_finalize_cuts_clamps_extended_edges_to_the_working_span():
    """A shot boundary sits at 5000ms; a burst right before it has enough
    follow-through padding to want to reach past 5000ms. The cut must clamp
    to the shot boundary, never leak into the next shot's own working span."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(46, 50):   # burst at 4600-5000ms, right at the shot edge
        ae[i] = 0.9
    motion["action_energy"] = ae
    scene = {"shot_points": [{"ts_ms": 5000}]}
    cuts = _segment(motion, scene=scene)
    for c in cuts:
        assert c.src_out_ms <= 5000 or c.src_in_ms >= 5000, c
    print("ok  test_finalize_cuts_clamps_extended_edges_to_the_working_span")


def test_finalize_cuts_merges_a_sub_floor_sliver_into_its_nearest_neighbor():
    """Force a degenerate short cut via the overlap clamp: two working spans
    separated by a 1ms shot boundary gap, each producing a cut whose padded
    edges collide right at the boundary -- the earlier cut gets clamped down
    to a sliver shorter than MIN_CUT_DURATION_MS and must be merged away
    rather than surviving as its own tiny cut."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(38, 42):    # burst just before the shot boundary
        ae[i] = 0.9
    for i in range(42, 46):    # burst just after -- close enough that both
        ae[i] = 0.9             # cuts' padding reaches the shared boundary
    motion["action_energy"] = ae
    scene = {"shot_points": [{"ts_ms": 4200}]}
    cuts = _segment(motion, scene=scene)
    for c in cuts:
        assert c.src_out_ms - c.src_in_ms >= MIN_CUT_DURATION_MS, \
            f"a sub-floor sliver survived unmerged: {c}"
    print("ok  test_finalize_cuts_merges_a_sub_floor_sliver_into_its_nearest_neighbor")


def test_lone_cut_below_the_floor_survives_with_no_neighbor_to_merge_into():
    """A single short point-event cut with nothing else in the file has no
    neighbor to merge into -- it must survive as-is rather than vanish
    (better a short cut than none). A tiny real burst (not a dead span --
    see test_uniform_static_yields_no_cut_at_all for that case) so a genuine
    cut is produced in the first place."""
    motion = _flat_motion(3, hop=100)
    motion["action_energy"] = [0.05, 0.9, 0.05]
    cuts = v4.segment_video(file_id="f1", duration_ms=300, speech_spans=[],
                            motion=motion, audio={}, scene={})
    assert len(cuts) == 1, cuts
    print("ok  test_lone_cut_below_the_floor_survives_with_no_neighbor_to_merge_into")


def test_sub_floor_sliver_never_welds_across_a_speech_gap():
    """Two working spans split by a speech span, each with a short burst near the
    speech edge that yields a sub-floor sliver. The min-duration merge must NOT
    weld the two slivers across the speech between them -- a cross-span union
    would swallow the speech, producing the exact video<->speech overlap that
    broke real ingests (f48da65f: [6860-9640] engulfing speech [8640-9560]).
    Every cut must stay wholly on its own side of the speech span."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(36, 39):    # burst just before the speech span
        ae[i] = 0.9
    for i in range(62, 65):    # burst just after the speech span
        ae[i] = 0.9
    motion["action_energy"] = ae
    speech = [(4000, 6000)]
    cuts = _segment(motion, speech_spans=speech)
    assert cuts, "expected video cuts around the speech"
    for c in cuts:
        assert c.src_out_ms <= 4000 or c.src_in_ms >= 6000, \
            f"a video cut welded into / across the speech span [4000-6000]: {c}"
    print("ok  test_sub_floor_sliver_never_welds_across_a_speech_gap")


# --------------------------------------------------------------------------
# Cluster + tree (v4_cluster_tree_cuts.plan.md section 10)
# --------------------------------------------------------------------------

def test_single_event_cluster_has_exactly_one_event():
    """A cluster of one is the degenerate, backward-compatible case: exactly
    one entry in salience.events, primary=0, top-level peak_ms/score/kind
    mirroring it."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = ([0.05] * 40
                                + [0.05, 0.1, 0.3, 0.7, 0.95, 0.9, 0.6, 0.3, 0.15, 0.08]
                                + [0.05] * 50)
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    sal = cuts[0].salience
    assert len(sal["events"]) == 1, sal
    assert sal["primary"] == 0, sal
    ev = sal["events"][0]
    assert sal["peak_ms"] == ev["peak_ms"] and sal["score"] == ev["score"] and sal["kind"] == ev["kind"]
    print("ok  test_single_event_cluster_has_exactly_one_event")


def test_two_pans_both_kept_not_just_the_longest():
    """The pan-loss regression (section 4.2): _camera_move_cores must return
    EVERY sustained move, not just the longest one via max()."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.0] * 10 + [0.08] * 15 + [0.0] * 10 + [0.08] * 15 + [0.0] * 50
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts   # close enough to fuse into one cluster
    sal = cuts[0].salience
    span_events = [ev for ev in sal["events"] if ev["kind"] == "span"]
    assert len(span_events) == 2, sal["events"]
    print("ok  test_two_pans_both_kept_not_just_the_longest")


def test_peak_and_pan_coexist_in_one_span():
    """A novelty peak and a camera move in the same working span must BOTH
    survive as events (section 4.1's "no first-match early-exit")."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.0] * 60 + [0.08] * 30 + [0.0] * 10
    ae = [0.05] * n
    for i in range(10, 15):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion)
    kinds = sorted(ev["kind"] for c in cuts for ev in c.salience["events"])
    assert "point" in kinds and "span" in kinds, kinds
    print("ok  test_peak_and_pan_coexist_in_one_span")


def test_tight_cluster_of_peaks_yields_one_cluster_with_n_events():
    n = 200
    motion = _flat_motion(n)
    ae = [0.05] * n
    for start in (20, 60, 100):
        for i in range(start, start + 4):
            ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion, duration_ms=20_000)
    assert len(cuts) == 1, cuts
    assert len(cuts[0].salience["events"]) == 3, cuts[0].salience["events"]
    print("ok  test_tight_cluster_of_peaks_yields_one_cluster_with_n_events")


def test_big_gap_between_bursts_yields_two_clusters():
    n = 200
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(10, 14):
        ae[i] = 0.9
    for i in range(180, 184):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion, duration_ms=20_000)
    assert len(cuts) == 2, cuts
    assert all(len(c.salience["events"]) == 1 for c in cuts), cuts
    print("ok  test_big_gap_between_bursts_yields_two_clusters")


def test_noisy_curve_near_one_burst_still_yields_one_event():
    """A wiggly rise/fall around a single real burst must not register as
    several near-duplicate events -- the dedup step keeps only the strongest
    overlapping detection."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = ([0.05] * 40
                                + [0.05, 0.1, 0.3, 0.7, 0.95, 0.9, 0.6, 0.3, 0.15, 0.08]
                                + [0.05] * 50)
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    assert len(cuts[0].salience["events"]) == 1, cuts[0].salience["events"]
    print("ok  test_noisy_curve_near_one_burst_still_yields_one_event")


def test_merged_sliver_unions_events_not_drops_them():
    """When _finalize_cuts welds a sub-floor sliver into a same-span
    neighbor, the merged cut's events must be the UNION of both -- never
    silently drop the sliver's own event."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    for i in range(38, 42):
        ae[i] = 0.9
    for i in range(42, 46):
        ae[i] = 0.9
    motion["action_energy"] = ae
    scene = {"shot_points": [{"ts_ms": 4200}]}
    cuts = _segment(motion, scene=scene)
    total_events = sum(len(c.salience["events"]) for c in cuts)
    assert total_events >= 2, cuts
    print("ok  test_merged_sliver_unions_events_not_drops_them")


# --------------------------------------------------------------------------
# cut_structure_and_scene_specificity.plan.md Part 1: structure-first
# reconciliation -- the camera-start-still fix + edge snapping.
# --------------------------------------------------------------------------

def test_dead_head_then_incoherent_shake_biases_to_the_real_activity():
    """The actual camera-start-still bug: a dead static head (no action, no
    camera motion) followed by a brief INCOHERENT shake -- real camera
    magnitude, but too incoherent to register as a _camera_move_cores span
    event, and no action/rms novelty to register as a point event either.
    Under the old steadiest+sharpest fallback this landed on the dead head
    (uniform stability/blur -> picks the first index, ts=0 -- the bug).
    The new energy-biased fallback must land near the real activity instead."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.0] * 50 + [0.05] * 10 + [0.0] * 40
    motion["camera_coherence"] = [0.9] * 50 + [0.2] * 10 + [0.9] * 40
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "none", c.salience
    assert c.src_in_ms >= 4000, \
        f"fallback window landed on the dead head, not the real activity: {c}"
    print("ok  test_dead_head_then_incoherent_shake_biases_to_the_real_activity")


def test_whip_between_two_holds_snaps_the_cut_edge():
    """A whip (transient camera_stability) sitting just outside a content
    event's padded edge, well within SNAP_FRAC/SNAP_MS_FLOOR of it, snaps
    the edge to the whip -- a clean trim, not the raw decay-walk pad."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = ([0.05] * 40
                               + [0.05, 0.1, 0.3, 0.7, 0.95, 0.9, 0.6, 0.3, 0.15, 0.08]
                               + [0.05] * 50)
    baseline_cuts = _segment(motion)
    assert baseline_cuts[0].src_in_ms == 3700, baseline_cuts  # no seam -> raw pad, ground truth

    motion["camera_stability"][35] = 0.3   # whip at 3500ms, near the 3700ms in-edge
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    assert cuts[0].src_in_ms == 3600, cuts[0]   # snapped to the whip's own boundary seam
    print("ok  test_whip_between_two_holds_snaps_the_cut_edge")


def test_seam_deep_inside_content_is_not_snapped():
    """A whip sitting at the event's own PEAK (dead center, not near either
    edge) must never pull an edge toward it -- that seam is inside real
    content, not free structure. The window must match the no-seam baseline
    exactly."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = ([0.05] * 40
                               + [0.05, 0.1, 0.3, 0.7, 0.95, 0.9, 0.6, 0.3, 0.15, 0.08]
                               + [0.05] * 50)
    motion["camera_stability"][44] = 0.3   # whip right at the peak itself (4400ms)
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    assert cuts[0].src_in_ms == 3700 and cuts[0].src_out_ms == 5300, cuts[0]
    print("ok  test_seam_deep_inside_content_is_not_snapped")


def test_coherent_move_overlapping_action_is_preserved_not_trimmed():
    """A coherent camera move that ALSO carries real action energy is
    content, not free structure -- its span event's own window must survive
    exactly, and the cluster's overall span must reach at least as wide as
    the move (never trimmed/split by the Step-3 reconcile, which only ever
    touches POINT-event padding, never a span event's own extent)."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.0] * 30 + [0.08] * 40 + [0.0] * 30   # move 3000-7000ms
    ae = [0.05] * n
    for i in range(45, 55):     # action burst INSIDE the move's own span
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion)
    span_events = [ev for c in cuts for ev in c.salience["events"] if ev["kind"] == "span"]
    assert len(span_events) == 1, span_events
    assert span_events[0]["span_ms"] == [3000, 7000], span_events[0]
    assert any(c.src_in_ms <= 3000 and c.src_out_ms >= 7000 for c in cuts), cuts
    print("ok  test_coherent_move_overlapping_action_is_preserved_not_trimmed")


# --------------------------------------------------------------------------
# cuts_content_first_segmentation.plan.md Part 1: clip-relative camera-move
# gate -- a sustained move well BELOW the old absolute REGIME_MAGNITUDE_
# MOVE_MIN (0.03, aerial/drone flow's mode-B root cause) must still register
# once it's clearly this clip's own upper range; a uniformly-present low
# magnitude (no real spread -- a genuinely locked/still clip's own sensor
# bias/jitter) must NOT be promoted just because it's technically the max.
# --------------------------------------------------------------------------

def test_below_old_floor_move_registers_when_clip_relatively_significant():
    """A sustained camera_dx of 0.025 -- under the old fixed 0.03 floor --
    against a 0.0 baseline for the rest of the clip: real spread, so the
    clip-relative gate engages and this reads as a genuine coherent-move
    span, exactly like the (much larger, 0.08) test_smooth_pan_... case."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.0] * 30 + [0.025] * 40 + [0.0] * 30
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "span", c.salience
    assert c.src_in_ms == 3000 and c.src_out_ms == 7000, c
    print("ok  test_below_old_floor_move_registers_when_clip_relatively_significant")


def test_uniform_low_magnitude_camera_drift_is_never_promoted_to_a_move():
    """The SAME 0.025 magnitude, but present on EVERY hop (no spread at
    all) -- a genuinely locked/still clip's own uniform sensor bias, not a
    deliberate move. The spread test must fall back to the old absolute
    floor here, so no camera span event registers at all; the clip's real
    action burst still produces its own point event untouched."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_dx"] = [0.025] * n
    ae = [0.05] * n
    for i in range(48, 52):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    kinds = [ev["kind"] for ev in cuts[0].salience["events"]]
    assert kinds == ["point"], kinds
    print("ok  test_uniform_low_magnitude_camera_drift_is_never_promoted_to_a_move")


# --------------------------------------------------------------------------
# cuts_content_first_segmentation.plan.md Part 2: composition_points as a
# first-class content boundary -- must open a cut even in an otherwise
# genuinely dead span (no action, no camera, no rms: exactly the static-
# camera content-change case action/camera can't see), and must respect the
# same discrete periodicity discipline as other point sources.
# --------------------------------------------------------------------------

def test_composition_point_opens_a_cut_in_an_otherwise_dead_span():
    """A totally flat/dead 10s span (see test_uniform_static_yields_no_cut_
    at_all) with ONE composition_point at 5000ms: must now produce exactly
    one cut around it, padded by the same RUN_UP_FLOOR_MS/FOLLOW_THROUGH_
    FLOOR_MS every other point event gets -- independent of Part 4's energy
    gate, since composition drift is this event's own justification."""
    scene = {"composition_points": [{"ts_ms": 5000, "kind": "composition_change", "score": 1.0}]}
    cuts = _segment(_flat_motion(100), scene=scene)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "point", c.salience
    assert c.src_in_ms == 4700 and c.src_out_ms == 5500, c
    print("ok  test_composition_point_opens_a_cut_in_an_otherwise_dead_span")


def test_evenly_spaced_composition_points_are_suppressed_as_periodic():
    """A composition point every 1000ms (a strobing reframe/flicker) must be
    suppressed entirely, not spam one boundary per flicker -- same discrete
    evenly-spaced discipline action_points/onsets already get."""
    scene = {"composition_points": [{"ts_ms": t, "kind": "composition_change", "score": 1.0}
                                    for t in range(1000, 9500, 1000)]}
    cuts = _segment(_flat_motion(100), scene=scene)
    assert cuts == [], cuts
    print("ok  test_evenly_spaced_composition_points_are_suppressed_as_periodic")


# --------------------------------------------------------------------------
# cuts_content_first_segmentation.plan.md Part 5: the protect gate
# (_inside_continuous_content) now also protects a whip/blur seam sitting
# inside a sustained action RUN (Part 3, no camera move needed) or where
# composition drift is LOW (no composition_point nearby) -- but must NOT
# protect a whip sitting right where composition drift is actually HIGH (an
# actual flagged change), so a genuine content change can still open a
# boundary.
# --------------------------------------------------------------------------

def test_whip_inside_a_sustained_action_run_is_not_a_seam():
    """A whip (transient camera_stability) sitting inside a sustained
    action run (no camera move at all) must be excluded -- it reads as
    incidental camera shake during ongoing content, not a clean edit
    point. The SAME whip with no action run around it (baseline
    everywhere) DOES register, proving the run itself is what protects."""
    n = 100
    protected = _flat_motion(n)
    protected["action_energy"] = [0.05] * 20 + [0.8] * 60 + [0.05] * 20
    protected["camera_stability"][50] = 0.3
    seams_protected = v4._structural_seams(
        protected, {}, (0, 10_000), 100, v4._series_lohi(protected["action_energy"]), (None, None))
    assert seams_protected == [], seams_protected

    unprotected = _flat_motion(n)
    unprotected["camera_stability"][50] = 0.3
    seams_unprotected = v4._structural_seams(
        unprotected, {}, (0, 10_000), 100, v4._series_lohi(unprotected["action_energy"]), (None, None))
    assert 5000 in seams_unprotected, seams_unprotected
    print("ok  test_whip_inside_a_sustained_action_run_is_not_a_seam")


def test_whip_far_from_any_composition_change_is_protected_by_low_drift():
    """No action run anywhere, but this file HAS composition data: a whip
    sitting far from the only flagged composition_point (drift is LOW
    right there) must be excluded -- composition continuity alone protects
    it, per Part 5's "OR composition drift is low" clause."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_stability"][50] = 0.3
    scene = {"composition_points": [{"ts_ms": 9000}]}
    seams = v4._structural_seams(motion, scene, (0, 10_000), 100,
                                 v4._series_lohi(motion["action_energy"]), (None, None))
    assert seams == [], seams
    print("ok  test_whip_far_from_any_composition_change_is_protected_by_low_drift")


def test_whip_right_at_a_composition_change_still_registers():
    """The SAME whip, but now sitting right where composition_points
    flags an actual change (drift is HIGH there) -- must NOT be protected;
    a genuine content change can still open a boundary."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_stability"][50] = 0.3
    scene = {"composition_points": [{"ts_ms": 5000}]}
    seams = v4._structural_seams(motion, scene, (0, 10_000), 100,
                                 v4._series_lohi(motion["action_energy"]), (None, None))
    assert 5000 in seams, seams
    print("ok  test_whip_right_at_a_composition_change_still_registers")


# --------------------------------------------------------------------------
# cuts_content_first_segmentation.plan.md Part 6 (lever A): the
# representative-window fallback cuts ONE CLEAN CYCLE of periodic content
# (period reused from _periodicity_score's own autocorrelation) instead of
# the generic fixed REPRESENTATIVE_WINDOW_MS.
# --------------------------------------------------------------------------

def test_representative_window_locks_to_one_period_of_periodic_content():
    """A clean 1000ms-period square wave (the SAME shape as
    test_blinking_periodic_energy_yields_none_not_split): the fallback
    window must now be exactly one period wide (1000ms), not the generic
    REPRESENTATIVE_WINDOW_MS (1500ms)."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = [0.1 if (i % 10) < 3 else 0.8 for i in range(n)]
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.salience["kind"] == "none", c.salience
    assert c.src_out_ms - c.src_in_ms == 1000, c
    print("ok  test_representative_window_locks_to_one_period_of_periodic_content")


# --------------------------------------------------------------------------
# cuts_content_first_segmentation.plan.md Part 3: action peaks -> peaks +
# runs/lulls -- a sustained plateau well beyond the novelty curve's own
# ~800ms local-contrast radius (so most of it has near-zero novelty, same
# as "constant-motion footage has no isolated peak") must still be covered,
# and a genuine internal dip must split it into separate moments.
# --------------------------------------------------------------------------

def test_sustained_plateau_beyond_novelty_reach_gets_covered_by_a_run_event():
    """A 6000ms elevated plateau (0.05 -> 0.7 -> 0.05): the rising/falling
    edges are real novelty (point events), but the plateau's own middle is
    far wider than NOVELTY_BASELINE_RADIUS_MS -- without Part 3 that middle
    stretch has near-zero novelty and is left uncovered. Must now appear as
    its own "span" leftover event, and the cut's coverage must reach deep
    into the plateau, not just its edges."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = [0.05] * 20 + [0.7] * 60 + [0.05] * 20
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    kinds = [ev["kind"] for ev in c.salience["events"]]
    assert kinds.count("span") == 1, kinds
    span_ev = next(ev for ev in c.salience["events"] if ev["kind"] == "span")
    # The leftover fragment sits INSIDE the plateau, well past where a bare
    # point event's own decay-walk pad would reach on its own.
    assert span_ev["span_ms"][0] >= 2000 and span_ev["span_ms"][1] <= 8000, span_ev
    assert span_ev["span_ms"][1] - span_ev["span_ms"][0] > 2000, span_ev
    print("ok  test_sustained_plateau_beyond_novelty_reach_gets_covered_by_a_run_event")


def test_lull_splits_a_long_active_stretch_into_separate_moments():
    """Two elevated plateaus (norm ~1.0) joined by a real, sustained dip
    (norm ~0.33 -- above the clip-baseline floor, so still ONE macro run by
    _true_runs, but well below LULL_LEVEL_FRACTION of the run's own mean
    level) must split into TWO disjoint leftover span fragments, one per
    plateau -- not one span covering the whole thing including the dip."""
    n = 100
    motion = _flat_motion(n)
    motion["action_energy"] = [0.05] * 20 + [0.95] * 30 + [0.35] * 10 + [0.95] * 30 + [0.05] * 10
    cuts = _segment(motion)
    span_fragments = sorted(
        tuple(ev["span_ms"]) for c in cuts for ev in c.salience["events"] if ev["kind"] == "span")
    assert len(span_fragments) == 2, span_fragments
    (a0, a1), (b0, b1) = span_fragments
    assert a1 <= 5000 <= b0, span_fragments   # disjoint, split apart by the dip
    print("ok  test_lull_splits_a_long_active_stretch_into_separate_moments")


# --------------------------------------------------------------------------
# cuts_content_first_segmentation.plan.md Part 4: energy-gated padding
# floors -- the "camera-start-still" bug at the PADDING level (RUN_UP_FLOOR_
# MS/FOLLOW_THROUGH_FLOOR_MS reaching backward/forward into a dead
# sub-region even when the working span's own peak amplitude is high).
# --------------------------------------------------------------------------

def test_sharp_spike_padding_floor_does_not_reach_into_a_dead_sub_region():
    """A one-hop spike deep in an otherwise totally flat/dead 10s span: the
    raw decay-walk bounds are [4800, 5000] (peak 4900), well inside
    RUN_UP_FLOOR_MS(300)/FOLLOW_THROUGH_FLOOR_MS(500) of the peak. Before
    Part 4 the floor unconditionally pads to [4600, 5400] -- both extra
    100-400ms slivers are genuinely dead (flat 0.05, no camera motion), so
    Part 4 must trim the edges back to the raw bounds instead of padding
    into them."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    ae[49] = 0.95
    motion["action_energy"] = ae
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.src_in_ms == 4800 and c.src_out_ms == 5000, c
    print("ok  test_sharp_spike_padding_floor_does_not_reach_into_a_dead_sub_region")


def test_padding_floor_still_applies_when_the_reach_carries_energy():
    """The SAME sharp spike, but now with real camera motion (well above
    REGIME_MAGNITUDE_MOVE_MIN, _has_energy_in's camera term) sitting in the
    floor's reach -- Part 4 must still pad there normally (never regress the
    base case): the floor is conditional on energy, not disabled outright.
    Camera motion (not rms) is used here specifically because it never feeds
    the novelty curve -- it can't shift the burst's own peak/decay bounds,
    so this isolates the padding-floor gate from curve-shape interaction."""
    n = 100
    motion = _flat_motion(n)
    ae = [0.05] * n
    ae[49] = 0.95
    motion["action_energy"] = ae
    motion["camera_dx"][46] = 0.04   # inside the run-up floor's reach [4600, 4800)
    motion["camera_dx"][51] = 0.04   # inside the follow-through floor's reach [5000, 5400)
    cuts = _segment(motion)
    assert len(cuts) == 1, cuts
    c = cuts[0]
    assert c.src_in_ms == 4600 and c.src_out_ms == 5400, c
    print("ok  test_padding_floor_still_applies_when_the_reach_carries_energy")


# --------------------------------------------------------------------------
# density (feeds post.compute_pace_envelope's content-aware min_ms)
# --------------------------------------------------------------------------

def test_density_is_higher_for_a_dense_span_than_a_sparse_one():
    n = 100
    sparse = _flat_motion(n)
    ae = [0.05] * n
    for i in range(40, 44):
        ae[i] = 0.9
    sparse["action_energy"] = ae
    sparse_cuts = _segment(sparse)

    dense = _flat_motion(n)
    ae2 = [0.05] * n
    for start in (5, 20, 35, 50, 65, 80):
        for i in range(start, start + 3):
            ae2[i] = 0.9
    dense["action_energy"] = ae2
    dense_cuts = _segment(dense)

    assert sparse_cuts and dense_cuts
    assert max(c.density for c in dense_cuts) > max(c.density for c in sparse_cuts)
    print("ok  test_density_is_higher_for_a_dense_span_than_a_sparse_one")


def main():
    test_burst_out_of_calm_yields_one_tight_point_cut_after_peak()
    test_blinking_periodic_energy_yields_none_not_split()
    test_uniform_static_yields_no_cut_at_all()
    test_smooth_pan_yields_span_cut_at_move_start_and_settle()
    test_two_separated_bursts_yield_two_cuts()
    test_two_near_bursts_consolidate_to_one()
    test_contrast_based_peak_beats_absolute_level_on_ramp_then_plateau()
    test_speech_spans_are_subtracted_from_working_spans()
    test_no_cut_ever_crosses_into_a_speech_span()
    test_finalize_cuts_are_always_disjoint_and_sorted()
    test_finalize_cuts_clamps_extended_edges_to_the_working_span()
    test_finalize_cuts_merges_a_sub_floor_sliver_into_its_nearest_neighbor()
    test_lone_cut_below_the_floor_survives_with_no_neighbor_to_merge_into()
    test_sub_floor_sliver_never_welds_across_a_speech_gap()
    test_single_event_cluster_has_exactly_one_event()
    test_two_pans_both_kept_not_just_the_longest()
    test_peak_and_pan_coexist_in_one_span()
    test_tight_cluster_of_peaks_yields_one_cluster_with_n_events()
    test_big_gap_between_bursts_yields_two_clusters()
    test_noisy_curve_near_one_burst_still_yields_one_event()
    test_merged_sliver_unions_events_not_drops_them()
    test_dead_head_then_incoherent_shake_biases_to_the_real_activity()
    test_whip_between_two_holds_snaps_the_cut_edge()
    test_seam_deep_inside_content_is_not_snapped()
    test_coherent_move_overlapping_action_is_preserved_not_trimmed()
    test_below_old_floor_move_registers_when_clip_relatively_significant()
    test_uniform_low_magnitude_camera_drift_is_never_promoted_to_a_move()
    test_composition_point_opens_a_cut_in_an_otherwise_dead_span()
    test_evenly_spaced_composition_points_are_suppressed_as_periodic()
    test_sustained_plateau_beyond_novelty_reach_gets_covered_by_a_run_event()
    test_lull_splits_a_long_active_stretch_into_separate_moments()
    test_representative_window_locks_to_one_period_of_periodic_content()
    test_whip_inside_a_sustained_action_run_is_not_a_seam()
    test_whip_far_from_any_composition_change_is_protected_by_low_drift()
    test_whip_right_at_a_composition_change_still_registers()
    test_sharp_spike_padding_floor_does_not_reach_into_a_dead_sub_region()
    test_padding_floor_still_applies_when_the_reach_carries_energy()
    test_density_is_higher_for_a_dense_span_than_a_sparse_one()
    print("\nall v4_segment tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
