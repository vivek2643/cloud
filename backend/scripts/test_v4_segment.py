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


def test_uniform_static_yields_single_representative_cut():
    cuts = _segment(_flat_motion(100))
    assert len(cuts) == 1, cuts
    assert cuts[0].salience["kind"] == "none", cuts[0].salience
    assert cuts[0].src_out_ms - cuts[0].src_in_ms < 10_000, "never the whole clip by default"
    print("ok  test_uniform_static_yields_single_representative_cut")


def test_smooth_pan_yields_span_cut_at_move_start_and_settle():
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
    """A single short representative-window cut with nothing else in the
    file has no neighbor to merge into -- it must survive as-is rather than
    vanish (better a short cut than none)."""
    cuts = v4.segment_video(file_id="f1", duration_ms=300, speech_spans=[],
                            motion=_flat_motion(3, hop=100), audio={}, scene={})
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
    test_uniform_static_yields_single_representative_cut()
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
    test_density_is_higher_for_a_dense_span_than_a_sparse_one()
    print("\nall v4_segment tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
