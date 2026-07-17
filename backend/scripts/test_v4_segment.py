"""
Pure unit tests for the V4 deterministic video segmenter
(``app.services.l3.v4_segment``) -- no DB, no model call.

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
from app.services.l3.lattice import Atom, Lattice  # noqa: E402


def _flat_motion(n, hop=100, action=0.05, stability=0.95, coh=0.9):
    return {
        "hop_ms": hop, "action_energy": [action] * n, "camera_stability": [stability] * n,
        "camera_coherence": [coh] * n, "camera_motion": [0.0] * n, "blur": [0.1] * n,
        "camera_dx": [0.0] * n, "camera_dy": [0.0] * n, "camera_zoom": [0.0] * n,
        "action_points": [], "transition_points": [],
    }


def _lattice(duration_ms=10_000, atoms=None):
    return Lattice(file_id="f1", duration_ms=duration_ms,
                   atoms=atoms if atoms is not None else
                   [Atom(0, "f1", 0, duration_ms, "clip_edge", "clip_edge", 0.05, 0.9)])


def _segment(motion, audio=None, scene=None, lattice=None, speech_spans=None, duration_ms=10_000):
    return v4.segment_video(
        file_id="f1", duration_ms=duration_ms, speech_spans=speech_spans or [],
        motion=motion, audio=audio or {}, scene=scene or {}, lattice=lattice or _lattice(duration_ms),
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
    assert cuts[0].src_out_ms < cuts[1].src_in_ms, "must not overlap"
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
# Speech subtraction + atom_ids mapping
# --------------------------------------------------------------------------

def test_speech_spans_are_subtracted_from_working_spans():
    atoms = [Atom(0, "f1", 0, 4000, "clip_edge", "clip_edge", 0.05, 0.9),
             Atom(1, "f1", 6000, 10_000, "clip_edge", "clip_edge", 0.05, 0.9)]
    motion = _flat_motion(100)
    ae = [0.05] * 100
    for i in range(15, 20):    # 1500-2000ms, INSIDE the speech span -> must vanish
        ae[i] = 0.9
    for i in range(70, 75):    # 7000-7500ms, outside speech -> must survive
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion, lattice=_lattice(atoms=atoms), speech_spans=[(1000, 3000)])
    assert all(c.src_in_ms >= 3000 or c.src_out_ms <= 1000 for c in cuts), cuts
    assert any(c.src_in_ms >= 6000 for c in cuts), "the surviving burst must still produce a cut"
    print("ok  test_speech_spans_are_subtracted_from_working_spans")


def test_atom_ids_map_back_to_covering_atoms():
    atoms = [Atom(0, "f1", 0, 4000, "clip_edge", "clip_edge", 0.05, 0.9),
             Atom(1, "f1", 6000, 10_000, "clip_edge", "clip_edge", 0.05, 0.9)]
    motion = _flat_motion(100)
    ae = [0.05] * 100
    for i in range(70, 75):
        ae[i] = 0.9
    motion["action_energy"] = ae
    cuts = _segment(motion, lattice=_lattice(atoms=atoms), speech_spans=[(1000, 3000)])
    burst = next(c for c in cuts if c.src_in_ms >= 6000)
    assert burst.atom_ids == [1], burst.atom_ids
    print("ok  test_atom_ids_map_back_to_covering_atoms")


def test_no_atom_overlap_falls_back_to_nearest_atom_id():
    """Never emit a video cut with an empty atom_ids list -- downstream
    (pass2.backfill_locators) treats that as an unresolved locator."""
    atoms = [Atom(0, "f1", 100, 200, "clip_edge", "clip_edge", 0.05, 0.9)]
    ids = v4._atom_ids_covering(_lattice(atoms=atoms), 5000, 5100)
    assert ids == [0], ids
    print("ok  test_no_atom_overlap_falls_back_to_nearest_atom_id")


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
    test_atom_ids_map_back_to_covering_atoms()
    test_no_atom_overlap_falls_back_to_nearest_atom_id()
    test_density_is_higher_for_a_dense_span_than_a_sparse_one()
    print("\nall v4_segment tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
