"""
Pure unit tests for app.services.vcut.resolve.resolve_cuts -- THE ALGORITHM,
flag-based and file-wide (vcut_moment_energy.plan.md section 11). No I/O --
synthetic MomentPlan + seam dicts only, same style as test_seam_curve.py.

Important fixture note: a completely FLAT S(t) array trivially ties its own
max at every point, so with the strong-seam-wall logic it would falsely wall
off EVERY pair of flags. Tests that expect genuine fusion use a LOW baseline
S with the file's "max" set by a spike placed well away from the flags under
test (see ``_low_seam``), not a flat array.

Run:  .venv/bin/python scripts/test_vcut_resolve.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.params import MIN_CUT_GAP_MS, MIN_CUT_MS, STRONG_SEAM_FRAC  # noqa: E402
from app.services.vcut.resolve import (  # noqa: E402
    FilePlan, MomentFlag, MomentPlan, ResolvedCut, _Candidate, _drop_dead_air,
    _enforce_min_gap, _strong_seam_between, _tag_window, _widen_to_min_cut, resolve_cuts,
)


def _spike(n, idx, value=1.0):
    arr = [0.0] * n
    if 0 <= idx < n:
        arr[idx] = value
    return arr


def _low_seam(n, hop_ms=100, high_at=0, baseline=0.1, high=1.0):
    """S with a LOW baseline everywhere and a single tall spike at
    ``high_at`` -- guarantees no strong-seam wall forms anywhere else in the
    array (baseline < STRONG_SEAM_FRAC * high), so tests can rely on
    genuine fusion happening wherever the reach/midpoint math says it
    should."""
    S = [baseline] * n
    if 0 <= high_at < n:
        S[high_at] = high
    return S


def _seam(n, hop_ms=100, S=None, action_energy=None, frame_diff=None):
    return {
        "hop_ms": hop_ms,
        "S": S if S is not None else _low_seam(n),
        "action_energy": action_energy if action_energy is not None else [0.0] * n,
        "frame_diff": frame_diff if frame_diff is not None else [0.0] * n,
    }


def _plan(file_id, flags):
    return MomentPlan(files=[FilePlan(file_id=file_id, flags=flags)])


# --------------------------------------------------------------------------
# single flag never divides; only shrinks with energy
# --------------------------------------------------------------------------

def test_single_flag_never_divides_only_shrinks_with_energy():
    n = 101
    plan = _plan("f1", [MomentFlag(t_ms=5000, shape="both")])
    seam = {"f1": _seam(n, action_energy=_spike(n, 50))}

    low = resolve_cuts(plan, seam, energy=0.0)
    mid = resolve_cuts(plan, seam, energy=0.5)
    high = resolve_cuts(plan, seam, energy=1.0)

    assert len(low) == 1 and len(mid) == 1 and len(high) == 1
    w_low = low[0].out_ms - low[0].in_ms
    w_mid = mid[0].out_ms - mid[0].in_ms
    w_high = high[0].out_ms - high[0].in_ms
    assert w_high < w_mid < w_low, (w_low, w_mid, w_high)
    assert low[0].peak_ms == 5000 and high[0].peak_ms == 5000
    print("ok  test_single_flag_never_divides_only_shrinks_with_energy")


# --------------------------------------------------------------------------
# section 11 test 1/2: fuse at energy 0, fall apart as energy rises,
# monotonicity
# --------------------------------------------------------------------------

def test_fuse_at_energy_zero_no_strong_seam():
    # "both" flags 2000ms apart: REACH_MAX_MS(5000)/2=2500 > the 1000ms
    # midpoint distance -> clamped windows touch at every midpoint -> the
    # whole chain fuses into ONE cut.
    n = 141
    flags = [MomentFlag(t_ms=ms, shape="both") for ms in (5000, 7000, 9000)]
    plan = _plan("f1", flags)
    ae = _spike(n, 50)
    for ms in (7000, 9000):
        ae[ms // 100] = 1.0
    seam = {"f1": _seam(n, S=_low_seam(n, high_at=5), action_energy=ae)}

    out = resolve_cuts(plan, seam, energy=0.0)
    assert len(out) == 1, out
    print("ok  test_fuse_at_energy_zero_no_strong_seam")


def test_fall_apart_as_energy_rises_same_flags():
    n = 141
    flags = [MomentFlag(t_ms=ms, shape="both") for ms in (5000, 7000, 9000)]
    plan = _plan("f1", flags)
    ae = _spike(n, 50)
    for ms in (7000, 9000):
        ae[ms // 100] = 1.0
    seam = {"f1": _seam(n, S=_low_seam(n, high_at=5), action_energy=ae)}

    out = resolve_cuts(plan, seam, energy=0.7)
    assert len(out) == 3, out
    for c in out:
        assert c.in_ms <= c.peak_ms <= c.out_ms
    print("ok  test_fall_apart_as_energy_rises_same_flags")


def test_monotonicity_count_nondecreasing_total_span_nonincreasing():
    n = 141
    flags = [MomentFlag(t_ms=ms, shape="both") for ms in (5000, 7000, 9000)]
    plan = _plan("f1", flags)
    ae = _spike(n, 50)
    for ms in (7000, 9000):
        ae[ms // 100] = 1.0
    seam = {"f1": _seam(n, S=_low_seam(n, high_at=5), action_energy=ae)}

    energies = [0.0, 0.2, 0.4, 0.5, 0.7, 1.0]
    counts = []
    spans = []
    for e in energies:
        out = resolve_cuts(plan, seam, energy=e)
        counts.append(len(out))
        spans.append(sum(c.out_ms - c.in_ms for c in out))
    assert all(b >= a for a, b in zip(counts, counts[1:])), counts
    assert all(b <= a + 1 for a, b in zip(spans, spans[1:])), spans  # +1 tolerates int rounding
    print("ok  test_monotonicity_count_nondecreasing_total_span_nonincreasing")


# --------------------------------------------------------------------------
# section 11 test 3: strong-seam wall
# --------------------------------------------------------------------------

def test_strong_seam_wall_prevents_fusion_even_at_energy_zero():
    # Same spacing/shape as test_fuse_at_energy_zero (would normally fuse at
    # e=0), but a tall spike sits BETWEEN the first two flags -- a strong
    # seam wall that must keep them apart at every energy.
    n = 141
    flags = [MomentFlag(t_ms=ms, shape="both") for ms in (5000, 7000, 9000)]
    plan = _plan("f1", flags)
    ae = _spike(n, 50)
    for ms in (7000, 9000):
        ae[ms // 100] = 1.0
    S = [0.1] * n
    S[60] = 1.0  # t=6000ms, strictly between the 5000/7000 flags

    seam = {"f1": _seam(n, S=S, action_energy=ae)}
    out = resolve_cuts(plan, seam, energy=0.0)
    # the 5000 flag stays isolated from 7000/9000 (which still fuse with
    # each other -- no wall between THEM) -> two groups, not one.
    assert len(out) == 2, out
    print("ok  test_strong_seam_wall_prevents_fusion_even_at_energy_zero")


def test_strong_seam_between_helper_respects_threshold():
    n = 101
    S = [0.1] * n
    S[50] = 1.0  # t=5000, between peak_i=2000 and peak_j=8000
    threshold_hit = STRONG_SEAM_FRAC * max(S)
    assert _strong_seam_between(2000, 8000, 100, S, threshold_hit) == 5000

    # a spike too weak to clear the threshold -> no wall.
    S2 = [0.1] * n
    S2[50] = 0.5
    threshold_miss = STRONG_SEAM_FRAC * 1.0  # threshold from a DIFFERENT, taller file max
    assert _strong_seam_between(2000, 8000, 100, S2, threshold_miss) is None
    print("ok  test_strong_seam_between_helper_respects_threshold")


# --------------------------------------------------------------------------
# section 11 test 4: shape asymmetry
# --------------------------------------------------------------------------

def test_tag_asymmetry_before_after_split():
    both_a, both_b = _tag_window(1000, "both", 1000.0)
    build_a, build_b = _tag_window(1000, "build", 1000.0)
    settle_a, settle_b = _tag_window(1000, "settle", 1000.0)

    assert both_a == 500.0 and both_b == 1500.0
    assert build_a == 300.0 and build_b == 1300.0
    assert (1000 - build_a) > (build_b - 1000)  # build: keep the run-up
    assert settle_a == 700.0 and settle_b == 1700.0
    assert (settle_b - 1000) > (1000 - settle_a)  # settle: keep the landing
    print("ok  test_tag_asymmetry_before_after_split")


# --------------------------------------------------------------------------
# edges snap to the local S maximum within SNAP_MS; never past a flag
# --------------------------------------------------------------------------

def test_edges_snap_to_local_s_maximum():
    n = 101
    S = [0.1] * n
    S[26] = 0.9  # 2600ms -- within SNAP_MS of the e=0 provisional in-edge (2500)
    S[74] = 0.9  # 7400ms -- within SNAP_MS of the e=0 provisional out-edge (7500)
    plan = _plan("f1", [MomentFlag(t_ms=5000, shape="both")])
    seam = {"f1": _seam(n, S=S, action_energy=_spike(n, 50))}

    out = resolve_cuts(plan, seam, energy=0.0)  # REACH_MAX_MS=5000 -> provisional [2500,7500]
    assert len(out) == 1
    assert out[0].in_ms == 2600, out[0].in_ms
    assert out[0].out_ms == 7400, out[0].out_ms
    print("ok  test_edges_snap_to_local_s_maximum")


def test_snap_never_crosses_past_the_peak():
    n = 101
    S = [0.1] * n
    S[52] = 0.99  # 5200ms -- just past the flag at 5000ms
    plan = _plan("f1", [MomentFlag(t_ms=5000, shape="both")])
    seam = {"f1": _seam(n, S=S, action_energy=_spike(n, 50))}

    out = resolve_cuts(plan, seam, energy=1.0)  # REACH_MIN_MS window includes 5200 in range
    assert len(out) == 1
    assert out[0].in_ms <= 5000, out[0].in_ms
    print("ok  test_snap_never_crosses_past_the_peak")


# --------------------------------------------------------------------------
# floors: dead-air / min-cut / min-gap (unchanged mechanics, summary field)
# --------------------------------------------------------------------------

def test_dead_air_floor_drops_only_peakless_low_median_s_windows():
    n = 101
    S = [1.0] * n
    S[0:10] = [0.01] * 10
    hop_ms = 100
    dead_peakless = _Candidate("f1", 0, 500, peak_ms=250, tag="both", summary="m",
                              span_lo=0, span_hi=1000, has_peak=False)
    dead_with_peak = _Candidate("f1", 0, 500, peak_ms=250, tag="both", summary="m",
                                span_lo=0, span_hi=1000, has_peak=True)
    loud = _Candidate("f1", 5000, 5500, peak_ms=5250, tag="both", summary="m",
                      span_lo=5000, span_hi=6000, has_peak=False)

    kept = _drop_dead_air([dead_peakless, dead_with_peak, loud], hop_ms, S)
    assert dead_peakless not in kept
    assert dead_with_peak in kept
    assert loud in kept
    print("ok  test_dead_air_floor_drops_only_peakless_low_median_s_windows")


def test_min_cut_floor_widens_short_windows():
    short = _Candidate("f1", 1000, 1000 + MIN_CUT_MS / 2, peak_ms=1000 + MIN_CUT_MS / 4,
                       tag="both", summary="m", span_lo=0, span_hi=10000)
    out = _widen_to_min_cut([short])
    assert out[0].b - out[0].a >= MIN_CUT_MS - 1e-6, (out[0].a, out[0].b)
    print("ok  test_min_cut_floor_widens_short_windows")


def test_min_cut_floor_is_a_noop_above_the_floor():
    fine = _Candidate("f1", 1000, 1000 + MIN_CUT_MS * 2, peak_ms=1500,
                      tag="both", summary="m", span_lo=0, span_hi=10000)
    out = _widen_to_min_cut([fine])
    assert out[0].a == fine.a and out[0].b == fine.b
    print("ok  test_min_cut_floor_is_a_noop_above_the_floor")


def test_min_gap_floor_pulls_close_cuts_apart_without_crossing_peaks():
    prev = ResolvedCut(file_id="f1", in_ms=0, out_ms=1000, peak_ms=500, tag="both", summary="m")
    gap = MIN_CUT_GAP_MS // 4
    cur = ResolvedCut(file_id="f1", in_ms=1000 + gap, out_ms=2000, peak_ms=1000 + gap + 500,
                      tag="both", summary="m")
    out = _enforce_min_gap([cur, prev])
    out_by_peak = {c.peak_ms: c for c in out}
    new_prev, new_cur = out_by_peak[500], out_by_peak[1000 + gap + 500]
    assert new_cur.in_ms - new_prev.out_ms >= MIN_CUT_GAP_MS - 1, (new_prev.out_ms, new_cur.in_ms)
    assert new_prev.out_ms >= new_prev.peak_ms
    assert new_cur.in_ms <= new_cur.peak_ms
    print("ok  test_min_gap_floor_pulls_close_cuts_apart_without_crossing_peaks")


def test_min_gap_floor_ignores_cuts_from_different_files():
    a = ResolvedCut(file_id="f1", in_ms=0, out_ms=1000, peak_ms=500, tag="both", summary="m")
    b = ResolvedCut(file_id="f2", in_ms=1000, out_ms=2000, peak_ms=1500, tag="both", summary="m")
    out = _enforce_min_gap([a, b])
    assert out[0].out_ms == 1000 and out[1].in_ms == 1000
    print("ok  test_min_gap_floor_ignores_cuts_from_different_files")


def test_missing_seam_for_a_file_contributes_no_cuts():
    plan = _plan("f1", [MomentFlag(t_ms=500, shape="both")])
    out = resolve_cuts(plan, {}, energy=0.5)
    assert out == []
    print("ok  test_missing_seam_for_a_file_contributes_no_cuts")


# --------------------------------------------------------------------------
# section 11 test 5: flag-only input / back-compat from_dict
# --------------------------------------------------------------------------

def test_file_plan_from_dict_new_shape():
    fp = FilePlan.from_dict("f1", {"flags": [
        {"t_ms": 1000, "shape": "build", "summary": "a"},
        {"t_ms": 2000, "shape": "settle", "summary": "b"},
    ]})
    assert fp.file_id == "f1"
    assert [f.t_ms for f in fp.flags] == [1000, 2000]
    assert [f.shape for f in fp.flags] == ["build", "settle"]
    assert [f.summary for f in fp.flags] == ["a", "b"]
    print("ok  test_file_plan_from_dict_new_shape")


def test_file_plan_from_dict_legacy_shape_flattens_peaks_to_flags():
    legacy = {
        "meaning": "doing push-ups", "question_ids": ["subject"],
        "loose_cuts": [
            {"span_ms": [0, 4000], "peaks": [{"t_ms": 500, "tag": "settle"}]},
            {"span_ms": [4000, 8000], "peaks": [
                {"t_ms": 4500, "tag": "build"}, {"t_ms": 5500, "tag": "build"},
            ]},
        ],
    }
    fp = FilePlan.from_dict("f1", legacy)
    assert [f.t_ms for f in fp.flags] == [500, 4500, 5500]
    assert [f.shape for f in fp.flags] == ["settle", "build", "build"]
    assert all(f.summary == "doing push-ups" for f in fp.flags)
    print("ok  test_file_plan_from_dict_legacy_shape_flattens_peaks_to_flags")


def test_moment_plan_from_dict_multi_file_both_shapes():
    data = {
        "f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x"}]},
        "f2": {"meaning": "y", "loose_cuts": [{"span_ms": [0, 100], "peaks": [{"t_ms": 50, "tag": "build"}]}]},
    }
    plan = MomentPlan.from_dict(data)
    by_file = {fp.file_id: fp for fp in plan.files}
    assert by_file["f1"].flags[0].t_ms == 100
    assert by_file["f2"].flags[0].t_ms == 50 and by_file["f2"].flags[0].summary == "y"
    print("ok  test_moment_plan_from_dict_multi_file_both_shapes")


def test_file_plan_to_dict_round_trips_through_from_dict():
    fp = FilePlan(file_id="f1", flags=[MomentFlag(t_ms=100, shape="build", summary="s")])
    round_tripped = FilePlan.from_dict("f1", fp.to_dict())
    assert round_tripped.flags == fp.flags
    print("ok  test_file_plan_to_dict_round_trips_through_from_dict")


def test_moment_flag_round_trips_question_ids_custom_questions_specifics():
    fp = FilePlan(file_id="f1", flags=[MomentFlag(
        t_ms=100, shape="build", summary="s",
        question_ids=["subject", "action"],
        custom_questions=[{"key": "brand", "prompt": "what brand is shown?"}],
        specifics={"subject": "a shoe"},
        subject_box=(0.1, 0.2, 0.3, 0.4),
    )])
    round_tripped = FilePlan.from_dict("f1", fp.to_dict())
    assert round_tripped.flags == fp.flags
    print("ok  test_moment_flag_round_trips_question_ids_custom_questions_specifics")


def test_legacy_from_dict_leaves_new_fields_at_their_defaults():
    legacy = {"meaning": "m", "loose_cuts": [{"span_ms": [0, 100], "peaks": [{"t_ms": 50, "tag": "build"}]}]}
    fp = FilePlan.from_dict("f1", legacy)
    assert fp.flags[0].question_ids == []
    assert fp.flags[0].custom_questions == []
    assert fp.flags[0].specifics == {}
    assert fp.flags[0].subject_box is None
    print("ok  test_legacy_from_dict_leaves_new_fields_at_their_defaults")


def test_moment_plan_genre_round_trips_through_reserved_meta_key():
    plan = MomentPlan(files=[FilePlan(file_id="f1", flags=[MomentFlag(t_ms=100)])], genre="tutorial")
    data = plan.to_dict()
    assert data["__meta__"] == {"genre": "tutorial"}
    assert "f1" in data
    round_tripped = MomentPlan.from_dict(data)
    assert round_tripped.genre == "tutorial"
    assert [fp.file_id for fp in round_tripped.files] == ["f1"]
    print("ok  test_moment_plan_genre_round_trips_through_reserved_meta_key")


def test_moment_plan_from_dict_no_meta_key_defaults_genre_to_empty():
    data = {"f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x"}]}}
    plan = MomentPlan.from_dict(data)
    assert plan.genre == ""
    print("ok  test_moment_plan_from_dict_no_meta_key_defaults_genre_to_empty")


# --------------------------------------------------------------------------
# vcut_pass2_video_specifics.plan.md section 7.2: composed specifics --
# single-flag cut passes through unchanged; merged multi-flag cut carries
# the representative flag's fields + a "moments" mini shot-list. Energy-
# invariant: a moment's specifics always land on whatever cut contains it.
# --------------------------------------------------------------------------

def test_composed_specifics_single_flag_passes_through_unchanged():
    n = 101
    plan = _plan("f1", [MomentFlag(t_ms=5000, shape="both", specifics={"subject": "a dog"})])
    seam = {"f1": _seam(n, action_energy=_spike(n, 50))}
    out = resolve_cuts(plan, seam, energy=0.5)
    assert len(out) == 1
    assert out[0].specifics == {"subject": "a dog"}
    print("ok  test_composed_specifics_single_flag_passes_through_unchanged")


def test_composed_specifics_merged_cut_carries_representative_plus_moments_list():
    n = 141
    flags = [
        MomentFlag(t_ms=5000, shape="both", summary="a", specifics={"subject": "dog"}),
        MomentFlag(t_ms=7000, shape="both", summary="b", specifics={"subject": "cat"}),
        MomentFlag(t_ms=9000, shape="both", summary="c", specifics={"subject": "bird"}),
    ]
    plan = _plan("f1", flags)
    ae = _spike(n, 70)   # strongest action_energy at 7000 -> the representative
    for ms in (5000, 9000):
        ae[ms // 100] = 0.5
    seam = {"f1": _seam(n, S=_low_seam(n, high_at=5), action_energy=ae)}

    out = resolve_cuts(plan, seam, energy=0.0)  # fuses into ONE cut (see test_fuse_at_energy_zero)
    assert len(out) == 1
    cut = out[0]
    assert cut.peak_ms == 7000
    assert cut.specifics["subject"] == "cat"   # representative flag's own field, at the top level
    moments = cut.specifics["moments"]
    assert len(moments) == 3
    by_t = {m["t_ms"]: m for m in moments}
    assert by_t[5000] == {"t_ms": 5000, "summary": "a", "subject": "dog"}
    assert by_t[7000] == {"t_ms": 7000, "summary": "b", "subject": "cat"}
    assert by_t[9000] == {"t_ms": 9000, "summary": "c", "subject": "bird"}
    print("ok  test_composed_specifics_merged_cut_carries_representative_plus_moments_list")


def test_composed_specifics_energy_invariant_lands_on_whatever_cut_contains_it():
    n = 141
    flags = [
        MomentFlag(t_ms=5000, shape="both", summary="a", specifics={"subject": "dog"}),
        MomentFlag(t_ms=7000, shape="both", summary="b", specifics={"subject": "cat"}),
        MomentFlag(t_ms=9000, shape="both", summary="c", specifics={"subject": "bird"}),
    ]
    plan = _plan("f1", flags)
    ae = _spike(n, 70)
    for ms in (5000, 9000):
        ae[ms // 100] = 0.5
    seam = {"f1": _seam(n, S=_low_seam(n, high_at=5), action_energy=ae)}

    low = resolve_cuts(plan, seam, energy=0.0)
    assert len(low) == 1
    low_subjects = {m["subject"] for m in low[0].specifics["moments"]}
    assert low_subjects == {"dog", "cat", "bird"}

    high = resolve_cuts(plan, seam, energy=1.0)
    assert len(high) == 3
    by_peak = {c.peak_ms: c for c in high}
    assert by_peak[5000].specifics == {"subject": "dog"}
    assert by_peak[7000].specifics == {"subject": "cat"}
    assert by_peak[9000].specifics == {"subject": "bird"}
    print("ok  test_composed_specifics_energy_invariant_lands_on_whatever_cut_contains_it")


# --------------------------------------------------------------------------
# reframe_vcut_geometry.plan.md section 3/testing: composed subject_box --
# a single-flag cut passes its box through; a merged cut carries the
# REPRESENTATIVE flag's own box (no merge, unlike specifics' moments-list);
# energy 0 and 1 give the same box for a given moment.
# --------------------------------------------------------------------------

def test_composed_subject_box_single_flag_passes_through_unchanged():
    n = 101
    plan = _plan("f1", [MomentFlag(t_ms=5000, shape="both", subject_box=(0.1, 0.2, 0.3, 0.4))])
    seam = {"f1": _seam(n, action_energy=_spike(n, 50))}
    out = resolve_cuts(plan, seam, energy=0.5)
    assert len(out) == 1
    assert out[0].subject_box == (0.1, 0.2, 0.3, 0.4)
    print("ok  test_composed_subject_box_single_flag_passes_through_unchanged")


def test_composed_subject_box_none_when_no_flag_has_one():
    n = 101
    plan = _plan("f1", [MomentFlag(t_ms=5000, shape="both")])
    seam = {"f1": _seam(n, action_energy=_spike(n, 50))}
    out = resolve_cuts(plan, seam, energy=0.5)
    assert out[0].subject_box is None
    print("ok  test_composed_subject_box_none_when_no_flag_has_one")


def test_composed_subject_box_merged_cut_carries_representative_box_energy_invariant():
    n = 141
    flags = [
        MomentFlag(t_ms=5000, shape="both", summary="a", subject_box=(0.0, 0.0, 0.1, 0.1)),
        MomentFlag(t_ms=7000, shape="both", summary="b", subject_box=(0.4, 0.4, 0.2, 0.2)),
        MomentFlag(t_ms=9000, shape="both", summary="c", subject_box=(0.8, 0.8, 0.1, 0.1)),
    ]
    plan = _plan("f1", flags)
    ae = _spike(n, 70)   # strongest action_energy at 7000 -> the representative
    for ms in (5000, 9000):
        ae[ms // 100] = 0.5
    seam = {"f1": _seam(n, S=_low_seam(n, high_at=5), action_energy=ae)}

    low = resolve_cuts(plan, seam, energy=0.0)  # fuses into ONE cut (see test_fuse_at_energy_zero)
    assert len(low) == 1
    assert low[0].peak_ms == 7000
    assert low[0].subject_box == (0.4, 0.4, 0.2, 0.2)  # the representative flag's own box, not merged

    high = resolve_cuts(plan, seam, energy=1.0)  # falls apart into 3 -- same box per moment either way
    assert len(high) == 3
    by_peak = {c.peak_ms: c for c in high}
    assert by_peak[5000].subject_box == (0.0, 0.0, 0.1, 0.1)
    assert by_peak[7000].subject_box == (0.4, 0.4, 0.2, 0.2)
    assert by_peak[9000].subject_box == (0.8, 0.8, 0.1, 0.1)
    print("ok  test_composed_subject_box_merged_cut_carries_representative_box_energy_invariant")


def test_back_compat_from_dict_resolves_correctly_end_to_end():
    n = 101
    legacy = {"meaning": "m", "loose_cuts": [{"span_ms": [0, 10000], "peaks": [{"t_ms": 5000, "tag": "both"}]}]}
    plan = MomentPlan(files=[FilePlan.from_dict("f1", legacy)])
    seam = {"f1": _seam(n, action_energy=_spike(n, 50))}
    out = resolve_cuts(plan, seam, energy=0.5)
    assert len(out) == 1
    assert out[0].summary == "m"
    print("ok  test_back_compat_from_dict_resolves_correctly_end_to_end")


# --------------------------------------------------------------------------
# section 4.6 / 11: the push-up worked example, reworked for the file-wide
# flag model -- strong-seam walls (not VLM loose-cut spans) now keep
# "setup"/"push-ups"/"shutdown" separate.
# --------------------------------------------------------------------------

def test_pushup_worked_example_low_and_high_energy():
    n = 181
    setup_peak = 3500
    pushup_peaks = [4500 + 1000 * i for i in range(8)]  # 4500..11500, build, 1000ms apart
    shutdown_peak = 13500

    flags = (
        [MomentFlag(t_ms=setup_peak, shape="settle", summary="setting up the camera")]
        + [MomentFlag(t_ms=ms, shape="build", summary="doing push-ups") for ms in pushup_peaks]
        + [MomentFlag(t_ms=shutdown_peak, shape="build", summary="walking over, switching off")]
    )
    plan = _plan("f1", flags)

    ae = _spike(n, setup_peak // 100)
    for ms in pushup_peaks:
        ae[ms // 100] = 1.0
    ae[shutdown_peak // 100] = 1.0

    # Strong-seam walls between setup<->pushups (a spike at 4000) and
    # pushups<->shutdown (a spike at 12500) -- everything else stays low.
    S = [0.1] * n
    S[40] = 1.0   # t=4000ms
    S[125] = 1.0  # t=12500ms
    seam = {"f1": _seam(n, S=S, action_energy=ae)}

    low = resolve_cuts(plan, seam, energy=0.0)
    high = resolve_cuts(plan, seam, energy=1.0)

    # Low energy: setup(1) + the whole push-up block fused(1) + shutdown(1).
    assert len(low) == 3, len(low)
    # High energy: setup(1) + one cut per rep(8) + shutdown(1).
    assert len(high) == 10, len(high)
    for c in low + high:
        assert 0 <= c.in_ms and c.out_ms <= 18000
    print("ok  test_pushup_worked_example_low_and_high_energy")


def main():
    test_single_flag_never_divides_only_shrinks_with_energy()
    test_fuse_at_energy_zero_no_strong_seam()
    test_fall_apart_as_energy_rises_same_flags()
    test_monotonicity_count_nondecreasing_total_span_nonincreasing()
    test_strong_seam_wall_prevents_fusion_even_at_energy_zero()
    test_strong_seam_between_helper_respects_threshold()
    test_tag_asymmetry_before_after_split()
    test_edges_snap_to_local_s_maximum()
    test_snap_never_crosses_past_the_peak()
    test_dead_air_floor_drops_only_peakless_low_median_s_windows()
    test_min_cut_floor_widens_short_windows()
    test_min_cut_floor_is_a_noop_above_the_floor()
    test_min_gap_floor_pulls_close_cuts_apart_without_crossing_peaks()
    test_min_gap_floor_ignores_cuts_from_different_files()
    test_missing_seam_for_a_file_contributes_no_cuts()
    test_file_plan_from_dict_new_shape()
    test_file_plan_from_dict_legacy_shape_flattens_peaks_to_flags()
    test_moment_plan_from_dict_multi_file_both_shapes()
    test_file_plan_to_dict_round_trips_through_from_dict()
    test_moment_flag_round_trips_question_ids_custom_questions_specifics()
    test_legacy_from_dict_leaves_new_fields_at_their_defaults()
    test_moment_plan_genre_round_trips_through_reserved_meta_key()
    test_moment_plan_from_dict_no_meta_key_defaults_genre_to_empty()
    test_composed_specifics_single_flag_passes_through_unchanged()
    test_composed_specifics_merged_cut_carries_representative_plus_moments_list()
    test_composed_specifics_energy_invariant_lands_on_whatever_cut_contains_it()
    test_composed_subject_box_single_flag_passes_through_unchanged()
    test_composed_subject_box_none_when_no_flag_has_one()
    test_composed_subject_box_merged_cut_carries_representative_box_energy_invariant()
    test_back_compat_from_dict_resolves_correctly_end_to_end()
    test_pushup_worked_example_low_and_high_energy()
    print("\nall vcut resolve tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
