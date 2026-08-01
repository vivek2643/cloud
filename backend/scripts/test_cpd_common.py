"""
Pure unit tests for scripts/cpd/common.py -- the shared consensus rule,
Rel.Dis F1 metric, peak-picking, and windowing helpers every
cpd_boundary_segmenter.plan.md phase script relies on. No network, no
video decode, no GEBD data -- everything here is synthetic and fast.

Run:  .venv/bin/python scripts/test_cpd_common.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
CPD_DIR = os.path.join(HERE, "cpd")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if CPD_DIR not in sys.path:
    sys.path.insert(0, CPD_DIR)

import numpy as np  # noqa: E402

import common as cpd_common  # noqa: E402
from extract_features import _fit_len  # noqa: E402

# --------------------------------------------------------------------------
# clip_id_for
# --------------------------------------------------------------------------

def test_clip_id_for_zero_pads_and_rounds():
    assert cpd_common.clip_id_for("abc123XYZ_-", 12.4, 22.6) == "abc123XYZ_-_000012_000023"
    print("ok  test_clip_id_for_zero_pads_and_rounds")


# --------------------------------------------------------------------------
# extract_features._fit_len -- pure length-alignment helper (the real
# video-decode path (compute_appearance_drift/extract_one) is exercised via
# an interactive ffmpeg-synthesized smoke test, not this suite -- matches
# this codebase's own test_audio_features.py convention of keeping the
# fast unit suite decode-free and testing signal-processing logic directly).
# --------------------------------------------------------------------------

def test_fit_len_pads_short_arrays_with_zero():
    assert _fit_len([1.0, 2.0], 4) == [1.0, 2.0, 0.0, 0.0]
    print("ok  test_fit_len_pads_short_arrays_with_zero")


def test_fit_len_truncates_long_arrays():
    assert _fit_len([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 2.0]
    print("ok  test_fit_len_truncates_long_arrays")


# --------------------------------------------------------------------------
# consensus_boundaries_sec
# --------------------------------------------------------------------------

def test_consensus_requires_min_agree_distinct_annotators():
    """Two annotators mark near 2.0s, a third marks somewhere else alone --
    only the 2-agreement cluster survives."""
    out = cpd_common.consensus_boundaries_sec(
        [[2.0], [2.05], [7.0]], min_agree=2, merge_window_s=0.3)
    assert len(out) == 1, out
    assert abs(out[0] - 2.025) < 1e-6, out
    print("ok  test_consensus_requires_min_agree_distinct_annotators")


def test_consensus_drops_singleton_boundary():
    out = cpd_common.consensus_boundaries_sec([[1.0], [], []], min_agree=2)
    assert out == [], out
    print("ok  test_consensus_drops_singleton_boundary")


def test_consensus_same_annotator_marking_twice_does_not_count_as_two():
    """Two marks from the SAME annotator close together must not fake a
    2-annotator agreement -- min_agree counts DISTINCT annotators."""
    out = cpd_common.consensus_boundaries_sec([[2.0, 2.1], [7.0]], min_agree=2)
    assert out == [], out
    print("ok  test_consensus_same_annotator_marking_twice_does_not_count_as_two")


def test_consensus_empty_input_yields_no_boundaries():
    assert cpd_common.consensus_boundaries_sec([]) == []
    assert cpd_common.consensus_boundaries_sec([[], [], []]) == []
    print("ok  test_consensus_empty_input_yields_no_boundaries")


def test_consensus_two_separate_clusters_both_survive():
    out = cpd_common.consensus_boundaries_sec(
        [[2.0, 8.0], [2.1, 8.1], [2.2, 8.2]], min_agree=2, merge_window_s=0.3)
    assert len(out) == 2, out
    assert abs(out[0] - 2.1) < 1e-6 and abs(out[1] - 8.1) < 1e-6, out
    print("ok  test_consensus_two_separate_clusters_both_survive")


# --------------------------------------------------------------------------
# hop_labels
# --------------------------------------------------------------------------

def test_hop_labels_marks_hops_within_tolerance():
    labels = cpd_common.hop_labels([2.0], duration_ms=5000, hop_ms=100, tolerance_ms=150)
    # 2.0s = hop 20; tolerance 150ms -> hops 18.5..21.5 -> hops 19,20,21 (ts 1900,2000,2100)
    assert labels[19] == 1.0 and labels[20] == 1.0 and labels[21] == 1.0, labels
    assert labels[17] == 0.0 and labels[23] == 0.0, labels
    print("ok  test_hop_labels_marks_hops_within_tolerance")


def test_hop_labels_length_matches_ceil_duration_over_hop():
    labels = cpd_common.hop_labels([], duration_ms=1050, hop_ms=100, tolerance_ms=100)
    assert len(labels) == 11, len(labels)  # ceil(1050/100) = 11
    print("ok  test_hop_labels_length_matches_ceil_duration_over_hop")


def test_hop_labels_empty_consensus_is_all_zero():
    labels = cpd_common.hop_labels([], duration_ms=2000, hop_ms=100, tolerance_ms=200)
    assert labels.sum() == 0, labels
    print("ok  test_hop_labels_empty_consensus_is_all_zero")


# --------------------------------------------------------------------------
# match_boundaries / f1_at_threshold / f1_curve (GEBD Rel.Dis protocol)
# --------------------------------------------------------------------------

def test_match_boundaries_perfect_match():
    matched, n_pred, n_gt = cpd_common.match_boundaries([2.0, 5.0], [2.05, 5.05], threshold_sec=0.5)
    assert (matched, n_pred, n_gt) == (2, 2, 2), (matched, n_pred, n_gt)
    print("ok  test_match_boundaries_perfect_match")


def test_match_boundaries_is_one_to_one_not_many_to_one():
    """Two predictions both near ONE ground truth boundary must match at
    most once -- a duplicate prediction shouldn't be free precision."""
    matched, n_pred, n_gt = cpd_common.match_boundaries([2.0, 2.05], [2.0], threshold_sec=0.5)
    assert matched == 1, (matched, n_pred, n_gt)
    print("ok  test_match_boundaries_is_one_to_one_not_many_to_one")


def test_match_boundaries_outside_threshold_does_not_match():
    matched, n_pred, n_gt = cpd_common.match_boundaries([2.0], [10.0], threshold_sec=0.5)
    assert matched == 0, (matched, n_pred, n_gt)
    print("ok  test_match_boundaries_outside_threshold_does_not_match")


def test_f1_at_threshold_perfect_predictions_score_one():
    pred = {"c1": [2.0, 5.0]}
    gt = {"c1": [2.0, 5.0]}
    dur = {"c1": 10.0}
    f1 = cpd_common.f1_at_threshold(pred, gt, dur, rel_threshold=0.05)
    assert abs(f1 - 1.0) < 1e-6, f1
    print("ok  test_f1_at_threshold_perfect_predictions_score_one")


def test_f1_at_threshold_no_predictions_and_no_ground_truth_scores_one():
    """Correctly predicting "nothing happens here" must not be penalized."""
    pred = {"c1": []}
    gt = {"c1": []}
    dur = {"c1": 10.0}
    f1 = cpd_common.f1_at_threshold(pred, gt, dur, rel_threshold=0.05)
    assert f1 == 1.0, f1
    print("ok  test_f1_at_threshold_no_predictions_and_no_ground_truth_scores_one")


def test_f1_at_threshold_false_positive_on_empty_ground_truth_scores_zero():
    pred = {"c1": [2.0]}
    gt = {"c1": []}
    dur = {"c1": 10.0}
    f1 = cpd_common.f1_at_threshold(pred, gt, dur, rel_threshold=0.05)
    assert f1 == 0.0, f1
    print("ok  test_f1_at_threshold_false_positive_on_empty_ground_truth_scores_zero")


def test_f1_at_threshold_is_relative_to_clip_duration():
    """The SAME absolute miss (0.4s off) passes at a threshold that's
    relative to a LONG clip's duration but fails on a SHORT one -- this is
    what makes it "Relative Distance", not an absolute-time tolerance."""
    pred = {"c1": [2.4]}
    gt = {"c1": [2.0]}
    f1_long = cpd_common.f1_at_threshold(pred, gt, {"c1": 100.0}, rel_threshold=0.05)  # thr=5s
    f1_short = cpd_common.f1_at_threshold(pred, gt, {"c1": 4.0}, rel_threshold=0.05)   # thr=0.2s
    assert f1_long == 1.0, f1_long
    assert f1_short == 0.0, f1_short
    print("ok  test_f1_at_threshold_is_relative_to_clip_duration")


def test_f1_curve_covers_the_standard_rel_dis_sweep():
    curve = cpd_common.f1_curve({"c1": [2.0]}, {"c1": [2.0]}, {"c1": 10.0})
    assert set(curve.keys()) == set(cpd_common.REL_DIS_THRESHOLDS), curve.keys()
    assert all(v == 1.0 for v in curve.values()), curve
    print("ok  test_f1_curve_covers_the_standard_rel_dis_sweep")


def test_precision_recall_f1_separates_a_precision_bottleneck_from_a_recall_one():
    """Two predictions, one ground truth: precision suffers (an extra false
    positive) but recall is perfect (the one true boundary was found)."""
    pred = {"c1": [2.0, 8.0]}
    gt = {"c1": [2.0]}
    dur = {"c1": 10.0}
    precision, recall, f1 = cpd_common.precision_recall_f1_at_threshold(pred, gt, dur, rel_threshold=0.05)
    assert recall == 1.0, recall
    assert precision == 0.5, precision
    assert abs(f1 - (2 * 0.5 * 1.0 / 1.5)) < 1e-9, f1
    print("ok  test_precision_recall_f1_separates_a_precision_bottleneck_from_a_recall_one")


def test_precision_recall_f1_agrees_with_f1_at_threshold():
    pred = {"c1": [2.0, 5.0], "c2": []}
    gt = {"c1": [2.1], "c2": [3.0]}
    dur = {"c1": 10.0, "c2": 8.0}
    _p, _r, f1_a = cpd_common.precision_recall_f1_at_threshold(pred, gt, dur, rel_threshold=0.1)
    f1_b = cpd_common.f1_at_threshold(pred, gt, dur, rel_threshold=0.1)
    assert abs(f1_a - f1_b) < 1e-12, (f1_a, f1_b)
    print("ok  test_precision_recall_f1_agrees_with_f1_at_threshold")


# --------------------------------------------------------------------------
# find_peak_indices / pick_peaks
# --------------------------------------------------------------------------

def test_find_peak_indices_returns_hop_indices_not_seconds():
    score = np.array([0.0, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0])
    idxs = cpd_common.find_peak_indices(score, hop_ms=100, min_gap_ms=200, min_score=0.5)
    assert idxs == [2, 7], idxs
    print("ok  test_find_peak_indices_returns_hop_indices_not_seconds")


def test_pick_peaks_matches_find_peak_indices_converted_to_seconds():
    score = np.array([0.0, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0])
    idxs = cpd_common.find_peak_indices(score, hop_ms=100, min_gap_ms=200, min_score=0.5)
    peaks = cpd_common.pick_peaks(score, hop_ms=100, min_gap_ms=200, min_score=0.5)
    assert peaks == [i * 100 / 1000.0 for i in idxs], (peaks, idxs)
    print("ok  test_pick_peaks_matches_find_peak_indices_converted_to_seconds")


# --------------------------------------------------------------------------
# pick_peaks
# --------------------------------------------------------------------------

def test_pick_peaks_finds_isolated_local_maxima():
    score = np.array([0.0, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0])
    peaks = cpd_common.pick_peaks(score, hop_ms=100, min_gap_ms=200, min_score=0.5)
    assert peaks == [0.2, 0.7], peaks
    print("ok  test_pick_peaks_finds_isolated_local_maxima")


def test_pick_peaks_suppresses_a_second_peak_too_close_to_a_stronger_one():
    score = np.array([0.0, 0.9, 0.85, 0.0])
    peaks = cpd_common.pick_peaks(score, hop_ms=100, min_gap_ms=300, min_score=0.5)
    assert peaks == [0.1], peaks
    print("ok  test_pick_peaks_suppresses_a_second_peak_too_close_to_a_stronger_one")


def test_pick_peaks_respects_min_score_floor():
    score = np.array([0.1, 0.2, 0.15])
    assert cpd_common.pick_peaks(score, hop_ms=100, min_score=0.5) == []
    print("ok  test_pick_peaks_respects_min_score_floor")


def test_pick_peaks_empty_score_yields_no_peaks():
    assert cpd_common.pick_peaks(np.array([]), hop_ms=100) == []
    print("ok  test_pick_peaks_empty_score_yields_no_peaks")


# --------------------------------------------------------------------------
# unsupervised_boundary_score -- honest classical CPD (Phase E.1)
# --------------------------------------------------------------------------

def test_unsupervised_score_peaks_at_a_clean_step_change():
    n, c = 40, 3
    features = np.zeros((n, c), dtype=np.float32)
    features[20:, :] = 5.0  # a clean step at hop 20
    score = cpd_common.unsupervised_boundary_score(features, window_hops=5)
    assert int(np.argmax(score)) == 20, (int(np.argmax(score)), score)
    print("ok  test_unsupervised_score_peaks_at_a_clean_step_change")


def test_unsupervised_score_is_near_zero_on_a_constant_signal():
    features = np.ones((30, 4), dtype=np.float32) * 3.0
    score = cpd_common.unsupervised_boundary_score(features, window_hops=5)
    assert float(score.max()) < 1e-6, score
    print("ok  test_unsupervised_score_is_near_zero_on_a_constant_signal")


# --------------------------------------------------------------------------
# windowed_features / smooth_curve
# --------------------------------------------------------------------------

def test_windowed_features_shape_and_edge_padding():
    t, c, w = 6, 2, 2
    features = np.arange(t * c, dtype=np.float32).reshape(t, c)
    out = cpd_common.windowed_features(features, window_hops=w)
    assert out.shape == (t, (2 * w + 1) * c), out.shape
    # The first row's window is edge-padded: the first `w` hops all repeat
    # row 0 (there's no real history before the clip starts).
    first_row_window = out[0].reshape(2 * w + 1, c)
    assert np.array_equal(first_row_window[0], features[0])
    assert np.array_equal(first_row_window[w], features[0])  # center = hop 0 itself
    print("ok  test_windowed_features_shape_and_edge_padding")


def test_smooth_curve_flattens_a_single_spike():
    values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    out = cpd_common.smooth_curve(values, window_hops=1)
    assert out[2] < 1.0, out
    assert out[2] == max(out), out  # still the peak, just softened
    print("ok  test_smooth_curve_flattens_a_single_spike")


def test_smooth_curve_no_op_for_zero_window():
    values = np.array([0.1, 0.9, 0.2])
    out = cpd_common.smooth_curve(values, window_hops=0)
    assert np.array_equal(out, values), out
    print("ok  test_smooth_curve_no_op_for_zero_window")


# --------------------------------------------------------------------------
# npz round-trip
# --------------------------------------------------------------------------

def test_save_and_load_clip_npz_roundtrip(tmp_path=None):
    import tempfile
    features = np.random.rand(15, len(cpd_common.CHANNEL_NAMES)).astype(np.float32)
    labels = np.zeros(15, dtype=np.float32)
    labels[7] = 1.0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip.npz")
        cpd_common.save_clip_npz(path, features, hop_ms=100, labels=labels, clip_id="clip1")
        loaded = cpd_common.load_clip_npz(path)
    assert np.allclose(loaded["features"], features, atol=1e-6), loaded["features"]
    assert loaded["hop_ms"] == 100, loaded["hop_ms"]
    assert loaded["channel_names"] == list(cpd_common.CHANNEL_NAMES), loaded["channel_names"]
    assert loaded["clip_id"] == "clip1", loaded["clip_id"]
    assert np.array_equal(loaded["labels"], labels), loaded["labels"]
    print("ok  test_save_and_load_clip_npz_roundtrip")


def test_load_clip_npz_without_labels_omits_the_key():
    import tempfile
    features = np.zeros((5, len(cpd_common.CHANNEL_NAMES)), dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip.npz")
        cpd_common.save_clip_npz(path, features, hop_ms=100)
        loaded = cpd_common.load_clip_npz(path)
    assert "labels" not in loaded, loaded
    print("ok  test_load_clip_npz_without_labels_omits_the_key")


def main():
    test_fit_len_pads_short_arrays_with_zero()
    test_fit_len_truncates_long_arrays()
    test_clip_id_for_zero_pads_and_rounds()
    test_consensus_requires_min_agree_distinct_annotators()
    test_consensus_drops_singleton_boundary()
    test_consensus_same_annotator_marking_twice_does_not_count_as_two()
    test_consensus_empty_input_yields_no_boundaries()
    test_consensus_two_separate_clusters_both_survive()
    test_hop_labels_marks_hops_within_tolerance()
    test_hop_labels_length_matches_ceil_duration_over_hop()
    test_hop_labels_empty_consensus_is_all_zero()
    test_match_boundaries_perfect_match()
    test_match_boundaries_is_one_to_one_not_many_to_one()
    test_match_boundaries_outside_threshold_does_not_match()
    test_f1_at_threshold_perfect_predictions_score_one()
    test_f1_at_threshold_no_predictions_and_no_ground_truth_scores_one()
    test_f1_at_threshold_false_positive_on_empty_ground_truth_scores_zero()
    test_f1_at_threshold_is_relative_to_clip_duration()
    test_f1_curve_covers_the_standard_rel_dis_sweep()
    test_precision_recall_f1_separates_a_precision_bottleneck_from_a_recall_one()
    test_precision_recall_f1_agrees_with_f1_at_threshold()
    test_find_peak_indices_returns_hop_indices_not_seconds()
    test_pick_peaks_matches_find_peak_indices_converted_to_seconds()
    test_pick_peaks_finds_isolated_local_maxima()
    test_pick_peaks_suppresses_a_second_peak_too_close_to_a_stronger_one()
    test_pick_peaks_respects_min_score_floor()
    test_pick_peaks_empty_score_yields_no_peaks()
    test_unsupervised_score_peaks_at_a_clean_step_change()
    test_unsupervised_score_is_near_zero_on_a_constant_signal()
    test_windowed_features_shape_and_edge_padding()
    test_smooth_curve_flattens_a_single_spike()
    test_smooth_curve_no_op_for_zero_window()
    test_save_and_load_clip_npz_roundtrip()
    test_load_clip_npz_without_labels_omits_the_key()
    print("\nall cpd common tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
