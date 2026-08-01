"""
Pure unit tests for app.services.vcut.speech.delivery (speech_cuts_pipeline
.plan.md section 10, part 1: per-take metrics + group normalization). No I/O.

Run:  .venv/bin/python scripts/test_speech_delivery.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.delivery import (  # noqa: E402
    TakeMetrics, _pace_score, compute_take_metrics, group_delivery_scores, is_voiced_beat,
)
from app.services.vcut.speech.inputs import Word  # noqa: E402
from app.services.vcut.speech.params import HESITATION_GAP_MS, PACE_HI, PACE_LO  # noqa: E402


def _word(idx, s, e, text="w", is_filler=False, probability=None):
    return Word(idx=idx, start_ms=s, end_ms=e, text=text, is_filler=is_filler, probability=probability)


# --------------------------------------------------------------------------
# compute_take_metrics
# --------------------------------------------------------------------------

def test_energy_and_dynamics_from_rms_slice():
    words = [_word(0, 0, 500)]
    rms_db = [-40, -40, -10, -10, -40]  # hop 200ms -> covers 0..1000ms
    m = compute_take_metrics(words, 0, 1000, rms_db, rms_hop_ms=200)
    assert m.energy == -40, m.energy  # median of [-40,-40,-10,-10,-40]
    assert m.dynamics > 0
    print("ok  test_energy_and_dynamics_from_rms_slice")


def test_pace_counts_only_non_filler_words_in_span():
    words = [_word(0, 0, 200), _word(1, 200, 400, is_filler=True), _word(2, 400, 600)]
    m = compute_take_metrics(words, 0, 1000, rms_db=[], rms_hop_ms=0)
    # 2 non-filler words over 1.0s
    assert abs(m.pace_wps - 2.0) < 1e-9, m.pace_wps
    print("ok  test_pace_counts_only_non_filler_words_in_span")


def test_hesitation_only_counts_gaps_above_the_threshold():
    big_gap = HESITATION_GAP_MS + 100
    words = [_word(0, 0, 200), _word(1, 200 + big_gap, 200 + big_gap + 100)]
    m = compute_take_metrics(words, 0, 5000, rms_db=[], rms_hop_ms=0)
    assert m.hesitation_ms == big_gap, m.hesitation_ms
    small_gap_words = [_word(0, 0, 200), _word(1, 200 + 50, 400)]
    m2 = compute_take_metrics(small_gap_words, 0, 5000, rms_db=[], rms_hop_ms=0)
    assert m2.hesitation_ms == 0.0
    print("ok  test_hesitation_only_counts_gaps_above_the_threshold")


def test_words_outside_the_span_are_excluded():
    words = [_word(0, -500, -100), _word(1, 0, 200), _word(2, 5000, 5200)]
    m = compute_take_metrics(words, 0, 1000, rms_db=[], rms_hop_ms=0)
    assert abs(m.pace_wps - 1.0) < 1e-9, m.pace_wps  # only word 1 counted
    print("ok  test_words_outside_the_span_are_excluded")


# --------------------------------------------------------------------------
# _pace_score -- fixed band, not group-normalized
# --------------------------------------------------------------------------

def test_pace_score_is_one_inside_the_band():
    assert _pace_score((PACE_LO + PACE_HI) / 2) == 1.0
    assert _pace_score(PACE_LO) == 1.0
    assert _pace_score(PACE_HI) == 1.0
    print("ok  test_pace_score_is_one_inside_the_band")


def test_pace_score_falls_off_outside_the_band():
    assert 0.0 < _pace_score(PACE_LO * 0.5) < 1.0
    assert 0.0 < _pace_score(PACE_HI * 1.5) < 1.0
    assert _pace_score(0.0) == 0.0
    print("ok  test_pace_score_falls_off_outside_the_band")


# --------------------------------------------------------------------------
# group_delivery_scores -- group normalization, neutral fallbacks
# --------------------------------------------------------------------------

def test_group_normalization_favors_the_relatively_better_take():
    quiet = TakeMetrics(energy=-40, dynamics=1, pace_wps=2.5, hesitation_ms=0)
    loud = TakeMetrics(energy=-10, dynamics=5, pace_wps=2.5, hesitation_ms=0)
    scores = group_delivery_scores([quiet, loud])
    assert scores[1] > scores[0], scores
    print("ok  test_group_normalization_favors_the_relatively_better_take")


def test_degenerate_group_all_equal_scores_everyone_the_same():
    a = TakeMetrics(energy=-20, dynamics=2, pace_wps=2.5, hesitation_ms=100)
    b = TakeMetrics(energy=-20, dynamics=2, pace_wps=2.5, hesitation_ms=100)
    scores = group_delivery_scores([a, b])
    assert abs(scores[0] - scores[1]) < 1e-9, scores
    print("ok  test_degenerate_group_all_equal_scores_everyone_the_same")


def test_visual_delivery_absent_for_whole_group_does_not_affect_ranking():
    a = TakeMetrics(energy=-15, dynamics=2, pace_wps=2.5, hesitation_ms=0)
    b = TakeMetrics(energy=-30, dynamics=2, pace_wps=2.5, hesitation_ms=0)
    scores = group_delivery_scores([a, b])
    assert scores[0] > scores[1]  # driven by energy alone, visual contributes 0 to both
    print("ok  test_visual_delivery_absent_for_whole_group_does_not_affect_ranking")


def test_empty_group_returns_empty():
    assert group_delivery_scores([]) == []
    print("ok  test_empty_group_returns_empty")


# --------------------------------------------------------------------------
# is_voiced_beat -- speech_noise_gate.plan.md section 2: the gate's DECISION
# MATH is unchanged (only inputs.py's floor calibration improved) -- regression
# guard that behavior here still matches the documented contract exactly.
# --------------------------------------------------------------------------

def test_is_voiced_beat_a_loud_beat_clears_the_gate():
    # spread = -10 - (-50) = 40 >= MIN_VOICED_SPREAD_DB; threshold = -50 + 0.30*40 = -38
    assert is_voiced_beat(-15.0, voiced_ref_db=-10.0, floor_ref_db=-50.0) is True
    print("ok  test_is_voiced_beat_a_loud_beat_clears_the_gate")


def test_is_voiced_beat_a_floor_level_beat_is_dropped():
    assert is_voiced_beat(-48.0, voiced_ref_db=-10.0, floor_ref_db=-50.0) is False
    print("ok  test_is_voiced_beat_a_floor_level_beat_is_dropped")


def test_is_voiced_beat_fails_open_when_the_spread_is_too_small_to_discriminate():
    # spread = -20 - (-25) = 5 < MIN_VOICED_SPREAD_DB (8) -> gate disabled, always True
    assert is_voiced_beat(-100.0, voiced_ref_db=-20.0, floor_ref_db=-25.0) is True
    print("ok  test_is_voiced_beat_fails_open_when_the_spread_is_too_small_to_discriminate")


def main():
    test_energy_and_dynamics_from_rms_slice()
    test_pace_counts_only_non_filler_words_in_span()
    test_hesitation_only_counts_gaps_above_the_threshold()
    test_words_outside_the_span_are_excluded()
    test_pace_score_is_one_inside_the_band()
    test_pace_score_falls_off_outside_the_band()
    test_group_normalization_favors_the_relatively_better_take()
    test_degenerate_group_all_equal_scores_everyone_the_same()
    test_visual_delivery_absent_for_whole_group_does_not_affect_ranking()
    test_empty_group_returns_empty()
    test_is_voiced_beat_a_loud_beat_clears_the_gate()
    test_is_voiced_beat_a_floor_level_beat_is_dropped()
    test_is_voiced_beat_fails_open_when_the_spread_is_too_small_to_discriminate()
    print("\nall speech delivery tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
