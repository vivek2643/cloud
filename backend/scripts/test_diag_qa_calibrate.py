#!/usr/bin/env python3
"""Tests for color_phase1.plan.md Part 1d's _diag_qa_calibrate.py -- pure
threshold-fitting/validation math over synthetic (signal, label) pairs, no
DB / real corpus / real human labels required.

Run:  .venv/bin/python scripts/test_diag_qa_calibrate.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _diag_qa_calibrate as cal  # noqa: E402


def test_fit_upper_finds_separating_threshold():
    # good shots have low residual (<=5), bad shots have high (>=9) -- a
    # clean gap, so a zero-false-good cut must exist somewhere in (5, 9).
    pairs = [(2.0, True), (3.0, True), (4.5, True), (9.0, False), (10.0, False), (12.0, False)]
    fit = cal._fit_upper(pairs)
    assert fit["zero_false_good_achieved"] is True, fit
    assert 4.5 <= fit["pass_threshold"] < 9.0, fit
    assert fit["calibration_agreement"] == 1.0, fit
    print("ok  _fit_upper: finds a zero-false-good threshold in a clean gap")


def test_fit_upper_prefers_conservative_zero_fp_over_a_mislabeled_outlier():
    # one bad shot (1.0) sits BELOW all good shots (5,6,7) -- with a single
    # monotonic threshold, the only way to never let that bad point through
    # as "good" is to reject everything (the trivial cut is always
    # available, since a threshold below the global min always has zero
    # false-goods). The plan's own preference order picks that over letting
    # the outlier through, even though it tanks calibration agreement --
    # this degenerate case is exactly what held-out validation (not this
    # fitter) is supposed to catch and reject at the adopt step.
    pairs = [(1.0, False), (5.0, True), (6.0, True), (7.0, True)]
    fit = cal._fit_upper(pairs)
    assert fit["zero_false_good_achieved"] is True, fit
    assert fit["pass_threshold"] < 1.0, fit
    assert fit["calibration_agreement"] == 0.25, fit
    print("ok  _fit_upper: a below-all-goods bad outlier forces the conservative reject-all cut")


def test_fit_window_finds_inner_pass_band():
    # good shots cluster near 1.0 (a ratio of "as intended"); bad shots are
    # extreme over/under-saturated.
    pairs = [(0.9, True), (1.0, True), (1.1, True), (0.1, False), (3.0, False)]
    fit = cal._fit_window(pairs)
    lo, hi = fit["pass_window"]
    assert lo <= 0.9 and hi >= 1.1, fit
    assert fit["zero_false_good_achieved"] is True, fit
    assert fit["calibration_agreement"] == 1.0, fit
    print("ok  _fit_window: finds an inner window containing all the good ratios")


def test_agreement_is_fraction_correct():
    pairs = [(1.0, True), (2.0, True), (10.0, False)]
    assert cal._agreement(pairs, lambda s: s <= 5.0) == 1.0
    assert cal._agreement(pairs, lambda s: s <= 0.5) == 1.0 / 3.0
    print("ok  _agreement: fraction of pairs where the prediction matches the label")


def test_validate_reuses_the_fitted_cut_on_held_out_pairs():
    fit = {"shape": "upper", "pass_threshold": 5.0}
    held = [(2.0, True), (8.0, False), (9.0, True)]  # 2/3 correct
    assert abs(cal._validate(fit, held) - (2.0 / 3.0)) < 1e-9
    print("ok  _validate: applies the calibration-set cut (not refit) to held-out pairs")


def test_validate_returns_none_with_no_held_out_pairs():
    fit = {"shape": "upper", "pass_threshold": 5.0}
    assert cal._validate(fit, []) is None
    print("ok  _validate: no held-out labels -> None, not a fabricated 0%/100%")


def test_calibrate_metric_adopts_when_held_out_clears_bar():
    labels = {}
    scoreboard_by_uid = {}
    # 6 calibration-set pairs with a clean separating gap, 4 held-out pairs
    # all inside the fitted threshold's correct side -> 100% held-out.
    cal_signal_good = [1.0, 1.5, 2.0]
    cal_signal_bad = [8.0, 9.0, 10.0]
    held_signal_good = [1.2, 1.8]
    held_signal_bad = [8.5, 9.5]
    i = 0
    for signal in cal_signal_good + cal_signal_bad:
        uid = f"u{i}"
        i += 1
        is_good = signal in cal_signal_good
        labels[uid] = {"verdict": "good" if is_good else "bad", "split": "calibration"}
        scoreboard_by_uid[uid] = {"metrics": {"skin_perp_residual": {"verdict": "warn", "value": signal}}}
    for signal in held_signal_good + held_signal_bad:
        uid = f"u{i}"
        i += 1
        is_good = signal in held_signal_good
        labels[uid] = {"verdict": "good" if is_good else "bad", "split": "held_out"}
        scoreboard_by_uid[uid] = {"metrics": {"skin_perp_residual": {"verdict": "warn", "value": signal}}}

    r = cal.calibrate_metric("skin_perp_residual", "upper", cal._plain_value_signal, labels, scoreboard_by_uid)
    assert r["calibration_n"] == 6 and r["held_out_n"] == 4, r
    assert r["adopt"] is True, r
    print("ok  calibrate_metric: adopts when held-out agreement clears the 90%% bar")


def test_calibrate_metric_skips_below_min_calibration_samples():
    labels = {"u0": {"verdict": "good", "split": "calibration"}}
    scoreboard_by_uid = {"u0": {"metrics": {"skin_perp_residual": {"verdict": "pass", "value": 1.0}}}}
    r = cal.calibrate_metric("skin_perp_residual", "upper", cal._plain_value_signal, labels, scoreboard_by_uid)
    assert r["fit"] is None and r["adopt"] is False, r
    print("ok  calibrate_metric: too few calibration labels -> skipped, not force-fit")


def test_calibrate_metric_ignores_na_verdicts():
    labels = {"u0": {"verdict": "good", "split": "calibration"}}
    scoreboard_by_uid = {"u0": {"metrics": {"skin_perp_residual": {"verdict": "na", "value": None}}}}
    r = cal.calibrate_metric("skin_perp_residual", "upper", cal._plain_value_signal, labels, scoreboard_by_uid)
    assert r["calibration_n"] == 0, r
    print("ok  calibrate_metric: na-verdict metric readouts are excluded from calibration")


def main():
    test_fit_upper_finds_separating_threshold()
    test_fit_upper_prefers_conservative_zero_fp_over_a_mislabeled_outlier()
    test_fit_window_finds_inner_pass_band()
    test_agreement_is_fraction_correct()
    test_validate_reuses_the_fitted_cut_on_held_out_pairs()
    test_validate_returns_none_with_no_held_out_pairs()
    test_calibrate_metric_adopts_when_held_out_clears_bar()
    test_calibrate_metric_skips_below_min_calibration_samples()
    test_calibrate_metric_ignores_na_verdicts()
    print("\nall _diag_qa_calibrate tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
