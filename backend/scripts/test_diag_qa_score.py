#!/usr/bin/env python3
"""Tests for color_phase1.plan.md's tiered-parity-bar rollup in
_diag_qa_score.py -- pure dict math over fake MetricResult.to_dict()-shaped
records, no DB / ffmpeg / R2 / actual frame loading (score_shot itself needs
real sampled frames and is exercised by the harness scripts, not here).

Run:  .venv/bin/python scripts/test_diag_qa_score.py
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

import _diag_qa_score as dqs  # noqa: E402


def _shot(metrics: dict) -> dict:
    return {"project_id": "p", "project_label": "P", "thread_id": "t", "shot_key": "s", "metrics": metrics}


def _group(metrics: dict) -> dict:
    return {"thread_id": "t", "group_id": "g", "project_id": "p", "project_label": "P", "metrics": metrics}


def _md(verdict: str, **extra) -> dict:
    return {"name": "x", "value": 0.0, "verdict": verdict, **extra}


# --------------------------------------------------------------------------
# Tier A
# --------------------------------------------------------------------------

def test_tier_a_all_pass_is_met():
    shots = [_shot({"crushed_black_fraction": _md("pass"), "exposure_band": _md("pass")})]
    r = dqs._tier_a_rollup(shots)
    assert r["met"] is True and r["fails"] == 0 and r["pass_pct"] == 100.0, r
    print("ok  tier_a: all-pass shots clear the zero-tolerance bar")


def test_tier_a_fail_without_raw_delta_counts_against_bar():
    shots = [_shot({"crushed_black_fraction": _md("fail")})]
    r = dqs._tier_a_rollup(shots)
    assert r["met"] is False and r["fails"] == 1, r
    print("ok  tier_a: a fail with no raw delta counts as a real failure")


def test_tier_a_exempts_grade_did_not_worsen_source_defect():
    shots = [_shot({"crushed_black_fraction": _md("fail", delta=-0.1)})]
    r = dqs._tier_a_rollup(shots)
    assert r["met"] is True and r["fails"] == 0 and r["exempted_source_defects"] == 1, r
    print("ok  tier_a: a fail whose grade didn't worsen it (delta<=0) is exempt")


def test_tier_a_does_not_exempt_grade_introduced_worsening():
    shots = [_shot({"clipped_highlight_fraction": _md("fail", delta=0.2)})]
    r = dqs._tier_a_rollup(shots)
    assert r["met"] is False and r["fails"] == 1 and r["exempted_source_defects"] == 0, r
    print("ok  tier_a: a fail the grade actually worsened (delta>0) still counts")


def test_tier_a_warn_never_counts_against_zero_tolerance():
    # This is the entire point of 1b's gross-vs-typical exposure split: a
    # deliberately dark/bright shot WARNs, never trips Tier A.
    shots = [_shot({"exposure_band": _md("warn")})]
    r = dqs._tier_a_rollup(shots)
    assert r["met"] is True and r["fails"] == 0, r
    print("ok  tier_a: WARN (outside-typical, not gross) never fails the bar")


def test_tier_a_na_excluded_from_denominator():
    shots = [_shot({"exposure_band": _md("na")})]
    r = dqs._tier_a_rollup(shots)
    assert r["scored"] == 0 and r["met"] is True, r
    print("ok  tier_a: na (e.g. synthetic content) is excluded from scoring")


def test_tier_a_ignores_non_tier_a_metrics():
    shots = [_shot({"saturation_band": _md("fail")})]
    r = dqs._tier_a_rollup(shots)
    assert r["scored"] == 0, r
    print("ok  tier_a: a Tier-B-only metric doesn't leak into the Tier-A count")


# --------------------------------------------------------------------------
# Tier B
# --------------------------------------------------------------------------

def test_tier_b_strict_pass_only_counts_as_pass():
    shots = [_shot({"saturation_band": _md("warn")})]
    r = dqs._tier_b_rollup(shots, [])
    assert r["pass"] == 0 and r["warn"] == 1 and r["pass_pct"] == 0.0, r
    print("ok  tier_b: WARN does not count toward the >=95%% pass bar")


def test_tier_b_met_at_exactly_95_percent():
    shots = [_shot({"saturation_band": _md("pass")}) for _ in range(19)]
    shots.append(_shot({"saturation_band": _md("fail")}))
    r = dqs._tier_b_rollup(shots, [])
    assert r["pass_pct"] == 95.0 and r["met"] is True, r
    print("ok  tier_b: exactly 95%% pass meets the bar")


def test_tier_b_banding_fail_is_downgraded_to_warn():
    shots = [_shot({"banding_score": _md("fail")})]
    r = dqs._tier_b_rollup(shots, [])
    assert r["fail"] == 0 and r["warn"] == 1, r
    print("ok  tier_b: banding_score can never count as a hard fail (contact-sheet judged)")


def test_tier_b_na_excluded_from_denominator():
    shots = [_shot({"skin_perp_residual": _md("na")})]
    r = dqs._tier_b_rollup(shots, [])
    assert r["scored"] == 0 and r["pass_pct"] == 100.0, r
    print("ok  tier_b: na is excluded from the Tier-B denominator")


def test_tier_b_includes_group_level_metrics():
    groups = [_group({"intra_group_luma_std": _md("pass")})]
    r = dqs._tier_b_rollup([], groups)
    assert r["scored"] == 1 and r["pass"] == 1, r
    print("ok  tier_b: group-level consistency metrics are counted too")


# --------------------------------------------------------------------------
# Tier C
# --------------------------------------------------------------------------

def test_group_clears_tier_c_when_luma_and_chroma_improved():
    metrics = {
        "intra_group_luma_std": _md("pass", value=0.01, raw=0.05, improved=True),
        "intra_group_chroma_std": _md("pass", value=1.0, raw=3.0, improved=True),
    }
    assert dqs._group_clears_tier_c(metrics) is True
    print("ok  tier_c: a group with improved luma+chroma clears")


def test_group_clears_tier_c_false_when_luma_worsened():
    metrics = {"intra_group_luma_std": _md("fail", value=0.08, raw=0.05, improved=False)}
    assert dqs._group_clears_tier_c(metrics) is False
    print("ok  tier_c: a group where the grade worsened luma spread does not clear")


def test_group_clears_tier_c_true_on_an_already_perfect_tie():
    # RAW members were already identical (raw_std == 0) -- nothing left to
    # improve. metrics.py's own `improved` field is a STRICT `<` and would
    # read this as False (0.0 is not < 0.0), but a tie is not "worse than
    # raw" -- the actual Tier C bar -- so it must still clear.
    metrics = {"intra_group_luma_std": _md("pass", value=0.0, raw=0.0, improved=False)}
    assert dqs._group_clears_tier_c(metrics) is True
    print("ok  tier_c: an already-perfect raw tie (0.0 == 0.0) clears, not a false failure")


def test_group_clears_tier_c_false_on_negative_convergence_delta():
    metrics = {"intra_group_subject_luma_std": _md("pass", convergence_delta=-0.02)}
    assert dqs._group_clears_tier_c(metrics) is False
    print("ok  tier_c: negative subject-luma convergence_delta does not clear")


def test_group_clears_tier_c_none_when_no_applicable_metric():
    # No raw comparison could be computed (e.g. single-frame group, or the
    # metric came back na) -- excluded from the denominator, not penalized.
    metrics = {"intra_group_luma_std": _md("na")}
    assert dqs._group_clears_tier_c(metrics) is None
    print("ok  tier_c: no applicable metric -> excluded (None), not counted either way")


def test_tier_c_rollup_100_percent_when_all_groups_clear():
    groups = [
        _group({"intra_group_luma_std": _md("pass", value=0.01, raw=0.05, improved=True)}),
        _group({"intra_group_subject_luma_std": _md("pass", convergence_delta=0.01)}),
    ]
    r = dqs._tier_c_rollup(groups)
    assert r["met"] is True and r["pass_pct"] == 100.0, r
    print("ok  tier_c: all applicable groups improving -> bar met")


def test_tier_c_rollup_not_met_when_one_group_regresses():
    groups = [
        _group({"intra_group_luma_std": _md("pass", value=0.01, raw=0.05, improved=True)}),
        _group({"intra_group_luma_std": _md("fail", value=0.08, raw=0.05, improved=False)}),
    ]
    r = dqs._tier_c_rollup(groups)
    assert r["met"] is False and r["passing_groups"] == 1 and r["scored_groups"] == 2, r
    print("ok  tier_c: any regressing group breaks the 100%% bar")


def test_tier_c_rollup_vacuously_met_with_no_applicable_groups():
    r = dqs._tier_c_rollup([_group({"intra_group_luma_std": _md("na")})])
    assert r["scored_groups"] == 0 and r["met"] is True, r
    print("ok  tier_c: no scoreable groups is vacuously met, not a false failure")


# --------------------------------------------------------------------------
# Tier annotation
# --------------------------------------------------------------------------

def test_annotate_tiers_marks_multi_tier_group_metric_with_both():
    shots = [_shot({"saturation_band": _md("pass")})]
    groups = [_group({"intra_group_luma_std": _md("pass", improved=True)})]
    dqs._annotate_tiers(shots, groups)
    assert shots[0]["metrics"]["saturation_band"]["tiers"] == ["B"]
    assert groups[0]["metrics"]["intra_group_luma_std"]["tiers"] == ["B", "C"]
    print("ok  _annotate_tiers: intra_group_luma_std is tagged both B and C")


def main():
    test_tier_a_all_pass_is_met()
    test_tier_a_fail_without_raw_delta_counts_against_bar()
    test_tier_a_exempts_grade_did_not_worsen_source_defect()
    test_tier_a_does_not_exempt_grade_introduced_worsening()
    test_tier_a_warn_never_counts_against_zero_tolerance()
    test_tier_a_na_excluded_from_denominator()
    test_tier_a_ignores_non_tier_a_metrics()
    test_tier_b_strict_pass_only_counts_as_pass()
    test_tier_b_met_at_exactly_95_percent()
    test_tier_b_banding_fail_is_downgraded_to_warn()
    test_tier_b_na_excluded_from_denominator()
    test_tier_b_includes_group_level_metrics()
    test_group_clears_tier_c_when_luma_and_chroma_improved()
    test_group_clears_tier_c_false_when_luma_worsened()
    test_group_clears_tier_c_true_on_an_already_perfect_tie()
    test_group_clears_tier_c_false_on_negative_convergence_delta()
    test_group_clears_tier_c_none_when_no_applicable_metric()
    test_tier_c_rollup_100_percent_when_all_groups_clear()
    test_tier_c_rollup_not_met_when_one_group_regresses()
    test_tier_c_rollup_vacuously_met_with_no_applicable_groups()
    test_annotate_tiers_marks_multi_tier_group_metric_with_both()
    print("\nall _diag_qa_score tier tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
