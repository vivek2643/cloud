#!/usr/bin/env python3
"""Tests for color_qa_harness.plan.md's qa/metrics.py -- pure array math, no
DB / ffmpeg / R2 (mirrors test_grade.py's "no DB" convention; this module is
explicitly built to be unit-testable without any of that).

Run:  .venv/bin/python scripts/test_qa_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from qa import metrics as m  # noqa: E402


def _solid(rgb: tuple, h: int = 12, w: int = 12) -> np.ndarray:
    return np.full((h, w, 3), rgb, dtype=np.float32)


# --------------------------------------------------------------------------
# 1. Exposure
# --------------------------------------------------------------------------

def test_luma01_matches_rec709_coefficients():
    frame = _solid((1.0, 0.0, 0.0))
    assert abs(float(m.luma01(frame).mean()) - 0.2126) < 1e-4
    frame = _solid((0.0, 1.0, 0.0))
    assert abs(float(m.luma01(frame).mean()) - 0.7152) < 1e-4
    print("ok  luma01: matches the Rec.709 coefficients (cdl.LUMA_R/G/B)")


def test_crushed_black_all_black_is_fail():
    r = m.exposure_metrics(_solid((0.0, 0.0, 0.0)))
    assert r["crushed_black_fraction"].value == 1.0
    assert r["crushed_black_fraction"].verdict == "fail"
    print("ok  exposure: an all-black graded frame is a crushed-black FAIL")


def test_clipped_highlight_all_white_is_fail():
    r = m.exposure_metrics(_solid((1.0, 1.0, 1.0)))
    assert r["clipped_highlight_fraction"].value == 1.0
    assert r["clipped_highlight_fraction"].verdict == "fail"
    print("ok  exposure: an all-white graded frame is a clipped-highlight FAIL")


def test_exposure_band_pass_at_target_mid_gray():
    r = m.exposure_metrics(_solid((0.42, 0.42, 0.42)))
    assert r["exposure_band"].verdict == "pass", r["exposure_band"]
    assert r["exposure_band"].extra["target"] == m.TARGET_MID_GRAY
    print("ok  exposure: exposure_band PASSes at TARGET_MID_GRAY")


def test_exposure_band_reports_raw_delta_direction():
    """The grade crushing MORE than raw must show as a positive delta -- the
    exact regression resolver.COMPOSITE_SHADOW_PROBE was added to guard."""
    raw = _solid((0.10, 0.10, 0.10))
    graded = _solid((0.0, 0.0, 0.0))
    r = m.exposure_metrics(graded, raw01=raw)
    assert r["crushed_black_fraction"].extra["delta"] > 0, r["crushed_black_fraction"]
    print("ok  exposure: crushed_black_fraction reports a positive RAW->GRADED delta when the grade crushes more")


# --------------------------------------------------------------------------
# 2. White balance / color cast
# --------------------------------------------------------------------------

def test_neutral_axis_deviation_zero_on_pure_gray():
    r = m.neutral_axis_deviation(_solid((0.5, 0.5, 0.5)))
    assert r.verdict == "pass" and abs(r.value) < 1e-6, r
    print("ok  cast: neutral_axis_deviation is ~0 on a pure neutral gray")


def test_neutral_axis_deviation_na_when_no_neutral_pixels():
    r = m.neutral_axis_deviation(_solid((0.9, 0.1, 0.1)))  # saturated red, no neutral mask
    assert r.verdict == "na", r
    print("ok  cast: neutral_axis_deviation is N/A when the neutral mask is too small")


def test_skin_perp_residual_na_outside_lightness_gate():
    dark = m.skin_perp_residual(_solid((0.02, 0.02, 0.02)), [0.0, 0.0, 1.0, 1.0])
    assert dark.verdict == "na", dark
    print("ok  skin: skin_perp_residual is N/A when L* is outside SKIN_L_MIN..MAX")


def test_skin_perp_residual_na_when_near_neutral():
    neutral = m.skin_perp_residual(_solid((0.5, 0.5, 0.5)), [0.0, 0.0, 1.0, 1.0])
    assert neutral.verdict == "na", neutral
    print("ok  skin: skin_perp_residual is N/A when the box reads near-neutral (chroma < SKIN_MIN_CHROMA)")


def test_skin_perp_residual_na_without_box():
    r = m.skin_perp_residual(_solid((0.7, 0.5, 0.4)), None)
    assert r.verdict == "na", r
    print("ok  skin: skin_perp_residual is N/A with no subject_box")


def test_crop_box_handles_degenerate_and_missing():
    frame = _solid((0.5, 0.5, 0.5))
    assert m.crop_box(frame, None) is None
    assert m.crop_box(frame, [0.5, 0.5, 0.0, 0.5]) is None   # zero width
    crop = m.crop_box(frame, [0.0, 0.0, 0.5, 0.5])
    assert crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0
    print("ok  crop_box: None for missing/degenerate box, a real crop for a valid one")


# --------------------------------------------------------------------------
# 3. Consistency
# --------------------------------------------------------------------------

def test_group_consistency_singleton_excluded():
    s = m.summarize_shot(_solid((0.4, 0.4, 0.4)))
    assert m.group_consistency_metrics([s]) == {}
    print("ok  consistency: a singleton group produces no metrics (nothing to match)")


def test_group_consistency_zero_std_on_identical_members():
    s = m.summarize_shot(_solid((0.4, 0.4, 0.4)))
    out = m.group_consistency_metrics([s, s, s])
    assert out["intra_group_luma_std"].value == 0.0
    assert out["intra_group_luma_std"].verdict == "pass"
    print("ok  consistency: identical members -> zero std, PASS")


def test_group_consistency_reports_raw_improved_flag():
    graded = [m.summarize_shot(_solid((0.40, 0.40, 0.40))), m.summarize_shot(_solid((0.41, 0.41, 0.41)))]
    raw = [m.summarize_shot(_solid((0.2, 0.2, 0.2))), m.summarize_shot(_solid((0.6, 0.6, 0.6)))]
    out = m.group_consistency_metrics(graded, raw=raw)
    assert out["intra_group_luma_std"].extra["improved"] is True
    print("ok  consistency: GRADED std tighter than RAW std reports improved=True")


def test_group_consistency_flags_when_grading_worsens_spread():
    graded = [m.summarize_shot(_solid((0.1, 0.1, 0.1))), m.summarize_shot(_solid((0.9, 0.9, 0.9)))]
    raw = [m.summarize_shot(_solid((0.4, 0.4, 0.4))), m.summarize_shot(_solid((0.44, 0.44, 0.44)))]
    out = m.group_consistency_metrics(graded, raw=raw)
    assert out["intra_group_luma_std"].extra["improved"] is False
    print("ok  consistency: GRADED std wider than RAW std reports improved=False (matching regressed)")


def test_group_subject_exposure_excludes_boxless_members():
    with_box = m.summarize_shot(_solid((0.3, 0.3, 0.3)), subject_box=[0.2, 0.2, 0.3, 0.3])
    no_box = m.summarize_shot(_solid((0.3, 0.3, 0.3)))
    assert no_box.subject_luma is None
    assert m.group_subject_exposure_metrics([with_box, no_box]) == {}   # only 1 usable subject_luma
    print("ok  subject-exposure: members with no box are excluded, not FAILed")


def test_group_subject_exposure_convergence_delta():
    a = m.summarize_shot(_solid((0.30, 0.30, 0.30)), subject_box=[0.0, 0.0, 1.0, 1.0])
    b = m.summarize_shot(_solid((0.34, 0.34, 0.34)), subject_box=[0.0, 0.0, 1.0, 1.0])
    ra = m.summarize_shot(_solid((0.15, 0.15, 0.15)), subject_box=[0.0, 0.0, 1.0, 1.0])
    rb = m.summarize_shot(_solid((0.55, 0.55, 0.55)), subject_box=[0.0, 0.0, 1.0, 1.0])
    out = m.group_subject_exposure_metrics([a, b], raw=[ra, rb])
    assert out["intra_group_subject_luma_std"].extra["convergence_delta"] > 0
    print("ok  subject-exposure: leveling that tightens subject_luma spread reports a positive convergence_delta")


# --------------------------------------------------------------------------
# 4. Over-processing
# --------------------------------------------------------------------------

def test_saturation_band_fail_on_dead_flat():
    r = m.saturation_band(_solid((0.5, 0.5, 0.5)))
    assert r.verdict == "fail", r   # chroma 0, below SATURATION_WARN's low end
    print("ok  over-processing: a dead-flat (zero chroma) frame FAILs saturation_band")


def test_saturation_band_chroma_increase_ratio_fail():
    raw = _solid((0.45, 0.40, 0.35))          # a mild, plausible warm cast
    graded = _solid((0.95, 0.05, 0.05))       # wildly oversaturated
    r = m.saturation_band(graded, raw01=raw)
    assert r.extra["chroma_increase_ratio"] > m.CHROMA_INCREASE_FAIL_RATIO
    assert r.verdict == "fail"
    print("ok  over-processing: chroma_increase_ratio > 2.0 forces a FAIL even if the band alone wouldn't")


def test_banding_score_flat_frame_is_pass():
    r = m.banding_score(_solid((0.4, 0.4, 0.4)))
    assert r.verdict == "pass" and r.value == 0.0
    print("ok  over-processing: a perfectly flat frame has zero banding")


def test_banding_score_sparse_levels_is_worse_than_full_ramp():
    h, w = 4, 256
    # A full 0..1 ramp across every column -- every 8-bit luma level occupied.
    ramp = np.zeros((h, w, 3), dtype=np.float32)
    for x in range(w):
        ramp[:, x, :] = x / (w - 1)
    full_ramp = m.banding_score(ramp)

    # The SAME ramp but posterized to only 8 distinct levels -- large empty
    # gaps between occupied bins.
    sparse = np.zeros((h, w, 3), dtype=np.float32)
    for x in range(w):
        level = round((x / (w - 1)) * 7) / 7.0
        sparse[:, x, :] = level
    posterized = m.banding_score(sparse)

    assert posterized.value > full_ramp.value, (posterized.value, full_ramp.value)
    print(f"ok  over-processing: banding_score is higher for a posterized ramp "
         f"({posterized.value:.3f}) than a full one ({full_ramp.value:.3f})")


# --------------------------------------------------------------------------
# 5. Look fidelity
# --------------------------------------------------------------------------

def test_look_fidelity_identical_shift_is_cosine_one():
    raw = _solid((0.4, 0.4, 0.4))
    graded = _solid((0.5, 0.42, 0.35))
    r = m.look_fidelity_metric(graded, raw, look_only01=graded)   # same shift as itself
    assert r.value is not None and r.value > 0.999, r
    assert r.verdict == "pass"
    print("ok  look_fidelity: a look-only shift identical to the graded shift -> cosine ~1.0, PASS")


def test_look_fidelity_opposite_shift_is_fail():
    raw = _solid((0.4, 0.4, 0.4))
    graded = _solid((0.6, 0.4, 0.2))          # warms up
    look_only = _solid((0.2, 0.4, 0.6))       # cools down -- opposite direction
    r = m.look_fidelity_metric(graded, raw, look_only01=look_only)
    assert r.value is not None and r.value < 0, r
    assert r.verdict == "fail"
    print("ok  look_fidelity: an opposite-direction look shift -> negative cosine, FAIL")


def test_look_fidelity_na_when_no_shift():
    raw = _solid((0.4, 0.4, 0.4))
    r = m.look_fidelity_metric(raw, raw, look_only01=raw)   # zero shift both sides
    assert r.verdict == "na", r
    print("ok  look_fidelity: N/A when both shift vectors are ~zero")


# --------------------------------------------------------------------------
# worst_verdict
# --------------------------------------------------------------------------

def test_worst_verdict_ranks_fail_over_warn_over_pass():
    assert m.worst_verdict(["pass"]) == "pass"
    assert m.worst_verdict(["pass", "warn"]) == "warn"
    assert m.worst_verdict(["pass", "warn", "fail"]) == "fail"
    print("ok  worst_verdict: fail > warn > pass")


def test_worst_verdict_ignores_na():
    assert m.worst_verdict(["na", "na"]) == "na"
    assert m.worst_verdict(["na", "pass"]) == "pass"
    print("ok  worst_verdict: N/A never masks or counts as a failure")


def main():
    test_luma01_matches_rec709_coefficients()
    test_crushed_black_all_black_is_fail()
    test_clipped_highlight_all_white_is_fail()
    test_exposure_band_pass_at_target_mid_gray()
    test_exposure_band_reports_raw_delta_direction()
    test_neutral_axis_deviation_zero_on_pure_gray()
    test_neutral_axis_deviation_na_when_no_neutral_pixels()
    test_skin_perp_residual_na_outside_lightness_gate()
    test_skin_perp_residual_na_when_near_neutral()
    test_skin_perp_residual_na_without_box()
    test_crop_box_handles_degenerate_and_missing()
    test_group_consistency_singleton_excluded()
    test_group_consistency_zero_std_on_identical_members()
    test_group_consistency_reports_raw_improved_flag()
    test_group_consistency_flags_when_grading_worsens_spread()
    test_group_subject_exposure_excludes_boxless_members()
    test_group_subject_exposure_convergence_delta()
    test_saturation_band_fail_on_dead_flat()
    test_saturation_band_chroma_increase_ratio_fail()
    test_banding_score_flat_frame_is_pass()
    test_banding_score_sparse_levels_is_worse_than_full_ramp()
    test_look_fidelity_identical_shift_is_cosine_one()
    test_look_fidelity_opposite_shift_is_fail()
    test_look_fidelity_na_when_no_shift()
    test_worst_verdict_ranks_fail_over_warn_over_pass()
    test_worst_verdict_ignores_na()
    print("\nall qa metrics tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
