"""
Pure unit tests for app.services.seam.curve.compute_seam_curve
(seam_function.plan.md §2/§8 step 2) -- no I/O, no DB, no video decode.
compute_seam_curve is a pure function of synthetic SeamSignals, exactly
like audio_features._detect_structure.

Run:  .venv/bin/python scripts/test_seam_curve.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.seam.curve import compute_seam_curve  # noqa: E402
from app.services.seam.params import GEST_COEFF, W_AUD_BASE, W_VIS  # noqa: E402
from app.services.seam.signals import SeamSignals  # noqa: E402


def _flat_signals(n=20, hop_ms=100, **overrides) -> SeamSignals:
    base = dict(
        file_id="f1", hop_ms=hop_ms, n=n,
        blur_n=[0.0] * n, act_n=[0.0] * n, cam_n=[0.0] * n, fd_n=[0.0] * n,
        beats_ms=[], onsets_ms=[], onset_strength=[], is_musical=False,
    )
    base.update(overrides)
    return SeamSignals(**base)


# --------------------------------------------------------------------------
# The plan's own §8 step 2 calibration cases
# --------------------------------------------------------------------------

def test_blurred_hop_scores_zero():
    sig = _flat_signals()
    sig.blur_n[5] = 1.0
    curve = compute_seam_curve(sig)
    assert curve.S[5] == 0.0, curve.S
    assert curve.g_sharp[5] == 0.0, curve.g_sharp
    print("ok  test_blurred_hop_scores_zero")


def test_still_and_on_beat_hop_scores_high():
    sig = _flat_signals(beats_ms=[1000], is_musical=True)
    curve = compute_seam_curve(sig)
    on_beat_i = 10  # 1000ms / 100ms hop
    assert curve.S[on_beat_i] == max(curve.S), curve.S
    # still=1 (cam/fd both quiet) + on-beat audio=1, salience=1 (musical):
    # S = 1 * 1 * (1.0*1 + 1.2*1) = 2.2
    assert abs(curve.S[on_beat_i] - (W_VIS + W_AUD_BASE)) < 1e-9, curve.S[on_beat_i]
    print("ok  test_still_and_on_beat_hop_scores_high")


def test_mid_gesture_is_attenuated_but_non_zero():
    sig = _flat_signals()
    sig.act_n[10] = 1.0  # full-strength gesture
    curve = compute_seam_curve(sig)
    assert curve.g_gest[10] == round(1.0 - GEST_COEFF, 6), curve.g_gest[10]
    assert 0.0 < curve.S[10] < curve.S[0], (curve.S[10], curve.S[0])
    print("ok  test_mid_gesture_is_attenuated_but_non_zero")


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def test_g_sharp_is_one_minus_blur():
    sig = _flat_signals()
    sig.blur_n = [0.0, 0.3, 0.7, 1.0] + [0.0] * 16
    curve = compute_seam_curve(sig)
    expected = [1.0, 0.7, 0.3, 0.0]
    assert all(abs(a - b) < 1e-9 for a, b in zip(curve.g_sharp[:4], expected)), curve.g_sharp[:4]
    print("ok  test_g_sharp_is_one_minus_blur")


def test_g_gest_floor_is_1_minus_gest_coeff():
    sig = _flat_signals()
    sig.act_n = [1.0] * sig.n  # maximal gesture everywhere
    curve = compute_seam_curve(sig)
    assert all(abs(g - (1.0 - GEST_COEFF)) < 1e-9 for g in curve.g_gest), curve.g_gest
    assert min(curve.g_gest) > 0.0, "gesture gate must never floor to exactly 0"
    print("ok  test_g_gest_floor_is_1_minus_gest_coeff")


def test_sharp_gate_disqualifies_regardless_of_gesture():
    """A blurred AND gesturing hop must still score exactly 0 -- g_sharp is
    the STRONG gate, multiplicative with everything else."""
    sig = _flat_signals()
    sig.blur_n[7] = 1.0
    sig.act_n[7] = 1.0
    curve = compute_seam_curve(sig)
    assert curve.S[7] == 0.0, curve.S[7]
    print("ok  test_sharp_gate_disqualifies_regardless_of_gesture")


# --------------------------------------------------------------------------
# Stillness attractor
# --------------------------------------------------------------------------

def test_still_requires_both_camera_and_frame_diff_quiet():
    """still(t) = 1 - max(cam_n, fd_n) -- EITHER channel being active must
    kill stillness (the plan's explicit `max`, not average)."""
    sig = _flat_signals()
    sig.cam_n[3] = 0.8   # camera active, frame-diff quiet
    sig.fd_n[6] = 0.8    # frame-diff active, camera quiet
    curve = compute_seam_curve(sig)
    assert abs(curve.still[3] - 0.2) < 1e-9, curve.still[3]
    assert abs(curve.still[6] - 0.2) < 1e-9, curve.still[6]
    assert curve.still[0] == 1.0, curve.still[0]
    print("ok  test_still_requires_both_camera_and_frame_diff_quiet")


def test_still_excludes_action_energy():
    """still(t) must NOT double-count action_energy -- that's already the
    g_gest gate, not part of the stillness attractor."""
    sig = _flat_signals()
    sig.act_n[4] = 1.0
    curve = compute_seam_curve(sig)
    assert curve.still[4] == 1.0, curve.still[4]
    print("ok  test_still_excludes_action_energy")


# --------------------------------------------------------------------------
# Audio attractor
# --------------------------------------------------------------------------

def test_audio_is_zero_with_no_events():
    sig = _flat_signals()
    curve = compute_seam_curve(sig)
    assert all(a == 0.0 for a in curve.audio), curve.audio
    print("ok  test_audio_is_zero_with_no_events")


def test_audio_decays_away_from_an_onset():
    sig = _flat_signals(onsets_ms=[500], onset_strength=[1.0])
    curve = compute_seam_curve(sig)
    on_i = 5
    assert curve.audio[on_i] == max(curve.audio), curve.audio
    assert curve.audio[on_i - 1] < curve.audio[on_i], curve.audio
    assert curve.audio[on_i + 1] < curve.audio[on_i], curve.audio
    assert curve.audio[on_i - 1] == curve.audio[on_i + 1], "symmetric Gaussian kernel"
    print("ok  test_audio_decays_away_from_an_onset")


def test_beats_and_onsets_both_feed_audio():
    sig = _flat_signals(beats_ms=[200], onsets_ms=[1500], onset_strength=[0.5], is_musical=True)
    curve = compute_seam_curve(sig)
    assert curve.audio[2] > 0.0, curve.audio[2]    # near the beat (200ms)
    assert curve.audio[15] > 0.0, curve.audio[15]  # near the onset (1500ms)
    print("ok  test_beats_and_onsets_both_feed_audio")


# --------------------------------------------------------------------------
# Salience / w_aud (§2's adaptive weighting, no footage-type branch)
# --------------------------------------------------------------------------

def test_musical_salience_is_flat_one():
    sig = _flat_signals(beats_ms=[0], is_musical=True)
    curve = compute_seam_curve(sig)
    assert all(abs(w - W_AUD_BASE) < 1e-9 for w in curve.w_aud), curve.w_aud
    print("ok  test_musical_salience_is_flat_one")


def test_non_musical_salience_follows_onset_strength_near_t():
    """Non-musical: salience(t) must be near 0 far from any onset (quiet
    ambient audio can't dominate) and high right at a strong onset."""
    sig = _flat_signals(onsets_ms=[1000], onset_strength=[1.0], is_musical=False)
    curve = compute_seam_curve(sig)
    assert curve.w_aud[10] > curve.w_aud[0], (curve.w_aud[10], curve.w_aud[0])
    assert curve.w_aud[0] < 0.01 * W_AUD_BASE, curve.w_aud[0]
    print("ok  test_non_musical_salience_follows_onset_strength_near_t")


def test_silent_non_musical_clip_reduces_to_pure_stillness():
    """No audio events + not musical -> audio(t)=0 everywhere -> S(t)
    reduces to the visual stillness term alone (§9's stated correct
    behavior: 'no audio -> stillness governs')."""
    sig = _flat_signals(is_musical=False)
    curve = compute_seam_curve(sig)
    assert all(a == 0.0 for a in curve.audio), curve.audio
    for i in range(sig.n):
        expected = curve.g_sharp[i] * curve.g_gest[i] * (W_VIS * curve.still[i])
        assert abs(curve.S[i] - expected) < 1e-9, (i, curve.S[i], expected)
    print("ok  test_silent_non_musical_clip_reduces_to_pure_stillness")


# --------------------------------------------------------------------------
# Structural invariants
# --------------------------------------------------------------------------

def test_grid_matches_hop_ms_and_n():
    sig = _flat_signals(n=7, hop_ms=250)
    curve = compute_seam_curve(sig)
    assert curve.t_ms == [0, 250, 500, 750, 1000, 1250, 1500], curve.t_ms
    assert len(curve.S) == 7
    print("ok  test_grid_matches_hop_ms_and_n")


def test_meta_carries_weights_and_provenance():
    sig = _flat_signals(is_musical=True)
    sig.meta = {"blur": "recomputed"}
    curve = compute_seam_curve(sig)
    assert curve.meta["w_vis"] == W_VIS
    assert curve.meta["w_aud_base"] == W_AUD_BASE
    assert curve.meta["gest_coeff"] == GEST_COEFF
    assert curve.meta["provenance"] == {"blur": "recomputed"}
    print("ok  test_meta_carries_weights_and_provenance")


def test_curve_never_produces_a_cut_decision():
    """No threshold, no peak list, no segmentation anywhere on SeamCurve --
    just the continuous per-term arrays (the plan's explicit non-goal)."""
    sig = _flat_signals()
    curve = compute_seam_curve(sig)
    field_names = set(vars(curve).keys())
    assert field_names == {"hop_ms", "t_ms", "S", "g_sharp", "g_gest", "still", "audio", "w_aud", "meta"}, field_names
    print("ok  test_curve_never_produces_a_cut_decision")


def main():
    test_blurred_hop_scores_zero()
    test_still_and_on_beat_hop_scores_high()
    test_mid_gesture_is_attenuated_but_non_zero()
    test_g_sharp_is_one_minus_blur()
    test_g_gest_floor_is_1_minus_gest_coeff()
    test_sharp_gate_disqualifies_regardless_of_gesture()
    test_still_requires_both_camera_and_frame_diff_quiet()
    test_still_excludes_action_energy()
    test_audio_is_zero_with_no_events()
    test_audio_decays_away_from_an_onset()
    test_beats_and_onsets_both_feed_audio()
    test_musical_salience_is_flat_one()
    test_non_musical_salience_follows_onset_strength_near_t()
    test_silent_non_musical_clip_reduces_to_pure_stillness()
    test_grid_matches_hop_ms_and_n()
    test_meta_carries_weights_and_provenance()
    test_curve_never_produces_a_cut_decision()
    print("\nall seam curve tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
