"""
Pure unit tests for the signal-processing helpers in
app.services.seam.signals that need no I/O -- camera-vector magnitude,
grid alignment, speech-span membership. The DB/proxy-recompute paths
(_persisted_motion, _recompute_motion_bundle, _extract_wav,
_recompute_onsets) need a live Postgres connection / ffmpeg / librosa and
are exercised via the cutviz overlay and the compute_all_seam.py driver
instead, matching this codebase's own convention of keeping the fast unit
suite decode/DB-free (see test_audio_features.py, test_cpd_common.py).

Run:  .venv/bin/python scripts/test_seam_signals.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.seam.signals import (  # noqa: E402
    _align_to_grid, _camera_magnitude_n, _in_speech,
)


# --------------------------------------------------------------------------
# _camera_magnitude_n
# --------------------------------------------------------------------------

def test_camera_magnitude_is_euclidean_norm_before_normalizing():
    # A single hop with dx=3, dy=4, zoom=0 -> raw magnitude 5, the clip's own
    # max -> normalizes to 1.0 (normalize_pctl clamps at the reference pctl).
    out = _camera_magnitude_n([3.0], [4.0], [0.0])
    assert out == [1.0], out
    print("ok  test_camera_magnitude_is_euclidean_norm_before_normalizing")


def test_camera_magnitude_handles_ragged_input_lengths():
    # dy/zoom shorter than dx -- missing entries treated as 0, never a crash.
    out = _camera_magnitude_n([1.0, 1.0, 1.0], [1.0], [])
    assert len(out) == 3, out
    print("ok  test_camera_magnitude_handles_ragged_input_lengths")


def test_camera_magnitude_all_zero_is_all_zero():
    out = _camera_magnitude_n([0.0] * 5, [0.0] * 5, [0.0] * 5)
    assert out == [0.0] * 5, out
    print("ok  test_camera_magnitude_all_zero_is_all_zero")


def test_camera_magnitude_relative_ordering_preserved():
    # A hop with bigger combined displacement must normalize to a bigger value.
    out = _camera_magnitude_n([0.01, 0.05, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert out[0] < out[1] < out[2], out
    print("ok  test_camera_magnitude_relative_ordering_preserved")


# --------------------------------------------------------------------------
# _align_to_grid
# --------------------------------------------------------------------------

def test_align_to_grid_truncates_long_arrays():
    assert _align_to_grid([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 2.0]
    print("ok  test_align_to_grid_truncates_long_arrays")


def test_align_to_grid_zero_pads_short_arrays():
    assert _align_to_grid([1.0, 2.0], 4) == [1.0, 2.0, 0.0, 0.0]
    print("ok  test_align_to_grid_zero_pads_short_arrays")


def test_align_to_grid_exact_length_is_unchanged():
    assert _align_to_grid([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]
    print("ok  test_align_to_grid_exact_length_is_unchanged")


# --------------------------------------------------------------------------
# _in_speech
# --------------------------------------------------------------------------

def test_in_speech_true_inside_a_span():
    assert _in_speech(1500, [(1000, 2000)]) is True
    print("ok  test_in_speech_true_inside_a_span")


def test_in_speech_false_outside_every_span():
    assert _in_speech(2500, [(1000, 2000), (3000, 4000)]) is False
    print("ok  test_in_speech_false_outside_every_span")


def test_in_speech_inclusive_at_span_boundaries():
    assert _in_speech(1000, [(1000, 2000)]) is True
    assert _in_speech(2000, [(1000, 2000)]) is True
    print("ok  test_in_speech_inclusive_at_span_boundaries")


def test_in_speech_false_with_no_spans():
    assert _in_speech(500, []) is False
    print("ok  test_in_speech_false_with_no_spans")


def main():
    test_camera_magnitude_is_euclidean_norm_before_normalizing()
    test_camera_magnitude_handles_ragged_input_lengths()
    test_camera_magnitude_all_zero_is_all_zero()
    test_camera_magnitude_relative_ordering_preserved()
    test_align_to_grid_truncates_long_arrays()
    test_align_to_grid_zero_pads_short_arrays()
    test_align_to_grid_exact_length_is_unchanged()
    test_in_speech_true_inside_a_span()
    test_in_speech_false_outside_every_span()
    test_in_speech_inclusive_at_span_boundaries()
    test_in_speech_false_with_no_spans()
    print("\nall seam signals tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
