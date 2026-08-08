"""
Pure unit tests for app.services.vcut.speech.inputs -- specifically
_energy_refs's silence-calibrated floor (speech_noise_gate.plan.md
Improvement B). No I/O -- the DB-touching loaders (_words_for_file etc.)
are exercised via the live speech-channel smoke run instead.

Run:  .venv/bin/python scripts/test_speech_inputs.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.inputs import _energy_refs, _median  # noqa: E402
from app.services.vcut.speech.params import SILENCE_FLOOR_MIN_FRAMES  # noqa: E402


def test_median_odd_and_even_and_empty():
    assert _median([3, 1, 2]) == 2
    assert _median([1, 2, 3, 4]) == 2.5
    assert _median([]) == 0.0
    print("ok  test_median_odd_and_even_and_empty")


# 10 "silence" frames (indices 0-9, hop=100ms -> covers ms [0,1000)) with a
# real spread, followed by 20 "loud" frames all at -10dB. Chosen so the
# silence-frame MEDIAN (-51.0) and the legacy 15th-percentile-of-everything
# fallback (-52.0) are DIFFERENT values -- a test that passed by coincidence
# either way would be worthless.
_SILENCE_VALS = [-60.0, -58.0, -56.0, -54.0, -52.0, -50.0, -48.0, -46.0, -44.0, -42.0]
_LOUD_VALS = [-10.0] * 20
_HOP_MS = 100


def test_energy_refs_uses_silence_median_as_floor_when_enough_silence():
    rms_db = list(_SILENCE_VALS) + list(_LOUD_VALS)
    assert len(_SILENCE_VALS) >= SILENCE_FLOOR_MIN_FRAMES
    silences = [(0, 999)]  # frames 0..9 inclusive at hop=100 -> exactly 10 frames
    voiced, floor = _energy_refs(rms_db, _HOP_MS, silences)
    assert floor == -51.0, floor  # median of _SILENCE_VALS, NOT the 15th pct (-52.0)
    assert voiced == -10.0, voiced  # 75th pct of all 30 values lands in the loud block
    print("ok  test_energy_refs_uses_silence_median_as_floor_when_enough_silence")


def test_energy_refs_falls_back_to_percentile_with_too_little_silence():
    rms_db = list(_SILENCE_VALS) + list(_LOUD_VALS)
    silences = [(0, 499)]  # frames 0..4 -> only 5 silence frames, < SILENCE_FLOOR_MIN_FRAMES
    voiced, floor = _energy_refs(rms_db, _HOP_MS, silences)
    assert floor == -52.0, floor  # legacy 15th-pct-of-everything fallback
    assert voiced == -10.0, voiced
    print("ok  test_energy_refs_falls_back_to_percentile_with_too_little_silence")


def test_energy_refs_no_silences_at_all_falls_back_to_percentile():
    rms_db = list(_SILENCE_VALS) + list(_LOUD_VALS)
    voiced, floor = _energy_refs(rms_db, _HOP_MS, [])
    assert floor == -52.0, floor
    print("ok  test_energy_refs_no_silences_at_all_falls_back_to_percentile")


def test_energy_refs_zero_hop_ms_falls_back_to_percentile():
    rms_db = list(_SILENCE_VALS) + list(_LOUD_VALS)
    voiced, floor = _energy_refs(rms_db, 0, [(0, 999)])
    assert floor == -52.0, floor
    print("ok  test_energy_refs_zero_hop_ms_falls_back_to_percentile")


def test_energy_refs_too_few_samples_disables_the_gate():
    rms_db = [-10.0] * 10  # < 20 valid samples
    assert _energy_refs(rms_db, _HOP_MS, [(0, 999)]) == (0.0, 0.0)
    print("ok  test_energy_refs_too_few_samples_disables_the_gate")


def main():
    test_median_odd_and_even_and_empty()
    test_energy_refs_uses_silence_median_as_floor_when_enough_silence()
    test_energy_refs_falls_back_to_percentile_with_too_little_silence()
    test_energy_refs_no_silences_at_all_falls_back_to_percentile()
    test_energy_refs_zero_hop_ms_falls_back_to_percentile()
    test_energy_refs_too_few_samples_disables_the_gate()
    print("\nall speech inputs tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
