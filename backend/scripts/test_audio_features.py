#!/usr/bin/env python3
"""Tests for audio_features._detect_structure (audio_and_audit.plan.md Phase 4):
the pure-numpy coarse musical-structure detector (sections + drop_ms), tested
directly against synthetic arrays -- no wav/ffmpeg/librosa I/O, unlike the rest
of this module (which needs real audio and isn't unit-tested elsewhere either).
Run:  .venv/bin/python scripts/test_audio_features.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l1 import audio_features as af  # noqa: E402


def _synth(total_ms=30000, hop_ms=100, peak_t=10100, jump_t=16000, peak_val=5.0):
    """A synthetic onset-strength curve: flat at 0.1, a sustained step up to
    0.6 at `jump_t` (a clean novelty boundary), and a single strong spike at
    `peak_t` (the 'drop'). `beat_times_ms` mirrors a steady 120bpm grid
    (500ms/beat)."""
    import numpy as np
    n = int(total_ms / hop_ms) + 1
    onset_times_ms = np.arange(n) * hop_ms
    onset_env = np.full(n, 0.1)
    onset_env[onset_times_ms >= jump_t] += 0.5
    onset_env[int(round(peak_t / hop_ms))] = peak_val
    beat_times_ms = list(range(0, total_ms, 500))
    return onset_env, onset_times_ms, beat_times_ms


def test_drop_snaps_the_loudest_onset_to_the_nearest_beat():
    onset_env, onset_times_ms, beat_times_ms = _synth(peak_t=10100)
    sections, drop_ms = af._detect_structure(onset_env, onset_times_ms, beat_times_ms, 120.0)
    # peak sits at 10100ms; nearest 500ms-spaced beat is 10000 (not 10500).
    assert drop_ms == 10000, drop_ms
    print("ok  drop snaps the loudest onset-strength instant to the nearest beat")


def test_sections_split_at_the_novelty_boundary_snapped_to_the_bar():
    onset_env, onset_times_ms, beat_times_ms = _synth(jump_t=16000)
    sections, _drop_ms = af._detect_structure(onset_env, onset_times_ms, beat_times_ms, 120.0)
    # bpm=120 -> bar_ms=2000; the step at 16000ms already sits on a bar line.
    assert sections == [{"start_ms": 0, "end_ms": 16000},
                        {"start_ms": 16000, "end_ms": 30000}], sections
    print("ok  sections split at the novelty boundary, snapped to the bar grid")


def test_short_track_yields_no_sections_but_still_a_drop():
    onset_env, onset_times_ms, beat_times_ms = _synth(total_ms=10000, peak_t=4000)
    sections, drop_ms = af._detect_structure(onset_env, onset_times_ms, beat_times_ms, 120.0)
    assert sections == [], sections   # too short to segment into coarse phrases
    assert drop_ms == 4000, drop_ms   # the single strongest moment is still reported
    print("ok  a short track yields no sections but still reports the drop")


def test_degenerate_inputs_never_fabricate_a_boundary():
    import numpy as np
    assert af._detect_structure(np.array([]), np.array([]), [0, 500, 1000, 1500], 120.0) == ([], None)
    onset_env, onset_times_ms, _ = _synth()
    assert af._detect_structure(onset_env, onset_times_ms, [0, 500], 120.0) == ([], None)   # <4 beats
    assert af._detect_structure(onset_env, onset_times_ms, [0, 500, 1000, 1500], 0.0) == ([], None)  # bpm<=0
    print("ok  degenerate inputs (empty/too-few-beats/no-bpm) never fabricate a boundary")


def main():
    test_drop_snaps_the_loudest_onset_to_the_nearest_beat()
    test_sections_split_at_the_novelty_boundary_snapped_to_the_bar()
    test_short_track_yields_no_sections_but_still_a_drop()
    test_degenerate_inputs_never_fabricate_a_boundary()
    print("\nall audio-features tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
