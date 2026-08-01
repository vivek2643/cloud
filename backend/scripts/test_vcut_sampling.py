"""
Pure unit tests for app.services.vcut.sampling.sample_timestamps -- no I/O,
no ffmpeg. sample_frames_for_files (the ffmpeg/R2-touching half) is
exercised via the vcut smoke run instead (this codebase's convention).

Run:  .venv/bin/python scripts/test_vcut_sampling.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.sampling import sample_timestamps  # noqa: E402


def test_samples_at_regular_interval_within_a_span():
    out = sample_timestamps([(0, 3000)], interval_ms=1000, max_frames=100)
    assert out == [0, 1000, 2000, 3000], out
    print("ok  test_samples_at_regular_interval_within_a_span")


def test_span_end_always_included():
    out = sample_timestamps([(0, 2500)], interval_ms=1000, max_frames=100)
    assert out[-1] == 2500, out
    print("ok  test_span_end_always_included")


def test_multiple_spans_sampled_independently():
    out = sample_timestamps([(0, 1000), (5000, 6000)], interval_ms=1000, max_frames=100)
    assert out == [0, 1000, 5000, 6000], out
    print("ok  test_multiple_spans_sampled_independently")


def test_empty_or_degenerate_spans_produce_nothing():
    assert sample_timestamps([], interval_ms=1000, max_frames=100) == []
    assert sample_timestamps([(1000, 1000)], interval_ms=1000, max_frames=100) == []
    assert sample_timestamps([(2000, 1000)], interval_ms=1000, max_frames=100) == []
    print("ok  test_empty_or_degenerate_spans_produce_nothing")


def test_downsamples_evenly_when_over_budget():
    spans = [(0, 20000)]  # 21 raw samples at interval_ms=1000
    out = sample_timestamps(spans, interval_ms=1000, max_frames=5)
    assert len(out) <= 5, out
    assert out == sorted(set(out)), "must stay sorted/unique"
    print("ok  test_downsamples_evenly_when_over_budget")


def test_no_cap_when_max_frames_is_zero_or_negative():
    out = sample_timestamps([(0, 5000)], interval_ms=1000, max_frames=0)
    assert out == [0, 1000, 2000, 3000, 4000, 5000], out
    print("ok  test_no_cap_when_max_frames_is_zero_or_negative")


def main():
    test_samples_at_regular_interval_within_a_span()
    test_span_end_always_included()
    test_multiple_spans_sampled_independently()
    test_empty_or_degenerate_spans_produce_nothing()
    test_downsamples_evenly_when_over_budget()
    test_no_cap_when_max_frames_is_zero_or_negative()
    print("\nall vcut sampling tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
