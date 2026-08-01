"""
Pure unit tests for app.services.vcut.spans.non_speech_spans_from_speech
(seam_cut_pipeline.plan.md section 5 / section 13.3) -- no I/O, no DB.

Run:  .venv/bin/python scripts/test_vcut_spans.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.params import MIN_NONSPEECH_SPAN_MS  # noqa: E402
from app.services.vcut.spans import non_speech_spans_from_speech  # noqa: E402


def test_no_transcript_is_the_whole_clip():
    assert non_speech_spans_from_speech([], 10_000) == [(0, 10_000)]
    print("ok  test_no_transcript_is_the_whole_clip")


def test_zero_duration_is_empty():
    assert non_speech_spans_from_speech([], 0) == []
    assert non_speech_spans_from_speech([(0, 100)], 0) == []
    print("ok  test_zero_duration_is_empty")


def test_speech_in_the_middle_leaves_two_gaps():
    out = non_speech_spans_from_speech([(4000, 6000)], 10_000)
    assert out == [(0, 4000), (6000, 10_000)], out
    print("ok  test_speech_in_the_middle_leaves_two_gaps")


def test_speech_at_the_very_start_leaves_only_the_tail_gap():
    out = non_speech_spans_from_speech([(0, 3000)], 10_000)
    assert out == [(3000, 10_000)], out
    print("ok  test_speech_at_the_very_start_leaves_only_the_tail_gap")


def test_speech_at_the_very_end_leaves_only_the_head_gap():
    out = non_speech_spans_from_speech([(7000, 10_000)], 10_000)
    assert out == [(0, 7000)], out
    print("ok  test_speech_at_the_very_end_leaves_only_the_head_gap")


def test_speech_covering_the_whole_clip_leaves_nothing():
    out = non_speech_spans_from_speech([(0, 10_000)], 10_000)
    assert out == [], out
    print("ok  test_speech_covering_the_whole_clip_leaves_nothing")


def test_tiny_gap_below_the_floor_is_dropped():
    tiny = MIN_NONSPEECH_SPAN_MS - 1
    second_start = 4000 + tiny
    out = non_speech_spans_from_speech([(0, 4000), (second_start, second_start + 2000)], 10_000)
    # the (4000, second_start) gap is below the floor and vanishes; the
    # trailing gap after the second speech segment is unaffected.
    assert out == [(second_start + 2000, 10_000)], out
    print("ok  test_tiny_gap_below_the_floor_is_dropped")


def test_gap_exactly_at_the_floor_survives():
    out = non_speech_spans_from_speech(
        [(0, 4000), (4000 + MIN_NONSPEECH_SPAN_MS, 10_000)], 10_000)
    assert out == [(4000, 4000 + MIN_NONSPEECH_SPAN_MS)], out
    print("ok  test_gap_exactly_at_the_floor_survives")


def test_overlapping_speech_segments_are_merged_not_double_counted():
    # Two segments that overlap (e.g. diarization/ASR produced messy bounds)
    # must merge before the complement is taken, or the "gap" between them
    # would be negative-length and corrupt the output.
    out = non_speech_spans_from_speech([(1000, 5000), (4000, 6000)], 10_000)
    assert out == [(0, 1000), (6000, 10_000)], out
    print("ok  test_overlapping_speech_segments_are_merged_not_double_counted")


def test_unsorted_speech_segments_handled():
    out = non_speech_spans_from_speech([(6000, 10_000), (0, 4000)], 10_000)
    assert out == [(4000, 6000)], out
    print("ok  test_unsorted_speech_segments_handled")


def test_speech_span_clamped_to_duration():
    # A segment extending past the clip's own duration (transcript drift)
    # never produces a negative-length or out-of-range gap.
    out = non_speech_spans_from_speech([(8000, 20_000)], 10_000)
    assert out == [(0, 8000)], out
    print("ok  test_speech_span_clamped_to_duration")


def main():
    test_no_transcript_is_the_whole_clip()
    test_zero_duration_is_empty()
    test_speech_in_the_middle_leaves_two_gaps()
    test_speech_at_the_very_start_leaves_only_the_tail_gap()
    test_speech_at_the_very_end_leaves_only_the_head_gap()
    test_speech_covering_the_whole_clip_leaves_nothing()
    test_tiny_gap_below_the_floor_is_dropped()
    test_gap_exactly_at_the_floor_survives()
    test_overlapping_speech_segments_are_merged_not_double_counted()
    test_unsorted_speech_segments_handled()
    test_speech_span_clamped_to_duration()
    print("\nall vcut spans tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
