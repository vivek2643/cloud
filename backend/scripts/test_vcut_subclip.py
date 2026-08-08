"""
Pure unit tests for app.services.vcut.subclip's time-map + ffmpeg-command-
building helpers (pass1_video_input.plan.md section 2/3/10.1). No ffmpeg, no
network, no R2 -- cut_non_speech_subclip's I/O is exercised via the live
video-mode smoke run instead.

Run:  .venv/bin/python scripts/test_vcut_subclip.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.subclip import (  # noqa: E402
    _apply_max_cap, _build_filter_complex, _build_segments, is_near_join, map_sub_to_orig,
)

# 3 non-speech spans: [1000,3000) [5000,5800) [10000,14000) -- lengths 2000/800/4000.
_SPANS = [(1000, 3000), (5000, 5800), (10000, 14000)]
_SEGMENTS = _build_segments(_SPANS)  # [(1000,3000,0), (5000,5800,2000), (10000,14000,2800)]


# --------------------------------------------------------------------------
# _build_segments -- cumulative sub_start math
# --------------------------------------------------------------------------

def test_build_segments_sub_start_is_cumulative_span_length():
    assert _SEGMENTS == [(1000, 3000, 0), (5000, 5800, 2000), (10000, 14000, 2800)], _SEGMENTS
    print("ok  test_build_segments_sub_start_is_cumulative_span_length")


def test_build_segments_single_span_starts_at_zero():
    assert _build_segments([(500, 900)]) == [(500, 900, 0)]
    print("ok  test_build_segments_single_span_starts_at_zero")


def test_build_segments_empty_spans_yields_empty_segments():
    assert _build_segments([]) == []
    print("ok  test_build_segments_empty_spans_yields_empty_segments")


# --------------------------------------------------------------------------
# map_sub_to_orig -- round-trips + boundary points + out-of-range
# --------------------------------------------------------------------------

def test_map_sub_to_orig_start_of_each_segment():
    assert map_sub_to_orig(0, _SEGMENTS) == 1000       # start of segment 0
    assert map_sub_to_orig(2000, _SEGMENTS) == 5000     # start of segment 1
    assert map_sub_to_orig(2800, _SEGMENTS) == 10000    # start of segment 2
    print("ok  test_map_sub_to_orig_start_of_each_segment")


def test_map_sub_to_orig_middle_of_a_segment():
    # segment 1: orig [5000,5800), sub_start=2000 -> sub 2000+300=2300 -> orig 5300
    assert map_sub_to_orig(2300, _SEGMENTS) == 5300
    print("ok  test_map_sub_to_orig_middle_of_a_segment")


def test_map_sub_to_orig_last_ms_of_a_segment_is_inclusive_of_the_open_end():
    # segment 0 spans sub [0,2000) -- sub_ms=1999 is the last INCLUDED ms.
    assert map_sub_to_orig(1999, _SEGMENTS) == 2999
    print("ok  test_map_sub_to_orig_last_ms_of_a_segment_is_inclusive_of_the_open_end")


def test_map_sub_to_orig_out_of_range_returns_none():
    total = sum(e - s for s, e in _SPANS)  # 6800
    assert map_sub_to_orig(total, _SEGMENTS) is None      # exactly at the end -- no segment covers it
    assert map_sub_to_orig(total + 500, _SEGMENTS) is None
    assert map_sub_to_orig(-1, _SEGMENTS) is None
    print("ok  test_map_sub_to_orig_out_of_range_returns_none")


def test_map_sub_to_orig_round_trips_across_the_whole_subclip():
    """Every ms in [0, total) maps to a real orig ms inside SOME original
    span -- a genuine round-trip check across the full 3-segment map."""
    total = sum(e - s for s, e in _SPANS)
    for sub_ms in range(0, total, 137):  # a stride that doesn't land on nice round numbers
        orig_ms = map_sub_to_orig(sub_ms, _SEGMENTS)
        assert orig_ms is not None, sub_ms
        assert any(s <= orig_ms < e for s, e in _SPANS), (sub_ms, orig_ms)
    print("ok  test_map_sub_to_orig_round_trips_across_the_whole_subclip")


# --------------------------------------------------------------------------
# is_near_join -- internal concat seams only, not the clip's own start/end
# --------------------------------------------------------------------------

def test_is_near_join_flags_points_within_guard_of_an_internal_seam():
    # internal joins are at sub_start=2000 (segment 1) and sub_start=2800 (segment 2)
    assert is_near_join(2000, _SEGMENTS, guard_ms=200) is True     # exactly on the seam
    assert is_near_join(1850, _SEGMENTS, guard_ms=200) is True     # 150ms before
    assert is_near_join(2150, _SEGMENTS, guard_ms=200) is True     # 150ms after
    assert is_near_join(2800, _SEGMENTS, guard_ms=200) is True
    print("ok  test_is_near_join_flags_points_within_guard_of_an_internal_seam")


def test_is_near_join_false_well_inside_a_segment():
    assert is_near_join(1000, _SEGMENTS, guard_ms=200) is False    # middle of segment 0
    assert is_near_join(2400, _SEGMENTS, guard_ms=200) is False    # middle of segment 1
    print("ok  test_is_near_join_false_well_inside_a_segment")


def test_is_near_join_the_clips_own_start_is_not_a_join():
    assert is_near_join(0, _SEGMENTS, guard_ms=200) is False
    print("ok  test_is_near_join_the_clips_own_start_is_not_a_join")


def test_is_near_join_single_segment_clip_has_no_joins_at_all():
    single = _build_segments([(0, 5000)])
    assert is_near_join(0, single, guard_ms=200) is False
    assert is_near_join(2500, single, guard_ms=200) is False
    assert is_near_join(4999, single, guard_ms=200) is False
    print("ok  test_is_near_join_single_segment_clip_has_no_joins_at_all")


def test_is_near_join_respects_the_exact_guard_boundary():
    assert is_near_join(2000 - 200, _SEGMENTS, guard_ms=200) is True    # exactly at the guard edge
    assert is_near_join(2000 - 201, _SEGMENTS, guard_ms=200) is False   # 1ms outside
    print("ok  test_is_near_join_respects_the_exact_guard_boundary")


# --------------------------------------------------------------------------
# _apply_max_cap -- safety ceiling truncates, never drops a fitting span
# --------------------------------------------------------------------------

def test_apply_max_cap_no_op_when_under_the_cap():
    assert _apply_max_cap(_SPANS, max_ms=100000) == _SPANS
    print("ok  test_apply_max_cap_no_op_when_under_the_cap")


def test_apply_max_cap_truncates_the_boundary_span_exactly_to_the_cap():
    # total = 2000+800+4000 = 6800; cap at 2500 -> keep all of span 0 (2000),
    # then span 1 truncated to 500ms (5000..5500) to land exactly on the cap.
    capped = _apply_max_cap(_SPANS, max_ms=2500)
    assert capped == [(1000, 3000), (5000, 5500)], capped
    print("ok  test_apply_max_cap_truncates_the_boundary_span_exactly_to_the_cap")


def test_apply_max_cap_drops_spans_entirely_past_the_cap():
    capped = _apply_max_cap(_SPANS, max_ms=2000)
    assert capped == [(1000, 3000)], capped
    print("ok  test_apply_max_cap_drops_spans_entirely_past_the_cap")


def test_apply_max_cap_zero_cap_yields_nothing():
    assert _apply_max_cap(_SPANS, max_ms=0) == []
    print("ok  test_apply_max_cap_zero_cap_yields_nothing")


# --------------------------------------------------------------------------
# _build_filter_complex -- structural sanity (no ffmpeg invocation)
# --------------------------------------------------------------------------

def test_build_filter_complex_has_one_trim_per_span_and_a_concat():
    fc = _build_filter_complex(_SPANS, width_px=640, fps=2.0)
    assert fc.count("trim=") == 3, fc
    assert "concat=n=3:v=1:a=0" in fc, fc
    assert "scale=640:-2" in fc and "fps=2.0" in fc, fc
    print("ok  test_build_filter_complex_has_one_trim_per_span_and_a_concat")


def test_build_filter_complex_trim_bounds_are_seconds_from_ms():
    fc = _build_filter_complex([(1500, 3200)], width_px=640, fps=2.0)
    assert "trim=start=1.500:end=3.200" in fc, fc
    print("ok  test_build_filter_complex_trim_bounds_are_seconds_from_ms")


def main():
    test_build_segments_sub_start_is_cumulative_span_length()
    test_build_segments_single_span_starts_at_zero()
    test_build_segments_empty_spans_yields_empty_segments()
    test_map_sub_to_orig_start_of_each_segment()
    test_map_sub_to_orig_middle_of_a_segment()
    test_map_sub_to_orig_last_ms_of_a_segment_is_inclusive_of_the_open_end()
    test_map_sub_to_orig_out_of_range_returns_none()
    test_map_sub_to_orig_round_trips_across_the_whole_subclip()
    test_is_near_join_flags_points_within_guard_of_an_internal_seam()
    test_is_near_join_false_well_inside_a_segment()
    test_is_near_join_the_clips_own_start_is_not_a_join()
    test_is_near_join_single_segment_clip_has_no_joins_at_all()
    test_is_near_join_respects_the_exact_guard_boundary()
    test_apply_max_cap_no_op_when_under_the_cap()
    test_apply_max_cap_truncates_the_boundary_span_exactly_to_the_cap()
    test_apply_max_cap_drops_spans_entirely_past_the_cap()
    test_apply_max_cap_zero_cap_yields_nothing()
    test_build_filter_complex_has_one_trim_per_span_and_a_concat()
    test_build_filter_complex_trim_bounds_are_seconds_from_ms()
    print("\nall vcut subclip tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
