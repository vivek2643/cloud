"""
Pure unit tests for the SRT sidecar exporter (``app.services.export.srt``) --
no DB, no ffmpeg, no model calls.

Run:  .venv/bin/python scripts/test_export_srt.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.export import srt  # noqa: E402


def _word(text, t_in, t_out, emphasized=False):
    return {"text": text, "t_in_ms": t_in, "t_out_ms": t_out, "emphasized": emphasized}


def _event(prog_start_ms, prog_end_ms, lines):
    """One resolved-caption event, styling fields present but irrelevant to
    SRT (build_srt must ignore box/style_ref/style/anim entirely)."""
    return {
        "prog_start_ms": prog_start_ms, "prog_end_ms": prog_end_ms,
        "lines": lines,
        "box": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.15},
        "style_ref": "bold-white", "style": {"colour": {"colour_id": "white"}},
        "anim": {"kind": "pop"},
    }


def test_timecode_formats_hh_mm_ss_mmm():
    assert srt._timecode(0) == "00:00:00,000"
    assert srt._timecode(999) == "00:00:00,999"
    assert srt._timecode(1000) == "00:00:01,000"
    assert srt._timecode(61_500) == "00:01:01,500"
    assert srt._timecode(3_661_250) == "01:01:01,250"
    print("ok  test_timecode_formats_hh_mm_ss_mmm")


def test_timecode_clamps_negative_to_zero():
    assert srt._timecode(-500) == "00:00:00,000"
    print("ok  test_timecode_clamps_negative_to_zero")


def test_build_srt_single_line_event():
    events = [
        _event(0, 1500, [{"words": [_word("Hello", 0, 400), _word("there.", 400, 1500)]}]),
    ]
    out = srt.build_srt(events)
    assert out == "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n", repr(out)
    print("ok  test_build_srt_single_line_event")


def test_build_srt_multi_line_event_joins_with_newline():
    events = [
        _event(0, 2000, [
            {"words": [_word("First", 0, 400), _word("line", 400, 800)]},
            {"words": [_word("second", 800, 1200), _word("line.", 1200, 2000)]},
        ]),
    ]
    out = srt.build_srt(events)
    assert "First line\nsecond line." in out, out
    print("ok  test_build_srt_multi_line_event_joins_with_newline")


def test_build_srt_numbers_cues_sequentially_and_orders_by_start():
    # Out of input order -- build_srt must sort by prog_start_ms itself.
    events = [
        _event(2000, 3000, [{"words": [_word("second", 2000, 3000)]}]),
        _event(0, 1000, [{"words": [_word("first", 0, 1000)]}]),
    ]
    out = srt.build_srt(events)
    blocks = out.strip("\n").split("\n\n")
    assert len(blocks) == 2, blocks
    assert blocks[0].splitlines()[0] == "1" and "first" in blocks[0]
    assert blocks[1].splitlines()[0] == "2" and "second" in blocks[1]
    print("ok  test_build_srt_numbers_cues_sequentially_and_orders_by_start")


def test_build_srt_skips_an_event_with_no_word_text():
    events = [
        _event(0, 500, [{"words": []}]),
        _event(600, 1200, [{"words": [_word("real", 600, 1200)]}]),
    ]
    out = srt.build_srt(events)
    assert out.count("-->") == 1, out
    assert "real" in out
    print("ok  test_build_srt_skips_an_event_with_no_word_text")


def test_build_srt_bumps_zero_duration_event_to_one_ms():
    events = [_event(1000, 1000, [{"words": [_word("blip", 1000, 1000)]}])]
    out = srt.build_srt(events)
    assert "00:00:01,000 --> 00:00:01,001" in out, out
    print("ok  test_build_srt_bumps_zero_duration_event_to_one_ms")


def test_build_srt_empty_input_is_empty_string():
    assert srt.build_srt([]) == ""
    print("ok  test_build_srt_empty_input_is_empty_string")


def test_build_srt_timecodes_are_monotonic_and_non_overlapping():
    # A realistic multi-event fixture -- assert every cue's timecodes parse,
    # each cue's end > start, and cues never overlap or go out of order.
    events = [
        _event(0, 1200, [{"words": [_word("Watch", 0, 500), _word("this.", 500, 1200)]}]),
        _event(1300, 2600, [{"words": [_word("It's", 1300, 1600), _word("great.", 1600, 2600)]}]),
        _event(2700, 4000, [{"words": [_word("Right?", 2700, 4000)]}]),
    ]
    out = srt.build_srt(events)
    blocks = [b for b in out.strip("\n").split("\n\n") if b]
    assert len(blocks) == 3, blocks
    spans = []
    pat = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)")
    for b in blocks:
        m = pat.search(b)
        assert m is not None, b
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
        start = ((h1 * 60 + m1) * 60 + s1) * 1000 + ms1
        end = ((h2 * 60 + m2) * 60 + s2) * 1000 + ms2
        assert end > start, (start, end)
        spans.append((start, end))
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert s1 >= e0, (s0, e0, s1, e1)   # non-overlapping
        assert s1 > s0, (s0, s1)            # monotonic
    print("ok  test_build_srt_timecodes_are_monotonic_and_non_overlapping")


def main():
    test_timecode_formats_hh_mm_ss_mmm()
    test_timecode_clamps_negative_to_zero()
    test_build_srt_single_line_event()
    test_build_srt_multi_line_event_joins_with_newline()
    test_build_srt_numbers_cues_sequentially_and_orders_by_start()
    test_build_srt_skips_an_event_with_no_word_text()
    test_build_srt_bumps_zero_duration_event_to_one_ms()
    test_build_srt_empty_input_is_empty_string()
    test_build_srt_timecodes_are_monotonic_and_non_overlapping()
    print("\nall export_srt tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
