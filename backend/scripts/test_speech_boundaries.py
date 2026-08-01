"""
Pure unit tests for app.services.vcut.speech.boundaries.compute_boundaries_ms
(speech_cuts_pipeline.plan.md section 9). No I/O.

Run:  .venv/bin/python scripts/test_speech_boundaries.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.boundaries import compute_boundaries_ms  # noqa: E402
from app.services.vcut.speech.inputs import Word  # noqa: E402
from app.services.vcut.speech.params import BREATH_PAD_MS  # noqa: E402


def _words(specs):
    """specs: list of (start_ms, end_ms, text)."""
    return [Word(idx=i, start_ms=s, end_ms=e, text=t) for i, (s, e, t) in enumerate(specs)]


def test_no_silence_leaves_raw_word_boundaries():
    words = _words([(1000, 1200, "a"), (1200, 1500, "b"), (1500, 1800, "c")])
    in_ms, out_ms = compute_boundaries_ms((1, 1), words, silences=[], duration_ms=5000)
    assert (in_ms, out_ms) == (1200, 1500), (in_ms, out_ms)
    print("ok  test_no_silence_leaves_raw_word_boundaries")


def test_extends_into_a_real_silence_up_to_the_pad():
    # a silence [1300, 1450) sits inside word 1's own BREATH_PAD_MS reach
    # window (word 1 starts at 1500, pad=250 -> window is [1250, 1500)) --
    # extend left into it, snapped to the silence's own midpoint.
    words = _words([(500, 800, "a"), (1500, 1800, "b")])
    silences = [(1300, 1450)]
    in_ms, _out = compute_boundaries_ms((1, 1), words, silences, duration_ms=5000)
    assert in_ms == 1375, in_ms  # midpoint of the clamped silence (1300, 1450)
    assert in_ms >= 800  # never crosses into the preceding word
    print("ok  test_extends_into_a_real_silence_up_to_the_pad")


def test_silence_outside_the_pad_reach_is_ignored():
    # the ONLY silence available ends well before the BREATH_PAD_MS window
    # even starts -- the edge must stay at the raw word boundary rather than
    # reaching further than the pad allows.
    words = _words([(500, 800, "a"), (1500, 1800, "b")])
    silences = [(800, 1200)]  # ends at 1200, but the pad window is [1250, 1500)
    in_ms, _out = compute_boundaries_ms((1, 1), words, silences, duration_ms=5000)
    assert in_ms == 1500, in_ms
    print("ok  test_silence_outside_the_pad_reach_is_ignored")


def test_never_clips_the_neighboring_word():
    # a "silence" interval that (erroneously) overlaps the previous word's
    # own span must never pull the edge past that word's end.
    words = _words([(500, 1000, "a"), (1000, 1300, "b")])
    silences = [(400, 1000)]  # touches word 0's own span
    in_ms, _out = compute_boundaries_ms((1, 1), words, silences, duration_ms=5000)
    assert in_ms >= 1000, in_ms  # word 0's end -- never clipped
    print("ok  test_never_clips_the_neighboring_word")


def test_pad_is_bounded_even_with_a_huge_silence():
    words = _words([(0, 200, "a"), (10000, 10300, "b")])
    silences = [(200, 10000)]  # a huge gap
    in_ms, _out = compute_boundaries_ms((1, 1), words, silences, duration_ms=20000)
    assert in_ms >= 10000 - BREATH_PAD_MS, in_ms
    print("ok  test_pad_is_bounded_even_with_a_huge_silence")


def test_first_word_left_edge_floors_at_zero():
    words = _words([(400, 700, "a"), (700, 1000, "b")])
    silences = [(0, 400)]
    in_ms, _out = compute_boundaries_ms((0, 1), words, silences, duration_ms=5000)
    assert in_ms >= 0
    print("ok  test_first_word_left_edge_floors_at_zero")


def test_last_word_right_edge_ceilings_at_duration():
    words = _words([(0, 300, "a"), (300, 600, "b")])
    silences = [(600, 5000)]
    _in, out_ms = compute_boundaries_ms((0, 1), words, silences, duration_ms=5000)
    assert out_ms <= 5000
    assert out_ms <= 600 + BREATH_PAD_MS
    print("ok  test_last_word_right_edge_ceilings_at_duration")


def test_out_of_range_span_is_clamped_not_crashed():
    words = _words([(0, 300, "a"), (300, 600, "b")])
    in_ms, out_ms = compute_boundaries_ms((-5, 99), words, silences=[], duration_ms=5000)
    assert in_ms == 0 and out_ms == 600
    print("ok  test_out_of_range_span_is_clamped_not_crashed")


def test_single_word_span_both_edges_extend_independently():
    words = _words([(0, 200, "a"), (700, 900, "b"), (1400, 1600, "c")])
    silences = [(200, 700), (900, 1400)]
    in_ms, out_ms = compute_boundaries_ms((1, 1), words, silences, duration_ms=5000)
    assert 200 <= in_ms <= 700
    assert 900 <= out_ms <= 1400
    print("ok  test_single_word_span_both_edges_extend_independently")


def main():
    test_no_silence_leaves_raw_word_boundaries()
    test_extends_into_a_real_silence_up_to_the_pad()
    test_silence_outside_the_pad_reach_is_ignored()
    test_never_clips_the_neighboring_word()
    test_pad_is_bounded_even_with_a_huge_silence()
    test_first_word_left_edge_floors_at_zero()
    test_last_word_right_edge_ceilings_at_duration()
    test_out_of_range_span_is_clamped_not_crashed()
    test_single_word_span_both_edges_extend_independently()
    print("\nall speech boundaries tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
