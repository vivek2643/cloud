"""
Tests for the seam-significance primitive (``app.services.l3.seam``) -- pure,
no DB, no API. Run:  .venv/bin/python scripts/test_seam.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.seam import Seam, classify_seam  # noqa: E402


def _seam(**over) -> Seam:
    base = dict(same_clip=True, same_speaker=True, gap_ms=500, bridged_speech_ms=4000,
                has_scene_or_transition=False, has_flagged_break=False)
    base.update(over)
    return Seam(**base)


def test_continuous_take_is_weldable():
    v = classify_seam(_seam())
    assert v.weldable and "continuous" in v.reason, v
    print("ok  test_continuous_take_is_weldable")


def test_cross_clip_is_hard():
    v = classify_seam(_seam(same_clip=False))
    assert not v.weldable and "cross-clip" in v.reason, v
    print("ok  test_cross_clip_is_hard")


def test_speaker_change_is_hard():
    v = classify_seam(_seam(same_speaker=False))
    assert not v.weldable and "speaker" in v.reason, v
    print("ok  test_speaker_change_is_hard")


def test_scene_or_transition_is_hard():
    v = classify_seam(_seam(has_scene_or_transition=True))
    assert not v.weldable and ("shot" in v.reason or "transition" in v.reason), v
    print("ok  test_scene_or_transition_is_hard")


def test_flagged_break_is_hard():
    v = classify_seam(_seam(has_flagged_break=True))
    assert not v.weldable and "flagged" in v.reason, v
    print("ok  test_flagged_break_is_hard")


def test_magnitude_backstop_hard_when_gap_exceeds_speech():
    # gap longer than the speech it would bridge -> hard, no tuned constant.
    v = classify_seam(_seam(gap_ms=5000, bridged_speech_ms=4000))
    assert not v.weldable and "longer than the speech" in v.reason, v
    # equal is still weldable (>, not >=): connective tissue == speech is fine.
    assert classify_seam(_seam(gap_ms=4000, bridged_speech_ms=4000)).weldable
    print("ok  test_magnitude_backstop_hard_when_gap_exceeds_speech")


def test_break_signals_take_priority_over_a_slack_backstop():
    # A short gap (backstop slack) still hard-splits on a real break.
    v = classify_seam(_seam(gap_ms=100, bridged_speech_ms=9000, has_scene_or_transition=True))
    assert not v.weldable, v
    print("ok  test_break_signals_take_priority_over_a_slack_backstop")


def main():
    test_continuous_take_is_weldable()
    test_cross_clip_is_hard()
    test_speaker_change_is_hard()
    test_scene_or_transition_is_hard()
    test_flagged_break_is_hard()
    test_magnitude_backstop_hard_when_gap_exceeds_speech()
    test_break_signals_take_priority_over_a_slack_backstop()
    print("\nall seam tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
