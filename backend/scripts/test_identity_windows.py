"""
Tests for the shared voice-window utilities (app.services.l3.identity.
windows) -- no DB, no network, no model call, synthetic turns/signals only.
Ported from test_identity_speaker_frames.py (voice_id_pass.plan.md): these
functions survived the still-frame -> video+audio clip redesign unchanged,
since finding a voice's clean turns is the same problem either way.

Run:  .venv/bin/python scripts/test_identity_windows.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.identity import windows as win  # noqa: E402


def test_merge_intervals_joins_overlapping_and_touching():
    merged = win._merge_intervals([(100, 200), (150, 300), (400, 500)])
    assert merged == [(100, 300), (400, 500)], merged
    print("ok  test_merge_intervals_joins_overlapping_and_touching")


def test_largest_clean_subspan_picks_the_bigger_piece():
    # [0, 1000) with a forbidden zone [200, 600) -> pieces [0,200) (len 200)
    # and [600,1000) (len 400): the second is bigger.
    span = win._largest_clean_subspan(0, 1000, [(200, 600)])
    assert span == (600, 1000), span
    print("ok  test_largest_clean_subspan_picks_the_bigger_piece")


def test_largest_clean_subspan_none_when_fully_covered():
    assert win._largest_clean_subspan(100, 200, [(0, 300)]) is None
    print("ok  test_largest_clean_subspan_none_when_fully_covered")


def test_clean_windows_carves_around_a_nearby_other_voice_turn():
    # V's turn [1000, 3000); another voice speaks [2800, 3200) -- guard=150
    # forbids [2650, 3350), so the clean sub-span is [1000, 2650) (len 1650).
    turns = [("f1", 1000, 3000)]
    others = {"f1": [(2800, 3200)]}
    windows = win.clean_windows(turns, others)
    assert windows == [("f1", 1000, 2650)], windows
    print("ok  test_clean_windows_carves_around_a_nearby_other_voice_turn")


def test_clean_windows_drops_a_turn_with_no_span_above_min_win_ms():
    # A 500ms turn entirely swallowed by a nearby other-voice guard zone.
    turns = [("f1", 1000, 1500)]
    others = {"f1": [(1500, 1600)]}   # guard 150 -> forbids [1350, 1750)
    windows = win.clean_windows(turns, others)
    assert windows == [], windows
    print("ok  test_clean_windows_drops_a_turn_with_no_span_above_min_win_ms")


def test_voice_turns_maps_local_speakers_through_voice_of():
    turns_by_file = {"f1": [(0, 2000, "S0"), (2500, 4500, "S1")], "f2": [(0, 1000, "S0")]}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1", ("f2", "S0"): "V0"}
    out = win.voice_turns(turns_by_file, voice_of)
    assert out["V0"] == [("f1", 0, 2000), ("f2", 0, 1000)], out["V0"]
    assert out["V1"] == [("f1", 2500, 4500)], out["V1"]
    print("ok  test_voice_turns_maps_local_speakers_through_voice_of")


def test_other_voice_turns_by_file_excludes_only_the_given_voice():
    turns_by_file = {"f1": [(0, 2000, "S0"), (2500, 4500, "S1")]}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    out = win._other_voice_turns_by_file(turns_by_file, voice_of, "V0")
    assert out == {"f1": [(2500, 4500)]}, out
    print("ok  test_other_voice_turns_by_file_excludes_only_the_given_voice")


def test_loudness_peak_ms_finds_the_argmax_hop():
    rms = [-40.0, -40.0, -10.0, -40.0]
    assert win._loudness_peak_ms(rms, 100, 0, 400) == 200
    print("ok  test_loudness_peak_ms_finds_the_argmax_hop")


def test_loudness_peak_ms_falls_back_to_midpoint_with_no_signal():
    assert win._loudness_peak_ms([], 0, 1000, 2000) == 1500
    print("ok  test_loudness_peak_ms_falls_back_to_midpoint_with_no_signal")


def main():
    test_merge_intervals_joins_overlapping_and_touching()
    test_largest_clean_subspan_picks_the_bigger_piece()
    test_largest_clean_subspan_none_when_fully_covered()
    test_clean_windows_carves_around_a_nearby_other_voice_turn()
    test_clean_windows_drops_a_turn_with_no_span_above_min_win_ms()
    test_voice_turns_maps_local_speakers_through_voice_of()
    test_other_voice_turns_by_file_excludes_only_the_given_voice()
    test_loudness_peak_ms_finds_the_argmax_hop()
    test_loudness_peak_ms_falls_back_to_midpoint_with_no_signal()
    print("\nall identity-windows tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
