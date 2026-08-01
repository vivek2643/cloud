"""
Pure unit tests for app.services.vcut.speech.select.select_winner
(speech_cuts_pipeline.plan.md section 10, part 2: gate + fuse + argmax). No I/O.

Run:  .venv/bin/python scripts/test_speech_select.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.select import TakeCandidate, select_winner  # noqa: E402


def test_singleton_take_always_wins_even_if_flagged():
    only = TakeCandidate(beat_id="b0", fluency_llm=0.1, flags=["incomplete"], delivery=0.0)
    result = select_winner([only])
    assert result.winner_beat_id == "b0"
    assert result.roles == {"b0": "winner"}
    print("ok  test_singleton_take_always_wins_even_if_flagged")


def test_higher_fused_score_wins():
    good = TakeCandidate(beat_id="good", fluency_llm=0.9, delivery=0.9)
    bad = TakeCandidate(beat_id="bad", fluency_llm=0.3, delivery=0.2)
    result = select_winner([good, bad])
    assert result.winner_beat_id == "good"
    assert result.roles == {"good": "winner", "bad": "take"}
    print("ok  test_higher_fused_score_wins")


def test_flag_gates_out_a_candidate_even_with_a_higher_raw_score():
    flagged_high = TakeCandidate(beat_id="flagged", fluency_llm=0.99, delivery=0.99,
                                 flags=["false_start"])
    clean_lower = TakeCandidate(beat_id="clean", fluency_llm=0.6, delivery=0.6)
    result = select_winner([flagged_high, clean_lower])
    assert result.winner_beat_id == "clean"
    assert result.finals["flagged"] == 0.0
    print("ok  test_flag_gates_out_a_candidate_even_with_a_higher_raw_score")


def test_every_candidate_flagged_falls_back_to_raw_score_ranking():
    better_but_flagged = TakeCandidate(beat_id="a", fluency_llm=0.8, delivery=0.8, flags=["incomplete"])
    worse_but_flagged = TakeCandidate(beat_id="b", fluency_llm=0.2, delivery=0.2, flags=["false_start"])
    result = select_winner([better_but_flagged, worse_but_flagged])
    assert result.winner_beat_id == "a"  # least-bad by raw score, never winner-less
    assert result.roles["a"] == "winner" and result.roles["b"] == "take"
    print("ok  test_every_candidate_flagged_falls_back_to_raw_score_ranking")


def test_multiple_alternates_all_marked_correctly():
    c1 = TakeCandidate(beat_id="c1", fluency_llm=0.9, delivery=0.9)
    c2 = TakeCandidate(beat_id="c2", fluency_llm=0.5, delivery=0.5)
    c3 = TakeCandidate(beat_id="c3", fluency_llm=0.3, delivery=0.3)
    result = select_winner([c1, c2, c3])
    assert result.winner_beat_id == "c1"
    assert result.roles == {"c1": "winner", "c2": "take", "c3": "take"}
    print("ok  test_multiple_alternates_all_marked_correctly")


def test_empty_group_raises():
    try:
        select_winner([])
        raise AssertionError("expected ValueError for an empty take group")
    except ValueError:
        pass
    print("ok  test_empty_group_raises")


def test_tie_breaks_deterministically_to_one_winner():
    a = TakeCandidate(beat_id="a", fluency_llm=0.7, delivery=0.5)
    b = TakeCandidate(beat_id="b", fluency_llm=0.7, delivery=0.5)
    result = select_winner([a, b])
    assert result.winner_beat_id in ("a", "b")
    winners = [bid for bid, role in result.roles.items() if role == "winner"]
    assert len(winners) == 1, winners
    print("ok  test_tie_breaks_deterministically_to_one_winner")


def main():
    test_singleton_take_always_wins_even_if_flagged()
    test_higher_fused_score_wins()
    test_flag_gates_out_a_candidate_even_with_a_higher_raw_score()
    test_every_candidate_flagged_falls_back_to_raw_score_ranking()
    test_multiple_alternates_all_marked_correctly()
    test_empty_group_raises()
    test_tie_breaks_deterministically_to_one_winner()
    print("\nall speech select tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
