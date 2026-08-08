"""
Pure unit tests for app.services.vcut.questions -- the closed question bank
(vcut_pass2_rich.plan.md section 4). No I/O.

Run:  .venv/bin/python scripts/test_vcut_questions.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.questions import (  # noqa: E402
    ALL_IDS, BANK, DEFAULT_QUESTION_IDS, bank_prompt_lines, validate_question_ids,
)


_ORIGINAL_NINE_IDS = {
    "subject", "action", "moment_type", "setting", "on_screen_text",
    "notable_object", "count", "motion_quality", "continuity_cue",
}

# vcut_pass2_video_specifics.plan.md section 4.3: edit-decision dimensions
# added for the question planner (qplan.py) to choose from.
_EDIT_DECISION_IDS = {
    "shot_size", "camera_move", "motion_direction", "subject_entry_exit",
    "headroom_lookroom", "usable", "energy_emotion", "hook_potential", "tags",
}


def test_bank_has_the_nine_documented_ids_plus_the_edit_decision_expansion():
    expected = _ORIGINAL_NINE_IDS | _EDIT_DECISION_IDS
    assert set(BANK.keys()) == expected, set(BANK.keys())
    assert set(ALL_IDS) == expected
    print("ok  test_bank_has_the_nine_documented_ids_plus_the_edit_decision_expansion")


def test_validate_drops_unknown_ids():
    out = validate_question_ids(["subject", "made_up_id", "action"])
    assert out == ["subject", "action"], out
    print("ok  test_validate_drops_unknown_ids")


def test_validate_dedupes_order_preserving():
    out = validate_question_ids(["action", "subject", "action", "subject"])
    assert out == ["action", "subject"], out
    print("ok  test_validate_dedupes_order_preserving")


def test_validate_empty_selection_applies_default_set():
    assert validate_question_ids([]) == list(DEFAULT_QUESTION_IDS)
    assert validate_question_ids(None) == list(DEFAULT_QUESTION_IDS)
    print("ok  test_validate_empty_selection_applies_default_set")


def test_validate_all_unknown_applies_default_set():
    assert validate_question_ids(["nope", "also_nope"]) == list(DEFAULT_QUESTION_IDS)
    print("ok  test_validate_all_unknown_applies_default_set")


def test_default_ids_are_themselves_in_the_bank():
    assert set(DEFAULT_QUESTION_IDS) <= set(BANK.keys())
    print("ok  test_default_ids_are_themselves_in_the_bank")


def test_bank_prompt_lines_mentions_every_id():
    text = bank_prompt_lines()
    for qid in BANK:
        assert qid in text, qid
    print("ok  test_bank_prompt_lines_mentions_every_id")


def main():
    test_bank_has_the_nine_documented_ids_plus_the_edit_decision_expansion()
    test_validate_drops_unknown_ids()
    test_validate_dedupes_order_preserving()
    test_validate_empty_selection_applies_default_set()
    test_validate_all_unknown_applies_default_set()
    test_default_ids_are_themselves_in_the_bank()
    test_bank_prompt_lines_mentions_every_id()
    print("\nall vcut questions tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
