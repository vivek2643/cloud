"""
Pure/mocked unit tests for app.services.vcut.qplan -- the question planner
(vcut_pass2_video_specifics.plan.md section 4). complete_gemini is
monkeypatched throughout; no real network, no money.

Run:  .venv/bin/python scripts/test_vcut_qplan.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut import qplan  # noqa: E402
from app.services.vcut.params import QPLAN_MAX_CUSTOM  # noqa: E402
from app.services.vcut.questions import DEFAULT_QUESTION_IDS  # noqa: E402
from app.services.vcut.resolve import FilePlan, MomentFlag, MomentPlan  # noqa: E402


def _plan(*file_flags):
    """``_plan(("f1", [flag, flag]), ("f2", [flag]))`` -> a MomentPlan."""
    return MomentPlan(files=[FilePlan(file_id=fid, flags=flags) for fid, flags in file_flags])


# --------------------------------------------------------------------------
# _local_context -- transcript window around a moment
# --------------------------------------------------------------------------

def test_local_context_joins_words_overlapping_the_window():
    words = [(0, 500, "hello"), (600, 1000, "world"), (9000, 9500, "far away")]
    ctx = qplan._local_context(700, words, ctx_ms=500)
    assert ctx == "hello world", ctx
    print("ok  test_local_context_joins_words_overlapping_the_window")


def test_local_context_empty_when_nothing_in_range():
    words = [(9000, 9500, "far away")]
    assert qplan._local_context(700, words, ctx_ms=500) == ""
    print("ok  test_local_context_empty_when_nothing_in_range")


def test_local_context_empty_words_list_is_empty():
    assert qplan._local_context(700, [], ctx_ms=500) == ""
    print("ok  test_local_context_empty_words_list_is_empty")


# --------------------------------------------------------------------------
# _moments_listing -- the prompt's per-moment lines
# --------------------------------------------------------------------------

def test_moments_listing_includes_file_moment_summary_and_context():
    plan = _plan(("f1", [MomentFlag(t_ms=1000, summary="a dog runs")]))
    words_by_file = {"f1": [(900, 1100, "look at that")]}
    text = qplan._moments_listing(plan, words_by_file)
    assert "file=f1" in text and "moment=0" in text and "t=1000ms" in text
    assert "a dog runs" in text
    assert "look at that" in text
    print("ok  test_moments_listing_includes_file_moment_summary_and_context")


def test_moments_listing_omits_context_part_when_no_speech_nearby():
    plan = _plan(("f1", [MomentFlag(t_ms=1000, summary="silent b-roll")]))
    text = qplan._moments_listing(plan, {})
    assert "nearby_speech" not in text
    print("ok  test_moments_listing_omits_context_part_when_no_speech_nearby")


# --------------------------------------------------------------------------
# _apply_defaults -- planner-failure fallback
# --------------------------------------------------------------------------

def test_apply_defaults_gives_every_flag_the_default_question_ids():
    plan = _plan(
        ("f1", [MomentFlag(t_ms=100), MomentFlag(t_ms=200)]),
        ("f2", [MomentFlag(t_ms=300)]),
    )
    out = qplan._apply_defaults(plan)
    for fp in out.files:
        for flag in fp.flags:
            assert flag.question_ids == list(DEFAULT_QUESTION_IDS)
            assert flag.custom_questions == []
    assert out.genre == plan.genre  # unchanged (still "")
    print("ok  test_apply_defaults_gives_every_flag_the_default_question_ids")


def test_apply_defaults_does_not_mutate_the_input_plan():
    original = MomentFlag(t_ms=100, question_ids=[], custom_questions=[])
    plan = _plan(("f1", [original]))
    qplan._apply_defaults(plan)
    assert plan.files[0].flags[0] is original
    assert original.question_ids == []  # untouched
    print("ok  test_apply_defaults_does_not_mutate_the_input_plan")


# --------------------------------------------------------------------------
# _apply_plan -- the planner's actual output applied onto the plan
# --------------------------------------------------------------------------

def test_apply_plan_writes_question_ids_and_genre():
    plan = _plan(("f1", [MomentFlag(t_ms=100, summary="a dog runs")]))
    parsed = qplan._QPlanSchema(genre="vlog", moments=[
        qplan._MomentPlanOut(file_id="f1", moment_index=0, question_ids=["subject", "action"]),
    ])
    out = qplan._apply_plan(plan, parsed)
    assert out.genre == "vlog"
    assert out.files[0].flags[0].question_ids == ["subject", "action"]
    print("ok  test_apply_plan_writes_question_ids_and_genre")


def test_apply_plan_falls_back_to_defaults_for_an_unmentioned_moment():
    plan = _plan(("f1", [MomentFlag(t_ms=100), MomentFlag(t_ms=200)]))
    # planner only covers moment 0 -- moment 1 must still get a sane default.
    parsed = qplan._QPlanSchema(genre="vlog", moments=[
        qplan._MomentPlanOut(file_id="f1", moment_index=0, question_ids=["subject"]),
    ])
    out = qplan._apply_plan(plan, parsed)
    assert out.files[0].flags[0].question_ids == ["subject"]
    assert out.files[0].flags[1].question_ids == list(DEFAULT_QUESTION_IDS)
    print("ok  test_apply_plan_falls_back_to_defaults_for_an_unmentioned_moment")


def test_apply_plan_validates_question_ids_against_the_closed_bank():
    """Even though the schema constrains question_ids to a Literal of real
    bank ids, _apply_plan re-validates defensively -- never trust the model
    blindly (the established pattern across this codebase)."""
    plan = _plan(("f1", [MomentFlag(t_ms=100)]))
    parsed = qplan._QPlanSchema(genre="other", moments=[
        qplan._MomentPlanOut(file_id="f1", moment_index=0, question_ids=[]),
    ])
    out = qplan._apply_plan(plan, parsed)
    # an empty selection whitelists down to the default set (questions.
    # validate_question_ids' own documented behavior).
    assert out.files[0].flags[0].question_ids == list(DEFAULT_QUESTION_IDS)
    print("ok  test_apply_plan_validates_question_ids_against_the_closed_bank")


def test_apply_plan_caps_custom_questions_and_drops_blank_entries():
    plan = _plan(("f1", [MomentFlag(t_ms=100)]))
    many_probes = [qplan._CustomProbe(key=f"k{i}", prompt=f"p{i}") for i in range(QPLAN_MAX_CUSTOM + 3)]
    many_probes.append(qplan._CustomProbe(key="", prompt="no key -- dropped"))
    many_probes.append(qplan._CustomProbe(key="no_prompt", prompt=""))
    parsed = qplan._QPlanSchema(genre="other", moments=[
        qplan._MomentPlanOut(file_id="f1", moment_index=0, question_ids=["subject"],
                             custom_questions=many_probes),
    ])
    out = qplan._apply_plan(plan, parsed)
    custom = out.files[0].flags[0].custom_questions
    assert len(custom) == QPLAN_MAX_CUSTOM, custom
    assert all(c["key"] and c["prompt"] for c in custom)
    print("ok  test_apply_plan_caps_custom_questions_and_drops_blank_entries")


def test_apply_plan_does_not_mutate_the_input_plan():
    original = MomentFlag(t_ms=100, question_ids=[])
    plan = _plan(("f1", [original]))
    parsed = qplan._QPlanSchema(genre="vlog", moments=[
        qplan._MomentPlanOut(file_id="f1", moment_index=0, question_ids=["subject"]),
    ])
    qplan._apply_plan(plan, parsed)
    assert original.question_ids == []
    print("ok  test_apply_plan_does_not_mutate_the_input_plan")


# --------------------------------------------------------------------------
# plan_questions -- full orchestration, complete_gemini mocked
# --------------------------------------------------------------------------

def test_plan_questions_empty_project_is_a_free_noop():
    plan = _plan(("f1", []))
    with patch("app.services.llm.ingest_gemini.complete_gemini") as mock_call:
        out, usage = qplan.plan_questions(plan, {})
    mock_call.assert_not_called()
    assert out is plan
    assert usage == {}
    print("ok  test_plan_questions_empty_project_is_a_free_noop")


def test_plan_questions_success_applies_genre_and_question_ids():
    plan = _plan(("f1", [MomentFlag(t_ms=100, summary="a dog runs")]))
    fake = MagicMock(
        data={"genre": "vlog", "moments": [{"file_id": "f1", "moment_index": 0, "question_ids": ["subject"]}]},
        usage={"input_tokens": 50, "output_tokens": 10},
    )
    with patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        out, usage = qplan.plan_questions(plan, {})
    assert mock_call.called
    assert out.genre == "vlog"
    assert out.files[0].flags[0].question_ids == ["subject"]
    assert usage == {"input_tokens": 50, "output_tokens": 10}
    print("ok  test_plan_questions_success_applies_genre_and_question_ids")


def test_plan_questions_failure_falls_back_to_defaults_and_empty_usage():
    plan = _plan(("f1", [MomentFlag(t_ms=100)]))
    with patch("app.services.llm.ingest_gemini.complete_gemini", side_effect=RuntimeError("boom")):
        out, usage = qplan.plan_questions(plan, {})
    assert out.files[0].flags[0].question_ids == list(DEFAULT_QUESTION_IDS)
    assert usage == {}
    print("ok  test_plan_questions_failure_falls_back_to_defaults_and_empty_usage")


def test_plan_questions_uses_the_explicit_model_kwarg_over_settings():
    plan = _plan(("f1", [MomentFlag(t_ms=100)]))
    fake = MagicMock(data={"genre": "other", "moments": []}, usage={})
    with patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        qplan.plan_questions(plan, {}, model="a-specific-model")
    assert mock_call.call_args.kwargs["model"] == "a-specific-model"
    print("ok  test_plan_questions_uses_the_explicit_model_kwarg_over_settings")


def main():
    test_local_context_joins_words_overlapping_the_window()
    test_local_context_empty_when_nothing_in_range()
    test_local_context_empty_words_list_is_empty()
    test_moments_listing_includes_file_moment_summary_and_context()
    test_moments_listing_omits_context_part_when_no_speech_nearby()
    test_apply_defaults_gives_every_flag_the_default_question_ids()
    test_apply_defaults_does_not_mutate_the_input_plan()
    test_apply_plan_writes_question_ids_and_genre()
    test_apply_plan_falls_back_to_defaults_for_an_unmentioned_moment()
    test_apply_plan_validates_question_ids_against_the_closed_bank()
    test_apply_plan_caps_custom_questions_and_drops_blank_entries()
    test_apply_plan_does_not_mutate_the_input_plan()
    test_plan_questions_empty_project_is_a_free_noop()
    test_plan_questions_success_applies_genre_and_question_ids()
    test_plan_questions_failure_falls_back_to_defaults_and_empty_usage()
    test_plan_questions_uses_the_explicit_model_kwarg_over_settings()
    print("\nall vcut qplan tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
