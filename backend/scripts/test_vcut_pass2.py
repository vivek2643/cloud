"""
Unit tests for app.services.vcut.pass2 (vcut_pass2_video_specifics.plan.md)
-- the pure/near-pure helpers directly, plus run_enrich_inline (video mode)
and run_enrich's (frames mode) orchestration with every I/O touchpoint
mocked (unittest.mock, stdlib -- no real network, no real DB). No pytest;
this codebase's plain test_*.py convention.

Run:  .venv/bin/python scripts/test_vcut_pass2.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut import pass2 as p2  # noqa: E402
from app.services.vcut.pass1 import VideoHandle  # noqa: E402
from app.services.vcut.resolve import FilePlan, MomentFlag, MomentPlan  # noqa: E402


def _flag(t_ms=1000, summary="m", question_ids=None, custom_questions=None, specifics=None):
    return MomentFlag(t_ms=t_ms, shape="both", summary=summary,
                      question_ids=question_ids or [], custom_questions=custom_questions or [],
                      specifics=specifics or {})


# --------------------------------------------------------------------------
# _flagged / _moments_listing / _task_text -- pure
# --------------------------------------------------------------------------

def test_flagged_keeps_only_flags_with_question_ids():
    flags = [_flag(question_ids=["subject"]), _flag(question_ids=[]), _flag(question_ids=["action"])]
    out = p2._flagged(flags)
    assert [i for i, _f in out] == [0, 2]
    print("ok  test_flagged_keeps_only_flags_with_question_ids")


def test_moments_listing_includes_index_time_summary_fields_and_custom():
    flags = [_flag(t_ms=5000, summary="a dog runs", question_ids=["subject", "action"],
                   custom_questions=[{"key": "breed", "prompt": "what breed?"}])]
    text = p2._moments_listing(p2._flagged(flags))
    assert "moment=0" in text and "t=5000ms" in text
    assert "a dog runs" in text
    assert "fields=[subject, action, subject_box]" in text
    assert "breed: what breed?" in text
    print("ok  test_moments_listing_includes_index_time_summary_fields_and_custom")


def test_task_text_video_vs_frames_preambles_differ():
    flags = [_flag(question_ids=["subject"])]
    indexed = p2._flagged(flags)
    video_text = p2._task_text(indexed, video=True)
    frames_text = p2._task_text(indexed, video=False)
    assert "shown this file's video" in video_text
    assert "shown this file's video" not in frames_text
    assert video_text != frames_text
    print("ok  test_task_text_video_vs_frames_preambles_differ")


def test_task_text_always_mentions_subject_box():
    # reframe_vcut_geometry.plan.md section 1: subject_box is requested for
    # EVERY moment, in both video and frames mode -- not gated by any
    # moment's own selected question_ids.
    flags = [_flag(question_ids=["subject"])]
    indexed = p2._flagged(flags)
    assert "subject_box" in p2._task_text(indexed, video=True)
    assert "subject_box" in p2._task_text(indexed, video=False)
    print("ok  test_task_text_always_mentions_subject_box")


# --------------------------------------------------------------------------
# _specifics_from_answer -- selected fields + matched custom probes only
# --------------------------------------------------------------------------

def test_specifics_from_answer_keeps_only_requested_bank_fields():
    ans = p2._MomentAnswerOut(moment_index=0, subject="a dog", action="running", notable_object="leash")
    out = p2._specifics_from_answer(ans, ["subject", "action"], [])
    assert out == {"subject": "a dog", "action": "running"}, out
    print("ok  test_specifics_from_answer_keeps_only_requested_bank_fields")


def test_specifics_from_answer_matches_custom_probes_by_key():
    ans = p2._MomentAnswerOut(moment_index=0, subject="a dog",
                              custom=[p2._CustomAnswer(key="breed", value="labrador"),
                                     p2._CustomAnswer(key="uninvited", value="should be dropped")])
    out = p2._specifics_from_answer(ans, ["subject"], [{"key": "breed", "prompt": "what breed?"}])
    assert out == {"subject": "a dog", "breed": "labrador"}, out
    print("ok  test_specifics_from_answer_matches_custom_probes_by_key")


def test_specifics_from_answer_drops_empty_custom_values():
    ans = p2._MomentAnswerOut(moment_index=0, custom=[p2._CustomAnswer(key="breed", value="")])
    out = p2._specifics_from_answer(ans, [], [{"key": "breed", "prompt": "what breed?"}])
    assert out == {}, out
    print("ok  test_specifics_from_answer_drops_empty_custom_values")


# --------------------------------------------------------------------------
# _write_answers_onto_flags -- functional, no mutation
# --------------------------------------------------------------------------

def test_write_answers_onto_flags_fills_in_answered_moments_only():
    flags = [_flag(question_ids=["subject"]), _flag(question_ids=["action"]), _flag(question_ids=[])]
    answers = {0: p2._MomentAnswerOut(moment_index=0, subject="a dog")}
    out = p2._write_answers_onto_flags(flags, answers)
    assert out[0].specifics == {"subject": "a dog"}
    assert out[1].specifics == {}   # flagged but not answered -- stays empty
    assert out[2].specifics == {}   # never flagged at all
    print("ok  test_write_answers_onto_flags_fills_in_answered_moments_only")


def test_write_answers_onto_flags_does_not_mutate_the_input():
    original = _flag(question_ids=["subject"])
    flags = [original]
    p2._write_answers_onto_flags(flags, {0: p2._MomentAnswerOut(moment_index=0, subject="a dog")})
    assert original.specifics == {}
    print("ok  test_write_answers_onto_flags_does_not_mutate_the_input")


def test_valid_normalized_box_rejects_absolute_pixel_coordinates():
    # The exact shape of a bad box a live smoke test caught: the model
    # returned absolute pixel coordinates instead of a normalized [0,1]
    # box for one ambiguous (no clear single subject) merged cut.
    assert p2._valid_normalized_box((0.0, 280.0, 1000.0, 720.0)) is None
    print("ok  test_valid_normalized_box_rejects_absolute_pixel_coordinates")


def test_valid_normalized_box_accepts_real_normalized_boxes():
    for box in [(0.347, 0.198, 0.465, 0.578), (0.605, 0.302, 0.395, 0.698), (0.2, 0.6, 0.8, 0.4)]:
        assert p2._valid_normalized_box(box) == box
    print("ok  test_valid_normalized_box_accepts_real_normalized_boxes")


def test_valid_normalized_box_rejects_non_positive_dims():
    assert p2._valid_normalized_box((0.1, 0.1, 0.0, 0.5)) is None
    assert p2._valid_normalized_box((0.1, 0.1, 0.5, -0.1)) is None
    print("ok  test_valid_normalized_box_rejects_non_positive_dims")


def test_valid_normalized_box_tolerates_small_edge_overshoot():
    assert p2._valid_normalized_box((-0.02, 0.0, 1.02, 1.0)) == (-0.02, 0.0, 1.02, 1.0)
    print("ok  test_valid_normalized_box_tolerates_small_edge_overshoot")


def test_valid_normalized_box_passes_none_through():
    assert p2._valid_normalized_box(None) is None
    print("ok  test_valid_normalized_box_passes_none_through")


def test_write_answers_onto_flags_drops_a_malformed_box():
    flags = [_flag(question_ids=["subject"])]
    answers = {0: p2._MomentAnswerOut(moment_index=0, subject="a dog", subject_box=(0.0, 280.0, 1000.0, 720.0))}
    out = p2._write_answers_onto_flags(flags, answers)
    assert out[0].subject_box is None
    assert out[0].specifics == {"subject": "a dog"}   # specifics unaffected by a bad box
    print("ok  test_write_answers_onto_flags_drops_a_malformed_box")


def test_write_answers_onto_flags_captures_subject_box():
    flags = [_flag(question_ids=["subject"])]
    answers = {0: p2._MomentAnswerOut(moment_index=0, subject="a dog", subject_box=(0.1, 0.2, 0.3, 0.4))}
    out = p2._write_answers_onto_flags(flags, answers)
    assert out[0].subject_box == (0.1, 0.2, 0.3, 0.4)
    assert out[0].specifics == {"subject": "a dog"}
    print("ok  test_write_answers_onto_flags_captures_subject_box")


def test_write_answers_onto_flags_captures_subject_box_even_with_no_question_ids():
    # subject_box is not a bank question -- captured regardless of
    # question_ids, unlike specifics (reframe_vcut_geometry.plan.md section
    # 1). Not reachable via the full pipeline today (qplan.py never leaves
    # a flag's question_ids empty), but _write_answers_onto_flags itself
    # must not couple the two.
    flags = [_flag(question_ids=[])]
    answers = {0: p2._MomentAnswerOut(moment_index=0, subject_box=(0.5, 0.5, 0.1, 0.1))}
    out = p2._write_answers_onto_flags(flags, answers)
    assert out[0].subject_box == (0.5, 0.5, 0.1, 0.1)
    assert out[0].specifics == {}
    print("ok  test_write_answers_onto_flags_captures_subject_box_even_with_no_question_ids")


# --------------------------------------------------------------------------
# _answer_file_video -- cached vs inline, mocked complete_gemini
# --------------------------------------------------------------------------

def test_answer_file_video_cached_sends_neutral_system_and_cached_content():
    flags = [_flag(question_ids=["subject"])]
    handle = VideoHandle(file_uri="files/abc", segments=[], cache_name="cachedContents/f1")
    fake = MagicMock(data={"answers": [{"moment_index": 0, "subject": "a dog"}]}, usage={"output_tokens": 5})
    with patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        new_flags, usage = p2._answer_file_video("f1", flags, handle, "gemini-3.1-flash-lite", 2.0, "low")
    args, kwargs = mock_call.call_args
    assert args[0] == p2.NEUTRAL_SYSTEM
    assert len(args[1]) == 1 and args[1][0]["type"] == "text"
    assert kwargs["cached_content"] == "cachedContents/f1"
    assert new_flags[0].specifics == {"subject": "a dog"}
    assert usage == {"output_tokens": 5}
    print("ok  test_answer_file_video_cached_sends_neutral_system_and_cached_content")


def test_answer_file_video_no_cache_sends_inline_video_block():
    flags = [_flag(question_ids=["subject"])]
    handle = VideoHandle(file_uri="files/abc", segments=[])  # cache_name defaults to None
    fake = MagicMock(data={"answers": []}, usage={})
    with patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        p2._answer_file_video("f1", flags, handle, "gemini-3.1-flash-lite", 2.0, "low")
    args, kwargs = mock_call.call_args
    assert args[0] != p2.NEUTRAL_SYSTEM   # task text IS the system here
    assert args[1][0]["type"] == "video_file" and args[1][0]["file_uri"] == "files/abc"
    assert kwargs["cached_content"] is None
    print("ok  test_answer_file_video_no_cache_sends_inline_video_block")


def test_answer_file_video_captures_subject_box_from_the_completion():
    flags = [_flag(question_ids=["subject"])]
    handle = VideoHandle(file_uri="files/abc", segments=[], cache_name="cachedContents/f1")
    fake = MagicMock(data={"answers": [
        {"moment_index": 0, "subject": "a dog", "subject_box": [0.1, 0.2, 0.3, 0.4]},
    ]}, usage={})
    with patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake):
        new_flags, _usage = p2._answer_file_video("f1", flags, handle, "m", 2.0, "low")
    assert new_flags[0].subject_box == (0.1, 0.2, 0.3, 0.4)
    print("ok  test_answer_file_video_captures_subject_box_from_the_completion")


def test_answer_file_video_no_flagged_moments_skips_the_call():
    flags = [_flag(question_ids=[])]
    handle = VideoHandle(file_uri="files/abc", segments=[])
    with patch("app.services.llm.ingest_gemini.complete_gemini") as mock_call:
        new_flags, usage = p2._answer_file_video("f1", flags, handle, "m", 2.0, "low")
    mock_call.assert_not_called()
    assert new_flags == flags and usage == {}
    print("ok  test_answer_file_video_no_flagged_moments_skips_the_call")


# --------------------------------------------------------------------------
# _answer_file_frames -- hero-still fallback, mocked complete_gemini + frames
# --------------------------------------------------------------------------

def test_answer_file_frames_builds_one_frame_block_pair_per_flagged_moment():
    flags = [_flag(t_ms=1000, question_ids=["subject"]), _flag(t_ms=2000, question_ids=["action"])]
    fake = MagicMock(data={"answers": [
        {"moment_index": 0, "subject": "a dog"}, {"moment_index": 1, "action": "running"},
    ]}, usage={})
    with patch("app.services.l3.frames.extract_for_planned_frames",
              return_value={("f1", 1000): "b64a", ("f1", 2000): "b64b"}), \
         patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        new_flags, _usage = p2._answer_file_frames("f1", flags, "proxy/key.mp4", "gemini-3.1-flash-lite")
    args, _kwargs = mock_call.call_args
    image_blocks = [b for b in args[1] if b["type"] == "image"]
    assert len(image_blocks) == 2
    assert new_flags[0].specifics == {"subject": "a dog"}
    assert new_flags[1].specifics == {"action": "running"}
    print("ok  test_answer_file_frames_builds_one_frame_block_pair_per_flagged_moment")


def test_answer_file_frames_no_frames_available_skips_the_call():
    flags = [_flag(question_ids=["subject"])]
    with patch("app.services.l3.frames.extract_for_planned_frames", return_value={}), \
         patch("app.services.llm.ingest_gemini.complete_gemini") as mock_call:
        new_flags, usage = p2._answer_file_frames("f1", flags, "proxy/key.mp4", "m")
    mock_call.assert_not_called()
    assert new_flags == flags and usage == {}
    print("ok  test_answer_file_frames_no_frames_available_skips_the_call")


def test_answer_file_frames_no_flagged_moments_skips_the_call():
    flags = [_flag(question_ids=[])]
    with patch("app.services.llm.ingest_gemini.complete_gemini") as mock_call:
        new_flags, usage = p2._answer_file_frames("f1", flags, "proxy/key.mp4", "m")
    mock_call.assert_not_called()
    assert new_flags == flags and usage == {}
    print("ok  test_answer_file_frames_no_flagged_moments_skips_the_call")


# --------------------------------------------------------------------------
# run_enrich_inline -- video mode, per-file fan-out, fail-open per file
# --------------------------------------------------------------------------

def test_run_enrich_inline_answers_every_file_with_a_handle_and_flagged_moments():
    plan = MomentPlan(files=[
        FilePlan(file_id="f1", flags=[_flag(question_ids=["subject"])]),
        FilePlan(file_id="f2", flags=[_flag(question_ids=["action"])]),
        FilePlan(file_id="f3", flags=[_flag(question_ids=[])]),   # nothing flagged -- skipped
    ])
    video_by_file = {
        "f1": VideoHandle(file_uri="files/f1", segments=[]),
        "f2": VideoHandle(file_uri="files/f2", segments=[]),
        # f3 has no handle at all either
    }

    def fake_answer(file_id, flags, handle, model, fps, media_resolution):
        answered = [_flag(question_ids=f.question_ids, specifics={"subject": f"answered-{file_id}"})
                   for f in flags]
        return answered, {"output_tokens": 1}

    with patch("app.services.vcut.pass2._answer_file_video", side_effect=fake_answer) as mock_answer:
        new_plan, usage = p2.run_enrich_inline(plan, video_by_file)

    assert mock_answer.call_count == 2   # f1, f2 only -- f3 skipped (no flagged moments)
    by_id = {fp.file_id: fp for fp in new_plan.files}
    assert by_id["f1"].flags[0].specifics == {"subject": "answered-f1"}
    assert by_id["f2"].flags[0].specifics == {"subject": "answered-f2"}
    assert by_id["f3"].flags[0].specifics == {}   # untouched
    assert usage == {"output_tokens": 2}
    print("ok  test_run_enrich_inline_answers_every_file_with_a_handle_and_flagged_moments")


def test_run_enrich_inline_is_fail_open_per_file():
    plan = MomentPlan(files=[
        FilePlan(file_id="f1", flags=[_flag(question_ids=["subject"])]),
        FilePlan(file_id="f2", flags=[_flag(question_ids=["action"])]),
    ])
    video_by_file = {
        "f1": VideoHandle(file_uri="files/f1", segments=[]),
        "f2": VideoHandle(file_uri="files/f2", segments=[]),
    }

    def fake_answer(file_id, flags, handle, model, fps, media_resolution):
        if file_id == "f1":
            raise RuntimeError("schema validation failed twice: boom")
        return [_flag(question_ids=f.question_ids, specifics={"subject": "ok"}) for f in flags], {}

    with patch("app.services.vcut.pass2._answer_file_video", side_effect=fake_answer):
        new_plan, _usage = p2.run_enrich_inline(plan, video_by_file)

    by_id = {fp.file_id: fp for fp in new_plan.files}
    assert by_id["f1"].flags[0].specifics == {}   # failed -- stays summary-only, run continues
    assert by_id["f2"].flags[0].specifics == {"subject": "ok"}
    print("ok  test_run_enrich_inline_is_fail_open_per_file")


def test_run_enrich_inline_skips_files_with_no_video_handle():
    plan = MomentPlan(files=[FilePlan(file_id="f1", flags=[_flag(question_ids=["subject"])])])
    with patch("app.services.vcut.pass2._answer_file_video") as mock_answer:
        new_plan, usage = p2.run_enrich_inline(plan, {})  # no handles at all
    mock_answer.assert_not_called()
    assert new_plan.files[0].flags[0].specifics == {}
    assert usage == {}
    print("ok  test_run_enrich_inline_skips_files_with_no_video_handle")


# --------------------------------------------------------------------------
# run_enrich -- frames mode, full orchestration, every I/O touchpoint mocked
# --------------------------------------------------------------------------

def test_run_enrich_no_seam_or_plan_is_a_silent_noop():
    with patch("app.services.vcut.store.load_seam_and_plan", return_value=({}, {})), \
         patch("app.services.vcut.pass2._proxy_keys_for_run") as proxy_mock:
        p2.run_enrich("proj1", "run1")
    proxy_mock.assert_not_called()
    print("ok  test_run_enrich_no_seam_or_plan_is_a_silent_noop")


def test_run_enrich_no_planned_moments_is_a_silent_noop():
    plan_dict = {"f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x"}]}}  # no question_ids
    with patch("app.services.vcut.store.load_seam_and_plan", return_value=({"f1": {}}, plan_dict)), \
         patch("app.services.vcut.pass2._proxy_keys_for_run") as proxy_mock:
        p2.run_enrich("proj1", "run1")
    proxy_mock.assert_not_called()
    print("ok  test_run_enrich_no_planned_moments_is_a_silent_noop")


def test_run_enrich_answers_persists_and_rewrites_cut_records():
    plan_dict = {
        "f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x", "question_ids": ["subject"]}]},
    }
    seam_cache = {"f1": {"hop_ms": 100, "S": [1.0]}}

    def fake_answer_frames(file_id, flags, proxy_key, model):
        return [_flag(question_ids=f.question_ids, specifics={"subject": "a dog"}) for f in flags], \
            {"output_tokens": 3}

    with patch("app.services.vcut.store.load_seam_and_plan", return_value=(seam_cache, plan_dict)), \
         patch("app.services.vcut.pass2._proxy_keys_for_run", return_value={"f1": "proxy/key.mp4"}), \
         patch("app.services.vcut.pass2._answer_file_frames", side_effect=fake_answer_frames), \
         patch("app.services.l3.ingest_store.accumulate_pass2_usage") as usage_mock, \
         patch("app.services.vcut.store.persist_seam_and_plan") as persist_mock, \
         patch("app.services.vcut.resolve.resolve_cuts", return_value=["resolved-cut"]) as resolve_mock, \
         patch("app.services.vcut.store.insert_video_cuts", return_value=["id1"]) as insert_mock:
        p2.run_enrich("proj1", "run1")

    usage_mock.assert_called_once_with("run1", {"output_tokens": 3})
    persist_mock.assert_called_once()
    persisted_plan_dict = persist_mock.call_args[0][2]
    assert persisted_plan_dict["f1"]["flags"][0]["specifics"] == {"subject": "a dog"}
    resolve_mock.assert_called_once()
    insert_mock.assert_called_once_with("run1", ["resolved-cut"], seam_cache)
    print("ok  test_run_enrich_answers_persists_and_rewrites_cut_records")


def test_run_enrich_is_fail_open_per_file_and_still_rewrites():
    plan_dict = {
        "f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x", "question_ids": ["subject"]}]},
        "f2": {"flags": [{"t_ms": 200, "shape": "both", "summary": "y", "question_ids": ["action"]}]},
    }
    seam_cache = {"f1": {}, "f2": {}}

    def fake_answer_frames(file_id, flags, proxy_key, model):
        if file_id == "f1":
            raise RuntimeError("boom")
        return [_flag(question_ids=f.question_ids, specifics={"action": "ok"}) for f in flags], {}

    with patch("app.services.vcut.store.load_seam_and_plan", return_value=(seam_cache, plan_dict)), \
         patch("app.services.vcut.pass2._proxy_keys_for_run",
              return_value={"f1": "proxy/1.mp4", "f2": "proxy/2.mp4"}), \
         patch("app.services.vcut.pass2._answer_file_frames", side_effect=fake_answer_frames), \
         patch("app.services.vcut.store.persist_seam_and_plan") as persist_mock, \
         patch("app.services.vcut.resolve.resolve_cuts", return_value=[]), \
         patch("app.services.vcut.store.insert_video_cuts", return_value=[]) as insert_mock:
        p2.run_enrich("proj1", "run1")

    persisted = persist_mock.call_args[0][2]
    assert persisted["f1"]["flags"][0]["specifics"] == {}     # failed -- stays empty
    assert persisted["f2"]["flags"][0]["specifics"] == {"action": "ok"}
    insert_mock.assert_called_once()   # the run still completes and rewrites
    print("ok  test_run_enrich_is_fail_open_per_file_and_still_rewrites")


def main():
    test_flagged_keeps_only_flags_with_question_ids()
    test_moments_listing_includes_index_time_summary_fields_and_custom()
    test_task_text_video_vs_frames_preambles_differ()
    test_task_text_always_mentions_subject_box()
    test_specifics_from_answer_keeps_only_requested_bank_fields()
    test_specifics_from_answer_matches_custom_probes_by_key()
    test_specifics_from_answer_drops_empty_custom_values()
    test_write_answers_onto_flags_fills_in_answered_moments_only()
    test_write_answers_onto_flags_does_not_mutate_the_input()
    test_valid_normalized_box_rejects_absolute_pixel_coordinates()
    test_valid_normalized_box_accepts_real_normalized_boxes()
    test_valid_normalized_box_rejects_non_positive_dims()
    test_valid_normalized_box_tolerates_small_edge_overshoot()
    test_valid_normalized_box_passes_none_through()
    test_write_answers_onto_flags_drops_a_malformed_box()
    test_write_answers_onto_flags_captures_subject_box()
    test_write_answers_onto_flags_captures_subject_box_even_with_no_question_ids()
    test_answer_file_video_cached_sends_neutral_system_and_cached_content()
    test_answer_file_video_no_cache_sends_inline_video_block()
    test_answer_file_video_captures_subject_box_from_the_completion()
    test_answer_file_video_no_flagged_moments_skips_the_call()
    test_answer_file_frames_builds_one_frame_block_pair_per_flagged_moment()
    test_answer_file_frames_no_frames_available_skips_the_call()
    test_answer_file_frames_no_flagged_moments_skips_the_call()
    test_run_enrich_inline_answers_every_file_with_a_handle_and_flagged_moments()
    test_run_enrich_inline_is_fail_open_per_file()
    test_run_enrich_inline_skips_files_with_no_video_handle()
    test_run_enrich_no_seam_or_plan_is_a_silent_noop()
    test_run_enrich_no_planned_moments_is_a_silent_noop()
    test_run_enrich_answers_persists_and_rewrites_cut_records()
    test_run_enrich_is_fail_open_per_file_and_still_rewrites()
    print("\nall vcut pass2 tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
