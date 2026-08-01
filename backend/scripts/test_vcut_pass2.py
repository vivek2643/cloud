"""
Unit tests for app.services.vcut.pass2 (vcut_pass2_rich.plan.md) -- the
pure helpers directly, and run_enrich's cache-ride/fallback orchestration
with every I/O touchpoint mocked (unittest.mock, stdlib -- no real network,
no real DB). No pytest; this codebase's plain test_*.py convention.

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


def _cut(id_="c1", file_id="f1", in_ms=0, out_ms=1000, hero_ms=500, meaning="m", qids=None):
    return {"id": id_, "file_id": file_id, "src_in_ms": in_ms, "src_out_ms": out_ms,
            "hero_ts_ms": hero_ms, "summary": meaning, "proxy_key": "k/proxy.mp4",
            "question_ids": qids or ["subject", "action"]}


# --------------------------------------------------------------------------
# _task_text / _cuts_listing
# --------------------------------------------------------------------------

def test_task_text_cached_vs_fallback_preambles_differ():
    cuts = [_cut()]
    cached = p2._task_text(cuts, cached=True)
    fallback = p2._task_text(cuts, cached=False)
    assert "cached frames" in cached
    assert "representative frame" in fallback
    assert cached != fallback
    print("ok  test_task_text_cached_vs_fallback_preambles_differ")


def test_task_text_mentions_bank_and_cut_listing():
    text = p2._task_text([_cut(qids=["subject", "notable_object"])], cached=True)
    assert "notable_object" in text  # from the bank definitions
    assert "cut_id=c1" in text
    assert "fields=[subject, notable_object]" in text
    print("ok  test_task_text_mentions_bank_and_cut_listing")


# --------------------------------------------------------------------------
# _question_ids_by_file / _cuts_with_meta
# --------------------------------------------------------------------------

def test_question_ids_by_file_is_the_default_set_for_every_file_phase2_interim():
    """vcut_moment_energy.plan.md section 9 (Phase 2, deferred): Pass 1 no
    longer selects question_ids at all (flags only) -- the new FilePlan
    shape has no such field, so every file gets the same small default set
    regardless of what's in loose_plan (a stale/legacy question_ids value,
    if present from a not-yet-re-ingested run, is simply ignored, not
    whitelisted)."""
    loose_plan_dict = {
        "f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "m"}]},
        "f2": {"meaning": "m2", "question_ids": ["subject", "bogus_id"], "loose_cuts": []},
    }
    out = p2._question_ids_by_file(loose_plan_dict)
    from app.services.vcut.questions import DEFAULT_QUESTION_IDS
    assert out["f1"] == list(DEFAULT_QUESTION_IDS), out["f1"]
    assert out["f2"] == list(DEFAULT_QUESTION_IDS), out["f2"]
    print("ok  test_question_ids_by_file_is_the_default_set_for_every_file_phase2_interim")


def test_cuts_with_meta_attaches_question_ids_falls_back_for_unknown_file():
    cuts = [{"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 1000,
            "hero_ts_ms": 500, "summary": "m", "proxy_key": "k"}]
    out = p2._cuts_with_meta(cuts, {"f1": ["setting"]})
    assert out[0]["question_ids"] == ["setting"]

    out2 = p2._cuts_with_meta(cuts, {})  # f1 missing from the map entirely
    from app.services.vcut.questions import DEFAULT_QUESTION_IDS
    assert out2[0]["question_ids"] == list(DEFAULT_QUESTION_IDS)
    print("ok  test_cuts_with_meta_attaches_question_ids_falls_back_for_unknown_file")


# --------------------------------------------------------------------------
# scene_specifics_from_answers -- selected fields only, unknown ids ignored
# --------------------------------------------------------------------------

def test_scene_specifics_keeps_only_selected_fields():
    known = {"c1": ["subject", "on_screen_text"]}
    parsed = p2.Pass2Schema(answers=[
        p2._AnswerOut(cut_id="c1", subject="a dog", action="running",
                     on_screen_text="SALE", notable_object="leash"),
    ])
    out = p2.scene_specifics_from_answers(parsed, known)
    assert out == {"c1": {"subject": "a dog", "on_screen_text": "SALE"}}, out
    print("ok  test_scene_specifics_keeps_only_selected_fields")


def test_scene_specifics_ignores_unknown_cut_ids():
    known = {"c1": ["subject"]}
    parsed = p2.Pass2Schema(answers=[
        p2._AnswerOut(cut_id="c1", subject="a dog"),
        p2._AnswerOut(cut_id="ghost", subject="nobody asked"),
    ])
    out = p2.scene_specifics_from_answers(parsed, known)
    assert set(out.keys()) == {"c1"}, out
    print("ok  test_scene_specifics_ignores_unknown_cut_ids")


def test_scene_specifics_empty_answers_is_empty():
    assert p2.scene_specifics_from_answers(p2.Pass2Schema(answers=[]), {"c1": ["subject"]}) == {}
    print("ok  test_scene_specifics_empty_answers_is_empty")


# --------------------------------------------------------------------------
# _call_cached / _call_fallback -- mocked complete_gemini
# --------------------------------------------------------------------------

def test_call_cached_sends_no_images_and_rides_the_cache_model():
    cuts = [_cut()]
    cache_handle = {"name": "cachedContents/abc", "model": "gemini-3.1-flash-lite"}
    fake = MagicMock(data={"answers": []}, usage={})
    with patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        p2._call_cached(cuts, cache_handle)
    args, kwargs = mock_call.call_args
    system, blocks = args[0], args[1]
    assert system == p2.NEUTRAL_SYSTEM
    assert len(blocks) == 1 and blocks[0]["type"] == "text"
    assert all(b["type"] != "image" for b in blocks)
    assert kwargs["cached_content"] == "cachedContents/abc"
    assert kwargs["model"] == "gemini-3.1-flash-lite"
    print("ok  test_call_cached_sends_no_images_and_rides_the_cache_model")


def test_call_fallback_fetches_hero_frames_and_includes_images():
    cuts = [_cut(id_="c1"), _cut(id_="c2", file_id="f2")]
    fake = MagicMock(data={"answers": []}, usage={})
    with patch("app.services.vcut.pass2._hero_frames",
              return_value={"c1": "b64img1", "c2": "b64img2"}) as mock_hero, \
        patch("app.services.llm.ingest_gemini.complete_gemini", return_value=fake) as mock_call:
        result = p2._call_fallback(cuts, "gemini-3.1-flash-lite")
    assert mock_hero.called
    assert result is fake
    args, kwargs = mock_call.call_args
    blocks = args[1]
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(image_blocks) == 2
    assert kwargs.get("cached_content") is None
    print("ok  test_call_fallback_fetches_hero_frames_and_includes_images")


def test_call_fallback_returns_none_when_no_frames_available():
    cuts = [_cut()]
    with patch("app.services.vcut.pass2._hero_frames", return_value={}):
        result = p2._call_fallback(cuts, "gemini-3.1-flash-lite")
    assert result is None
    print("ok  test_call_fallback_returns_none_when_no_frames_available")


# --------------------------------------------------------------------------
# run_enrich -- full orchestration, every I/O touchpoint mocked
# --------------------------------------------------------------------------

def _patched_run_enrich(cuts, loose_plan_dict, cache_handle, cached_completion=None,
                        fallback_completion=None):
    """Context manager stack shared by the run_enrich tests below -- patches
    every DB/model touchpoint run_enrich reaches for."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("app.services.vcut.pass2._video_cuts_for_run", return_value=cuts))
    stack.enter_context(patch("app.services.vcut.store.load_seam_and_plan",
                              return_value=({}, loose_plan_dict)))
    stack.enter_context(patch("app.services.vcut.store.load_vcut_cache", return_value=cache_handle))
    stack.enter_context(patch("app.services.l3.ingest_store.accumulate_pass2_usage"))
    update_mock = stack.enter_context(patch("app.services.l3.ingest_store.update_cut_scene_specifics"))
    delete_mock = stack.enter_context(patch("app.services.llm.ingest_gemini.delete_pass2_cache"))

    def _fake_call_cached(cuts_with_meta, handle):
        if cached_completion is None:
            raise RuntimeError("cache expired")
        return cached_completion
    stack.enter_context(patch("app.services.vcut.pass2._call_cached", side_effect=_fake_call_cached))
    stack.enter_context(patch("app.services.vcut.pass2._call_fallback", return_value=fallback_completion))
    return stack, update_mock, delete_mock


def test_run_enrich_uses_cached_path_when_handle_present():
    cuts = [_cut()]
    loose_plan_dict = {"f1": {"meaning": "m", "question_ids": ["subject"], "loose_cuts": []}}
    cache_handle = {"name": "cachedContents/abc", "model": "gemini-3.1-flash-lite"}
    completion = MagicMock(data={"answers": [{"cut_id": "c1", "subject": "a dog"}]}, usage={})

    stack, update_mock, delete_mock = _patched_run_enrich(
        cuts, loose_plan_dict, cache_handle, cached_completion=completion)
    with stack:
        p2.run_enrich("proj1", "run1")

    update_mock.assert_called_once()
    assert update_mock.call_args[0][0] == "c1"
    # Phase 2 interim (vcut_moment_energy.plan.md section 9): every file
    # gets the full DEFAULT_QUESTION_IDS set now, not just "subject" --
    # unset fields still ride along at their own schema default (empty str).
    assert update_mock.call_args[0][1] == {"subject": "a dog", "action": "", "moment_type": ""}
    delete_mock.assert_called_once_with("cachedContents/abc")
    print("ok  test_run_enrich_uses_cached_path_when_handle_present")


def test_run_enrich_falls_back_to_hero_frames_when_no_cache_handle():
    cuts = [_cut()]
    loose_plan_dict = {"f1": {"meaning": "m", "question_ids": ["subject"], "loose_cuts": []}}
    completion = MagicMock(data={"answers": [{"cut_id": "c1", "subject": "a dog"}]}, usage={})

    stack, update_mock, delete_mock = _patched_run_enrich(
        cuts, loose_plan_dict, None, fallback_completion=completion)
    with stack:
        p2.run_enrich("proj1", "run1")

    update_mock.assert_called_once()
    delete_mock.assert_not_called()
    print("ok  test_run_enrich_falls_back_to_hero_frames_when_no_cache_handle")


def test_run_enrich_falls_back_when_cached_call_raises():
    cuts = [_cut()]
    loose_plan_dict = {"f1": {"meaning": "m", "question_ids": ["subject"], "loose_cuts": []}}
    cache_handle = {"name": "cachedContents/abc", "model": "gemini-3.1-flash-lite"}
    completion = MagicMock(data={"answers": [{"cut_id": "c1", "subject": "a dog"}]}, usage={})

    # cached_completion=None makes the patched _call_cached raise, forcing
    # the fallback path -- exactly the "cache expired between passes" case.
    stack, update_mock, delete_mock = _patched_run_enrich(
        cuts, loose_plan_dict, cache_handle, cached_completion=None, fallback_completion=completion)
    with stack:
        p2.run_enrich("proj1", "run1")

    update_mock.assert_called_once()
    delete_mock.assert_called_once_with("cachedContents/abc")  # teardown still runs
    print("ok  test_run_enrich_falls_back_when_cached_call_raises")


def test_run_enrich_ignores_unknown_cut_ids_in_the_response():
    cuts = [_cut(id_="c1")]
    loose_plan_dict = {"f1": {"meaning": "m", "question_ids": ["subject"], "loose_cuts": []}}
    cache_handle = {"name": "cachedContents/abc", "model": "gemini-3.1-flash-lite"}
    completion = MagicMock(data={"answers": [
        {"cut_id": "c1", "subject": "a dog"}, {"cut_id": "ghost", "subject": "nope"},
    ]}, usage={})

    stack, update_mock, _delete_mock = _patched_run_enrich(
        cuts, loose_plan_dict, cache_handle, cached_completion=completion)
    with stack:
        p2.run_enrich("proj1", "run1")

    update_mock.assert_called_once()
    assert update_mock.call_args[0][0] == "c1"
    print("ok  test_run_enrich_ignores_unknown_cut_ids_in_the_response")


def test_run_enrich_no_cuts_is_a_silent_noop():
    with patch("app.services.vcut.pass2._video_cuts_for_run", return_value=[]), \
        patch("app.services.vcut.store.load_seam_and_plan") as seam_mock:
        p2.run_enrich("proj1", "run1")
    seam_mock.assert_not_called()
    print("ok  test_run_enrich_no_cuts_is_a_silent_noop")


def main():
    test_task_text_cached_vs_fallback_preambles_differ()
    test_task_text_mentions_bank_and_cut_listing()
    test_question_ids_by_file_is_the_default_set_for_every_file_phase2_interim()
    test_cuts_with_meta_attaches_question_ids_falls_back_for_unknown_file()
    test_scene_specifics_keeps_only_selected_fields()
    test_scene_specifics_ignores_unknown_cut_ids()
    test_scene_specifics_empty_answers_is_empty()
    test_call_cached_sends_no_images_and_rides_the_cache_model()
    test_call_fallback_fetches_hero_frames_and_includes_images()
    test_call_fallback_returns_none_when_no_frames_available()
    test_run_enrich_uses_cached_path_when_handle_present()
    test_run_enrich_falls_back_to_hero_frames_when_no_cache_handle()
    test_run_enrich_falls_back_when_cached_call_raises()
    test_run_enrich_ignores_unknown_cut_ids_in_the_response()
    test_run_enrich_no_cuts_is_a_silent_noop()
    print("\nall vcut pass2 tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
