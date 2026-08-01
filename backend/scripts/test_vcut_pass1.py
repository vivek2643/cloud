"""
Pure unit tests for the non-model parts of app.services.vcut.pass1 --
vcut_moment_energy.plan.md section 2/2.5: the per-file flag-only schema,
the defensive t_ms clamp, and run_pass1's per-file fan-out / merge /
failure-isolation / usage-aggregation orchestration. The Gemini call is
MOCKED (complete_gemini is monkeypatched) -- no real network, no money.

Run:  .venv/bin/python scripts/test_vcut_pass1.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.pass1 import (  # noqa: E402
    VideoHandle, _MomentOut, _Pass1FileSchema, _clamp_moment, _find_enclosing_span,
    _task_text, run_pass1,
)


# --------------------------------------------------------------------------
# A stand-in for llm.client.Completion -- only .data / .usage are read.
# --------------------------------------------------------------------------

class _FakeCompletion:
    def __init__(self, data, usage):
        self.data = data
        self.usage = usage


def _fake_complete_gemini(returns_by_file, calls_seen=None):
    """Build a thread-safe complete_gemini stand-in. ``returns_by_file`` maps
    a file_id -> either (data_dict, usage_dict) or an Exception instance to
    raise. The scoped file is recovered from the ``system`` text, which
    _run_pass1_for_file passes as _task_text(file_id, filename) ("FILE
    <id>")."""
    def fake(system, blocks, schema, **kwargs):
        assert schema is _Pass1FileSchema, schema
        file_id = next(fid for fid in returns_by_file if f"FILE {fid} " in system)
        if calls_seen is not None:
            calls_seen.append(file_id)
        spec = returns_by_file[file_id]
        if isinstance(spec, Exception):
            raise spec
        data, usage = spec
        return _FakeCompletion(data, usage)
    return fake


def _patch_gemini(fake):
    # _run_pass1_for_file does `from app.services.llm.ingest_gemini import
    # complete_gemini` at call time, so patch it at the source module.
    return patch("app.services.llm.ingest_gemini.complete_gemini", side_effect=fake)


_FILE_ROWS = [("f1", "one.mp4", 10000), ("f2", "two.mp4", 20000), ("f3", "three.mp4", 30000)]
# Every t_ms below sits inside its file's single non-speech span, so the
# defensive clamp is a no-op unless a test deliberately drifts out of range.
_SPANS = {"f1": [(0, 10000)], "f2": [(0, 20000)], "f3": [(0, 30000)]}


# --------------------------------------------------------------------------
# Schema / clamp -- pure, no orchestration
# --------------------------------------------------------------------------

def test_moment_fully_inside_a_non_speech_span_is_unchanged():
    out = _clamp_moment(_MomentOut(t_ms=1500, shape="both", summary="s"), [(0, 5000)])
    assert out.t_ms == 1500 and out.shape == "both" and out.summary == "s"
    print("ok  test_moment_fully_inside_a_non_speech_span_is_unchanged")


def test_moment_outside_every_span_snaps_to_the_nearest_edge():
    out = _clamp_moment(_MomentOut(t_ms=5800, shape="settle"), [(0, 5000)])
    assert out.t_ms == 5000, out.t_ms
    print("ok  test_moment_outside_every_span_snaps_to_the_nearest_edge")


def test_moment_picks_the_nearest_of_several_non_speech_spans():
    out = _clamp_moment(_MomentOut(t_ms=2500, shape="both"), [(0, 1000), (2000, 3000), (5000, 6000)])
    assert out.t_ms == 2500, out.t_ms
    print("ok  test_moment_picks_the_nearest_of_several_non_speech_spans")


def test_no_non_speech_spans_at_all_drops_the_moment():
    assert _clamp_moment(_MomentOut(t_ms=500, shape="both"), []) is None
    print("ok  test_no_non_speech_spans_at_all_drops_the_moment")


def test_find_enclosing_span_exact_match():
    assert _find_enclosing_span(2500, [(2000, 3000)]) == (2000, 3000)
    print("ok  test_find_enclosing_span_exact_match")


def test_find_enclosing_span_nearest_when_outside_all():
    assert _find_enclosing_span(9000, [(0, 1000), (5000, 6000)]) == (5000, 6000)
    print("ok  test_find_enclosing_span_nearest_when_outside_all")


def test_per_call_schema_is_flat_moments_only_no_file_wrapper():
    """Section 2.5: each call is scoped to one file (named in the prompt,
    not echoed by the model) -- the wire schema is flat moments only, no
    files/file_id wrapper."""
    assert set(_Pass1FileSchema.model_fields.keys()) == {"moments"}
    assert set(_MomentOut.model_fields.keys()) == {"t_ms", "shape", "summary"}
    print("ok  test_per_call_schema_is_flat_moments_only_no_file_wrapper")


def test_task_text_names_the_one_file_it_scopes_to():
    text = _task_text("f1", "clip one.mp4")
    assert "FILE f1" in text and "clip one.mp4" in text
    print("ok  test_task_text_names_the_one_file_it_scopes_to")


# --------------------------------------------------------------------------
# run_pass1 -- per-file fan-out / merge / isolation / usage (mocked model)
# --------------------------------------------------------------------------

def test_n_files_yield_n_calls_merged_in_file_rows_order():
    """N files -> N calls; merged MomentPlan has one FilePlan per file in
    file_rows order regardless of completion order (each file's moment
    encodes its own id, so a mis-order can't pass by accident)."""
    calls_seen: list = []
    returns = {
        fid: ({"moments": [{"t_ms": int(fid[1:]) * 1000, "shape": "both", "summary": fid}]},
              {"input_tokens": 10, "output_tokens": 5})
        for fid, _n, _d in _FILE_ROWS
    }
    with _patch_gemini(_fake_complete_gemini(returns, calls_seen)):
        plan, usage = run_pass1(_FILE_ROWS, _SPANS, {}, cached_content="cachedContents/shared")

    assert sorted(calls_seen) == ["f1", "f2", "f3"]          # exactly one call per file
    assert [f.file_id for f in plan.files] == ["f1", "f2", "f3"]  # deterministic order
    assert [f.flags[0].summary for f in plan.files] == ["f1", "f2", "f3"]
    assert usage == {"input_tokens": 30, "output_tokens": 15}
    print("ok  test_n_files_yield_n_calls_merged_in_file_rows_order")


def test_a_single_files_failure_aborts_the_whole_run():
    """NO FALLBACK: a single file's persistent Pass-1 failure propagates out
    of run_pass1 and aborts the whole project -- it is no longer masked as an
    empty FilePlan (which would silently hide that file's missing cuts)."""
    returns = {
        "f1": ({"moments": [{"t_ms": 1000}]}, {"output_tokens": 1}),
        "f2": RuntimeError("schema validation failed twice: boom"),
        "f3": ({"moments": [{"t_ms": 3000}]}, {"output_tokens": 1}),
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        try:
            run_pass1(_FILE_ROWS, _SPANS, {}, cached_content=None)
        except RuntimeError as e:
            assert "boom" in str(e), e
        else:
            raise AssertionError("expected run_pass1 to propagate the per-file failure")
    print("ok  test_a_single_files_failure_aborts_the_whole_run")


def test_a_file_with_zero_moments_yields_an_empty_fileplan_not_an_error():
    returns = {
        "f1": ({"moments": [{"t_ms": 1000}]}, {"output_tokens": 1}),
        "f2": ({"moments": []}, {"output_tokens": 1}),   # legitimately nothing
        "f3": ({"moments": [{"t_ms": 3000}]}, {"output_tokens": 1}),
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        plan, usage = run_pass1(_FILE_ROWS, _SPANS, {}, cached_content=None)

    by_id = {f.file_id: f for f in plan.files}
    assert by_id["f2"].flags == []
    assert len(by_id["f1"].flags) == 1 and len(by_id["f3"].flags) == 1
    assert usage == {"output_tokens": 3}    # the zero-moment call still counts
    print("ok  test_a_file_with_zero_moments_yields_an_empty_fileplan_not_an_error")


def test_moment_clamping_still_applies_through_the_full_call():
    """A model t_ms outside the file's non-speech spans snaps to the nearest
    span edge -- the defensive clamp still runs on every per-file call."""
    returns = {
        "f1": ({"moments": [{"t_ms": 999999, "shape": "settle", "summary": "way past the end"}]},
               {"output_tokens": 1}),
        "f2": ({"moments": []}, {}),
        "f3": ({"moments": []}, {}),
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        plan, _usage = run_pass1(_FILE_ROWS, _SPANS, {}, cached_content=None)

    f1 = next(f for f in plan.files if f.file_id == "f1")
    assert f1.flags[0].t_ms == 10000, f1.flags[0].t_ms   # snapped to f1's span end
    assert f1.flags[0].shape == "settle" and f1.flags[0].summary == "way past the end"
    print("ok  test_moment_clamping_still_applies_through_the_full_call")


def test_usage_is_summed_across_files_with_missing_keys_as_zero():
    returns = {
        "f1": ({"moments": [{"t_ms": 1000}]}, {"input_tokens": 100, "output_tokens": 10}),
        "f2": ({"moments": [{"t_ms": 2000}]}, {"input_tokens": 200, "cache_read_input_tokens": 3}),
        "f3": ({"moments": [{"t_ms": 3000}]}, {}),   # a call that reported no usage at all
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        _plan, usage = run_pass1(_FILE_ROWS, _SPANS, {}, cached_content=None)

    assert usage == {"input_tokens": 300, "output_tokens": 10, "cache_read_input_tokens": 3}
    print("ok  test_usage_is_summed_across_files_with_missing_keys_as_zero")


def test_every_file_failing_aborts_the_whole_run():
    """NO FALLBACK: when every file's call fails, run_pass1 raises rather than
    returning all-empty plans (which would look like a project with no cuts)."""
    returns = {fid: RuntimeError("boom") for fid, _n, _d in _FILE_ROWS}
    with _patch_gemini(_fake_complete_gemini(returns)):
        try:
            run_pass1(_FILE_ROWS, _SPANS, {}, cached_content=None)
        except RuntimeError as e:
            assert "boom" in str(e), e
        else:
            raise AssertionError("expected run_pass1 to raise when a file fails")
    print("ok  test_every_file_failing_aborts_the_whole_run")


# --------------------------------------------------------------------------
# run_pass1 -- video mode (pass1_video_input.plan.md section 5, mocked model)
# --------------------------------------------------------------------------

# 2 segments: orig [0,5000) -> sub[0,5000); orig [8000,10000) -> sub[5000,7000).
# The internal join sits at sub_start=5000 (segment 1's own start).
_VIDEO_SEGMENTS = [(0, 5000, 0), (8000, 10000, 5000)]
_VIDEO_HANDLE = VideoHandle(file_uri="files/abc123", segments=_VIDEO_SEGMENTS)


def _fake_complete_gemini_capturing(returns_by_file, calls_info):
    """Like _fake_complete_gemini but also records {file_id: {system,
    blocks, kwargs}} so video-mode tests can inspect what was actually
    sent, not just the parsed result."""
    def fake(system, blocks, schema, **kwargs):
        assert schema is _Pass1FileSchema, schema
        file_id = next(fid for fid in returns_by_file if f"FILE {fid} " in system)
        calls_info[file_id] = {"system": system, "blocks": blocks, "kwargs": kwargs}
        spec = returns_by_file[file_id]
        if isinstance(spec, Exception):
            raise spec
        data, usage = spec
        return _FakeCompletion(data, usage)
    return fake


def test_video_handle_sends_a_video_file_block_not_frame_blocks():
    calls_info: dict = {}
    returns = {"f1": ({"moments": []}, {})}
    with _patch_gemini(_fake_complete_gemini_capturing(returns, calls_info)):
        run_pass1([_FILE_ROWS[0]], _SPANS, {}, cached_content=None, video_by_file={"f1": _VIDEO_HANDLE})

    info = calls_info["f1"]
    assert len(info["blocks"]) == 1
    assert info["blocks"][0]["type"] == "video_file"
    assert info["blocks"][0]["file_uri"] == "files/abc123"
    assert "ONE CONTINUOUS CLIP" in info["system"]
    print("ok  test_video_handle_sends_a_video_file_block_not_frame_blocks")


def test_video_mode_remaps_sub_clip_t_ms_to_original_file_ms():
    # sub_ms=100 -> segment 0 (sub_start=0) -> orig = 0+100 = 100
    # sub_ms=5300 -> segment 1 (sub_start=5000) -> orig = 8000+300 = 8300
    returns = {
        "f1": ({"moments": [
            {"t_ms": 100, "shape": "both", "summary": "early"},
            {"t_ms": 5300, "shape": "settle", "summary": "later"},
        ]}, {}),
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        plan, _usage = run_pass1([_FILE_ROWS[0]], _SPANS, {}, cached_content=None,
                                 video_by_file={"f1": _VIDEO_HANDLE})

    t_by_summary = {f.summary: f.t_ms for f in plan.files[0].flags}
    assert t_by_summary == {"early": 100, "later": 8300}, t_by_summary
    print("ok  test_video_mode_remaps_sub_clip_t_ms_to_original_file_ms")


def test_video_mode_drops_moments_landing_on_a_join_artifact():
    returns = {
        "f1": ({"moments": [
            {"t_ms": 5000, "shape": "both", "summary": "on the seam"},
            {"t_ms": 2500, "shape": "both", "summary": "well inside segment 0"},
        ]}, {}),
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        plan, _usage = run_pass1([_FILE_ROWS[0]], _SPANS, {}, cached_content=None,
                                 video_by_file={"f1": _VIDEO_HANDLE})

    summaries = [f.summary for f in plan.files[0].flags]
    assert summaries == ["well inside segment 0"], summaries
    print("ok  test_video_mode_drops_moments_landing_on_a_join_artifact")


def test_video_mode_drops_out_of_range_moments():
    total = 7000  # sum of both segments' lengths -- exactly at/past the end maps nowhere
    returns = {
        "f1": ({"moments": [
            {"t_ms": total + 500, "shape": "both", "summary": "past the end"},
            {"t_ms": 1000, "shape": "both", "summary": "in range"},
        ]}, {}),
    }
    with _patch_gemini(_fake_complete_gemini(returns)):
        plan, _usage = run_pass1([_FILE_ROWS[0]], _SPANS, {}, cached_content=None,
                                 video_by_file={"f1": _VIDEO_HANDLE})

    summaries = [f.summary for f in plan.files[0].flags]
    assert summaries == ["in range"], summaries
    print("ok  test_video_mode_drops_out_of_range_moments")


def test_video_mode_still_clamps_the_remapped_ms_to_non_speech_spans():
    # f1's non-speech span (_SPANS) is [(0, 10000)]; this handle remaps
    # sub_ms=1200 -> orig 9000+1200=10200, just past the span -- confirms
    # _clamp_moment still runs AFTER the remap, not just for the frames path.
    handle = VideoHandle(file_uri="files/x", segments=[(9000, 20000, 0)])
    returns = {"f1": ({"moments": [{"t_ms": 1200, "shape": "settle", "summary": "past f1's span"}]}, {})}
    with _patch_gemini(_fake_complete_gemini(returns)):
        plan, _usage = run_pass1([_FILE_ROWS[0]], _SPANS, {}, cached_content=None,
                                 video_by_file={"f1": handle})

    assert plan.files[0].flags[0].t_ms == 10000, plan.files[0].flags[0].t_ms  # snapped to span end
    print("ok  test_video_mode_still_clamps_the_remapped_ms_to_non_speech_spans")


def test_a_file_absent_from_video_by_file_falls_back_to_frames_per_file():
    """f1 has a video handle (video path); f2/f3 don't (frames path) -- one
    run_pass1 call exercises both, proving the choice is genuinely per-file,
    not project-wide."""
    calls_info: dict = {}
    returns = {
        "f1": ({"moments": [{"t_ms": 100}]}, {}),
        "f2": ({"moments": [{"t_ms": 2000}]}, {}),
        "f3": ({"moments": [{"t_ms": 3000}]}, {}),
    }
    with _patch_gemini(_fake_complete_gemini_capturing(returns, calls_info)):
        plan, _usage = run_pass1(_FILE_ROWS, _SPANS, {}, cached_content=None,
                                 video_by_file={"f1": _VIDEO_HANDLE})

    assert calls_info["f1"]["blocks"][0]["type"] == "video_file"
    assert calls_info["f2"]["blocks"][0]["type"] == "text"   # build_frame_blocks' header block
    assert calls_info["f3"]["blocks"][0]["type"] == "text"
    by_id = {f.file_id: f for f in plan.files}
    assert by_id["f1"].flags[0].t_ms == 100    # remapped via segment 0
    assert by_id["f2"].flags[0].t_ms == 2000   # frames path -- no remap
    print("ok  test_a_file_absent_from_video_by_file_falls_back_to_frames_per_file")


def main():
    test_moment_fully_inside_a_non_speech_span_is_unchanged()
    test_moment_outside_every_span_snaps_to_the_nearest_edge()
    test_moment_picks_the_nearest_of_several_non_speech_spans()
    test_no_non_speech_spans_at_all_drops_the_moment()
    test_find_enclosing_span_exact_match()
    test_find_enclosing_span_nearest_when_outside_all()
    test_per_call_schema_is_flat_moments_only_no_file_wrapper()
    test_task_text_names_the_one_file_it_scopes_to()
    test_n_files_yield_n_calls_merged_in_file_rows_order()
    test_a_single_files_failure_aborts_the_whole_run()
    test_a_file_with_zero_moments_yields_an_empty_fileplan_not_an_error()
    test_moment_clamping_still_applies_through_the_full_call()
    test_usage_is_summed_across_files_with_missing_keys_as_zero()
    test_every_file_failing_aborts_the_whole_run()
    test_video_handle_sends_a_video_file_block_not_frame_blocks()
    test_video_mode_remaps_sub_clip_t_ms_to_original_file_ms()
    test_video_mode_drops_moments_landing_on_a_join_artifact()
    test_video_mode_drops_out_of_range_moments()
    test_video_mode_still_clamps_the_remapped_ms_to_non_speech_spans()
    test_a_file_absent_from_video_by_file_falls_back_to_frames_per_file()
    print("\nall vcut pass1 tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
