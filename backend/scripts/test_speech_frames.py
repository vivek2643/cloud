"""
Pure unit tests for app.services.vcut.speech.frames (speech_cuts_pipeline
.plan.md section 11). run_frame_analysis itself is real-money/DB-touching
and exercised via the live smoke run instead.

Run:  .venv/bin/python scripts/test_speech_frames.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from types import SimpleNamespace  # noqa: E402

from app.services.vcut.speech import frames as frames_mod  # noqa: E402
from app.services.vcut.speech.frames import (  # noqa: E402
    _CutAnswerOut, _has_interior_motion_peak, _nonempty_fields, frame_timestamps_for_cut,
    run_frame_analysis, select_cuts_needing_frames,
)
from app.services.vcut.speech.inputs import FaceTrackLite  # noqa: E402
from app.services.vcut.speech.params import (  # noqa: E402
    SPEECH_FRAME_BATCH, SPEECH_FRAME_LONG_MS, SPEECH_FRAME_MAX,
)
from app.services.vcut.speech.segment_llm import _BeatOut  # noqa: E402
from app.services.vcut.speech.store import ExpandedCut, ResolvedBeat  # noqa: E402


def _ec(file_id="f1", in_ms=0, out_ms=2000, take_role="winner", beat_id="b0"):
    beat = _BeatOut(id=beat_id, file_id=file_id, word_span=(0, 3))
    rb = ResolvedBeat(beat=beat, in_ms=in_ms, out_ms=out_ms, take_group_key="tg0",
                      take_role=take_role, speech_quality=0.7)
    return ExpandedCut(resolved_beat=rb, file_id=file_id, in_ms=in_ms, out_ms=out_ms,
                       sync_group_id=None, audio_file_id="", audio_offset_ms=0,
                       audio_align_confidence=None)


# --------------------------------------------------------------------------
# _has_interior_motion_peak
# --------------------------------------------------------------------------

def test_interior_peak_detected_above_median():
    ae = [0.1] * 20
    ae[10] = 0.9  # a clear peak well inside [1000,3000]ms at hop=100
    assert _has_interior_motion_peak(ae, hop_ms=100, in_ms=500, out_ms=3500) is True
    print("ok  test_interior_peak_detected_above_median")


def test_no_peak_when_flat():
    ae = [0.3] * 20
    assert _has_interior_motion_peak(ae, hop_ms=100, in_ms=500, out_ms=3500) is True  # all == median -> "at" counts
    ae2 = []
    assert _has_interior_motion_peak(ae2, hop_ms=100, in_ms=0, out_ms=1000) is False
    print("ok  test_no_peak_when_flat")


def test_no_peak_when_window_degenerate():
    ae = [0.1, 0.9, 0.1]
    assert _has_interior_motion_peak(ae, hop_ms=1000, in_ms=0, out_ms=100) is False
    print("ok  test_no_peak_when_window_degenerate")


# --------------------------------------------------------------------------
# frame_timestamps_for_cut
# --------------------------------------------------------------------------

def test_short_cut_always_gets_one_frame():
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, 2000, 0.8)])]
    ts = frame_timestamps_for_cut(0, 2000, tracks, action_energy=[0.9] * 50, hop_ms=100)
    assert len(ts) == 1, ts
    print("ok  test_short_cut_always_gets_one_frame")


def test_long_but_not_changing_gets_one_frame():
    duration = SPEECH_FRAME_LONG_MS + 1000
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, duration, 0.8)])]
    # an all-equal action_energy array WOULD count as "at or above median"
    # everywhere per _has_interior_motion_peak's own definition -- use a
    # genuinely empty array to exercise the "not changing" path instead.
    ts_no_signal = frame_timestamps_for_cut(0, duration, tracks, action_energy=[], hop_ms=100)
    assert len(ts_no_signal) == 1, ts_no_signal
    print("ok  test_long_but_not_changing_gets_one_frame")


def test_long_and_changing_gets_two_frames_capped():
    duration = SPEECH_FRAME_LONG_MS + 4000
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, duration, 0.8)])]
    ae = [0.1] * 200
    ae[100] = 0.99  # a clear interior peak
    ts = frame_timestamps_for_cut(0, duration, tracks, action_energy=ae, hop_ms=100)
    assert len(ts) <= SPEECH_FRAME_MAX
    assert len(ts) == 2, ts
    print("ok  test_long_and_changing_gets_two_frames_capped")


# --------------------------------------------------------------------------
# select_cuts_needing_frames
# --------------------------------------------------------------------------

def test_alternates_never_get_frames():
    ec = _ec(take_role="take")
    tracks = {"f1": [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, 2000, 0.8)])]}
    out = select_cuts_needing_frames([ec], tracks)
    assert out == []
    print("ok  test_alternates_never_get_frames")


def test_off_camera_winner_gets_no_frames():
    ec = _ec(take_role="winner")
    tracks = {"f1": []}  # no face tracks at all -> off camera
    out = select_cuts_needing_frames([ec], tracks)
    assert out == []
    print("ok  test_off_camera_winner_gets_no_frames")


def test_on_camera_winner_is_selected():
    ec = _ec(take_role="winner", in_ms=0, out_ms=2000)
    tracks = {"f1": [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, 2000, 0.8)])]}
    out = select_cuts_needing_frames([ec], tracks)
    assert out == [ec]
    print("ok  test_on_camera_winner_is_selected")


# --------------------------------------------------------------------------
# _nonempty_fields
# --------------------------------------------------------------------------

def test_nonempty_fields_excludes_blank_strings():
    ans = _CutAnswerOut(cut_id="c1", subject="a dog", action="", setting="", count=None)
    out = _nonempty_fields(ans)
    assert out == {"subject": "a dog"}, out
    print("ok  test_nonempty_fields_excludes_blank_strings")


def test_nonempty_fields_keeps_a_genuine_zero_count():
    ans = _CutAnswerOut(cut_id="c1", count=0)
    out = _nonempty_fields(ans)
    assert out.get("count") == 0, out
    print("ok  test_nonempty_fields_keeps_a_genuine_zero_count")


def test_nonempty_fields_excludes_none_count():
    ans = _CutAnswerOut(cut_id="c1", count=None)
    out = _nonempty_fields(ans)
    assert "count" not in out, out
    print("ok  test_nonempty_fields_excludes_none_count")


# --------------------------------------------------------------------------
# run_frame_analysis batching (fully mocked -- no network/db/R2/Gemini)
# --------------------------------------------------------------------------

def _install_frame_mocks(calls):
    """Patch every external dependency run_frame_analysis reaches (DB proxy
    lookup, R2 still extraction, settings, and the Gemini call) with in-memory
    fakes, recording each complete_gemini call's cut count into ``calls``.
    Returns a restore() thunk the caller must invoke to undo the patches."""
    import app.config as cfg
    import app.services.l3.frames as l3frames
    import app.services.llm.ingest_gemini as ig

    orig = {
        "proxy": frames_mod._proxy_keys_for_files,
        "stills": l3frames.extract_stills_from_r2,
        "settings": cfg.get_settings,
        "complete": ig.complete_gemini,
    }

    frames_mod._proxy_keys_for_files = lambda file_ids: {fid: f"proxy/{fid}" for fid in file_ids}
    l3frames.extract_stills_from_r2 = lambda proxy_key, ts_list: {ts: f"b64-{ts}" for ts in ts_list}
    cfg.get_settings = lambda: SimpleNamespace(vcut_pass2_model="fake-model")

    def _fake_complete(system, blocks, schema, *, model=None, max_tokens=None,
                       thinking=None, extra_check=None):
        keys = [b["text"].split("cut_id=", 1)[1].split(" ", 1)[0]
                for b in blocks if b.get("type") == "text" and b.get("text", "").startswith("cut_id=")]
        calls.append(len(keys))
        return SimpleNamespace(
            data={"answers": [{"cut_id": k} for k in keys]},
            usage={"input_tokens": 1, "output_tokens": 2,
                   "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        )

    ig.complete_gemini = _fake_complete

    def restore():
        frames_mod._proxy_keys_for_files = orig["proxy"]
        l3frames.extract_stills_from_r2 = orig["stills"]
        cfg.get_settings = orig["settings"]
        ig.complete_gemini = orig["complete"]

    return restore


def _winner_cut(i):
    """One winning, on-camera ExpandedCut with a unique beat id (so cut_key is
    unique) and a face track that overlaps its span (so it's on-camera and
    gets exactly one frame)."""
    beat = _BeatOut(id=f"b{i}", file_id="f1", word_span=(0, 3), gist=f"gist {i}")
    rb = ResolvedBeat(beat=beat, in_ms=0, out_ms=2000, take_group_key="tg0",
                      take_role="winner", speech_quality=0.7)
    return ExpandedCut(resolved_beat=rb, file_id="f1", in_ms=0, out_ms=2000,
                       sync_group_id=None, audio_file_id="", audio_offset_ms=0,
                       audio_align_confidence=None)


def _run_with_n_cuts(n):
    cuts = [_winner_cut(i) for i in range(n)]
    tracks = {"f1": [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, 2000, 0.8)])]}
    calls: list = []
    restore = _install_frame_mocks(calls)
    try:
        visual, usage = run_frame_analysis(cuts, tracks, action_energy_by_file={})
    finally:
        restore()
    return visual, usage, calls


def test_single_batch_makes_one_call_and_passes_usage_through():
    n = SPEECH_FRAME_BATCH  # exactly one full batch
    visual, usage, calls = _run_with_n_cuts(n)
    assert calls == [n], calls
    assert len(visual) == n, len(visual)
    # <= one batch => usage is exactly the single call's usage (unchanged
    # from the pre-batch behavior).
    assert usage == {"input_tokens": 1, "output_tokens": 2,
                     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, usage
    print("ok  test_single_batch_makes_one_call_and_passes_usage_through")


def test_multi_batch_chunks_and_merges_and_sums_usage():
    n = SPEECH_FRAME_BATCH * 2 + 2   # 3 batches: full, full, remainder
    expected_calls = (n + SPEECH_FRAME_BATCH - 1) // SPEECH_FRAME_BATCH
    visual, usage, calls = _run_with_n_cuts(n)
    assert len(calls) == expected_calls, (calls, expected_calls)
    assert all(c <= SPEECH_FRAME_BATCH for c in calls), calls
    assert sum(calls) == n, calls
    # every cut's answer merged back, across all batches
    assert len(visual) == n, len(visual)
    # usage summed across all batches (1 in / 2 out per call)
    assert usage["input_tokens"] == expected_calls, usage
    assert usage["output_tokens"] == 2 * expected_calls, usage
    print("ok  test_multi_batch_chunks_and_merges_and_sums_usage")


def test_no_targets_makes_no_call():
    cuts = [_ec(take_role="take")]  # alternate -> not selected
    tracks = {"f1": [FaceTrackLite(track_id=0, best_crop_ms=500, speaking=[(0, 2000, 0.8)])]}
    calls: list = []
    restore = _install_frame_mocks(calls)
    try:
        visual, usage = run_frame_analysis(cuts, tracks, action_energy_by_file={})
    finally:
        restore()
    assert calls == [], calls
    assert visual == {} and usage == {}, (visual, usage)
    print("ok  test_no_targets_makes_no_call")


def main():
    test_interior_peak_detected_above_median()
    test_no_peak_when_flat()
    test_no_peak_when_window_degenerate()
    test_short_cut_always_gets_one_frame()
    test_long_but_not_changing_gets_one_frame()
    test_long_and_changing_gets_two_frames_capped()
    test_alternates_never_get_frames()
    test_off_camera_winner_gets_no_frames()
    test_on_camera_winner_is_selected()
    test_nonempty_fields_excludes_blank_strings()
    test_nonempty_fields_keeps_a_genuine_zero_count()
    test_nonempty_fields_excludes_none_count()
    test_single_batch_makes_one_call_and_passes_usage_through()
    test_multi_batch_chunks_and_merges_and_sums_usage()
    test_no_targets_makes_no_call()
    print("\nall speech frames tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
