"""
Pure unit tests for the non-DB parts of app.services.vcut.store --
build_cut_records and its small helpers. The DB-touching functions
(persist_seam_and_plan/load_seam_and_plan) need a
live Postgres connection and are exercised via the vcut smoke run instead,
matching this codebase's convention (see test_ingest_store.py).

Run:  .venv/bin/python scripts/test_vcut_store.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.resolve import ResolvedCut  # noqa: E402
from app.services.vcut.store import (  # noqa: E402
    _energy_grade, _mean_in_range, _short_label, build_cut_records, insert_video_cuts,
)


def _seam(n=101, hop_ms=100, S=None, action_energy=None):
    return {
        "hop_ms": hop_ms,
        "S": S if S is not None else [1.0] * n,
        "action_energy": action_energy if action_energy is not None else [0.0] * n,
        "frame_diff": [0.0] * n,
    }


def test_energy_grade_buckets():
    assert _energy_grade(0.0) == "calm"
    assert _energy_grade(0.32) == "calm"
    assert _energy_grade(0.33) == "medium"
    assert _energy_grade(0.65) == "medium"
    assert _energy_grade(0.66) == "high"
    assert _energy_grade(1.0) == "high"
    print("ok  test_energy_grade_buckets")


def test_short_label_truncates_to_max_words():
    assert _short_label("a b c d e f g h") == "a b c d e f"
    assert _short_label("one two") == "one two"
    assert _short_label("") == "clip"
    print("ok  test_short_label_truncates_to_max_words")


def test_mean_in_range_averages_the_hop_slice():
    track = [0.0, 1.0, 1.0, 0.0]
    assert _mean_in_range(100, 250, 100, track) == 1.0
    assert _mean_in_range(0, 400, 100, track) == 0.5
    print("ok  test_mean_in_range_averages_the_hop_slice")


def test_mean_in_range_empty_track_is_zero():
    assert _mean_in_range(0, 1000, 100, []) == 0.0
    print("ok  test_mean_in_range_empty_track_is_zero")


def test_build_cut_records_shape_and_channel():
    cuts = [ResolvedCut(file_id="f1", in_ms=1000, out_ms=3000, peak_ms=2000,
                        tag="build", summary="doing push-ups")]
    seam = {"f1": _seam(S=[0.8] * 101)}
    records = build_cut_records(cuts, seam)
    assert len(records) == 1
    r = records[0]
    assert r.kind == "video" and r.channel == "shown"
    assert r.file_id == "f1" and r.src_in_ms == 1000 and r.src_out_ms == 3000
    assert r.hero_ts_ms == 2000
    assert r.summary == "doing push-ups"
    assert r.label == "doing push-ups"
    assert abs(r.total_quality - 0.8) < 1e-9
    assert r.pace.min_ms == r.pace.natural_ms == r.pace.max_ms == 2000
    print("ok  test_build_cut_records_shape_and_channel")


def test_build_cut_records_missing_seam_falls_back_to_zero_quality():
    cuts = [ResolvedCut(file_id="missing", in_ms=0, out_ms=1000, peak_ms=500,
                        tag="both", summary="m")]
    records = build_cut_records(cuts, {})
    assert records[0].total_quality == 0.0
    print("ok  test_build_cut_records_missing_seam_falls_back_to_zero_quality")


# --------------------------------------------------------------------------
# reframe_vcut_geometry.plan.md: build_cut_records writes a fully-populated
# framing (subject_box + all three crops + rotation_deg) from the seam
# entry's own probed proxy dims -- absent dims (older run/probe failure)
# degrades to an empty framing, never a hard failure.
# --------------------------------------------------------------------------

def test_build_cut_records_writes_full_framing_when_dims_present():
    cuts = [ResolvedCut(file_id="f1", in_ms=1000, out_ms=3000, peak_ms=2000,
                        tag="build", summary="doing push-ups", subject_box=(0.4, 0.4, 0.2, 0.2))]
    seam = _seam(S=[0.8] * 101)
    seam["src_w"], seam["src_h"] = 1920, 1080
    records = build_cut_records(cuts, {"f1": seam})
    framing = records[0].framing
    assert framing["subject_box"] == [0.4, 0.4, 0.2, 0.2]
    assert framing["rotation_deg"] == 0.0
    for key in ("crop_16x9", "crop_9x16", "crop_1x1"):
        assert framing[key] is not None, framing
    print("ok  test_build_cut_records_writes_full_framing_when_dims_present")


def test_build_cut_records_no_subject_box_still_yields_centered_crops():
    cuts = [ResolvedCut(file_id="f1", in_ms=1000, out_ms=3000, peak_ms=2000,
                        tag="build", summary="m")]  # no subject_box
    seam = _seam(S=[0.8] * 101)
    seam["src_w"], seam["src_h"] = 1920, 1080
    records = build_cut_records(cuts, {"f1": seam})
    framing = records[0].framing
    assert framing["subject_box"] is None
    assert framing["crop_16x9"] is not None
    print("ok  test_build_cut_records_no_subject_box_still_yields_centered_crops")


def test_build_cut_records_missing_dims_yields_empty_framing():
    cuts = [ResolvedCut(file_id="f1", in_ms=1000, out_ms=3000, peak_ms=2000,
                        tag="build", summary="m", subject_box=(0.1, 0.1, 0.1, 0.1))]
    records = build_cut_records(cuts, {"f1": _seam(S=[0.8] * 101)})  # no src_w/src_h in this seam entry
    assert records[0].framing == {}
    print("ok  test_build_cut_records_missing_dims_yields_empty_framing")


# --------------------------------------------------------------------------
# insert_video_cuts -- vcut_pass2_video_specifics.plan.md section 7.3: build
# -> delete -> insert -> write each cut's own composed specifics, with the
# new ids zipped DIRECTLY against resolved (no hero-containment matching).
# Every DB touchpoint mocked.
# --------------------------------------------------------------------------

def test_insert_video_cuts_builds_deletes_inserts_and_writes_specifics():
    cuts = [
        ResolvedCut(file_id="f1", in_ms=0, out_ms=1000, peak_ms=500, tag="both",
                    summary="m1", specifics={"subject": "a dog"}),
        ResolvedCut(file_id="f1", in_ms=2000, out_ms=3000, peak_ms=2500, tag="both",
                    summary="m2", specifics={}),  # no specifics -- update must be skipped for this one
    ]
    with patch("app.services.vcut.store.delete_video_cuts_for_run") as delete_mock, \
         patch("app.services.l3.ingest_store.insert_cut_records", return_value=["id1", "id2"]) as insert_mock, \
         patch("app.services.l3.ingest_store.update_cut_scene_specifics") as update_mock:
        ids = insert_video_cuts("run1", cuts, {})

    assert ids == ["id1", "id2"]
    delete_mock.assert_called_once_with("run1")
    insert_mock.assert_called_once()
    assert insert_mock.call_args[0][0] == "run1"
    assert len(insert_mock.call_args[0][1]) == 2   # the built CutRecords
    update_mock.assert_called_once_with("id1", {"subject": "a dog"})
    print("ok  test_insert_video_cuts_builds_deletes_inserts_and_writes_specifics")


def test_insert_video_cuts_no_cut_has_specifics_writes_nothing():
    cuts = [ResolvedCut(file_id="f1", in_ms=0, out_ms=1000, peak_ms=500, tag="both", summary="m")]
    with patch("app.services.vcut.store.delete_video_cuts_for_run"), \
         patch("app.services.l3.ingest_store.insert_cut_records", return_value=["id1"]), \
         patch("app.services.l3.ingest_store.update_cut_scene_specifics") as update_mock:
        insert_video_cuts("run1", cuts, {})
    update_mock.assert_not_called()
    print("ok  test_insert_video_cuts_no_cut_has_specifics_writes_nothing")


def main():
    test_energy_grade_buckets()
    test_short_label_truncates_to_max_words()
    test_mean_in_range_averages_the_hop_slice()
    test_mean_in_range_empty_track_is_zero()
    test_build_cut_records_shape_and_channel()
    test_build_cut_records_missing_seam_falls_back_to_zero_quality()
    test_build_cut_records_writes_full_framing_when_dims_present()
    test_build_cut_records_no_subject_box_still_yields_centered_crops()
    test_build_cut_records_missing_dims_yields_empty_framing()
    test_insert_video_cuts_builds_deletes_inserts_and_writes_specifics()
    test_insert_video_cuts_no_cut_has_specifics_writes_nothing()
    print("\nall vcut store tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
