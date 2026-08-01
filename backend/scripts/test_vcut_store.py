"""
Pure unit tests for the non-DB parts of app.services.vcut.store --
build_cut_records and its small helpers. The DB-touching functions
(persist_seam_and_plan/load_seam_and_plan/copy_prior_speech_cuts) need a
live Postgres connection and are exercised via the vcut smoke run instead,
matching this codebase's convention (see test_ingest_store.py).

Run:  .venv/bin/python scripts/test_vcut_store.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.resolve import ResolvedCut  # noqa: E402
from app.services.vcut.store import (  # noqa: E402
    _energy_grade, _mean_in_range, _short_label, build_cut_records,
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


def main():
    test_energy_grade_buckets()
    test_short_label_truncates_to_max_words()
    test_mean_in_range_averages_the_hop_slice()
    test_mean_in_range_empty_track_is_zero()
    test_build_cut_records_shape_and_channel()
    test_build_cut_records_missing_seam_falls_back_to_zero_quality()
    print("\nall vcut store tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
