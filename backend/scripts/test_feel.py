#!/usr/bin/env python3
"""Tests for the feel simulator (pure; no DB/LLM).

Run:  PYTHONPATH=. python scripts/test_feel.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import feel  # noqa: E402


def _seg(seg_id, file_id, in_ms, out_ms, content="", axis="speech", ref=None):
    return {"seg_id": seg_id, "file_id": file_id, "in_ms": in_ms, "out_ms": out_ms,
            "content": content, "axis": axis, "ref": ref}


def test_empty_timeline():
    r = feel.simulate([])
    assert r.cuts == [] and r.total_ms == 0
    assert "empty" in r.narrate().lower()
    print("ok  empty timeline narrates gracefully")


def test_pace_and_totals():
    # 4 words over 2s -> 2.0 wps; 10 words over 2s -> 5.0 wps.
    tl = [
        _seg("a", "f1", 0, 2000, content="one two three four", ref="f1:m00"),
        _seg("b", "f1", 2000, 4000, content="a b c d e f g h i j", ref="f1:m01"),
    ]
    r = feel.simulate(tl)
    assert r.total_ms == 4000, r.total_ms
    assert r.cuts[0].pace_wps == 2.0 and r.cuts[1].pace_wps == 5.0, r.cuts
    assert r.avg_pace == 3.5, r.avg_pace
    print("ok  pace + totals computed from the timeline alone")


def test_video_cut_has_no_pace():
    tl = [_seg("a", "f1", 0, 3000, content="", axis="any", ref="f1:m00")]
    r = feel.simulate(tl, meta_by_ref={"f1:m00": {"channel": "shown"}})
    c = r.cuts[0]
    assert c.pace_wps == 0.0 and not c.is_speech and c.channel == "shown"
    print("ok  silent video cut carries no pace, channel from map")


def test_fast_run_flagged():
    # Five back-to-back sub-2s cuts -> a fast burst spanning cuts 1-5.
    tl = [_seg(f"s{i}", "f1", i * 1000, i * 1000 + 900,
               content="quick", ref=f"f1:m{i:02d}") for i in range(5)]
    r = feel.simulate(tl)
    txt = r.narrate().lower()
    assert "race" in txt and "1-5" in txt, txt
    print("ok  fast run of short cuts flagged with anchors")


def test_same_speaker_jump_cut_risk():
    tl = [
        _seg("a", "f1", 0, 3000, content="hello there friend", ref="f1:m00"),
        _seg("b", "f1", 3000, 6000, content="and one more thing", ref="f1:m01"),
        _seg("c", "f2", 6000, 9000, content="different person now", ref="f2:m00"),
    ]
    meta = {"f1:m00": {"speaker_person": "P0"}, "f1:m01": {"speaker_person": "P0"},
            "f2:m00": {"speaker_person": "P1"}}
    r = feel.simulate(tl, meta_by_ref=meta)
    txt = r.narrate().lower()
    assert "jump-cut risk" in txt and "1-2" in txt, txt
    print("ok  adjacent same-speaker cuts flagged as jump-cut risk")


def main():
    test_empty_timeline()
    test_pace_and_totals()
    test_video_cut_has_no_pace()
    test_fast_run_flagged()
    test_same_speaker_jump_cut_risk()
    print("\nall feel tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
