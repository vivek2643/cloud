#!/usr/bin/env python3
"""Tests for the WELD pass (l3.arrange._weld_segments) + keep_spans expansion
(l3.act._segments_from_cut). Welding now runs live in observe.resolve_doc: adjacent
same-clip source-contiguous main-line cuts merge into one continuous segment, while
real intra-clip jumps and a cut's own keep_spans jump-cuts stay separate. Run:
    PYTHONPATH=. .venv/bin/python scripts/test_arrange.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import act, arrange  # noqa: E402
from app.services.l3.arrange import ResolvedCut, _weld_segments  # noqa: E402

FID_A = "aaaaaaaa-0000-0000-0000-000000000000"
FID_B = "bbbbbbbb-0000-0000-0000-000000000000"


def _seg(file_id, in_ms, out_ms, *, channel="said"):
    """A main-line segment as act.place would emit it (pre-weld)."""
    return {"seg_id": f"x{in_ms}", "file_id": file_id, "in_ms": in_ms, "out_ms": out_ms,
            "axis": "speech" if channel == "said" else "any",
            "content": f"{in_ms}-{out_ms}", "mute": None,
            "ref": None, "level": "balanced"}


def _rc(file_id, in_ms, out_ms, *, channel="said", keep_spans=None, level="balanced"):
    return ResolvedCut(
        file_id=file_id, src_in_ms=in_ms, src_out_ms=out_ms, keep_spans=keep_spans,
        channel=channel, label=f"{in_ms}-{out_ms}", track=arrange._MAIN_TRACK,
        from_ms=None, reason="", ref="", level=level,
    )


def test_contiguous_same_clip_welds():
    """Two adjacent slices of one continuous shot -> ONE segment."""
    segs = _weld_segments([_seg(FID_A, 0, 2000), _seg(FID_A, 2000, 4000)])
    assert len(segs) == 1, segs
    assert (segs[0]["in_ms"], segs[0]["out_ms"]) == (0, 4000), segs
    print("ok  contiguous same-clip cuts weld into one segment")


def test_near_touching_within_tolerance_welds():
    """A sub-frame gap (<= _WELD_TOL_MS) still reads as continuous -> welded."""
    gap = arrange._WELD_TOL_MS - 1
    segs = _weld_segments([_seg(FID_A, 0, 2000), _seg(FID_A, 2000 + gap, 4000)])
    assert len(segs) == 1, segs
    assert segs[0]["out_ms"] == 4000, segs
    print("ok  near-touching (within tolerance) welds")


def test_overlapping_same_clip_welds_to_union():
    """Overlapping source spans weld to their union (no double-played footage)."""
    segs = _weld_segments([_seg(FID_A, 1000, 3000), _seg(FID_A, 2000, 4000)])
    assert len(segs) == 1, segs
    assert (segs[0]["in_ms"], segs[0]["out_ms"]) == (1000, 4000), segs
    print("ok  overlapping same-clip cuts weld to their union")


def test_intra_clip_jump_stays_separate():
    """Two DISTANT slices of the same clip are an intentional jump -> NOT welded."""
    segs = _weld_segments([_seg(FID_A, 0, 2000), _seg(FID_A, 30000, 32000)])
    assert len(segs) == 2, segs
    print("ok  intra-clip jump (distant slices) stays separate")


def test_different_clips_never_weld():
    """Adjacency across DIFFERENT clips is always a real cut."""
    segs = _weld_segments([_seg(FID_A, 0, 2000), _seg(FID_B, 0, 2000)])
    assert len(segs) == 2, segs
    print("ok  different clips never weld")


def test_keep_spans_jumpcut_survives():
    """A cut's own windup|payoff keep_spans expand into separate segments (act),
    and being non-contiguous they SURVIVE the weld (not stitched back shut)."""
    cut = _rc(FID_A, 0, 8000, channel="done",
              keep_spans=[{"in_ms": 0, "out_ms": 1000}, {"in_ms": 6000, "out_ms": 8000}])
    segs = _weld_segments(act._segments_from_cut(cut))
    assert len(segs) == 2, segs
    assert (segs[0]["in_ms"], segs[0]["out_ms"]) == (0, 1000), segs
    assert (segs[1]["in_ms"], segs[1]["out_ms"]) == (6000, 8000), segs
    print("ok  keep_spans jump-cut survives (not welded shut)")


def test_welded_axis_is_speech_if_either_side_is():
    """A welded run is marked speech when any merged slice carried audio, so
    downstream coverage knows the audio is load-bearing."""
    segs = _weld_segments([
        _seg(FID_A, 0, 2000, channel="shown"),
        _seg(FID_A, 2000, 4000, channel="said"),
    ])
    assert len(segs) == 1, segs
    assert segs[0]["axis"] == "speech", segs[0]
    print("ok  welded axis is speech if either side is speech")


def test_seg_ids_unique_after_weld():
    segs = _weld_segments([
        _seg(FID_A, 0, 2000), _seg(FID_A, 2000, 4000),   # weld -> 1
        _seg(FID_B, 0, 1000),                            # -> 1
        _seg(FID_A, 0, 1000),                            # -> 1
    ])
    ids = [s["seg_id"] for s in segs]
    assert len(ids) == len(set(ids)) == 3, ids
    print("ok  seg ids unique after weld")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall arrange tests passed")
