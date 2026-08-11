#!/usr/bin/env python3
"""Tests for the shared HEAL pass (l3.arrange.heal_adjacent_cuts) + keep_spans
expansion (l3.act._segments_from_cut). Healing runs live on BOTH composition
paths: the auto-assembly/brain path (observe.resolve_doc) and the manual/SNAP
path (put_document). Adjacent same-source contiguous main-line cuts merge into
one continuous segment, while real intra-clip jumps, a cut's own keep_spans
jump-cuts, and deliberately tightened (hard_seam) pauses stay separate. Run:
    PYTHONPATH=. .venv/bin/python scripts/test_arrange.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import act, arrange, layers  # noqa: E402
from app.services.l3.arrange import (  # noqa: E402
    HEAL_GAP_MS_DEFAULT,
    ResolvedCut,
    heal_adjacent_cuts,
    _remap_seam_ops_list,
)

FID_A = "aaaaaaaa-0000-0000-0000-000000000000"
FID_B = "bbbbbbbb-0000-0000-0000-000000000000"


def _seg(file_id, in_ms, out_ms, *, channel="said", **extra):
    """A main-line segment as act.place would emit it (pre-heal)."""
    seg = {"seg_id": f"x{in_ms}", "file_id": file_id, "in_ms": in_ms, "out_ms": out_ms,
           "axis": "speech" if channel == "said" else "any",
           "content": f"{in_ms}-{out_ms}", "mute": None,
           "ref": None, "level": "balanced"}
    seg.update(extra)
    return seg


def _heal(segments, *, gap_ms=HEAL_GAP_MS_DEFAULT, reindex=True):
    return heal_adjacent_cuts(segments, gap_ms=gap_ms, reindex=reindex)


def _rc(file_id, in_ms, out_ms, *, channel="said", keep_spans=None, level="balanced"):
    return ResolvedCut(
        file_id=file_id, src_in_ms=in_ms, src_out_ms=out_ms, keep_spans=keep_spans,
        channel=channel, label=f"{in_ms}-{out_ms}", track=arrange._MAIN_TRACK,
        from_ms=None, reason="", ref="", level=level,
    )


def test_contiguous_same_clip_heals():
    """Two adjacent slices of one continuous shot -> ONE segment."""
    segs, merged = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 2000, 4000)])
    assert len(segs) == 1, segs
    assert (segs[0]["in_ms"], segs[0]["out_ms"]) == (0, 4000), segs
    assert merged == {"x2000": "a000"}, merged
    print("ok  contiguous same-clip cuts heal into one segment")


def test_tiny_gap_demo_case_heals():
    """The demo regression: a 130ms source drop (< 200) between two adjacent
    spine segments of the same file (a002/a003, run 6f31d648) -> ONE segment
    that plays the bridged gap straight through."""
    segs, _merged = _heal([_seg(FID_A, 0, 60805), _seg(FID_A, 60935, 70000)])
    assert len(segs) == 1, segs
    assert segs[0]["out_ms"] == 70000, segs           # bridged gap is played
    print("ok  130ms micro-gap (demo a002/a003) heals into one continuous take")


def test_big_gap_stays_separate():
    """A gap well beyond the threshold is a real cut -> NOT healed."""
    segs, merged = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 2500, 4000)])
    assert len(segs) == 2, segs
    assert merged == {}, merged
    print("ok  big-gap (500ms > 200) stays separate")


def test_just_over_threshold_stays_separate():
    """gap == gap_ms+1 must stay split (boundary test)."""
    gap = HEAL_GAP_MS_DEFAULT + 1
    segs, _merged = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 2000 + gap, 4000)])
    assert len(segs) == 2, segs
    # exactly at the threshold heals
    segs2, _m2 = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 2000 + HEAL_GAP_MS_DEFAULT, 4000)])
    assert len(segs2) == 1, segs2
    print("ok  just-over-threshold stays split; exactly-at-threshold heals")


def test_overlapping_same_clip_heals_to_union():
    """Overlapping source spans heal to their union (no double-played footage)."""
    segs, _merged = _heal([_seg(FID_A, 1000, 3000), _seg(FID_A, 2000, 4000)])
    assert len(segs) == 1, segs
    assert (segs[0]["in_ms"], segs[0]["out_ms"]) == (1000, 4000), segs
    print("ok  overlapping same-clip cuts heal to their union")


def test_intra_clip_jump_stays_separate():
    """Two DISTANT slices of the same clip are an intentional jump -> NOT healed."""
    segs, _merged = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 30000, 32000)])
    assert len(segs) == 2, segs
    print("ok  intra-clip jump (distant slices) stays separate")


def test_different_clips_never_heal():
    """Adjacency across DIFFERENT clips is always a real cut."""
    segs, _merged = _heal([_seg(FID_A, 0, 2000), _seg(FID_B, 0, 2000)])
    assert len(segs) == 2, segs
    print("ok  different clips never heal")


def test_keep_spans_jumpcut_survives():
    """A cut's own windup|payoff keep_spans expand into separate segments (act),
    and being non-contiguous they SURVIVE the heal (not stitched back shut)."""
    cut = _rc(FID_A, 0, 8000, channel="done", keep_spans=[(0, 1000), (6000, 8000)])
    segs, _merged = _heal(act._segments_from_cut(cut))
    assert len(segs) == 2, segs
    assert (segs[0]["in_ms"], segs[0]["out_ms"]) == (0, 1000), segs
    assert (segs[1]["in_ms"], segs[1]["out_ms"]) == (6000, 8000), segs
    print("ok  keep_spans jump-cut survives (not healed shut)")


def test_healed_axis_is_speech_if_either_side_is():
    """A healed run is marked speech when any merged slice carried audio, so
    downstream coverage knows the audio is load-bearing."""
    segs, _merged = _heal([
        _seg(FID_A, 0, 2000, channel="shown"),
        _seg(FID_A, 2000, 4000, channel="said"),
    ])
    assert len(segs) == 1, segs
    assert segs[0]["axis"] == "speech", segs[0]
    print("ok  healed axis is speech if either side is speech")


def test_hard_seam_blocks_heal():
    """A pair within gap_ms but with hard_seam=True on the SECOND segment (a
    deliberately tightened pause) is NOT merged -- tighten protection."""
    segs, merged = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 2050, 4000, hard_seam=True)])
    assert len(segs) == 2, segs
    assert merged == {}, merged
    print("ok  hard_seam guard blocks healing a tightened pause")


def test_different_audio_coupling_not_healed():
    """Two contiguous same-file segments with different audio_file_id stay
    split -- merging would silently drop one side's routing."""
    a = _seg(FID_A, 0, 2000, audio_file_id=FID_A)
    b = _seg(FID_A, 2000, 4000, audio_file_id=FID_B)
    segs, _merged = _heal([a, b])
    assert len(segs) == 2, segs
    # differing audio_offset_ms also blocks
    c = _seg(FID_A, 0, 2000, audio_file_id=FID_A, audio_offset_ms=0)
    d = _seg(FID_A, 2000, 4000, audio_file_id=FID_A, audio_offset_ms=500)
    segs2, _m2 = _heal([c, d])
    assert len(segs2) == 2, segs2
    print("ok  different audio coupling is not healed")


def test_id_policy_reindex_vs_preserve():
    """reindex=True re-issues a000,a001,...; reindex=False keeps predecessor ids
    and only drops absorbed ids. merged_map maps absorbed->survivor in both."""
    base = lambda: [_seg(FID_A, 0, 2000), _seg(FID_A, 2000, 4000), _seg(FID_B, 5000, 6000)]
    segs_r, merged_r = _heal(base(), reindex=True)
    assert [s["seg_id"] for s in segs_r] == ["a000", "a001"], segs_r
    assert merged_r == {"x2000": "a000"}, merged_r

    segs_p, merged_p = _heal(base(), reindex=False)
    # the survivor keeps its own id "x0"; the untouched FID_B keeps "x5000"
    assert [s["seg_id"] for s in segs_p] == ["x0", "x5000"], segs_p
    assert merged_p == {"x2000": "x0"}, merged_p
    print("ok  id policy: reindex vs preserve, merged_map in both")


def test_idempotent():
    """A second heal pass over its own output is a no-op with an empty merged_map."""
    once, _m1 = _heal([_seg(FID_A, 0, 2000), _seg(FID_A, 2000, 4000),
                       _seg(FID_A, 30000, 32000)])
    twice, m2 = _heal(list(once))
    assert [(s["in_ms"], s["out_ms"]) for s in once] == \
        [(s["in_ms"], s["out_ms"]) for s in twice], (once, twice)
    assert m2 == {}, m2
    print("ok  heal is idempotent (empty merged_map on re-run)")


def test_audio_stays_aligned_after_heal():
    """After healing two contiguous same-file segments, layers.resolve emits
    exactly ONE spine dialogue AudioLayer spanning the merged [in,out] with a
    source window that bridges the gap -- audio needs no separate handling."""
    healed, _merged = _heal([_seg(FID_A, 0, 60805), _seg(FID_A, 60935, 70000)])
    doc = {"timeline": healed, "operations": [],
           "format": {"aspect": "landscape"}}
    resolved = layers.resolve(doc, {FID_A: 200000})
    spine_audio = [a for a in resolved.audio_layers if a.kind == "spine"]
    assert len(spine_audio) == 1, spine_audio
    a = spine_audio[0]
    assert (a.src_in_ms, a.src_out_ms) == (0, 70000), (a.src_in_ms, a.src_out_ms)
    spine_video = [v for v in resolved.video_layers if v.kind == "spine"]
    assert len(spine_video) == 1, spine_video
    print("ok  audio stays aligned after heal (one bridged dialogue layer)")


def test_remap_seam_ops_list_drops_healed_away():
    """The shared op-remap helper drops a split_edit/crossfade whose seam was
    absorbed (id is a key in merged_map) and keeps ops on surviving seams."""
    ops = [
        {"op_id": "se1", "type": "split_edit", "seam_seg_id": "x2000"},   # absorbed
        {"op_id": "xf1", "type": "crossfade", "seam_seg_id": "x2000"},    # absorbed
        {"op_id": "se2", "type": "split_edit", "seam_seg_id": "x9999"},   # survives
        {"op_id": "lv1", "type": "level"},                                # untouched
    ]
    kept = _remap_seam_ops_list(ops, {"x2000": "x0"})
    kept_ids = [o["op_id"] for o in kept]
    assert kept_ids == ["se2", "lv1"], kept_ids
    print("ok  _remap_seam_ops_list drops healed-away seam ops (split_edit+crossfade)")


def test_seg_ids_unique_after_heal():
    segs, _merged = _heal([
        _seg(FID_A, 0, 2000), _seg(FID_A, 2000, 4000),   # heal -> 1
        _seg(FID_B, 0, 1000),                            # -> 1
        _seg(FID_A, 0, 1000),                            # -> 1
    ])
    ids = [s["seg_id"] for s in segs]
    assert len(ids) == len(set(ids)) == 3, ids
    print("ok  seg ids unique after heal")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall arrange tests passed")
