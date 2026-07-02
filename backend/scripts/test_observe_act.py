#!/usr/bin/env python3
"""Tests for the observe (senses) + act (verbs) engine -- pure, no DB/LLM.

Builds a real moment-tree map, hand-builds an EditContext (so no DB), and checks
that verbs mutate the document immutably + correctly and that the senses read it
back faithfully. Run:  PYTHONPATH=. python scripts/test_observe_act.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import act, layers, observe  # noqa: E402
from app.services.l3 import footage_map as fm  # noqa: E402
from app.services.l3.arrange import Placement, _MapIndex, _weld_segments  # noqa: E402


def _rung(level, in_ms, out_ms, text="", score=0.6):
    return {"level": level, "spans": [{"in_ms": in_ms, "out_ms": out_ms}],
            "in_ms": in_ms, "out_ms": out_ms, "play_ms": out_ms - in_ms,
            "text": text, "score": score}


def _cut(hero_id, in_ms, out_ms, label="", modality="speech", score=0.6, speaker="S0", ladder=None):
    return {"hero_id": hero_id, "file_id": "ffffffff-1111", "modality": modality,
            "channel": "said" if modality == "speech" else "shown",
            "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
            "play_ms": out_ms - in_ms, "keep_spans": None, "score": score,
            "speaker": speaker, "affordances": [modality], "flags": [], "ladder": ladder}


def _map():
    c0 = _cut("f:t0", 0, 4000, "we almost shut down", score=0.82, ladder=[
        _rung("broad", 0, 8000, "whole answer", 0.8),
        _rung("balanced", 0, 4000, "we almost shut down", 0.82),
        _rung("tight", 0, 2000, "we almost", 0.7),
        _rung("sharp", 500, 1500, "shut down", 0.7),
    ])
    c1 = _cut("f:t1", 4000, 8000, "one customer changed everything", score=0.78, ladder=[
        _rung("broad", 0, 8000, "whole answer", 0.8),
        _rung("balanced", 4000, 8000, "one customer changed everything", 0.78),
        _rung("tight", 4000, 6000, "one customer", 0.7),
    ])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [c0, c1])
    return {"clips": [tree]}


def _ctx(struct):
    return observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, valence_by_file={"ffffffff-1111": "tense"},
        dup_groups=[])


def _doc(struct, refs):
    """Build a starting document from (ref, level) main-line picks via the real
    act.place path, then weld -- exactly what observe.resolve_doc does on persist."""
    idx = _MapIndex(struct)
    doc = {"brief": {"aspect": "landscape"}, "format": {"aspect": "landscape"},
           "timeline": [], "operations": []}
    for ref, lv in refs:
        doc = act.place(doc, idx, ref, level=lv, channel="V1")
    doc["timeline"] = _weld_segments(doc["timeline"])
    return doc


def test_read_state_reports_cuts_and_feel():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced"), ("ffffffff:m01", "balanced")])
    st = observe.read_state(doc, _ctx(struct))
    # m00 (0-4000) welds with m01 (4000-8000) -> one continuous main-line segment.
    assert st["cut_count"] == 1, st
    assert st["channels"] == ["V1", "A1"], st["channels"]
    assert st["total_ms"] == 8000, st["total_ms"]
    assert "feel" in st and isinstance(st["feel"], str)
    print("ok  read_state reports cuts + channels + feel")


def test_place_adds_main_line_cut():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    idx = _MapIndex(struct)
    before = len(doc["timeline"])
    doc2 = act.place(doc, idx, "ffffffff:m01", level="balanced", channel="V1")
    assert len(doc2["timeline"]) == before + 1, doc2["timeline"]
    assert doc is not doc2 and "resolved" not in doc2      # immutable + stale dropped
    assert len(doc["timeline"]) == before                  # original untouched
    print("ok  place appends a main-line cut immutably")


def test_place_v2_cutaway_op():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    idx = _MapIndex(struct)
    doc2 = act.place(doc, idx, "ffffffff:m01", channel="V2", from_ms=500)
    ops = [o for o in doc2["operations"] if o["type"] == "place_video"]
    assert len(ops) == 1 and ops[0]["from_ms"] == 500, ops
    assert ops[0]["mute"] is True                          # silent cutaway by default
    print("ok  place V2 -> silent place_video cutaway")


def test_remove_and_move():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "tight"), ("ffffffff:m01", "tight")])
    ids = [s["seg_id"] for s in doc["timeline"]]
    assert len(ids) == 2, ids                              # tight cuts don't weld (non-contiguous)
    doc2 = act.remove(doc, ids[0])
    assert [s["seg_id"] for s in doc2["timeline"]] == [ids[1]]
    doc3 = act.move(doc, ids[1], 0)
    assert [s["seg_id"] for s in doc3["timeline"]] == [ids[1], ids[0]]
    print("ok  remove drops, move reorders")


def test_trim_and_set_audio():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "tight")])
    sid = doc["timeline"][0]["seg_id"]
    doc2 = act.trim(doc, sid, delta_in_ms=200)
    assert doc2["timeline"][0]["in_ms"] == doc["timeline"][0]["in_ms"] + 200
    # a trim that would invert the span is rejected (unchanged doc)
    assert act.trim(doc, sid, out_ms=0) is doc
    doc3 = act.set_audio(doc, sid, mute=True)
    assert doc3["timeline"][0]["mute"] is True
    print("ok  trim nudges span (guards inversion); set_audio mutes")


def test_tighten_reshapes_span():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    idx = _MapIndex(struct)
    sid = doc["timeline"][0]["seg_id"]
    doc2 = act.tighten(doc, idx, seg_id=sid, level="tight")
    seg = doc2["timeline"][0]
    assert (seg["in_ms"], seg["out_ms"]) == (0, 2000), seg   # tight variant of m00
    assert seg["level"] == "tight" and seg["seg_id"] == sid  # id preserved
    print("ok  tighten re-takes a cut at a tighter level")


def test_predict_length_under_tighten():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])   # 4000ms
    p = observe.predict(doc, _ctx(struct), set_level="tight")
    assert p["current_ms"] == 4000, p
    assert p["projected_ms"] == 2000, p                  # tight m00 = 2000ms
    assert p["delta_ms"] == -2000, p
    print("ok  predict projects length under a level change")


def test_validate_flags_bad_span():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    # Corrupt the span to exceed the source duration.
    doc["timeline"][0]["out_ms"] = 999999
    doc.pop("resolved", None)
    issues = observe.validate(doc, _ctx(struct))
    assert any("exceeds source" in i["message"] for i in issues), issues
    print("ok  validate flags an out-of-range span")


def _rung_multi(level, spans, text="", score=0.7):
    """A ladder rung whose take is SPLIT across >1 source span (a breath-excised
    jump-cut). footage_map turns this into a variant with keep_spans PAIRS -- the
    real production shape that a single-span fixture never exercises."""
    return {"level": level, "spans": [{"in_ms": a, "out_ms": b} for a, b in spans],
            "in_ms": spans[0][0], "out_ms": spans[-1][1],
            "play_ms": sum(b - a for a, b in spans), "text": text, "score": score}


def test_place_multispan_keep_spans_survives():
    """Regression: a moment whose chosen take is multi-span resolves keep_spans as
    [in,out] PAIRS. act.place must place one segment per span (not crash on a
    dict-shaped keep_span). This is the exact path that broke the live loop."""
    c0 = _cut("f:t0", 0, 3000, "jump cut line", ladder=[
        _rung_multi("balanced", [(0, 1000), (1500, 3000)], "jump cut line")])
    tree = fm.build_clip_tree("ffffffff-2222", {"name": "J", "duration_ms": 3000}, [c0])
    struct = {"clips": [tree]}
    idx = _MapIndex(struct)
    # The map really carries pairs, and resolve normalizes to (in, out) tuples.
    rc = idx.resolve(Placement(ref="ffffffff:m00", level="balanced"))
    assert rc is not None and rc.keep_spans == [(0, 1000), (1500, 3000)], rc.keep_spans
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    doc2 = act.place(doc, idx, "ffffffff:m00", level="balanced", channel="V1")
    assert doc2 is not doc, "place no-oped -- the multi-span crash is back"
    spans = [(s["in_ms"], s["out_ms"]) for s in doc2["timeline"]]
    assert spans == [(0, 1000), (1500, 3000)], spans
    print("ok  place survives multi-span keep_spans (pairs) -- the live-loop bug")


def test_norm_keep_spans_accepts_both_shapes():
    from app.services.l3.arrange import _norm_keep_spans
    assert _norm_keep_spans([[0, 1000], [1500, 3000]]) == [(0, 1000), (1500, 3000)]
    assert _norm_keep_spans([{"in_ms": 0, "out_ms": 1000}]) == [(0, 1000)]
    assert _norm_keep_spans(None) is None
    assert _norm_keep_spans([[500, 400]]) is None      # inverted span dropped
    print("ok  _norm_keep_spans canonicalizes pairs + dicts, drops junk")


def test_split_screen_adds_op_and_region():
    struct = _map()
    # One long welded main-line cut (0-8000ms) to place a split over.
    doc = _doc(struct, [("ffffffff:m00", "balanced"), ("ffffffff:m01", "balanced")])
    idx = _MapIndex(struct)
    doc2 = act.split_screen(doc, idx, "ffffffff:m01", template="split_h",
                            from_ms=1000, to_ms=3000)
    assert doc2 is not doc
    ops = [o for o in doc2["operations"] if o["type"] == "place_video"]
    assert len(ops) == 1 and ops[0]["from_ms"] == 1000, ops
    assert ops[0]["mute"] is True                          # added cell silent by default
    regs = doc2["layout_regions"]
    assert len(regs) == 1 and regs[0]["template"] == "split_h", regs
    cells = regs[0]["cells"]
    assert cells["left"]["layer"] == "spine", cells
    assert cells["right"]["layer"] == ops[0]["op_id"], cells
    # bad template / window -> unchanged
    assert act.split_screen(doc, idx, "ffffffff:m01", template="nope",
                            from_ms=0, to_ms=1000) is doc
    assert act.split_screen(doc, idx, "ffffffff:m01", template="pip",
                            from_ms=1000, to_ms=1000) is doc
    print("ok  split_screen adds a place_video op + a layout region")


def test_split_screen_resolves_to_dest_rects():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced"), ("ffffffff:m01", "balanced")])
    idx = _MapIndex(struct)
    doc = act.split_screen(doc, idx, "ffffffff:m01", template="split_h",
                           from_ms=1000, to_ms=3000)
    rt = layers.resolve(doc, {"ffffffff-1111": 8000})
    # In-window: two picture layers (spine slice + op), each in its half.
    stack = rt.video_stack_at(1500)
    dests = sorted((v.transform.get("dest") for v in stack if layers.is_rect(v.transform.get("dest"))),
                   key=lambda d: d["x"])
    assert len(dests) == 2, [v.to_dict() for v in stack]
    assert dests[0] == {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}, dests
    assert dests[1] == {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}, dests
    # Outside the window the spine is full frame again (no dest rect).
    top = rt.video_at(500)
    assert not layers.is_rect((top.transform or {}).get("dest")), top.to_dict()
    print("ok  split resolves to two dest sub-rects inside the window")


def test_remove_tears_down_region():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced"), ("ffffffff:m01", "balanced")])
    idx = _MapIndex(struct)
    doc = act.split_screen(doc, idx, "ffffffff:m01", template="pip",
                           from_ms=1000, to_ms=3000)
    op_id = doc["operations"][0]["op_id"]
    doc2 = act.remove(doc, op_id)
    assert not doc2["operations"], doc2["operations"]
    assert not doc2.get("layout_regions"), doc2.get("layout_regions")  # dangling region dropped
    print("ok  removing a split op tears down its layout region")


def test_solve_layout_templates():
    assert layers.solve_layout("split_v")["bottom"] == {"x": 0.0, "y": 0.5, "w": 1.0, "h": 0.5}
    assert layers.solve_layout("pip")["inset"]["w"] < 0.5
    assert layers.solve_layout("nope") == {}
    print("ok  solve_layout maps templates to normalized rects")


def test_affordances_menu():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    aff = observe.affordances(doc, _ctx(struct))
    cut = aff["cuts"][0]
    assert "tight" in cut["can_tighten_to"], cut
    assert "broad" in cut["can_widen_to"], cut
    assert aff["verbs"] and "V2" in aff["can_add_channel"]
    assert "place_span" in aff["verbs"] and "source_awareness" in aff["senses"], aff
    print("ok  affordances lists retake levels + channels + continuous verbs")


def test_place_span_arbitrary_main_line():
    """place_span lifts ANY source window onto V1 -- no map ref needed."""
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    n0 = len(doc["timeline"])
    doc2 = act.place_span(doc, "ffffffff-1111", in_ms=1200, out_ms=1900,
                          channel="V1", axis="any", content="silent reaction")
    assert doc2 is not doc and len(doc2["timeline"]) == n0 + 1, doc2["timeline"]
    seg = doc2["timeline"][-1]
    assert (seg["in_ms"], seg["out_ms"]) == (1200, 1900) and seg["ref"] is None, seg
    assert seg["level"] == "span" and seg["file_id"] == "ffffffff-1111", seg
    print("ok  place_span puts an arbitrary window on the main line")


def test_place_span_v2_cutaway_and_bad_span_noop():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    prog = act._program_end(doc)
    doc2 = act.place_span(doc, "ffffffff-1111", in_ms=3000, out_ms=3500,
                          channel="V2", from_ms=prog, audio="keep")
    ops = [o for o in doc2["operations"] if o["type"] == "place_video"]
    assert ops and ops[-1]["from_ms"] == prog and ops[-1]["mute"] is False, ops
    # empty/invalid span -> unchanged doc (total verb)
    assert act.place_span(doc, "ffffffff-1111", in_ms=2000, out_ms=2000) is doc
    assert act.place_span(doc, "", in_ms=0, out_ms=100) is doc
    print("ok  place_span V2 cutaway keeps sound; bad span is a no-op")


def main():
    test_read_state_reports_cuts_and_feel()
    test_place_adds_main_line_cut()
    test_place_v2_cutaway_op()
    test_remove_and_move()
    test_trim_and_set_audio()
    test_tighten_reshapes_span()
    test_predict_length_under_tighten()
    test_validate_flags_bad_span()
    test_place_multispan_keep_spans_survives()
    test_norm_keep_spans_accepts_both_shapes()
    test_split_screen_adds_op_and_region()
    test_split_screen_resolves_to_dest_rects()
    test_remove_tears_down_region()
    test_solve_layout_templates()
    test_affordances_menu()
    test_place_span_arbitrary_main_line()
    test_place_span_v2_cutaway_and_bad_span_noop()
    print("\nall observe/act tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
