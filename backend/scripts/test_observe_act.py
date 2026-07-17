#!/usr/bin/env python3
"""Tests for the observe (senses) + act (verbs) engine -- pure, no DB/LLM.

Builds a real moment-tree map, hand-builds an EditContext (so no DB), and checks
that verbs mutate the document immutably + correctly and that the senses read it
back faithfully. Run:  PYTHONPATH=. python scripts/test_observe_act.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import act, arrange, layers, observe  # noqa: E402
from app.services.l3 import footage_map as fm  # noqa: E402
from app.services.l3.arrange import Placement, _MapIndex, _weld_segments  # noqa: E402

# build_clip_tree calls _said_text_for_span -> _sentences_for_file for every
# "said" cut (beat_transcript.plan.md), which would otherwise hit a real DB.
# Stub it process-wide (same as test_footage_map.py/test_tools_loop.py) --
# tests that need real transcript content override it locally.
fm._sentences_for_file = lambda file_id: ()


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
        durations={"ffffffff-1111": 8000},
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


def test_place_snaps_speech_cut_to_sentences():
    """A placed SPEECH cut whose keep-span seam falls mid-sentence is merged to
    one contiguous segment (footage_map sentence snap), so a later remove can't
    sever the sentence -- the m02 fix. Non-speech / no-transcript paths no-op."""
    c0 = _cut("f:t0", 0, 6000, "the question", ladder=[
        _rung_multi("balanced", [(0, 3000), (3060, 6000)], "the question")])
    tree = fm.build_clip_tree("ffffffff-2222", {"name": "Q", "duration_ms": 6000}, [c0])
    idx = _MapIndex({"clips": [tree]})
    sents = ({"speaker": "S0", "text": "so what was the biggest challenge here?",
              "src_in_ms": 0, "src_out_ms": 6000},)
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        doc2 = act.place(doc, idx, "ffffffff:m00", level="balanced", channel="V1")
    spans = [(s["in_ms"], s["out_ms"]) for s in doc2["timeline"]]
    assert spans == [(0, 6000)], spans   # 60ms mid-sentence seam swallowed
    print("ok  place snaps a speech cut's mid-sentence seam to one contiguous span")


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


def test_split_screen_window_path():
    """A cell source can be a raw (file, in, out) window, not just a map ref --
    the continuous-source path, mirroring place_span."""
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced"), ("ffffffff:m01", "balanced")])
    idx = _MapIndex(struct)
    doc2 = act.split_screen(doc, idx, None, file="ffffffff-1111",
                            in_ms=5000, out_ms=6500, template="pip",
                            from_ms=1000, to_ms=2500)
    assert doc2 is not doc, "window path should apply"
    ops = [o for o in doc2["operations"] if o["type"] == "place_video"]
    assert len(ops) == 1 and ops[0]["source_file_id"] == "ffffffff-1111", ops
    assert ops[0]["src_in_ms"] == 5000 and ops[0]["purpose"] == "split_cell", ops
    regs = doc2["layout_regions"]
    assert regs[0]["cells"]["inset"]["layer"] == ops[0]["op_id"], regs
    # neither ref nor a full window -> no-op
    assert act.split_screen(doc, idx, None, template="pip",
                            from_ms=1000, to_ms=2500) is doc
    print("ok  split_screen fills a cell from a raw source window")


def test_remove_region_tears_down_its_op():
    """Teardown symmetry: removing a REGION also retires the coverage op it fed
    (no orphaned full-frame silent paste-over left behind)."""
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced"), ("ffffffff:m01", "balanced")])
    idx = _MapIndex(struct)
    doc = act.split_screen(doc, idx, "ffffffff:m01", template="split_h",
                           from_ms=1000, to_ms=3000)
    region_id = doc["layout_regions"][0]["region_id"]
    doc2 = act.remove(doc, region_id)
    assert not doc2.get("layout_regions"), doc2.get("layout_regions")
    assert not [o for o in doc2["operations"] if o["type"] == "place_video"], \
        doc2["operations"]     # coverage op retired with its region
    print("ok  removing a region tears down its coverage op (symmetry)")


def test_validate_flags_orphaned_split_cell():
    """A split_cell op with no region referencing it is an orphan; an ordinary
    V2 cutaway (region-less by design) is NOT flagged."""
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    # An orphaned split cell (purpose marker, no region):
    doc["operations"].append({
        "op_id": "sp_orphan", "type": "place_video", "purpose": "split_cell",
        "source_file_id": "ffffffff-1111", "src_in_ms": 0, "src_out_ms": 1000,
        "from_ms": 0, "to_ms": 1000, "layout": layers.DEFAULT_LAYOUT})
    # A legitimate cutaway (no purpose marker) that should stay clean:
    doc["operations"].append({
        "op_id": "ov_ok", "type": "place_video",
        "source_file_id": "ffffffff-1111", "src_in_ms": 0, "src_out_ms": 1000,
        "from_ms": 0, "to_ms": 1000, "layout": layers.DEFAULT_LAYOUT})
    msgs = [(i["id"], i["message"]) for i in observe.validate(doc, _ctx(struct))]
    assert any(i == "sp_orphan" and "orphaned split cell" in m for i, m in msgs), msgs
    assert not any(i == "ov_ok" for i, _ in msgs), msgs
    print("ok  validate flags an orphaned split cell, not a normal cutaway")


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
    # cuts_v3_continuity.plan.md: the cut-centric loop drops the raw-footage
    # verbs/senses (source_awareness/scan_source/place_span) from the menu.
    assert "place" in aff["verbs"] and "place_span" not in aff["verbs"], aff
    assert "source_awareness" not in aff["senses"] and "scan_source" not in aff["senses"], aff
    print("ok  affordances lists retake levels + channels + cut-centric verbs")


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


def test_split_edit_add_replace_clear_and_guards():
    """split_edit decouples the audio edge at a seam: adds one op per seam,
    re-issuing replaces, 0 clears, and the first cut (no seam) is a no-op."""
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "tight"), ("ffffffff:m01", "tight")])
    ids = [s["seg_id"] for s in doc["timeline"]]
    assert len(ids) == 2, ids

    doc2 = act.split_edit(doc, ids[1], audio_offset_ms=-400)   # J-cut
    ses = [o for o in doc2["operations"] if o["type"] == "split_edit"]
    assert len(ses) == 1 and ses[0]["audio_offset_ms"] == -400, ses
    doc3 = act.split_edit(doc2, ids[1], audio_offset_ms=300)   # replace, not stack
    ses = [o for o in doc3["operations"] if o["type"] == "split_edit"]
    assert len(ses) == 1 and ses[0]["audio_offset_ms"] == 300, ses
    doc4 = act.split_edit(doc3, ids[1], audio_offset_ms=0)     # clear
    assert not [o for o in doc4["operations"] if o["type"] == "split_edit"]
    assert act.split_edit(doc, ids[0], audio_offset_ms=-400) is doc   # first cut
    assert act.split_edit(doc, "nope", audio_offset_ms=-400) is doc  # unknown
    print("ok  split_edit adds/replaces/clears; first-cut + unknown are no-ops")


def test_split_edit_resolves_decoupled_audio():
    """Through layers.resolve, a J-cut moves the AUDIO boundary and leaves the
    video boundary alone -- per-channel edges, end to end."""
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "tight"), ("ffffffff:m01", "tight")])
    ids = [s["seg_id"] for s in doc["timeline"]]
    doc = act.split_edit(doc, ids[1], audio_offset_ms=-400)
    res = layers.resolve(doc, {"ffffffff-1111": 8000})
    aud = [a for a in res.audio_layers if a.kind == "spine"]
    vid = [v for v in res.video_layers if v.kind == "spine"]
    # video seam unchanged; audio boundary moved 400ms earlier
    assert vid[0].prog_end_ms == vid[1].prog_start_ms, (vid[0], vid[1])
    assert aud[1].prog_start_ms == vid[1].prog_start_ms - 400, aud[1]
    assert aud[0].prog_end_ms == aud[1].prog_start_ms, (aud[0], aud[1])
    print("ok  split_edit J-cut decouples audio from video through resolve")


def test_validate_flags_bad_split_edit():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "tight"), ("ffffffff:m01", "tight")])
    ids = [s["seg_id"] for s in doc["timeline"]]
    doc["operations"].append({"op_id": "se_x", "type": "split_edit",
                              "seam_seg_id": "gone", "audio_offset_ms": -300})
    doc["operations"].append({"op_id": "se_y", "type": "split_edit",
                              "seam_seg_id": ids[0], "audio_offset_ms": -300})
    msgs = [i["message"] for i in observe.validate(doc, _ctx(struct))]
    assert any("not a main-line seg_id" in m for m in msgs), msgs
    assert any("first cut" in m for m in msgs), msgs
    print("ok  validate flags orphaned + first-cut split edits")


def test_weld_remaps_split_edit_seams():
    """resolve_doc's weld re-issues seg_ids; a split at a SURVIVING seam is
    remapped, a split at a seam that welded away is dropped."""
    struct = _map()
    ctx = _ctx(struct)
    # m00 tight (0-2000) + m01 tight (4000-6000): non-contiguous, both survive.
    doc = _doc(struct, [("ffffffff:m00", "tight"), ("ffffffff:m01", "tight")])
    ids = [s["seg_id"] for s in doc["timeline"]]
    doc = act.split_edit(doc, ids[1], audio_offset_ms=-400)
    out = observe.resolve_doc(doc, ctx)
    ses = [o for o in out["operations"] if o["type"] == "split_edit"]
    new_ids = [s["seg_id"] for s in out["timeline"]]
    assert len(ses) == 1 and ses[0]["seam_seg_id"] == new_ids[1], (ses, new_ids)

    # m00 balanced (0-4000) + m01 balanced (4000-8000): contiguous -> weld away.
    doc2 = _doc(struct, [("ffffffff:m00", "balanced")])
    idx = _MapIndex(struct)
    doc2 = act.place(doc2, idx, "ffffffff:m01", level="balanced", channel="V1")
    seam_id = doc2["timeline"][1]["seg_id"]
    doc2 = act.split_edit(doc2, seam_id, audio_offset_ms=-400)
    out2 = observe.resolve_doc(doc2, ctx)
    assert len(out2["timeline"]) == 1, out2["timeline"]          # welded
    assert not [o for o in out2["operations"] if o["type"] == "split_edit"], \
        out2["operations"]                                        # split moot -> dropped
    print("ok  weld remaps surviving split seams and drops welded-away ones")


def test_split_screen_snap_cap_keeps_edge_and_suggests():
    """Snap sovereignty: an edge whose nearest cut-boundary is beyond the cap
    stays where the brain put it; the boundary arrives as a suggestion
    instead. The v3-native seam source (cleanup.plan.md B1) is the clean
    src_in_ms/src_out_ms edges from cut_records, not the old hero-substrate
    fused seam field."""
    import json as _json_mod

    from app.services.l3 import cuts_v3_read, tools

    struct = _map()
    ctx = _ctx(struct)
    ctx.run_id = "run-1"
    orig = cuts_v3_read.rows_for_run
    cuts_v3_read.rows_for_run = lambda run_id, file_ids: [
        {"src_in_ms": 0, "src_out_ms": 1300},      # boundary at 1300ms -- 700ms
        #                                             from the raw in (beyond the 400 cap)
        {"src_in_ms": 3100, "src_out_ms": 8000},   # boundary at 3100ms -- 100ms from raw out
    ]
    try:
        doc = _doc(struct, [("ffffffff:m00", "balanced")])
        obs, new, changed = tools._dispatch(
            "split_screen",
            {"file": "ffffffff", "in_ms": 2000, "out_ms": 3000,
             "template": "pip", "from_ms": 0, "to_ms": 2000},
            ctx, doc)
    finally:
        cuts_v3_read.rows_for_run = orig
    assert changed
    op = new["operations"][-1]
    assert op["src_in_ms"] == 2000, op          # kept: boundary was 700ms away (> cap)
    assert op["src_out_ms"] == 3100, op         # snapped: 100ms move (< cap)
    payload = _json_mod.loads(obs)
    assert payload["snap"]["in_suggested_ms"] == 1300, payload
    print("ok  snap cap keeps the brain's edge and only SUGGESTS the far boundary")


def test_split_screen_snap_off_places_raw():
    from app.services.l3 import tools

    struct = _map()
    ctx = _ctx(struct)
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    _, new, changed = tools._dispatch(
        "split_screen",
        {"file": "ffffffff", "in_ms": 1150, "out_ms": 3000,
         "template": "pip", "from_ms": 0, "to_ms": 2000, "snap": "off"},
        ctx, doc)
    assert changed
    op = new["operations"][-1]
    assert (op["src_in_ms"], op["src_out_ms"]) == (1150, 3000), op   # untouched
    print("ok  snap:'off' places the raw span (deliberate mid-motion edge)")


def test_split_screen_snaps_to_seam_via_dispatch():
    """split_screen's raw-window cell, through the tool loop, snaps to the
    clean cut-boundary points from cut_records (the v3-native seam source)
    and reports the deltas."""
    import json as _json_mod

    from app.services.l3 import cuts_v3_read, tools

    struct = _map()
    ctx = _ctx(struct)
    ctx.run_id = "run-1"
    orig = cuts_v3_read.rows_for_run
    cuts_v3_read.rows_for_run = lambda run_id, file_ids: [
        {"src_in_ms": 0, "src_out_ms": 1200},      # boundary at 1200ms
        {"src_in_ms": 3200, "src_out_ms": 8000},   # boundary at 3200ms
    ]
    try:
        doc = _doc(struct, [("ffffffff:m00", "balanced")])
        n0 = len(doc["operations"])
        obs, new, changed = tools._dispatch(
            "split_screen",
            {"file": "ffffffff", "in_ms": 1000, "out_ms": 3000,
             "template": "pip", "from_ms": 0, "to_ms": 2500},
            ctx, doc)
    finally:
        cuts_v3_read.rows_for_run = orig
    assert changed and len(new["operations"]) == n0 + 1, new["operations"]
    op = new["operations"][-1]
    assert (op["src_in_ms"], op["src_out_ms"]) == (1200, 3200), op   # snapped, not 1000/3000
    payload = _json_mod.loads(obs)
    assert payload.get("snap"), payload
    assert payload["snap"]["in_delta_ms"] == 200 and payload["snap"]["out_delta_ms"] == 200, payload
    assert payload["snap"]["in_q"] == 1.0 and payload["snap"]["out_q"] == 1.0, payload
    print("ok  split_screen snaps to the cut-boundary seam source through the tool loop")


def test_seams_for_file_fails_open_with_no_run_id():
    ctx = observe.EditContext(file_ids=["f1"], index=None, map_struct={},
                              durations={})
    assert observe._seams_for_file(ctx, "f1") == []
    print("ok  _seams_for_file fails open with no pinned/resolved run")


def test_snap_span_to_seams_pure_logic():
    # no boundaries -> unchanged no-op
    assert observe.snap_span_to_seams([], 100, 200) == {"in_ms": 100, "out_ms": 200, "snapped": False}
    # empty/inverted span -> unchanged no-op
    assert observe.snap_span_to_seams([500], 200, 200) == {"in_ms": 200, "out_ms": 200, "snapped": False}
    # snaps each edge to its nearest boundary; every boundary is quality 1.0
    r = observe.snap_span_to_seams([1200, 3200], 1000, 3000, max_move_ms=400)
    assert r["in_ms"] == 1200 and r["out_ms"] == 3200, r
    assert r["in_delta_ms"] == 200 and r["out_delta_ms"] == 200, r
    assert r["in_q"] == 1.0 and r["out_q"] == 1.0, r
    # beyond the cap -> kept + suggested, never inverted
    r2 = observe.snap_span_to_seams([1300, 3100], 2000, 3000, max_move_ms=400)
    assert r2["in_ms"] == 2000 and r2["in_suggested_ms"] == 1300, r2
    assert r2["out_ms"] == 3100 and "out_suggested_ms" not in r2, r2
    print("ok  snap_span_to_seams: nearest-boundary snap + sovereignty cap")


# --------------------------------------------------------------------------
# retime (pacing verb) + pace surfacing
# --------------------------------------------------------------------------

def _paced_struct():
    """A map with one VIDEO cut (speed room) + one SPEECH cut (dead-air budget),
    carrying real pace envelopes -- the shape cutrecord_map hands the tree."""
    return {"clips": [{
        "file_id": "ffffffff-1111", "name": "clipA", "duration_ms": 60000,
        "content_type": "reel", "primary_axis": "action", "people": ["G1"],
        "moment_count": 2,
        "moments": [
            {"moment_id": "ffffffff:m00", "file_id": "ffffffff-1111", "kind": "video",
             "channel": "shown", "in_ms": 1000, "out_ms": 5000, "play_ms": 4000,
             "variants": {"balanced": {"level": "balanced", "in_ms": 1000, "out_ms": 5000,
                                       "play_ms": 4000, "keep_spans": None}},
             "pace": {"levels": [0.5, 0.8, 1.0, 1.3, 1.8], "remove_spans": [], "min_ms": 1500}},
            {"moment_id": "ffffffff:m01", "file_id": "ffffffff-1111", "kind": "speech",
             "channel": "said", "speaker": "G1", "in_ms": 10000, "out_ms": 18000, "play_ms": 8000,
             "variants": {"balanced": {"level": "balanced", "in_ms": 10000, "out_ms": 18000,
                                       "play_ms": 8000, "keep_spans": None}},
             "pace": {"levels": [1.0] * 5, "remove_spans": [[12000, 13000], [15000, 15500]]}},
        ]}]}


def _paced_doc(struct):
    idx = _MapIndex(struct)
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    doc = act.place(doc, idx, "ffffffff:m00", level="balanced", channel="V1")
    doc = act.place(doc, idx, "ffffffff:m01", level="balanced", channel="V1")
    return doc


def test_retime_video_stamps_speed_not_span():
    struct = _paced_struct()
    idx = _MapIndex(struct)
    doc = _paced_doc(struct)
    vid = next(s for s in doc["timeline"] if s["ref"] == "ffffffff:m00")
    span0 = (vid["in_ms"], vid["out_ms"])
    d2 = act.retime(doc, idx, seg_id=vid["seg_id"], pace="faster")
    v = next(s for s in d2["timeline"] if s["ref"] == "ffffffff:m00")
    assert v["speed"] == 1.3 and v["pace_level"] == "faster", v      # levels[3]
    assert (v["in_ms"], v["out_ms"]) == span0, v                     # span untouched (phase-2 render)
    assert doc is not d2 and "speed" not in vid                      # immutable
    print("ok  retime video stamps cross-clip speed, leaves the span alone")


def test_retime_speech_trims_dead_air_and_is_idempotent():
    struct = _paced_struct()
    idx = _MapIndex(struct)
    doc = _paced_doc(struct)
    sid = next(s for s in doc["timeline"] if s["ref"] == "ffffffff:m01")["seg_id"]

    def spans(d):
        return [(s["in_ms"], s["out_ms"]) for s in d["timeline"] if s.get("ref") == "ffffffff:m01"]

    d = act.retime(doc, idx, seg_id=sid, pace="much_faster")
    assert spans(d) == [(10000, 12000), (13000, 15000), (15500, 18000)], spans(d)  # both gaps cut
    assert all("speed" not in s for s in d["timeline"] if s.get("ref") == "ffffffff:m01")
    d = act.retime(d, idx, seg_id=sid, pace="faster")
    assert spans(d) == [(10000, 12000), (13000, 18000)], spans(d)     # only the longest gap
    d = act.retime(d, idx, seg_id=sid, pace="natural")
    assert spans(d) == [(10000, 18000)], spans(d)                     # full delivery restored
    print("ok  retime speech trims dead-air into jump-cuts, idempotent widen/restore")


def test_retime_unknown_pace_is_noop():
    struct = _paced_struct()
    idx = _MapIndex(struct)
    doc = _paced_doc(struct)
    assert act.retime(doc, idx, pace="turbo") is doc
    print("ok  retime with an unknown pace is a no-op")


def test_pace_tag_and_affordances_surface_room():
    struct = _paced_struct()
    m_vid, m_spe = struct["clips"][0]["moments"]
    assert fm._pace_tag(m_vid) == " · pace:0.5-1.8x", fm._pace_tag(m_vid)
    assert fm._pace_tag(m_spe) == " · trim\u22641.5s", fm._pace_tag(m_spe)

    ctx = _ctx(struct)
    doc = _paced_doc(struct)
    aff = observe.affordances(doc, ctx)
    assert "retime" in aff["verbs"], aff["verbs"]
    by_ref = {c["ref"]: c for c in aff["cuts"]}
    assert by_ref["ffffffff:m00"]["retime_kind"] == "video_speed"
    assert by_ref["ffffffff:m00"]["speed_by_step"]["faster"] == 1.3
    assert by_ref["ffffffff:m01"]["retime_kind"] == "speech_trim"
    assert by_ref["ffffffff:m01"]["trim_budget_ms"] == 1500

    d = act.retime(doc, _MapIndex(struct),
                   seg_id=by_ref["ffffffff:m00"]["seg_id"], pace="faster")
    st = observe.read_state(d, ctx)
    vcut = next(c for c in st["cuts"] if c["ref"] == "ffffffff:m00")
    assert vcut["pace_level"] == "faster" and vcut["speed"] == 1.3 and "speed_note" in vcut, vcut
    print("ok  pace tag + affordances + read_state surface the pacing room")


# --------------------------------------------------------------------------
# A/V coupling audio-source priority (av_coupling_authoritative.plan.md)
# --------------------------------------------------------------------------

def _seg(seg_id, file_id, in_ms, out_ms, **extra):
    d = {"seg_id": seg_id, "file_id": file_id, "in_ms": in_ms, "out_ms": out_ms}
    d.update(extra)
    return d


def test_resolve_uses_baked_coupling_over_legacy_audio_routes():
    doc = {"timeline": [_seg("s0", "f1", 0, 1000, audio_file_id="f2", audio_offset_ms=150)]}
    # A legacy route also exists for this seg_id -- baked coupling must win.
    audio_routes = {"s0": {"source_file_id": "f9", "src_in_ms": 999, "src_out_ms": 1999}}
    rt = layers.resolve(doc, {}, {}, audio_routes)
    a = rt.audio_layers[0]
    assert a.source_file_id == "f2", a
    assert (a.src_in_ms, a.src_out_ms) == (150, 1150), (a.src_in_ms, a.src_out_ms)
    print("ok  resolve: baked coupling wins over legacy audio_routes")


def test_resolve_falls_back_to_legacy_audio_routes_with_no_baked_coupling():
    # No audio_file_id at all -- an edit document built before this feature,
    # or a place_span-placed span -- must still get the SS8 route correction.
    doc = {"timeline": [_seg("s0", "f1", 0, 1000)]}
    audio_routes = {"s0": {"source_file_id": "f2", "src_in_ms": 300, "src_out_ms": 1300}}
    rt = layers.resolve(doc, {}, {}, audio_routes)
    a = rt.audio_layers[0]
    assert a.source_file_id == "f2", a
    assert (a.src_in_ms, a.src_out_ms) == (300, 1300), (a.src_in_ms, a.src_out_ms)
    print("ok  resolve: legacy audio_routes still applies with no baked coupling")


def test_resolve_plain_coupled_audio_with_no_coupling_or_route_at_all():
    doc = {"timeline": [_seg("s0", "f1", 0, 1000)]}
    rt = layers.resolve(doc, {}, {}, {})
    a = rt.audio_layers[0]
    assert a.source_file_id == "f1", a
    assert (a.src_in_ms, a.src_out_ms) == (0, 1000), (a.src_in_ms, a.src_out_ms)
    print("ok  resolve: plain coupled audio with no coupling or route at all")


def test_resolve_audio_override_wins_over_baked_coupling():
    doc = {"timeline": [_seg("s0", "f1", 0, 1000, audio_file_id="f2", audio_offset_ms=150,
                             audio_override={"source_file_id": "f3", "src_in_ms": 5000, "src_out_ms": 6000})]}
    rt = layers.resolve(doc, {}, {}, {})
    a = rt.audio_layers[0]
    assert a.source_file_id == "f3", a
    assert (a.src_in_ms, a.src_out_ms) == (5000, 6000), (a.src_in_ms, a.src_out_ms)
    print("ok  resolve: audio_override wins over baked coupling")


def test_resolve_same_source_baked_coupling_is_zero_offset():
    doc = {"timeline": [_seg("s0", "f1", 500, 1500, audio_file_id="f1", audio_offset_ms=0)]}
    rt = layers.resolve(doc, {}, {}, {})
    a = rt.audio_layers[0]
    assert a.source_file_id == "f1", a
    assert (a.src_in_ms, a.src_out_ms) == (500, 1500), (a.src_in_ms, a.src_out_ms)
    print("ok  resolve: same-source baked coupling is zero-offset identity")


# --------------------------------------------------------------------------
# edso_pacing_audit_timing.plan.md Group 1
# --------------------------------------------------------------------------

def test_program_map_renders_video_and_audio_tables_with_stacking():
    """A coverage layer (V2) appears with its own z/layout/program window,
    distinct from the spine (V1) row it stacks over -- overlap is visible
    from the shared clock + z, no prose needed."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 4000,
             "ref": "ffffffff:m00", "level": "balanced", "content": "we almost shut down"},
        ],
        "operations": [
            {"op_id": "ov1", "type": "place_video", "source_file_id": "9f0c1234-2222",
             "src_in_ms": 0, "src_out_ms": 2000, "from_ms": 1000, "to_ms": 3000,
             "layout": "pip", "z": layers.Z_COVERAGE, "opacity": 1.0,
             "rationale": "inset detail", "mute": True},
        ],
    }
    text = arrange.render_program_map(doc)
    assert text.startswith("PROGRAM MAP"), text
    assert "VIDEO" in text and "AUDIO" in text, text
    assert "V1 s0" in text and "0-4000ms" in text, text
    assert f"V2 ov1  z{layers.Z_COVERAGE}" in text, text
    assert "1000-3000ms" in text and "pip" in text, text   # the coverage layer's OWN window
    print("ok  Program Map renders VIDEO+AUDIO tables with correct z/layout/program window")


def test_program_map_shows_per_layer_loudness_and_audio_gaps():
    """audio_and_audit.plan.md Phase 2: each AUDIO row shows its own source
    LUFS when known, and a trailing GAPS line names any stretch with no
    audible layer at all (a muted seg between two audible ones)."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000},
            {"seg_id": "s1", "file_id": "ffffffff-1111", "in_ms": 2000, "out_ms": 4000, "mute": True},
            {"seg_id": "s2", "file_id": "ffffffff-1111", "in_ms": 4000, "out_ms": 6000},
        ],
        "operations": [],
    }
    text = arrange.render_program_map(
        doc, audio_features={"ffffffff-1111": {"integrated_lufs": -18.3}})
    assert "lufs:-18.3" in text, text
    assert "GAPS (no audio): 0:02-0:04" in text, text
    print("ok  Program Map shows per-layer loudness and a GAPS line")


def test_program_map_empty_document_is_empty_string():
    assert arrange.render_program_map(None) == ""
    assert arrange.render_program_map({"timeline": [], "operations": []}) == ""
    print("ok  Program Map is empty for an empty document")


def test_read_state_reports_video_stack_and_audio_layers():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    idx = _MapIndex(struct)
    doc = act.place(doc, idx, "ffffffff:m01", channel="V2", from_ms=500)
    st = observe.read_state(doc, _ctx(struct))
    assert "video_stack" in st and len(st["video_stack"]) == 1, st
    assert st["video_stack"][0]["z"] == layers.Z_COVERAGE, st["video_stack"]
    assert st["video_stack"][0]["layout"] == "full_frame", st["video_stack"]
    print("ok  read_state reports the z-stack (coverage layers)")


def test_read_state_omits_z_stack_keys_when_theres_no_coverage():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    st = observe.read_state(doc, _ctx(struct))
    assert "video_stack" not in st and "audio_layers" not in st, st
    print("ok  read_state omits the z-stack keys when there's no coverage")


def test_read_state_shows_loudness_gaps_and_candidates():
    """audio_and_audit.plan.md Phase 2: read_state ALONE (no separate
    audio_state call) surfaces a bed's own loudness, a mid-program silence
    hole (a muted seg between two audible ones), and this user's unused
    audio assets (each with is_musical/bpm)."""
    struct = _map()
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000},
            {"seg_id": "s1", "file_id": "ffffffff-1111", "in_ms": 2000, "out_ms": 4000, "mute": True},
            {"seg_id": "s2", "file_id": "ffffffff-1111", "in_ms": 4000, "out_ms": 6000},
        ],
        "operations": [
            {"op_id": "pa1", "type": "place_audio", "role": "music",
             "source_file_id": "bed1", "src_in_ms": 0, "src_out_ms": 1000,
             "from_ms": 0, "to_ms": 1000, "gain_db": 0.0, "duck_db": 0.0},
        ],
    }
    ctx = observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, dup_groups=[],
        audio_features={"bed1": {"integrated_lufs": -14.0}},
        audio_assets=[{"file_id": "unused1", "name": "song.mp3", "dur_ms": 30000,
                       "is_musical": True, "bpm": 120.0}],
    )
    st = observe.read_state(doc, ctx)
    assert st["audio_layers"][0]["loudness_lufs"] == -14.0, st["audio_layers"]
    assert st["audio_gaps"] == [{"from_ms": 2000, "to_ms": 4000}], st.get("audio_gaps")
    assert st["audio_candidates"][0]["name"] == "song.mp3", st.get("audio_candidates")
    assert st["audio_candidates"][0]["is_musical"] is True, st["audio_candidates"]
    print("ok  read_state surfaces per-layer loudness, audio gaps, and candidate beds")


def test_beat_grid_maps_sections_and_drop_into_program_time():
    """audio_and_audit.plan.md Phase 4: a musical bed's OPTIONAL sections/
    drop_ms (when its L1 analysis detected them) map into PROGRAM time
    alongside onsets_ms -- clipped to the window it's actually playing, and
    simply omitted (never fabricated) when they fall outside it."""
    struct = _map()
    doc = {"timeline": [], "operations": [
        {"op_id": "pa1", "type": "place_audio", "role": "music",
         "source_file_id": "bed1", "src_in_ms": 0, "src_out_ms": 4000,
         "from_ms": 1000, "to_ms": 5000},
    ]}
    ctx = observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, dup_groups=[],
        audio_features={"bed1": {"is_musical": True, "bpm": 120.0, "onsets_ms": [500],
                                 "sections": [{"start_ms": 0, "end_ms": 2000},
                                              {"start_ms": 2000, "end_ms": 4000}],
                                 "drop_ms": 1500}})
    grid = observe._beat_grid(doc, ctx)
    assert len(grid) == 1, grid
    entry = grid[0]
    # source time -> program time is +1000ms (from_ms) throughout.
    assert entry["sections"] == [{"from_ms": 1000, "to_ms": 3000},
                                 {"from_ms": 3000, "to_ms": 5000}], entry
    assert entry["drop_ms"] == 2500, entry
    print("ok  beat_grid maps a bed's sections/drop into program time")


def test_beat_grid_omits_sections_and_drop_when_undetected():
    struct = _map()
    doc = {"timeline": [], "operations": [
        {"op_id": "pa1", "type": "place_audio", "role": "music",
         "source_file_id": "bed1", "src_in_ms": 0, "src_out_ms": 4000,
         "from_ms": 0, "to_ms": 4000},
    ]}
    ctx = observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, dup_groups=[],
        audio_features={"bed1": {"is_musical": True, "bpm": 120.0, "onsets_ms": [500]}})
    entry = observe._beat_grid(doc, ctx)[0]
    assert "sections" not in entry and "drop_ms" not in entry, entry
    print("ok  beat_grid omits sections/drop_ms when the source's analysis never detected them")


def test_read_state_seg_id_detail_gives_word_program_offsets():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    seg_id = doc["timeline"][0]["seg_id"]
    transcript_segments = [{"words": [
        {"text": "we", "start_ms": 0, "end_ms": 300},
        {"text": "almost", "start_ms": 400, "end_ms": 900},
    ]}]
    with mock.patch.object(observe.captions_resolver, "fetch_transcripts",
                           return_value={"ffffffff-1111": {"segments": transcript_segments}}):
        st = observe.read_state(doc, _ctx(struct), seg_id=seg_id)
    words = st["word_offsets"]["words"]
    assert words == [
        {"text": "we", "prog_start_ms": 0, "prog_end_ms": 300},
        {"text": "almost", "prog_start_ms": 400, "prog_end_ms": 900},
    ], words
    print("ok  read_state seg_id detail resolves word->program offsets")


def test_word_offsets_stay_correct_after_an_upstream_trim():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    seg_id = doc["timeline"][0]["seg_id"]
    # Trim 500ms off the front: in_ms moves 0 -> 500, but this seg is still
    # FIRST on the main line so its own prog_start_ms stays 0 -- a word at
    # src [600,900) should now land at prog [100,400).
    doc = act.trim(doc, seg_id, delta_in_ms=500)
    transcript_segments = [{"words": [{"text": "almost", "start_ms": 600, "end_ms": 900}]}]
    with mock.patch.object(observe.captions_resolver, "fetch_transcripts",
                           return_value={"ffffffff-1111": {"segments": transcript_segments}}):
        st = observe.read_state(doc, _ctx(struct), seg_id=seg_id)
    assert st["word_offsets"]["words"] == [
        {"text": "almost", "prog_start_ms": 100, "prog_end_ms": 400}
    ], st["word_offsets"]
    print("ok  word offsets stay correct after an upstream trim")


def test_review_flags_speaker_mismatch_filler_and_dead_air():
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 5000,
         "axis": "speech", "ref": "ffffffff:m00"},
    ], "operations": [], "brief": {"target_duration_s": 3}}
    sentences = (
        {"speaker": "S9", "text": "um", "src_in_ms": 0, "src_out_ms": 300},
        {"speaker": "S0", "text": "so the real point is this", "src_in_ms": 500, "src_out_ms": 3000},
        {"speaker": "S0", "text": "and that changed everything", "src_in_ms": 3200, "src_out_ms": 4500},
    )
    struct = _map()
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        out = observe.review(doc, _ctx(struct))
    assert out["total_ms"] == 5000 and out["target_ms"] == 3000, out
    msgs = [f["message"] for f in out["flags"]]
    assert any("dominant speaker" in m for m in msgs), msgs
    assert any("filler/backchannel" in m for m in msgs), msgs
    assert any("dead air" in m for m in msgs), msgs
    assert all("trim to" not in m.lower() for m in msgs)   # never a prescribed fix
    print("ok  review flags speaker mismatch, filler lead-in, and trailing dead air")


def test_review_played_text_reflects_the_excised_span_not_the_whole_cut():
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 1000, "out_ms": 2000,
         "axis": "speech", "ref": "ffffffff:m00"},
    ], "operations": [], "brief": {}}
    sentences = (
        {"speaker": "S0", "text": "before the trim", "src_in_ms": 0, "src_out_ms": 900},
        {"speaker": "S0", "text": "inside the kept span", "src_in_ms": 1000, "src_out_ms": 2000},
    )
    struct = _map()
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        out = observe.review(doc, _ctx(struct))
    assert out["items"][0]["played_text"] == "inside the kept span", out["items"][0]
    print("ok  review's played_text reflects only the segment's own (excised) span")


def test_review_flags_an_overrunning_overlay():
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000, "axis": "any"},
    ], "operations": [
        {"op_id": "ov1", "type": "place_video", "source_file_id": "9f0c1234-2222",
         "src_in_ms": 0, "src_out_ms": 3000, "from_ms": 0, "to_ms": 3000,
         "layout": "full_frame", "z": layers.Z_COVERAGE, "opacity": 1.0, "mute": True},
    ], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct))
    msgs = [f["message"] for f in out["flags"]]
    assert any("extends past the beat" in m for m in msgs), msgs
    assert all("trim to" not in m.lower() and "set " not in m.lower() for m in msgs)
    print("ok  review flags an overlay that overruns the beat it sits over")


def test_review_flags_an_underfilling_overlay():
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 4000, "axis": "any"},
    ], "operations": [
        {"op_id": "ov1", "type": "place_video", "source_file_id": "9f0c1234-2222",
         "src_in_ms": 0, "src_out_ms": 500, "from_ms": 0, "to_ms": 500,
         "layout": "full_frame", "z": layers.Z_COVERAGE, "opacity": 1.0, "mute": True},
    ], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct))
    msgs = [f["message"] for f in out["flags"]]
    assert any("underfills" in m for m in msgs), msgs
    print("ok  review flags an overlay that underfills the beat it sits over")


def test_review_flags_a_mid_program_audio_gap():
    """audio_and_audit.plan.md Phase 5 Stage 3: a stretch with no audible
    layer at all (a muted seg between two audible ones) surfaces as a
    craft-category flag -- the 'sound randomly missing in the middle' case."""
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000, "axis": "any"},
        {"seg_id": "s1", "file_id": "ffffffff-1111", "in_ms": 2000, "out_ms": 4000,
         "axis": "any", "mute": True},
        {"seg_id": "s2", "file_id": "ffffffff-1111", "in_ms": 4000, "out_ms": 6000, "axis": "any"},
    ], "operations": [], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct))
    gap_flags = [f for f in out["flags"] if f.get("category") == "craft" and "no audio" in f["message"]]
    assert len(gap_flags) == 1 and "2.0s" in gap_flags[0]["message"], out["flags"]
    print("ok  review flags a mid-program audio gap (craft category)")


def test_review_flags_a_loudness_imbalance():
    """A layer whose own source loudness sits well off the program's median
    is checkable, not just advised (guidance §5 'keep it level')."""
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000, "axis": "any"},
    ], "operations": [
        {"op_id": "pa1", "type": "place_audio", "role": "music",
         "source_file_id": "bed1", "src_in_ms": 0, "src_out_ms": 2000,
         "from_ms": 0, "to_ms": 2000, "gain_db": 0.0, "duck_db": 0.0},
    ], "brief": {}}
    struct = _map()
    ctx = observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, dup_groups=[],
        audio_features={"ffffffff-1111": {"integrated_lufs": -24.0},
                        "bed1": {"integrated_lufs": -8.0}})
    out = observe.review(doc, ctx)
    imbalance_flags = [f for f in out["flags"] if f.get("category") == "craft" and "loudness" in f["message"]]
    # median picks one of the two layers as the reference (0 deviation from
    # itself); only the OTHER layer's real deviation from it can flag.
    assert len(imbalance_flags) == 1 and "-16.0dB" in imbalance_flags[0]["message"], out["flags"]
    print("ok  review flags a loudness imbalance across layers (craft category)")


def test_offcam_speaker_flag_only_when_an_oncam_angle_exists():
    # off camera + a sibling angle shows the speaker -> flagged (with the ref)
    f = observe._offcam_speaker_flag(
        {"speaker_person": "P1", "on_camera": False,
         "alt_pic": [{"moment_id": "1e529bed:m06", "visible_persons": ["P1"]}]},
        "cut 1 (s0)")
    assert f and "off camera" in f[0]["message"] and "1e529bed:m06" in f[0]["message"], f
    assert "trim" not in f[0]["message"].lower()   # never a prescribed fix
    # speaker already on camera -> nothing to switch to
    assert observe._offcam_speaker_flag(
        {"speaker_person": "P1", "on_camera": True,
         "alt_pic": [{"moment_id": "x:m1", "visible_persons": ["P1"]}]}, "a") == []
    # off camera but NO sibling shows the speaker (narration / voiceover-over-b-roll)
    assert observe._offcam_speaker_flag(
        {"speaker_person": "P1", "on_camera": False,
         "alt_pic": [{"moment_id": "x:m1", "visible_persons": ["P0"]}]}, "a") == []
    # no speaker at all -> silent
    assert observe._offcam_speaker_flag({"on_camera": False, "alt_pic": []}, "a") == []
    print("ok  off-camera-speaker flag fires only when an on-camera angle exists")


def test_review_wires_the_offcam_speaker_flag():
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 4000,
         "axis": "speech", "ref": "ffffffff:m00"},
    ], "operations": [], "brief": {}}
    ctx = _ctx(_map())
    ctx.index.moments["ffffffff:m00"] = {
        "speaker_person": "P1", "on_camera": False,
        "alt_pic": [{"moment_id": "1e529bed:m06", "visible_persons": ["P1"]}],
    }
    with mock.patch.object(fm, "_sentences_for_file", return_value=()):
        out = observe.review(doc, ctx)
    msgs = [f["message"] for f in out["flags"]]
    assert any("off camera" in m and "1e529bed:m06" in m for m in msgs), msgs
    print("ok  review surfaces the off-camera-speaker flag end-to-end")


# --------------------------------------------------------------------------
# edso_think_act_check.plan.md change 4: requested-feature presence check
# --------------------------------------------------------------------------

def test_review_flags_a_named_feature_missing_from_the_edit():
    doc = {"timeline": [], "operations": [], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct), user_ask="make it a split screen with music")
    msgs = [f["message"] for f in out["flags"]]
    assert any("split screen" in m and "doesn't have it" in m for m in msgs), msgs
    assert any("a music bed" in m and "doesn't have it" in m for m in msgs), msgs
    cats = {f["category"] for f in out["flags"]}
    assert cats == {"ask"}, cats
    print("ok  review flags a named feature (split screen, music) missing from the edit")


def test_review_no_feature_flag_when_the_feature_is_present():
    doc = {"timeline": [], "operations": [
        {"op_id": "ov1", "type": "place_audio", "role": "music", "from_ms": 0, "to_ms": 1000,
         "source_file_id": "f9", "src_in_ms": 0, "src_out_ms": 1000},
    ], "layout_regions": [
        {"region_id": "r1", "template": "split_h", "from_ms": 0, "to_ms": 1000, "cells": {}},
    ], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct), user_ask="make it a split screen with music")
    ask_flags = [f for f in out["flags"] if f.get("category") == "ask"]
    assert ask_flags == [], ask_flags
    print("ok  review doesn't flag a named feature that IS present")


def test_review_no_feature_flag_without_an_ask():
    doc = {"timeline": [], "operations": [], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct))   # default user_ask=""
    assert out["flags"] == [], out["flags"]
    print("ok  review runs no feature check at all with no user_ask given")


def test_review_feature_flags_never_prescribe_a_fix():
    doc = {"timeline": [], "operations": [], "brief": {}}
    struct = _map()
    out = observe.review(doc, _ctx(struct), user_ask="split screen please")
    for f in out["flags"]:
        msg = f["message"].lower()
        assert "add " not in msg and "use split_screen" not in msg and "call " not in msg, f
    print("ok  requested-feature flags state a fact, never a prescribed fix")


# --------------------------------------------------------------------------
# v4_cluster_read_act.plan.md Part B/C: multi-event cluster piece read + act
# --------------------------------------------------------------------------

def _cluster_struct():
    """One video moment holding a 3-event cluster (same fixture as
    test_footage_map.py/test_cutrecord_map.py's _cluster_events): peaks at
    1000/5000/9000ms inside a 0-10000ms cluster, far enough apart that
    resolve_cluster at energy=1.0 keeps them as three separate pieces. Carries
    a hand-built ladder ("broad" = the whole span, "sharp" = the 3-piece
    split) so `place(ref, level=...)` can be regression-tested without
    depending on the real ingest pipeline's ladder construction."""
    events = [
        {"peak_ms": 1000, "score": 0.6, "kind": "point", "onset_ms": 700, "settle_ms": 1500, "span_ms": None},
        {"peak_ms": 5000, "score": 1.0, "kind": "point", "onset_ms": 4600, "settle_ms": 5500, "span_ms": None},
        {"peak_ms": 9000, "score": 0.4, "kind": "point", "onset_ms": 8700, "settle_ms": 9400, "span_ms": None},
    ]
    salience = {"peak_ms": 5000, "score": 1.0, "kind": "point", "span_ms": None,
                "events": events, "primary": 1, "density": 0.5, "shape": "center"}
    c0 = {"hero_id": "f:cl", "file_id": "ffffffff-1111", "channel": "shown",
          "subject": "person", "label": "cluster", "src_in_ms": 0, "src_out_ms": 10000,
          "play_ms": 10000, "keep_spans": None, "score": 0.6, "speaker": None,
          "flags": [], "take_count": 1, "salience": salience,
          "ladder": [_rung("broad", 0, 10000, "whole cluster", 0.6),
                     _rung_multi("sharp", [(700, 1500), (4700, 5500), (8700, 9500)], "punchy", 0.9)]}
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "C", "duration_ms": 10000}, [c0])
    return {"clips": [tree]}


def test_read_state_piece_breakdown_matches_beat_index_for_cluster():
    struct = _cluster_struct()
    idx = _MapIndex(struct)
    m = idx.moments["ffffffff:m00"]
    doc = act.place({"format": {"aspect": "landscape"}, "timeline": [], "operations": []},
                    idx, "ffffffff:m00", level="broad", channel="V1")
    st = observe.read_state(doc, _ctx(struct))
    cut = st["cuts"][0]
    # Same shared helper (footage_map.piece_breakdown) the Beat Index's
    # _piece_lines renders from -- a deep look must agree with the up-front map.
    assert cut["pieces"] == fm.piece_breakdown(m), cut.get("pieces")
    assert cut["pieces"]["tight_count"] == 3, cut["pieces"]
    # The new filtering fields flow through unchanged, so a deep look shows the
    # same punchy floor + per-beat core marking the Beat Index does.
    assert cut["pieces"]["punchy_count"] == 1, cut["pieces"]
    assert "punchy_avg_s" in cut["pieces"], cut["pieces"]
    assert [p["core"] for p in cut["pieces"]["pieces"]] == [False, True, False], cut["pieces"]
    print("ok  read_state's piece breakdown mirrors the Beat Index for a cluster")


def test_read_state_omits_pieces_for_single_event_cut():
    struct = _map()
    doc = _doc(struct, [("ffffffff:m00", "balanced")])
    st = observe.read_state(doc, _ctx(struct))
    assert "pieces" not in st["cuts"][0], st["cuts"][0]
    print("ok  read_state omits piece breakdown for a single-event cut")


def test_place_cluster_whole_vs_pieces_via_level():
    struct = _cluster_struct()
    idx = _MapIndex(struct)
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    whole = act.place(doc, idx, "ffffffff:m00", level="broad", channel="V1")
    assert [(s["in_ms"], s["out_ms"]) for s in whole["timeline"]] == [(0, 10000)], whole["timeline"]
    pieces_doc = act.place(doc, idx, "ffffffff:m00", level="sharp", channel="V1")
    spans = [(s["in_ms"], s["out_ms"]) for s in pieces_doc["timeline"]]
    assert spans == [(700, 1500), (4700, 5500), (8700, 9500)], spans
    print("ok  place(ref, level) resolves a cluster whole (broad) <-> pieces (sharp)")


def test_place_piece_resolves_to_one_beat_span():
    struct = _cluster_struct()
    idx = _MapIndex(struct)
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    doc2 = act.place(doc, idx, "ffffffff:m00", piece=2, channel="V1")
    assert len(doc2["timeline"]) == 1, doc2["timeline"]
    seg = doc2["timeline"][0]
    assert (seg["in_ms"], seg["out_ms"]) == (4700, 5500), seg
    assert seg["level"] == "sharp", seg
    print("ok  place(ref, piece=k) places just that one beat")


def test_trim_works_on_a_placed_piece():
    struct = _cluster_struct()
    idx = _MapIndex(struct)
    doc = act.place({"format": {"aspect": "landscape"}, "timeline": [], "operations": []},
                    idx, "ffffffff:m00", piece=2, channel="V1")
    seg_id = doc["timeline"][0]["seg_id"]
    doc2 = act.trim(doc, seg_id, delta_in_ms=100)
    assert doc2["timeline"][0]["in_ms"] == 4800, doc2["timeline"][0]
    print("ok  trim works on a placed single-piece segment")


def test_place_bad_piece_is_clean_noop():
    struct = _cluster_struct()
    idx = _MapIndex(struct)
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    assert act.place(doc, idx, "ffffffff:m00", piece=0, channel="V1") is doc
    assert act.place(doc, idx, "ffffffff:m00", piece=99, channel="V1") is doc
    print("ok  place(ref, piece=<out-of-range>) is a clean no-op")


def test_place_piece_on_single_event_moment_is_noop():
    struct = _map()
    idx = _MapIndex(struct)
    doc = {"format": {"aspect": "landscape"}, "timeline": [], "operations": []}
    assert act.place(doc, idx, "ffffffff:m00", piece=1, channel="V1") is doc
    print("ok  place(ref, piece=1) on a single-event moment is a clean no-op")


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
    test_place_snaps_speech_cut_to_sentences()
    test_norm_keep_spans_accepts_both_shapes()
    test_split_screen_adds_op_and_region()
    test_split_screen_resolves_to_dest_rects()
    test_split_screen_window_path()
    test_remove_tears_down_region()
    test_remove_region_tears_down_its_op()
    test_validate_flags_orphaned_split_cell()
    test_solve_layout_templates()
    test_affordances_menu()
    test_place_span_arbitrary_main_line()
    test_place_span_v2_cutaway_and_bad_span_noop()
    test_split_screen_snaps_to_seam_via_dispatch()
    test_split_edit_add_replace_clear_and_guards()
    test_split_edit_resolves_decoupled_audio()
    test_validate_flags_bad_split_edit()
    test_weld_remaps_split_edit_seams()
    test_split_screen_snap_cap_keeps_edge_and_suggests()
    test_split_screen_snap_off_places_raw()
    test_seams_for_file_fails_open_with_no_run_id()
    test_snap_span_to_seams_pure_logic()
    test_retime_video_stamps_speed_not_span()
    test_retime_speech_trims_dead_air_and_is_idempotent()
    test_retime_unknown_pace_is_noop()
    test_pace_tag_and_affordances_surface_room()
    test_resolve_uses_baked_coupling_over_legacy_audio_routes()
    test_resolve_falls_back_to_legacy_audio_routes_with_no_baked_coupling()
    test_resolve_plain_coupled_audio_with_no_coupling_or_route_at_all()
    test_resolve_audio_override_wins_over_baked_coupling()
    test_resolve_same_source_baked_coupling_is_zero_offset()
    test_program_map_renders_video_and_audio_tables_with_stacking()
    test_program_map_shows_per_layer_loudness_and_audio_gaps()
    test_program_map_empty_document_is_empty_string()
    test_read_state_reports_video_stack_and_audio_layers()
    test_read_state_omits_z_stack_keys_when_theres_no_coverage()
    test_read_state_shows_loudness_gaps_and_candidates()
    test_beat_grid_maps_sections_and_drop_into_program_time()
    test_beat_grid_omits_sections_and_drop_when_undetected()
    test_read_state_seg_id_detail_gives_word_program_offsets()
    test_word_offsets_stay_correct_after_an_upstream_trim()
    test_review_flags_speaker_mismatch_filler_and_dead_air()
    test_review_played_text_reflects_the_excised_span_not_the_whole_cut()
    test_review_flags_an_overrunning_overlay()
    test_review_flags_an_underfilling_overlay()
    test_review_flags_a_mid_program_audio_gap()
    test_review_flags_a_loudness_imbalance()
    test_offcam_speaker_flag_only_when_an_oncam_angle_exists()
    test_review_wires_the_offcam_speaker_flag()
    test_review_flags_a_named_feature_missing_from_the_edit()
    test_review_no_feature_flag_when_the_feature_is_present()
    test_review_no_feature_flag_without_an_ask()
    test_review_feature_flags_never_prescribe_a_fix()
    test_read_state_piece_breakdown_matches_beat_index_for_cluster()
    test_read_state_omits_pieces_for_single_event_cut()
    test_place_cluster_whole_vs_pieces_via_level()
    test_place_piece_resolves_to_one_beat_span()
    test_trim_works_on_a_placed_piece()
    test_place_bad_piece_is_clean_noop()
    test_place_piece_on_single_event_moment_is_noop()
    print("\nall observe/act tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
