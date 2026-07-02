#!/usr/bin/env python3
"""Tests for the Clip Timeline fusion (l3.clip_timeline) -- the continuous-editing
substrate: change-point lanes over the clip clock, reused seams, peaks, energy,
person cards, and the fresh scored cut index. Pure, no DB / no VLM. Run:
    PYTHONPATH=. .venv/bin/python scripts/test_clip_timeline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l1.fused_seams import FusedField, FusedSeam  # noqa: E402
from app.services.l3 import clip_timeline as ct  # noqa: E402
from app.services.l3.clip_timeline import (  # noqa: E402
    LANE_ACTION, LANE_GAZE, LANE_PRESENCE, LANE_SHOT, LANE_SPEAKING, LANE_SPEECH,
    Interval, TimelineInputs, build_clip_timeline,
)

FID = "aaaaaaaa-0000-0000-0000-000000000000"


def _word(a, b, text, speaker="S0"):
    return {"start_ms": a, "end_ms": b, "text": text, "speaker": speaker, "is_filler": False}


# --------------------------------------------------------------------------
# change-point primitives
# --------------------------------------------------------------------------

def test_merge_adjacent_collapses_equal_values():
    ivs = ct._merge_adjacent([
        Interval(0, 100, {"state": "on"}),
        Interval(100, 300, {"state": "on"}),   # equal -> merge
        Interval(300, 400, {"state": "off"}),  # change -> boundary
    ])
    assert [(i.start_ms, i.end_ms) for i in ivs] == [(0, 300), (300, 400)], ivs
    print("ok  adjacent equal-value intervals collapse to one change point")


def test_full_coverage_fills_gaps_and_tiles_clock():
    """A single mid-clip span yields default-fill on both sides, tiling [0,dur]."""
    ivs = ct._full_coverage([(200, 500, {"state": "on"})], 1000, {"state": "off"})
    assert (ivs[0].start_ms, ivs[-1].end_ms) == (0, 1000), ivs
    # contiguous, no gaps
    for a, b in zip(ivs, ivs[1:]):
        assert a.end_ms == b.start_ms, (a, b)
    assert ivs[1].value == {"state": "on"} and ivs[1].start_ms == 200, ivs
    print("ok  full coverage tiles the whole clock and fills gaps with default")


def test_full_coverage_later_span_wins_overlap():
    ivs = ct._full_coverage(
        [(0, 600, {"who": "a"}), (400, 1000, {"who": "b"})], 1000, {"who": None})
    # overlap [400,600] must read 'b' (later wins)
    mid = next(i for i in ivs if i.start_ms <= 500 < i.end_ms)
    assert mid.value["who"] == "b", ivs
    print("ok  overlapping spans resolve later-wins")


def test_merge_turns_groups_by_speaker_and_gap():
    words = [_word(0, 400, "hello"), _word(500, 900, "there"),      # same turn
             _word(5000, 5400, "later", speaker="S1")]              # new speaker+gap
    turns = ct._merge_turns(words)
    assert len(turns) == 2, turns
    assert turns[0]["text"] == "hello there" and turns[0]["end_ms"] == 900, turns
    assert turns[1]["speaker"] == "S1", turns
    print("ok  words merge into speaker turns by speaker + gap")


# --------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------

def test_speech_lane_labels_speech_and_silence():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        words=[_word(1000, 2000, "hi"), _word(2200, 3000, "world")]))
    speech = tl.lane(LANE_SPEECH)
    assert speech.full_coverage
    assert speech.value_at(0)["state"] == "silence", speech.intervals
    assert speech.value_at(1500)["state"] == "speech", speech.intervals
    assert speech.value_at(9000)["state"] == "silence", speech.intervals
    print("ok  speech lane covers clock with speech + silence")


def test_presence_lane_per_person():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        persons=[{"local_id": "p1", "enters_ms": 2000, "exits_ms": 8000,
                  "canonical_description": "tall man, beard"}]))
    pres = tl.lane(f"{LANE_PRESENCE}:p1")
    assert pres.value_at(1000)["state"] == "off"
    assert pres.value_at(5000)["state"] == "on"
    assert pres.value_at(9000)["state"] == "off"
    # facet_at collapses presence to on-screen id list
    assert tl.facet_at(5000)[LANE_PRESENCE] == ["p1"], tl.facet_at(5000)
    assert tl.facet_at(1000)[LANE_PRESENCE] == []
    print("ok  per-person presence lane + facet_at on-screen list")


def test_speaking_gaze_shot_lanes_full_coverage():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        speaking=[{"start_ms": 1000, "end_ms": 4000, "subject": "p1"}],
        gaze=[{"start_ms": 0, "end_ms": 5000, "subject": "p1", "direction": "to_camera"}],
        camera_craft=[{"start_ms": 0, "end_ms": 6000, "shot_size": "close_up",
                       "movement": "static"}]))
    assert tl.lane(LANE_SPEAKING).value_at(2000)["subject"] == "p1"
    assert tl.lane(LANE_SPEAKING).value_at(8000)["subject"] is None
    assert tl.lane(LANE_GAZE).value_at(1000)["direction"] == "to_camera"
    assert tl.lane(LANE_SHOT).value_at(1000)["shot_size"] == "close_up"
    assert tl.lane(LANE_SHOT).value_at(8000)["shot_size"] == "unsure"
    print("ok  speaking/gaze/shot lanes are full coverage")


def test_action_lane_is_sparse_events():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        atoms=[{"channel": "done", "subject": "person", "start_ms": 1000,
                "end_ms": 2000, "peak_ms": 1500, "label": "kick", "confidence": 0.8},
               {"channel": "shown", "subject": "object", "start_ms": 4000,
                "end_ms": 5000, "label": "the trophy", "confidence": 0.9}]))
    action = tl.lane(LANE_ACTION)
    assert not action.full_coverage
    assert len(action.intervals) == 2, action.intervals
    assert action.value_at(1500)["label"] == "kick"
    assert action.value_at(3000) is None       # gap between events is empty
    print("ok  action lane is sparse (events, not coverage)")


# --------------------------------------------------------------------------
# seams / peaks / energy / cut index
# --------------------------------------------------------------------------

def test_seams_passthrough_from_fused_field():
    fld = FusedField(hop_ms=100, cost=[0.1] * 100,
                     seams=[FusedSeam(1500, 0.9, "sentence_end", ["dialogue"])])
    tl = build_clip_timeline(TimelineInputs(file_id=FID, duration_ms=10000, field=fld))
    assert tl.seams and tl.seams[0]["ts_ms"] == 1500 and tl.seams[0]["kind"] == "sentence_end"
    print("ok  seams pass through from the reused fused field")


def test_peaks_from_atoms_and_motion():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        atoms=[{"channel": "done", "start_ms": 1000, "end_ms": 2000, "peak_ms": 1500,
                "actor": "p1", "confidence": 0.7, "label": "x"}],
        action_points=[{"ts_ms": 3000, "kind": "action_impact", "score": 1.0}]))
    kinds = {(p.ts_ms, p.kind) for p in tl.peaks}
    assert (1500, "done") in kinds and (3000, "action_impact") in kinds, tl.peaks
    print("ok  peaks fuse atom peaks + motion impacts")


def test_energy_normalized_0_1():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=1000, rms_db=[-40.0, -20.0, -60.0], prosody_hop_ms=100))
    assert tl.energy == [0.5, 1.0, 0.0], tl.energy
    assert tl.energy_hop_ms == 100
    print("ok  energy curve normalized to 0..1")


def test_cut_index_said_and_video_snapped_and_scored():
    fld = FusedField(hop_ms=100, cost=[0.0] * 200,
                     seams=[FusedSeam(0, 1.0, "rest", []), FusedSeam(6000, 1.0, "sentence_end", [])])
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=12000, field=fld,
        words=[_word(500, 5800, "a long substantive spoken turn here")],
        rms_db=[0.2] * 60, prosody_hop_ms=100,
        atoms=[{"channel": "shown", "subject": "place", "start_ms": 8000,
                "end_ms": 10000, "peak_ms": 9000, "label": "vista", "confidence": 0.95}]))
    kinds = {c.kind for c in tl.cuts}
    assert "said" in kinds and "shown" in kinds, tl.cuts
    # highest-scored first; the 0.95-confidence shown atom should top the said turn
    assert tl.cuts[0].kind == "shown" and tl.cuts[0].score == 0.95, tl.cuts
    said = next(c for c in tl.cuts if c.kind == "said")
    assert said.peak_ms is not None and said.speaker == "S0"
    print("ok  cut index: said + video bookmarks, seam-snapped and scored")


def test_scan_and_handles_queries():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        speaking=[{"start_ms": 1000, "end_ms": 4000, "subject": "p1"}]))
    hits = tl.scan(LANE_SPEAKING, subject="p1")
    assert len(hits) == 1 and (hits[0].start_ms, hits[0].end_ms) == (1000, 4000), hits
    h = tl.handles(3000, 7000)
    assert h == {"lead_ms": 3000, "tail_ms": 3000}, h
    print("ok  scan() facet query + handles() room query")


def test_presence_lane_v8_dense_coverage_and_reentry():
    """The dense v8 presence_lane handles re-entry (off in the middle) that
    coarse enters/exits cannot express."""
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=10000,
        persons=[{"local_id": "p1"}, {"local_id": "p2"}],
        presence_lane=[
            {"start_ms": 0, "end_ms": 3000, "present": ["p1"]},
            {"start_ms": 3000, "end_ms": 6000, "present": ["p1", "p2"]},
            {"start_ms": 6000, "end_ms": 10000, "present": ["p2"]},  # p1 left
        ]))
    p1 = tl.lane(f"{LANE_PRESENCE}:p1")
    assert p1.value_at(1000)["state"] == "on"
    assert p1.value_at(8000)["state"] == "off"       # re-entry/exit expressed
    assert tl.facet_at(4000)[LANE_PRESENCE] == ["p1", "p2"], tl.facet_at(4000)
    assert tl.facet_at(8000)[LANE_PRESENCE] == ["p2"]
    print("ok  v8 presence_lane gives dense per-person coverage w/ re-entry")


def test_activity_lane_v8_held_reaction_is_addressable():
    """A silent HELD reaction in activity_lane becomes an action-lane event AND a
    placeable cut-index bookmark -- the case the sparse atoms miss."""
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=12000,
        activity_lane=[
            {"start_ms": 0, "end_ms": 4000, "mode": "action", "subject": "person",
             "actor": "p1", "label": "makes the point", "peak_ms": 2000, "confidence": 0.8},
            {"start_ms": 4000, "end_ms": 5000, "mode": "idle"},          # dropped
            {"start_ms": 5000, "end_ms": 9000, "mode": "held", "subject": "person",
             "actor": "p2", "label": "listens, silent", "peak_ms": 7000, "confidence": 0.7},
        ]))
    action = tl.lane(LANE_ACTION)
    assert len(action.intervals) == 2, action.intervals    # idle skipped
    held = action.value_at(7000)
    assert held["channel"] == "shown" and held["label"] == "listens, silent", held
    # the held reaction is a placeable bookmark in the index
    assert any(c.kind == "shown" and c.label == "listens, silent" for c in tl.cuts), tl.cuts
    # peaks fuse from activity
    assert any(p.ts_ms == 7000 and p.kind == "shown" for p in tl.peaks), tl.peaks
    print("ok  v8 activity_lane: silent held reaction is addressable (event+bookmark+peak)")


def test_render_awareness_digest():
    fld = FusedField(hop_ms=100, cost=[0.0] * 120,
                     seams=[FusedSeam(6000, 0.95, "sentence_end", ["dialogue"])])
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=12000, field=fld,
        words=[_word(500, 5800, "a substantive spoken turn")],
        rms_db=[0.3] * 60, prosody_hop_ms=100,
        persons=[{"local_id": "p1", "role": "main subject",
                  "canonical_description": "tall man, beard", "enters_ms": 0,
                  "exits_ms": 12000, "voice_speaker_id": "S0"}],
        speaking=[{"start_ms": 500, "end_ms": 5800, "subject": "p1"}],
        camera_craft=[{"start_ms": 0, "end_ms": 12000, "shot_size": "close_up"}],
        atoms=[{"channel": "shown", "subject": "place", "start_ms": 8000,
                "end_ms": 10000, "peak_ms": 9000, "label": "vista", "confidence": 0.9}]))
    text = ct.render_awareness(tl)
    assert "CLIP" in text and "PEOPLE:" in text
    assert "tall man, beard" in text
    assert "CUT INDEX" in text and "[shown]" in text
    assert "SEAMS" in text and "PEAKS" in text
    print("ok  render_awareness produces a complete digest")
    print("--- sample awareness digest ---\n" + text + "\n-------------------------------")


def test_serialization_roundtrips_shape():
    tl = build_clip_timeline(TimelineInputs(
        file_id=FID, duration_ms=5000,
        words=[_word(0, 1000, "hi")],
        persons=[{"local_id": "p1", "canonical_description": "person"}]))
    d = tl.to_dict()
    assert d["version"] == ct.CLIP_TIMELINE_VERSION
    assert {"lanes", "seams", "peaks", "energy", "cuts", "persons"} <= set(d)
    assert any(ln["name"] == LANE_SPEECH for ln in d["lanes"])
    print("ok  to_dict serializes the expected shape")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall clip_timeline tests passed")
