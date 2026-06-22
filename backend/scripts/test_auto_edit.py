#!/usr/bin/env python3
"""Tests for the L3 v2 auto-editor pipeline (hermetic; FakeLLM, no network/DB).

Run:  PYTHONPATH=. python scripts/test_auto_edit.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import auto_edit as ae  # noqa: E402
from app.services.llm.base import LLMResponse  # noqa: E402


class FakeLLM:
    """Returns canned JSON per stage, keyed off a marker in the system prompt."""

    def __init__(self, director="{}", editor="{}", coverage="{}"):
        self._r = {"DIRECTOR": director, "EDITOR": editor, "COVERAGE": coverage}
        self.calls = []

    @property
    def model(self) -> str:
        return "fake-model"

    def run(self, *, system, messages, **kwargs):
        for key, text in self._r.items():
            if key in system:
                self.calls.append(key)
                return LLMResponse(text=text)
        self.calls.append("?")
        return LLMResponse(text="{}")


def _cut(hid, fid, modality="speech", label="x", t_in=0, t_out=2000,
         score=0.5, keep_spans=None, speaker="S0"):
    return {
        "hero_id": hid, "file_id": fid, "modality": modality, "label": label,
        "src_in_ms": t_in, "src_out_ms": t_out, "duration_ms": t_out - t_in,
        "play_ms": (sum(s["out_ms"] - s["in_ms"] for s in keep_spans)
                    if keep_spans else t_out - t_in),
        "keep_spans": keep_spans, "score": score, "speaker": speaker,
    }


def test_coerce_plan():
    p = ae._coerce_plan({"energy": 2.0, "aspect": "vertical", "spine_kind": "xyz",
                         "target_duration_ms": "30000",
                         "beats": [{"purpose": "hook", "intent": "grab"}, {}]})
    assert p.energy == 1.0                      # clamped
    assert p.aspect == "landscape"              # unknown -> default
    assert p.spine_kind == "dialogue"           # unknown -> default
    assert p.target_duration_ms == 30000
    assert len(p.beats) == 1                    # empty beat dropped
    empty = ae._coerce_plan(None)
    assert empty.energy == 0.5 and empty.aspect == "landscape"
    print("ok  coerce_plan clamps + defaults")


def test_editor_validates_orders_dedupes():
    pool = [_cut("f1:a", "f1"), _cut("f1:b", "f1"), _cut("f2:c", "f2")]
    llm = FakeLLM(editor=json.dumps({"picks": [
        {"hero_id": "f2:c", "reason": "open"},
        {"hero_id": "ghost", "reason": "hallucinated"},   # dropped (unknown)
        {"hero_id": "f1:a", "reason": "mid"},
        {"hero_id": "f2:c", "reason": "dupe"},            # dropped (seen)
    ]}))
    picks = ae._editor("brief", ae.Plan(), pool, llm)
    assert [p["hero_id"] for p in picks] == ["f2:c", "f1:a"], picks
    print("ok  editor validates + orders + dedupes")


def test_segments_keep_spans_expansion():
    by_id = {
        "f1:a": _cut("f1:a", "f1", t_in=0, t_out=3000,
                     keep_spans=[{"in_ms": 0, "out_ms": 1000},
                                 {"in_ms": 2000, "out_ms": 3000}]),
        "f1:b": _cut("f1:b", "f1", t_in=5000, t_out=6000),
    }
    picks = [{"hero_id": "f1:a", "beat": "hook", "reason": "r"},
             {"hero_id": "f1:b", "beat": None, "reason": "r2"}]
    segs = ae._segments_from_picks(picks, by_id)
    # First pick (2 kept spans) -> 2 segments; second -> 1. Order preserved.
    assert len(segs) == 3, segs
    assert segs[0]["in_ms"] == 0 and segs[0]["out_ms"] == 1000
    assert segs[1]["in_ms"] == 2000 and segs[1]["out_ms"] == 3000
    assert segs[2]["in_ms"] == 5000
    assert len({s["seg_id"] for s in segs}) == 3        # unique ids
    assert segs[0]["content"] == "x" and segs[0]["beat_id"] == "hook"
    print("ok  keep_spans -> back-to-back segments")


def test_operations_from_coverage():
    by_id = {"b1": _cut("b1", "fb", modality="broll", t_in=1000, t_out=9000)}
    cov = {"overlays": [
        {"hero_id": "b1", "from_ms": 2000, "to_ms": 4000, "reason": "illustrate"},
        {"hero_id": "ghost", "from_ms": 0, "to_ms": 1000},      # dropped
        {"hero_id": "b1", "from_ms": 0, "to_ms": 100},          # too short, dropped
    ]}
    ops = ae._operations_from_coverage(cov, by_id, total_ms=10000)
    assert len(ops) == 1, ops
    op = ops[0]
    assert op["type"] == "place_video" and op["source_file_id"] == "fb"
    assert op["from_ms"] == 2000 and op["to_ms"] == 4000        # 2s span
    assert op["src_in_ms"] == 1000 and op["src_out_ms"] == 3000  # span from cut start
    print("ok  coverage overlays -> place_video ops")


def test_apply_trims():
    segs = [{"seg_id": "s0", "in_ms": 0, "out_ms": 5000, "priority": 3},
            {"seg_id": "s1", "in_ms": 0, "out_ms": 5000, "priority": 5},
            {"seg_id": "s2", "in_ms": 0, "out_ms": 5000, "priority": 1}]
    # Explicit drop of s1; target 6s forces another drop (weakest priority first).
    kept = ae._apply_trims(segs, {"trims": [{"seg_id": "s1"}]},
                           ae.Plan(target_duration_ms=6000))
    ids = [s["seg_id"] for s in kept]
    assert "s1" not in ids                       # explicit drop
    assert "s2" in ids                           # highest-priority survives
    assert sum(s["out_ms"] - s["in_ms"] for s in kept) <= 6000 * 1.1
    print("ok  apply_trims drops + fits target")


def test_make_edit_end_to_end(monkeypatch=None):
    """Full pipeline with stubbed clip cards / feed / durations + FakeLLM."""
    feed = [
        _cut("f1:a", "f1", label="the hook line", score=0.9),
        _cut("f1:b", "f1", label="the middle point", score=0.7),
        _cut("fb:x", "fb", modality="broll", label="b-roll pour", t_out=4000),
    ]
    orig_cards, orig_feed = ae._clip_cards, ae.hero_store.get_hero_feed
    import app.services.render.tasks as rt
    orig_dur = rt._durations
    ae._clip_cards = lambda fids: {f: {"file_id": f, "name": f, "duration_ms": 600000,
                                       "best_use": [], "topics": [], "people": []}
                                   for f in fids}
    ae.hero_store.get_hero_feed = lambda fids, energy=0.5, **kw: feed
    rt._durations = lambda fids: {f: 600000 for f in fids}
    llm = FakeLLM(
        director=json.dumps({"energy": 0.7, "aspect": "portrait",
                             "spine_kind": "dialogue", "intent": "tell it",
                             "beats": [{"purpose": "hook", "intent": "open"}]}),
        editor=json.dumps({"picks": [
            {"hero_id": "f1:a", "beat": "hook", "reason": "strong open"},
            {"hero_id": "f1:b", "beat": None, "reason": "support"},
        ]}),
        coverage=json.dumps({"overlays": [
            {"hero_id": "fb:x", "from_ms": 2000, "to_ms": 3500, "reason": "cover"}],
            "notes": "kept it sparse"}),
    )
    # Default: pure assembler -- no coverage pass, no overlay operations.
    try:
        result = ae.make_edit(["f1", "fb"], "make a punchy reel", llm=llm)
    finally:
        ae._clip_cards, ae.hero_store.get_hero_feed = orig_cards, orig_feed
        rt._durations = orig_dur

    doc = result.document
    assert result.plan.energy == 0.7 and result.plan.aspect == "portrait"
    assert doc["format"]["aspect"] == "portrait"
    assert [s["hero_id"] for s in doc["timeline"]] == ["f1:a", "f1:b"], doc["timeline"]
    assert doc["operations"] == [], doc["operations"]
    assert doc["resolved"]["video_layers"], "resolved layers must exist"
    assert llm.calls == ["DIRECTOR", "EDITOR"], llm.calls
    print("ok  make_edit end-to-end (pure assembler: pick + order, no ops)")


def test_make_edit_coverage_when_enabled():
    """With autoedit_coverage on, the coverage pass runs and lays an overlay."""
    feed = [
        _cut("f1:a", "f1", label="the hook line", score=0.9),
        _cut("f1:b", "f1", label="the middle point", score=0.7),
        _cut("fb:x", "fb", modality="broll", label="b-roll pour", t_out=4000),
    ]
    orig_cards, orig_feed = ae._clip_cards, ae.hero_store.get_hero_feed
    import app.services.render.tasks as rt
    orig_dur = rt._durations
    ae._clip_cards = lambda fids: {f: {"file_id": f, "name": f, "duration_ms": 600000,
                                       "best_use": [], "topics": [], "people": []}
                                   for f in fids}
    ae.hero_store.get_hero_feed = lambda fids, energy=0.5, **kw: feed
    rt._durations = lambda fids: {f: 600000 for f in fids}
    llm = FakeLLM(
        director=json.dumps({"energy": 0.7, "aspect": "portrait",
                             "spine_kind": "dialogue", "intent": "tell it",
                             "beats": [{"purpose": "hook", "intent": "open"}]}),
        editor=json.dumps({"picks": [
            {"hero_id": "f1:a", "beat": "hook", "reason": "strong open"},
            {"hero_id": "f1:b", "beat": None, "reason": "support"},
        ]}),
        coverage=json.dumps({"overlays": [
            {"hero_id": "fb:x", "from_ms": 2000, "to_ms": 3500, "reason": "cover"}],
            "notes": "kept it sparse"}),
    )
    from app.config import get_settings
    settings = get_settings()
    prev = settings.autoedit_coverage
    settings.autoedit_coverage = True
    try:
        result = ae.make_edit(["f1", "fb"], "make a punchy reel", llm=llm)
    finally:
        settings.autoedit_coverage = prev
        ae._clip_cards, ae.hero_store.get_hero_feed = orig_cards, orig_feed
        rt._durations = orig_dur

    doc = result.document
    assert len(doc["operations"]) == 1 and doc["operations"][0]["type"] == "place_video"
    assert llm.calls == ["DIRECTOR", "EDITOR", "COVERAGE"], llm.calls
    print("ok  make_edit coverage path (enabled -> 1 overlay op)")


def test_make_edit_fallback_on_editor_failure():
    feed = [_cut("f1:a", "f1", score=0.9), _cut("f1:b", "f1", score=0.4)]
    orig_cards, orig_feed = ae._clip_cards, ae.hero_store.get_hero_feed
    import app.services.render.tasks as rt
    orig_dur = rt._durations
    ae._clip_cards = lambda fids: {f: {"file_id": f, "name": f, "duration_ms": 60000,
                                       "best_use": [], "topics": [], "people": []}
                                   for f in fids}
    ae.hero_store.get_hero_feed = lambda fids, energy=0.5, **kw: feed
    rt._durations = lambda fids: {f: 60000 for f in fids}
    # Editor returns no usable picks -> deterministic fallback kicks in.
    llm = FakeLLM(director=json.dumps({"energy": 0.5, "spine_kind": "dialogue"}),
                  editor=json.dumps({"picks": []}))
    try:
        result = ae.make_edit(["f1"], "", llm=llm)
    finally:
        ae._clip_cards, ae.hero_store.get_hero_feed = orig_cards, orig_feed
        rt._durations = orig_dur
    assert result.selected >= 1, "fallback must still produce a draft"
    print("ok  make_edit fallback on editor failure")


def main():
    test_coerce_plan()
    test_editor_validates_orders_dedupes()
    test_segments_keep_spans_expansion()
    test_operations_from_coverage()
    test_apply_trims()
    test_make_edit_end_to_end()
    test_make_edit_coverage_when_enabled()
    test_make_edit_fallback_on_editor_failure()
    print("\nall auto-edit tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
