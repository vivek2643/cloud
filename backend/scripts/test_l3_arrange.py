"""
Tests for the free-canvas arranger (no DB, fake LLM).

Builds a real moment-tree map, feeds the arranger a model response, and checks
that ids are validated, illegal energy levels normalise, placements resolve to
the right source spans, and the main line compiles to contiguous segments (the
no-gap critic). Run:  .venv/bin/python scripts/test_l3_arrange.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import arrange as ar  # noqa: E402
from app.services.l3 import footage_map as fm  # noqa: E402


class _FakeLLM:
    """Returns a fixed JSON body for the one arranger call."""
    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def model(self) -> str:
        return "fake"

    def run(self, *, system, messages, max_tokens=2048, effort=None, **kw):
        class _R:
            text = self._text
        return _R()


class _Plan:
    energy = 0.5
    aspect = "landscape"
    spine_kind = "dialogue"
    target_duration_ms = None
    intent = "the pivot story"
    rationale = ""
    beats: list = []


def _cut(hero_id, in_ms, out_ms, label="", modality="speech", score=0.6, keep_spans=None):
    return {"hero_id": hero_id, "file_id": "ffffffff-1111", "modality": modality,
            "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
            "play_ms": out_ms - in_ms, "keep_spans": keep_spans, "score": score,
            "speaker": "S0", "affordances": [modality], "flags": []}


def _map():
    bands = {
        0: [_cut("f:ans", 0, 8000, "whole answer", score=0.8)],
        1: [_cut("f:ans", 0, 8000, "whole answer", score=0.8)],
        2: [_cut("f:t0", 0, 4000, "we almost shut down", score=0.82),
            _cut("f:t1", 4000, 8000, "one customer changed everything", score=0.78)],
        3: [_cut("f:s0", 0, 2000, "we almost shut down", score=0.7),
            _cut("f:s1", 2000, 4000, "out of money", score=0.66)],
        4: [_cut("f:s0", 0, 2000, "we almost shut down", score=0.7),
            _cut("f:s1", 2000, 4000, "out of money", score=0.66)],
    }
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, bands)
    return {"clips": [tree]}


def test_validates_and_normalises():
    body = """{"timeline": [
        {"ref": "ffffffff:m01", "level": "balanced", "track": 0},
        {"ref": "ffffffff:m00", "level": "calm", "track": 0},
        {"ref": "ffffffff:m00", "level": "balanced", "track": 0},
        {"ref": "bogus:m99", "level": "balanced", "track": 0}
    ], "notes": "hi"}"""
    places = ar.arrange("the pivot", _map(), _Plan(), llm=_FakeLLM(body), map_text="(map)")
    refs = [(p.ref, p.level) for p in places]
    # bogus dropped, duplicate m00 dropped, order preserved, m00 'calm' is a real
    # widen level so it stays 'calm'.
    assert refs == [("ffffffff:m01", "balanced"), ("ffffffff:m00", "calm")], refs
    print("ok  test_validates_and_normalises")


def test_illegal_level_falls_back():
    body = '{"timeline": [{"ref": "ffffffff:m01", "level": "broad", "track": 0}]}'
    # m01 spans 4-8s; the broad answer (0-8s) contains it, so 'broad' is legal.
    places = ar.arrange("x", _map(), _Plan(), llm=_FakeLLM(body), map_text="(m)")
    assert places[0].level == "broad", places[0].level
    body2 = '{"timeline": [{"ref": "ffffffff:m01", "level": "nonsense", "track": 0}]}'
    places2 = ar.arrange("x", _map(), _Plan(), llm=_FakeLLM(body2), map_text="(m)")
    assert places2[0].level == "balanced", places2[0].level
    print("ok  test_illegal_level_falls_back")


def test_resolve_and_main_line_is_contiguous():
    places = [ar.Placement("ffffffff:m00", "broad", 0),
              ar.Placement("ffffffff:m01", "balanced", 0)]
    cuts = ar.resolve_placements(places, _map())
    assert len(cuts) == 2
    assert (cuts[0].src_in_ms, cuts[0].src_out_ms) == (0, 8000)   # m00 widened to answer
    assert (cuts[1].src_in_ms, cuts[1].src_out_ms) == (4000, 8000)  # m01 balanced
    segs = ar._segments_from_main(cuts)
    assert len(segs) == 2
    assert [s["seg_id"] for s in segs] == ["a000_0", "a001_0"], segs
    assert segs[0]["file_id"] == "ffffffff-1111"
    print("ok  test_resolve_and_main_line_is_contiguous")


def test_atom_ref_resolves_to_subcut():
    places = [ar.Placement("f:s1", "sharp", 0)]   # an atom id
    cuts = ar.resolve_placements(places, _map())
    assert len(cuts) == 1
    assert (cuts[0].src_in_ms, cuts[0].src_out_ms) == (2000, 4000), cuts
    assert cuts[0].label == "out of money"
    print("ok  test_atom_ref_resolves_to_subcut")


def main():
    test_validates_and_normalises()
    test_illegal_level_falls_back()
    test_resolve_and_main_line_is_contiguous()
    test_atom_ref_resolves_to_subcut()
    print("\nall arranger tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
