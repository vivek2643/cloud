"""
Tests for the footage moment-tree builder (no DB).

Exercises the owned-ladder breakdown: each cut carries its own zoom ladder, so a
moment reads its VARIANTS straight off the rungs (no cross-band re-matching, no
atoms; a split is a multi-span rung). Then the compact Tier-0 map text and
Tier-1 moment record. Run:  .venv/bin/python scripts/test_footage_map.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import footage_map as fm  # noqa: E402


def _rung(level, in_ms, out_ms, text="", score=0.5, spans=None):
    sp = spans or [(in_ms, out_ms)]
    return {
        "level": level,
        "spans": [{"in_ms": a, "out_ms": b} for a, b in sp],
        "in_ms": min(a for a, _ in sp), "out_ms": max(b for _, b in sp),
        "play_ms": sum(b - a for a, b in sp), "text": text, "score": score,
    }


def _cut(hero_id, in_ms, out_ms, label="", channel="said", subject="person",
         speaker="S0", score=0.5, play_ms=None, keep_spans=None, ladder=None, **extra):
    d = {
        "hero_id": hero_id, "file_id": "ffffffff-1111", "channel": channel,
        "subject": subject, "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
        "play_ms": play_ms if play_ms is not None else (out_ms - in_ms),
        "keep_spans": keep_spans, "score": score, "speaker": speaker,
        "flags": [], "take_count": 1, "ladder": ladder,
    }
    d.update(extra)
    return d


def _thought_cut():
    """One thought cut whose ladder is the five nested zooms: the turn / run-up
    (broad/calm) -> the thought (balanced) -> core sentence (tight) -> punchline
    clause (sharp)."""
    return _cut("f:th", 1000, 4000, "we almost shut the company down", score=0.82,
                ladder=[
                    _rung("broad", 0, 8000, "so anyway we almost shut the company down", 0.8),
                    _rung("calm", 500, 4000, "so we almost shut the company down", 0.8),
                    _rung("balanced", 1000, 4000, "we almost shut the company down", 0.82),
                    _rung("tight", 1000, 3000, "we almost shut down", 0.7),
                    _rung("sharp", 1500, 3000, "shut down", 0.7),
                ])


def test_thought_levels_become_variants():
    """Every nested level (incl. tight=core) is a selectable VARIANT read off the
    cut's ladder; a single thought yields NO atoms."""
    tree = fm.build_clip_tree("ffffffff-1111",
                              {"name": "Take 2", "duration_ms": 8000,
                               "content_type": "interview", "primary_axis": "dialogue"},
                              [_thought_cut()])
    assert tree["moment_count"] == 1, tree["moment_count"]
    m = tree["moments"][0]
    assert m["moment_id"] == "ffffffff:m00", m["moment_id"]
    assert set(m["variants"].keys()) == {"broad", "calm", "balanced", "tight", "sharp"}, \
        m["variants"].keys()
    # The moment anchors on balanced (one complete thought per cut).
    assert (m["in_ms"], m["out_ms"]) == (1000, 4000), (m["in_ms"], m["out_ms"])
    assert m["variants"]["broad"]["out_ms"] == 8000
    assert m["variants"]["tight"]["out_ms"] == 3000
    assert m["variants"]["sharp"]["in_ms"] == 1500
    assert m["atoms"] == [], m["atoms"]
    print("ok  test_thought_levels_become_variants")


def test_split_rung_becomes_keep_spans():
    """A multi-span rung (a jump-cut / breath-excised split) surfaces as the
    variant's keep_spans, not separate atoms."""
    cut = _cut("f:sp", 1000, 4000, "the product really changes everything", score=0.7,
               ladder=[
                   _rung("balanced", 1000, 4000, "the product really changes everything", 0.7),
                   _rung("sharp", 1000, 4000, "really changes everything", 0.8,
                         spans=[(1000, 1800), (2600, 4000)]),
               ])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    sharp = tree["moments"][0]["variants"]["sharp"]
    assert sharp["keep_spans"] == [[1000, 1800], [2600, 4000]], sharp["keep_spans"]
    assert sharp["in_ms"] == 1000 and sharp["out_ms"] == 4000
    assert tree["moments"][0]["variants"]["balanced"]["keep_spans"] is None
    print("ok  test_split_rung_becomes_keep_spans")


def test_no_ladder_uses_flat_span():
    """A legacy cut with no ladder still yields a moment (balanced variant from
    its flat span)."""
    cut = _cut("f:legacy", 0, 3000, "held wide shot", channel="shown", subject="place",
               speaker=None, ladder=None)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "B", "duration_ms": 3000}, [cut])
    assert tree["moment_count"] == 1, tree["moment_count"]
    m = tree["moments"][0]
    assert m["channel"] == "shown" and m["subject"] == "place"
    assert m["variants"]["balanced"]["out_ms"] == 3000
    print("ok  test_no_ladder_uses_flat_span")


def test_facets_surface_on_moment():
    """People / framing / quality facets ride along onto the moment for the
    brain to read."""
    cut = _cut("f:th", 1000, 4000, "a clean line", score=0.8,
               ladder=[_rung("balanced", 1000, 4000, "a clean line", 0.8)],
               people=[{"voice_speaker_id": "S0", "person_id": "p1", "role": "host",
                        "on_camera": True}],
               framing={"shot_size": "medium", "region": {"x": 0.3}},
               quality={"delivery": 0.81, "on_camera": 1.0})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["people"][0]["person_id"] == "p1"
    assert m["framing"]["shot_size"] == "medium"
    assert m["quality"]["delivery"] == 0.81
    print("ok  test_facets_surface_on_moment")


def test_map_text_lists_variants_no_atoms():
    tree = fm.build_clip_tree("ffffffff-1111",
                              {"name": "Take 2", "duration_ms": 8000,
                               "content_type": "interview"}, [_thought_cut()])
    block = fm._clip_block(tree)
    lines = block.splitlines()
    assert lines[0].startswith('CLIP ffffffff "Take 2"'), lines[0]
    assert len(lines) == 1 + tree["moment_count"]
    assert "nrg:broad|calm|balanced|tight|sharp" in lines[1], lines[1]
    assert "atoms" not in lines[1], lines[1]
    print("ok  test_map_text_lists_variants_no_atoms")


def test_moment_line_flags_offcamera():
    """The brain's one-line index keys on CHANNEL.SUBJECT and flags an off-camera
    voice (an off-screen interviewer / voiceover)."""
    cut = _cut("f:mo", 1000, 4000, "what made you start this", channel="said",
               subject="person", speaker="interviewer", score=0.7,
               ladder=[_rung("balanced", 1000, 4000, "what made you start this", 0.7)],
               flags=["offscreen"],
               people=[{"voice_speaker_id": "S9", "person_id": None, "on_camera": False}])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "said.person" in line, line
    assert "interviewer off-cam" in line, line
    print("ok  test_moment_line_flags_offcamera")


def test_moment_line_shows_channel_subject_v2():
    """cuts-v2: the resident line keys on CHANNEL.SUBJECT."""
    cut = _cut("f:v0", 1000, 4000, "kicks the ball", channel="done", subject="person",
               speaker="p1", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "kicks the ball", 0.6)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["channel"] == "done" and m["subject"] == "person"
    assert fm._capture_tag(m) == "done.person", fm._capture_tag(m)
    print("ok  test_moment_line_shows_channel_subject_v2")


def test_default_energy_from_genre():
    """The tree opens the slider per genre: long-form calm, short-form punchy."""
    calm = fm.build_clip_tree("ffffffff-2222",
                              {"name": "Pod", "duration_ms": 8000, "content_type": "interview"},
                              [_thought_cut()])
    punchy = fm.build_clip_tree("ffffffff-3333",
                                {"name": "Ad", "duration_ms": 8000, "content_type": "product"},
                                [_thought_cut()])
    assert calm["default_energy"] < 0.5 < punchy["default_energy"], (
        calm["default_energy"], punchy["default_energy"])
    print("ok  test_default_energy_from_genre")


def main():
    test_thought_levels_become_variants()
    test_moment_line_flags_offcamera()
    test_moment_line_shows_channel_subject_v2()
    test_split_rung_becomes_keep_spans()
    test_no_ladder_uses_flat_span()
    test_facets_surface_on_moment()
    test_map_text_lists_variants_no_atoms()
    test_default_energy_from_genre()
    print("\nall footage-map tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
