"""
Tests for the footage moment-tree builder (no DB).

Exercises the pure breakdown: five energy bands of hero cuts for one clip
collapsed into balanced-anchored MOMENTS that fold in their coarser "widen"
variants and finer "atom" splits, then the compact Tier-0 map text and Tier-1
moment record. Run:  .venv/bin/python scripts/test_footage_map.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import footage_map as fm  # noqa: E402


def _cut(hero_id, in_ms, out_ms, label="", modality="speech",
         speaker="S0", score=0.5, play_ms=None, keep_spans=None):
    return {
        "hero_id": hero_id, "file_id": "ffffffff-1111", "modality": modality,
        "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
        "play_ms": play_ms if play_ms is not None else (out_ms - in_ms),
        "keep_spans": keep_spans, "score": score, "speaker": speaker,
        "affordances": [modality], "flags": [], "take_count": 1,
    }


def _bands():
    """A clip with one long answer (0-8s) that, at balanced, is two thoughts,
    and at sharp splits into four sentence atoms."""
    return {
        0: [_cut("f:ans", 0, 8000, "the whole answer about the pivot", score=0.8)],     # broad
        1: [_cut("f:ans", 0, 8000, "the whole answer about the pivot", score=0.8)],     # calm
        2: [_cut("f:t0", 0, 4000, "we almost shut the company down", score=0.82),       # balanced
            _cut("f:t1", 4000, 8000, "then one customer changed everything", score=0.78)],
        3: [_cut("f:s0", 0, 2000, "we almost shut down", score=0.7),
            _cut("f:s1", 2000, 4000, "out of money", score=0.66),
            _cut("f:s2", 4000, 6000, "then one customer", score=0.72),
            _cut("f:s3", 6000, 8000, "changed everything", score=0.71)],
        4: [_cut("f:s0", 0, 2000, "we almost shut down", score=0.7),
            _cut("f:s1", 2000, 4000, "out of money", score=0.66),
            _cut("f:s2", 4000, 6000, "then one customer", score=0.72),
            _cut("f:s3", 6000, 8000, "changed everything", score=0.71)],
    }


def test_moments_anchor_on_balanced():
    tree = fm.build_clip_tree("ffffffff-1111",
                              {"name": "Take 2", "duration_ms": 8000,
                               "content_type": "interview", "primary_axis": "dialogue"},
                              _bands())
    assert tree["moment_count"] == 2, tree["moment_count"]
    ids = [m["moment_id"] for m in tree["moments"]]
    assert ids == ["ffffffff:m00", "ffffffff:m01"], ids
    assert tree["moments"][0]["gist"] == "we almost shut the company down"
    print("ok  test_moments_anchor_on_balanced")


def test_widen_and_atoms_fold_in():
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, _bands())
    m0 = tree["moments"][0]
    # The broad/calm answer (0-8s) contains m0's center (2s) -> widen variants.
    assert "broad" in m0["variants"] and "calm" in m0["variants"], m0["variants"].keys()
    assert m0["variants"]["broad"]["out_ms"] == 8000
    assert m0["variants"]["balanced"]["out_ms"] == 4000
    # m0 spans 0-4s -> its two sharp sentences (0-2, 2-4) fold in as atoms.
    assert [a["atom_id"] for a in m0["atoms"]] == ["f:s0", "f:s1"], m0["atoms"]
    m1 = tree["moments"][1]
    assert [a["atom_id"] for a in m1["atoms"]] == ["f:s2", "f:s3"], m1["atoms"]
    print("ok  test_widen_and_atoms_fold_in")


def test_map_text_is_compact_and_complete():
    tree = fm.build_clip_tree("ffffffff-1111",
                              {"name": "Take 2", "duration_ms": 8000,
                               "content_type": "interview"}, _bands())
    block = fm._clip_block(tree)
    lines = block.splitlines()
    assert lines[0].startswith('CLIP ffffffff "Take 2"'), lines[0]
    assert len(lines) == 1 + tree["moment_count"]            # header + one line per moment
    assert "nrg:broad|calm|balanced" in lines[1], lines[1]
    # Resident mode lists each atom by id + gist (not just a count).
    assert "· atoms:" in lines[1] and 's0="' in lines[1], lines[1]
    # Compact mode falls back to the count (paged path leans on inspect_moment).
    cblock = fm._clip_block(tree, compact=True)
    assert "(+2 atoms)" in cblock.splitlines()[1], cblock
    print("ok  test_map_text_is_compact_and_complete")


def _thought_bands():
    """A clip with ONE thought whose five levels are single nested spans: the
    turn / run-up (broad/calm) -> the thought (balanced) -> the core sentence
    (tight) -> the punchline clause (sharp)."""
    return {
        0: [_cut("f:turn", 0, 8000, "so anyway we almost shut the company down", score=0.8)],
        1: [_cut("f:calm", 500, 4000, "so we almost shut the company down", score=0.8)],
        2: [_cut("f:th", 1000, 4000, "we almost shut the company down", score=0.82)],
        3: [_cut("f:core", 1000, 3000, "we almost shut down", score=0.7)],
        4: [_cut("f:punch", 1500, 3000, "shut down", score=0.7)],
    }


def test_thought_levels_become_variants():
    """Each nested level (incl. tight=core) is a selectable VARIANT, and a single
    thought yields NO atoms (nothing to sub-pick)."""
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000},
                              _thought_bands())
    assert tree["moment_count"] == 1, tree["moment_count"]
    m = tree["moments"][0]
    assert set(m["variants"].keys()) == {"broad", "calm", "balanced", "tight", "sharp"}, \
        m["variants"].keys()
    assert m["variants"]["tight"]["out_ms"] == 3000, m["variants"]["tight"]
    assert m["variants"]["sharp"]["in_ms"] == 1500, m["variants"]["sharp"]
    assert m["atoms"] == [], m["atoms"]
    line = fm._moment_line(m)
    assert "nrg:broad|calm|balanced|tight|sharp" in line, line
    assert "atoms" not in line, line
    print("ok  test_thought_levels_become_variants")


def test_anchor_fallback_when_balanced_empty():
    """A modality present only in coarser bands still yields moments (anchor
    falls back rather than dropping the clip)."""
    bands = {0: [], 1: [_cut("f:a", 0, 3000, "held wide shot", modality="broll", speaker=None)],
             2: [], 3: [], 4: []}
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "B", "duration_ms": 3000}, bands)
    assert tree["moment_count"] == 1, tree["moment_count"]
    assert tree["moments"][0]["modality"] == "broll"
    print("ok  test_anchor_fallback_when_balanced_empty")


def main():
    test_moments_anchor_on_balanced()
    test_widen_and_atoms_fold_in()
    test_thought_levels_become_variants()
    test_map_text_is_compact_and_complete()
    test_anchor_fallback_when_balanced_empty()
    print("\nall footage-map tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
