#!/usr/bin/env python3
"""Tests for brain_perception_blindness.plan.md Part A (A1 takes:, A2 honest
quoting) and Part B1 (vcut continuity, video+speech).
No network, no real DB (all DB-touching functions are mocked).

Part A3 (the material digest) is not covered here: it lived in the plan/
mirror work, which is not part of this codebase.

Run:  .venv/bin/python scripts/test_perception_blindness.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import cuts_read  # noqa: E402
from app.services.l3 import footage_map as fm  # noqa: E402
from app.services.l3 import ingest_store  # noqa: E402
from app.services.vcut import continuity as vcut_continuity  # noqa: E402
from test_footage_map import _cut, _rung  # noqa: E402

fm._sentences_for_file = lambda file_id: ()  # no-DB stub, same as test_footage_map.py

_FILE_ID = "ffffffff-1111"


def _sent(a, b, text="x", spk="S0"):
    return {"speaker": spk, "text": text, "src_in_ms": a, "src_out_ms": b}


# ==========================================================================
# A1 -- takes: (every OTHER take-group member, unfiltered by person)
# ==========================================================================

def test_takes_tag_lists_every_sibling_unfiltered_by_person():
    """A1's whole point: same-person is NOT a reason to drop here (unlike
    _alt_pic_segment) -- a 5-member group must show all 4 siblings."""
    cuts = []
    for i in range(5):
        cuts.append(_cut(f"f{i}:m0", 1000, 1000 + (i + 1) * 800, "thanks for the opportunity",
                         channel="said", speaker="S0",
                         ladder=[_rung("balanced", 1000, 1000 + (i + 1) * 800, "thanks", 0.5)],
                         take_group_id="tg1", take_role="winner" if i == 0 else "take",
                         score=0.3 + i * 0.1))
    trees = [fm.build_clip_tree(f"f{i}-uuid", {"name": f"c{i}", "duration_ms": 20000}, [c])
            for i, c in enumerate(cuts)]
    fm._annotate_dups(trees)
    m0 = trees[0]["moments"][0]
    line = fm._moment_line(m0)
    assert "takes: 4 other versions of this line" in line, line
    for i in range(1, 5):
        assert trees[i]["moments"][0]["cut_id"] in line, (i, line)
    print("ok  A1: a 5-member take group's beat lists all 4 siblings, unfiltered by person")


def test_takes_tag_reachability_on_the_measured_five_member_group():
    """Reachability assertion from the plan: the 5-member speech group MUST
    produce a takes: tag -- if the index contains no takes: line, the
    feature did not fire."""
    cuts = [_cut(f"f{i}:m0", 0, 2000, "line", channel="said", speaker="S0",
                ladder=[_rung("balanced", 0, 2000, "line", 0.5)],
                take_group_id="tg-measured", take_role="take")
           for i in range(5)]
    trees = [fm.build_clip_tree(f"f{i}-uuid", {"name": f"c{i}", "duration_ms": 8000}, [c])
            for i, c in enumerate(cuts)]
    fm._annotate_dups(trees)
    text = "\n".join(fm._moment_line(t["moments"][0]) for t in trees)
    assert "takes:" in text, text
    print("ok  A1 reachability: the 5-member group renders a takes: line")


def test_takes_tag_absent_without_take_group_siblings():
    cut = _cut("f:m0", 0, 2000, "line", channel="said", speaker="S0",
              ladder=[_rung("balanced", 0, 2000, "line", 0.5)])
    tree = fm.build_clip_tree("f-uuid", {"name": "c", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "takes:" not in line, line
    print("ok  A1: no take-group siblings -> no takes: line (zero cost on the common case)")


def test_takes_tag_absent_for_outlook_members():
    """Outlook (alternate cameras of the SAME simultaneous audio) is a
    different question -- already fully served by alt-PIC/outlook:, and
    'what else says this line' doesn't apply (it's the same audio)."""
    cut_a = _cut("fa:m0", 0, 2000, "line", channel="said", speaker="S0",
                ladder=[_rung("balanced", 0, 2000, "line", 0.5)],
                take_group_id="tg-outlook", take_role="outlook")
    cut_b = _cut("fb:m0", 0, 2000, "line", channel="said", speaker="S0",
                ladder=[_rung("balanced", 0, 2000, "line", 0.5)],
                take_group_id="tg-outlook", take_role="outlook")
    tree_a = fm.build_clip_tree("fa-uuid", {"name": "a", "duration_ms": 8000}, [cut_a])
    tree_b = fm.build_clip_tree("fb-uuid", {"name": "b", "duration_ms": 8000}, [cut_b])
    tree_a["moments"][0]["outlook_group_id"] = "og1"
    tree_b["moments"][0]["outlook_group_id"] = "og1"
    fm._annotate_dups([tree_a, tree_b])
    line = fm._moment_line(tree_a["moments"][0])
    assert "takes:" not in line, line
    print("ok  A1: outlook members never get a takes: tag (already served by alt-PIC)")


# ==========================================================================
# A2 -- honest quoting (requote to what survives sentence-snapping)
# ==========================================================================

def test_said_text_requotes_to_the_surviving_sentence():
    """The m14 failure: a trailing partial sentence in the raw span must not
    stay quoted once it wouldn't survive place()'s own sentence snap."""
    sents = (
        _sent(1000, 5000, "people are trying to make video editing through commands."),
        _sent(5000, 9000, "but it doesn't seem to be working out well because lms lack the visual."),
    )
    # the cut's OWN span only catches a partial sliver of the second sentence
    # (ends well before 9000) -- exactly the "promised more than it can keep" shape.
    cut = _cut("f:m0", 1000, 6500, "commands wipe editing", channel="said", speaker="S0",
              ladder=[_rung("balanced", 1000, 6500, "commands wipe editing", 0.5)])
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        tree = fm.build_clip_tree("f-uuid", {"name": "c", "duration_ms": 20000}, [cut])
    said = tree["moments"][0]["said_text"]
    assert "lms lack the visual" not in said, said
    assert "people are trying to make video editing through commands" in said, said
    print("ok  A2: a trailing partial sentence is dropped from the quote, not promised")


def test_said_text_unchanged_when_the_full_span_survives_snapping():
    sents = (_sent(1000, 5000, "a clean whole sentence."),)
    cut = _cut("f:m0", 1000, 5000, "a clean whole sentence", channel="said", speaker="S0",
              ladder=[_rung("balanced", 1000, 5000, "a clean whole sentence", 0.5)])
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        tree = fm.build_clip_tree("f-uuid", {"name": "c", "duration_ms": 8000}, [cut])
    assert tree["moments"][0]["said_text"] == "a clean whole sentence.", tree["moments"][0]["said_text"]
    print("ok  A2: a span that already lands on sentence boundaries is unchanged")


def test_said_text_snap_never_applied_to_non_speech_channel():
    """A 'shown'/'done' cut's incidental said_text (band-aid C's twin) must
    not be snap-adjusted -- the snap is specifically place()'s speech-cut
    behavior (act._snap_cut_to_sentences gates on channel=='said')."""
    sents = (_sent(1000, 5000, "incidental words under a picture cut."),)
    cut = _cut("f:m0", 1000, 3000, "watch this", channel="shown", subject="object", speaker=None,
              ladder=[_rung("balanced", 1000, 3000, "watch this", 0.5)])
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        tree = fm.build_clip_tree("f-uuid", {"name": "c", "duration_ms": 8000}, [cut])
    # _said_text_for_span's own overlap-only text, untouched by any snap logic
    assert tree["moments"][0]["said_text"] == "incidental words under a picture cut.", \
        tree["moments"][0]["said_text"]
    print("ok  A2: a non-speech-channel cut's incidental words are never snap-adjusted")


# ==========================================================================
# B1 -- vcut continuity (video + speech, together)
# ==========================================================================

def test_continuity_welds_a_clean_seam_and_breaks_on_a_shot_point():
    cuts = [
        {"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000, "voice_ids": []},
        {"id": "c2", "file_id": "f1", "src_in_ms": 2000, "src_out_ms": 5000, "voice_ids": []},
        {"id": "c3", "file_id": "f1", "src_in_ms": 5100, "src_out_ms": 8000, "voice_ids": []},
    ]
    shot_points = [{"ts_ms": 5050}]
    out = vcut_continuity.compute_continuity_for_file(cuts, shot_points)
    assert out["c1"]["cut_no"] == 1 and out["c1"]["of"] == 3, out["c1"]
    assert out["c1"]["next_contiguous"] is True, out["c1"]
    assert out["c2"]["prev_contiguous"] is True, out["c2"]
    assert out["c2"]["next_contiguous"] is False, out["c2"]
    assert out["c2"]["seam_reason_next"] == "shot/scene boundary or transition inside the gap", out["c2"]
    assert out["c3"]["prev_contiguous"] is False, out["c3"]
    print("ok  B1: a clean seam welds, a shot point in the gap breaks it")


def test_continuity_covers_mixed_video_and_speech_cuts_on_one_file():
    """B1's own requirement: this must cover speech cuts, not just video --
    continuity spans BOTH kinds together, in source order. voice_ids=[] on
    both sides matches today's actual vcut data (never populated for either
    kind yet -- a real, separate gap outside this plan's scope), so this is
    the realistic shape, not an idealized one."""
    cuts = [
        {"id": "v1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000, "voice_ids": []},
        {"id": "s1", "file_id": "f1", "src_in_ms": 2000, "src_out_ms": 4000, "voice_ids": []},
    ]
    out = vcut_continuity.compute_continuity_for_file(cuts, [])
    assert out["v1"]["of"] == 2 and out["s1"]["of"] == 2, out
    assert out["v1"]["next_contiguous"] is True, out["v1"]
    print("ok  B1: continuity spans video and speech cuts on the same file together")


def test_continuity_speaker_change_is_a_hard_break():
    cuts = [
        {"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000, "voice_ids": ["V1"]},
        {"id": "c2", "file_id": "f1", "src_in_ms": 2000, "src_out_ms": 4000, "voice_ids": ["V2"]},
    ]
    out = vcut_continuity.compute_continuity_for_file(cuts, [])
    assert out["c1"]["next_contiguous"] is False, out["c1"]
    assert out["c1"]["seam_reason_next"] == "speaker change across the seam", out["c1"]
    print("ok  B1: a voice_ids change across the seam is a hard break")


def test_continuity_gap_longer_than_bridged_speech_is_a_hard_break():
    cuts = [
        {"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 500, "voice_ids": []},
        {"id": "c2", "file_id": "f1", "src_in_ms": 20000, "src_out_ms": 20500, "voice_ids": []},
    ]
    out = vcut_continuity.compute_continuity_for_file(cuts, [])
    assert out["c1"]["next_contiguous"] is False, out["c1"]
    assert out["c1"]["seam_reason_next"] == "gap is longer than the speech it would bridge", out["c1"]
    print("ok  B1: a gap much longer than the bridged speech is a hard break (magnitude backstop)")


def test_continuity_reachability_every_cut_gets_a_nonempty_block():
    """Reachability: after this runs, no cut should be left with continuity
    == {} the way EVERY cut in the measured run was."""
    cuts = [{"id": f"c{i}", "file_id": "f1", "src_in_ms": i * 1000, "src_out_ms": i * 1000 + 900,
            "voice_ids": []} for i in range(5)]
    out = vcut_continuity.compute_continuity_for_file(cuts, [])
    assert len(out) == 5, out
    for cid, cont in out.items():
        assert cont.get("cut_no") and cont.get("of") == 5, (cid, cont)
    print("ok  B1 reachability: every cut on a file gets a real (non-empty) continuity block")


class _FakeContUpdateStore:
    def __init__(self):
        self.calls = []

    def update_cut_continuity(self, cut_id, continuity):
        self.calls.append((cut_id, continuity))


def test_write_continuity_for_run_groups_by_file_and_updates_every_row():
    rows = [
        {"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 1000, "voice_ids": []},
        {"id": "c2", "file_id": "f1", "src_in_ms": 1000, "src_out_ms": 2000, "voice_ids": []},
        {"id": "c3", "file_id": "f2", "src_in_ms": 0, "src_out_ms": 1000, "voice_ids": []},
    ]
    fake_store = _FakeContUpdateStore()
    with mock.patch.object(cuts_read, "rows_for_run", return_value=rows), \
         mock.patch.object(ingest_store, "update_cut_continuity", side_effect=fake_store.update_cut_continuity):
        n = vcut_continuity.write_continuity_for_run("run1", seam_cache={})
    assert n == 3, n
    updated_ids = {cid for cid, _ in fake_store.calls}
    assert updated_ids == {"c1", "c2", "c3"}, updated_ids
    # c1/c2 (same file) welded; c3 is alone on its own file (of=1, no neighbors).
    by_id = dict(fake_store.calls)
    assert by_id["c1"]["of"] == 2, by_id["c1"]
    assert by_id["c3"]["of"] == 1, by_id["c3"]
    print("ok  B1: write_continuity_for_run groups by file and updates every row")


def test_write_continuity_for_run_is_fail_open_on_read_failure():
    with mock.patch.object(cuts_read, "rows_for_run", side_effect=RuntimeError("db down")):
        n = vcut_continuity.write_continuity_for_run("run1", seam_cache={})
    assert n == 0
    print("ok  B1: write_continuity_for_run fails open (never raises) on a read failure")


def test_write_continuity_for_run_uses_shot_points_from_seam_cache():
    rows = [
        {"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 1000, "voice_ids": []},
        {"id": "c2", "file_id": "f1", "src_in_ms": 1100, "src_out_ms": 2000, "voice_ids": []},
    ]
    seam_cache = {"f1": {"shot_points": [{"ts_ms": 1050}]}}
    fake_store = _FakeContUpdateStore()
    with mock.patch.object(cuts_read, "rows_for_run", return_value=rows), \
         mock.patch.object(ingest_store, "update_cut_continuity", side_effect=fake_store.update_cut_continuity):
        vcut_continuity.write_continuity_for_run("run1", seam_cache=seam_cache)
    by_id = dict(fake_store.calls)
    assert by_id["c1"]["seam_reason_next"] == "shot/scene boundary or transition inside the gap", by_id["c1"]
    print("ok  B1: write_continuity_for_run reads shot_points straight off the in-memory seam_cache")


def test_continuity_tag_renders_from_a_computed_block():
    """End-to-end: a computed continuity block renders the cut:N/M weld
    marks _continuity_tag/_weld_mark already know how to draw."""
    cuts = [
        {"id": "c1", "file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000, "voice_ids": []},
        {"id": "c2", "file_id": "f1", "src_in_ms": 2000, "src_out_ms": 5000, "voice_ids": []},
    ]
    out = vcut_continuity.compute_continuity_for_file(cuts, [])
    m = {"continuity": out["c2"]}
    tag = fm._continuity_tag(m)
    assert tag == " · ↔cut:2/2", tag
    print("ok  B1: a computed continuity block renders cut:N/M with weld marks via _continuity_tag")


def test_every_cut_rewrite_site_recomputes_continuity():
    """B1 STRUCTURAL INVARIANT, over the real source tree.

    `insert_video_cuts` DELETES and rebuilds every kind='video' row of a run,
    and a fresh CutRecord carries continuity={} by default -- so continuity is
    destroyed by construction at every call site, and must be recomputed right
    after. There are three such sites today (ingest, the energy dial, and the
    frames-mode enrich pass), and the failure is SILENT: _continuity_tag
    renders '' on a missing block, so a forgotten recompute just makes every
    cut: tag quietly disappear from the brain's index.

    A behavioural test would only cover the paths someone remembered to write
    one for. This asserts the invariant on the SOURCE instead, so a fourth
    rewrite site added later fails here the day it is written rather than the
    day someone notices the index looks thin.
    """
    import ast
    import pathlib

    app_root = pathlib.Path(BACKEND) / "app"
    offenders = []
    for path in app_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = {
                (n.func.attr if isinstance(n.func, ast.Attribute) else
                 n.func.id if isinstance(n.func, ast.Name) else "")
                for n in ast.walk(fn) if isinstance(n, ast.Call)
            }
            if "insert_video_cuts" in called and "write_continuity_for_run" not in called:
                offenders.append(f"{path.relative_to(BACKEND)}::{fn.name}")

    assert not offenders, (
        "these functions rewrite cut_records via insert_video_cuts but never "
        "recompute continuity, silently blanking every video cut's block: "
        + ", ".join(sorted(offenders))
    )
    print(f"ok  B1 invariant: every insert_video_cuts site recomputes continuity "
          f"({len(offenders)} offenders)")


def main():
    tests = [
        test_takes_tag_lists_every_sibling_unfiltered_by_person,
        test_takes_tag_reachability_on_the_measured_five_member_group,
        test_takes_tag_absent_without_take_group_siblings,
        test_takes_tag_absent_for_outlook_members,
        test_said_text_requotes_to_the_surviving_sentence,
        test_said_text_unchanged_when_the_full_span_survives_snapping,
        test_said_text_snap_never_applied_to_non_speech_channel,
        test_continuity_welds_a_clean_seam_and_breaks_on_a_shot_point,
        test_continuity_covers_mixed_video_and_speech_cuts_on_one_file,
        test_continuity_speaker_change_is_a_hard_break,
        test_continuity_gap_longer_than_bridged_speech_is_a_hard_break,
        test_continuity_reachability_every_cut_gets_a_nonempty_block,
        test_write_continuity_for_run_groups_by_file_and_updates_every_row,
        test_write_continuity_for_run_is_fail_open_on_read_failure,
        test_write_continuity_for_run_uses_shot_points_from_seam_cache,
        test_continuity_tag_renders_from_a_computed_block,
        test_every_cut_rewrite_site_recomputes_continuity,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} perception-blindness tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
