"""
Tests for the cuts-v3 pass-1 module (``app.services.l3.pass1``) -- no DB, NO
REAL API CALLS. Reuses the exact fake-SDK-client pattern from
``test_ingest_client.py`` (monkeypatch ``llm.client._sdk_client``) so the
orchestration path is exercised for $0.

Run:  .venv/bin/python scripts/test_pass1.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass1  # noqa: E402
from app.services.l3.lattice import Atom, Lattice  # noqa: E402
from app.services.llm import client as ic  # noqa: E402
from test_ingest_client import FakeBlock, FakeClient, FakeResponse  # noqa: E402


def _with_fake_client(responses):
    fake = FakeClient(responses)
    orig = ic._sdk_client
    ic._sdk_client = lambda: fake
    return fake, orig


def _make_lattice() -> Lattice:
    words = [
        {"start_ms": 0, "end_ms": 200, "text": "So", "speaker": "S1"},
        {"start_ms": 200, "end_ms": 500, "text": "anyway,", "speaker": "S1"},
        {"start_ms": 550, "end_ms": 900, "text": "yeah.", "speaker": "S1"},
        {"start_ms": 1500, "end_ms": 1800, "text": "Right,", "speaker": "S2"},
        {"start_ms": 1800, "end_ms": 2100, "text": "exactly.", "speaker": "S2"},
    ]
    atoms = [
        Atom(atom_id=0, file_id="f1", start_ms=2100, end_ms=2600,
             state_in="speech_edge", state_out="shot", action_energy=0.2,
             coherence=0.9, anchor_ms=[2300]),
        Atom(atom_id=1, file_id="f1", start_ms=2600, end_ms=3200,
             state_in="shot", state_out="clip_edge", action_energy=0.4,
             coherence=0.8, anchor_ms=[]),
    ]
    return Lattice(file_id="f1", duration_ms=3200, words=words, turns=[],
                   hints=["speaker change after word 2 (0.6s gap)"], atoms=atoms)


def test_render_clip_block_includes_transcript_and_atoms():
    lat = _make_lattice()
    block = pass1._render_clip_block("f1", "clip_one.mp4", 3200, lat)
    assert "CLIP f1" in block and "clip_one.mp4" in block, block
    assert "0:So" in block and "[S2]3:Right," in block, block
    assert "HINTS: speaker change after word 2" in block, block
    assert "ATOM 0 [2100-2600]" in block, block
    assert "ATOM 1 [2600-3200]" in block, block
    print("ok  test_render_clip_block_includes_transcript_and_atoms")


def test_render_clip_block_handles_no_speech():
    lat = Lattice(file_id="f2", duration_ms=1000, words=[], turns=[], hints=[], atoms=[])
    block = pass1._render_clip_block("f2", "b.mp4", 1000, lat)
    assert "(no speech)" in block, block
    print("ok  test_render_clip_block_handles_no_speech")


def test_build_pass1_blocks_one_per_clip():
    lat1 = _make_lattice()
    lat2 = Lattice(file_id="f2", duration_ms=500, words=[], turns=[], hints=[], atoms=[])
    blocks = pass1.build_pass1_blocks([("f1", "a.mp4", 3200, lat1), ("f2", "b.mp4", 500, lat2)])
    assert len(blocks) == 2, blocks
    assert blocks[0]["type"] == "text" and "CLIP f1" in blocks[0]["text"]
    assert blocks[1]["type"] == "text" and "CLIP f2" in blocks[1]["text"]
    print("ok  test_build_pass1_blocks_one_per_clip")


def test_run_pass1_empty_file_rows_raises():
    try:
        pass1.run_pass1([])
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_run_pass1_empty_file_rows_raises")


def test_run_pass1_calls_complete_with_pass1_stage_and_schema():
    good = {
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 2], "label": "intro", "speaker_ids": ["S1"]}],
        "take_candidates": [],
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0, 1]}],
        "junk_suspects": [],
        "project_summary": "one clip of two people talking",
        "clip_summaries": [{"file_id": "f1", "summary": "a conversation"}],
    }
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        result = pass1.run_pass1([("f1", "a.mp4", 3200, _make_lattice())])
    finally:
        ic._sdk_client = orig

    call = fake.messages.calls[0]
    assert call["tools"][0]["input_schema"] == pass1.Pass1Output.model_json_schema()
    assert result.attempts == 1
    parsed = pass1.Pass1Output.model_validate(result.data)
    assert parsed.speech_cuts[0].word_span == (0, 2)
    assert parsed.video_tentative_groups[0].atom_ids == [0, 1]
    assert parsed.project_summary == good["project_summary"]
    print("ok  test_run_pass1_calls_complete_with_pass1_stage_and_schema")


def test_pass1_output_schema_round_trip_with_junk_suspects():
    raw = {
        "speech_cuts": [],
        "take_candidates": [
            {"group_id": "tg1", "members": [
                {"file_id": "f1", "word_span": [0, 1]},
                {"file_id": "f2", "word_span": [3, 4]},
            ]},
        ],
        "video_tentative_groups": [],
        "junk_suspects": [
            {"file_id": "f1", "word_span": [4, 4], "atom_ids": None, "reason": "false start"},
            {"file_id": "f1", "word_span": None, "atom_ids": [3], "reason": "out of focus"},
        ],
        "project_summary": "",
        "clip_summaries": [],
    }
    parsed = pass1.Pass1Output.model_validate(raw)
    assert len(parsed.take_candidates[0].members) == 2
    assert parsed.junk_suspects[0].atom_ids is None
    assert parsed.junk_suspects[1].word_span is None
    print("ok  test_pass1_output_schema_round_trip_with_junk_suspects")


def test_pass1_output_defaults_are_empty_not_missing():
    parsed = pass1.Pass1Output.model_validate({})
    assert parsed.speech_cuts == []
    assert parsed.take_candidates == []
    assert parsed.video_tentative_groups == []
    assert parsed.junk_suspects == []
    assert parsed.project_summary == ""
    assert parsed.clip_summaries == []
    print("ok  test_pass1_output_defaults_are_empty_not_missing")


def test_enforce_splits_a_speech_cut_at_an_atom_owned_gap():
    # Words 0-2 end at 900ms; words 3-4 run 1500-2100ms. Put an atom inside
    # the 900-1500 gap: a cut grouping words [0-4] crosses atom territory and
    # must come back split into [0-2] + [3-4], deterministically.
    lat = _make_lattice()
    lat.atoms.insert(0, Atom(atom_id=99, file_id="f1", start_ms=950, end_ms=1450,
                             state_in="speech_edge", state_out="speech_edge",
                             action_energy=0.1, coherence=1.0))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "whole thing"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    spans = [tuple(sc.word_span) for sc in fixed.speech_cuts]
    assert spans == [(0, 2), (3, 4)], spans
    assert "(1/2)" in fixed.speech_cuts[0].label and "(2/2)" in fixed.speech_cuts[1].label
    print("ok  test_enforce_splits_a_speech_cut_at_an_atom_owned_gap")


def test_enforce_leaves_a_clean_grouping_untouched():
    lat = _make_lattice()
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 2], "label": "first"},
            {"file_id": "f1", "word_span": [3, 4], "label": "second"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 2), (3, 4)]
    assert fixed.speech_cuts[0].label == "first"   # no (1/n) suffix when untouched
    print("ok  test_enforce_leaves_a_clean_grouping_untouched")


def _take_line_lattice(fid: str) -> Lattice:
    """Two clips of the SAME line -- what a real take actually is (same words,
    different capture). Content tokens (stop-words dropped): launch/new/product/today."""
    words = [
        {"start_ms": 0, "end_ms": 400, "text": "Launch", "speaker": "S1"},
        {"start_ms": 400, "end_ms": 800, "text": "the", "speaker": "S1"},
        {"start_ms": 800, "end_ms": 1200, "text": "new", "speaker": "S1"},
        {"start_ms": 1200, "end_ms": 1600, "text": "product", "speaker": "S1"},
        {"start_ms": 1600, "end_ms": 2000, "text": "today.", "speaker": "S1"},
    ]
    atoms = [Atom(atom_id=0, file_id=fid, start_ms=2000, end_ms=2600,
                  state_in="speech_edge", state_out="clip_edge",
                  action_energy=0.1, coherence=1.0)]
    return Lattice(file_id=fid, duration_ms=2600, words=words, turns=[], hints=[], atoms=atoms)


def test_enforce_keeps_a_genuine_same_line_take_group():
    lats = {"f1": _take_line_lattice("f1"), "f2": _take_line_lattice("f2")}
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "line"},
                        {"file_id": "f2", "word_span": [0, 4], "label": "line"}],
        "take_candidates": [{"group_id": "tg1", "members": [
            {"file_id": "f1", "word_span": [0, 4]},
            {"file_id": "f2", "word_span": [0, 4]},
        ]}],
    })
    fixed = pass1.enforce_lattice_partition(out, lats)
    assert len(fixed.take_candidates) == 1, fixed.take_candidates
    fids = sorted(m.file_id for m in fixed.take_candidates[0].members)
    assert fids == ["f1", "f2"], fids
    print("ok  test_enforce_keeps_a_genuine_same_line_take_group")


def test_enforce_drops_an_over_grouped_take_member():
    # Same-line guard: a member that says DIFFERENT words is dropped, and a group
    # left with <2 same-line members is dropped entirely (the closing_thanks bug).
    f3 = Lattice(file_id="f3", duration_ms=2600, words=[
        {"start_ms": 0, "end_ms": 500, "text": "Architecture", "speaker": "S1"},
        {"start_ms": 500, "end_ms": 1000, "text": "print", "speaker": "S1"},
        {"start_ms": 1000, "end_ms": 1500, "text": "shop", "speaker": "S1"},
        {"start_ms": 1500, "end_ms": 2000, "text": "idea.", "speaker": "S1"},
    ], turns=[], hints=[], atoms=[Atom(atom_id=0, file_id="f3", start_ms=2000, end_ms=2600,
                                       state_in="speech_edge", state_out="clip_edge",
                                       action_energy=0.1, coherence=1.0)])
    lats = {"f1": _take_line_lattice("f1"), "f2": _take_line_lattice("f2"), "f3": f3}

    # f1 + f2 are the same line, f3 is different -> f3 dropped, group of 2 kept.
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "line"},
                        {"file_id": "f2", "word_span": [0, 4], "label": "line"},
                        {"file_id": "f3", "word_span": [0, 3], "label": "other"}],
        "take_candidates": [{"group_id": "tg1", "members": [
            {"file_id": "f1", "word_span": [0, 4]},
            {"file_id": "f2", "word_span": [0, 4]},
            {"file_id": "f3", "word_span": [0, 3]},
        ]}],
    })
    fixed = pass1.enforce_lattice_partition(out, lats)
    assert len(fixed.take_candidates) == 1, fixed.take_candidates
    assert sorted(m.file_id for m in fixed.take_candidates[0].members) == ["f1", "f2"]

    # A group of ONLY mismatched lines (f1 vs f3) collapses to <2 -> dropped.
    out2 = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "line"},
                        {"file_id": "f3", "word_span": [0, 3], "label": "other"}],
        "take_candidates": [{"group_id": "tg2", "members": [
            {"file_id": "f1", "word_span": [0, 4]},
            {"file_id": "f3", "word_span": [0, 3]},
        ]}],
    })
    fixed2 = pass1.enforce_lattice_partition(out2, lats)
    assert fixed2.take_candidates == [], fixed2.take_candidates
    print("ok  test_enforce_drops_an_over_grouped_take_member")


def test_coverage_fill_recovers_uncovered_speech_words():
    # The model grouped only words [0-1]; words 2,3,4 were left out of every
    # speech_cut. Coverage-fill surfaces them as recovered cuts (split at the
    # atom-owned gap between word 2 and word 3), so no speech is ever dropped.
    lat = _make_lattice()
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 1], "label": "kept"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    covered = set()
    for sc in fixed.speech_cuts:
        for k in range(sc.word_span[0], sc.word_span[1] + 1):
            covered.add(k)
    assert covered == {0, 1, 2, 3, 4}, [tuple(sc.word_span) for sc in fixed.speech_cuts]
    assert any(sc.label == "(recovered)" for sc in fixed.speech_cuts), fixed.speech_cuts
    print("ok  test_coverage_fill_recovers_uncovered_speech_words")


def _bridge_lattice(gap_atoms, *, second_speaker="S1", gap_span=(2000, 2600)):
    """A beat that continues across ONE wordless moment: 'Watch this ... and
    topspin', with ``gap_atoms`` occupying [gap_span] between word 1 and word
    2. 2s of speech on each side, so the magnitude backstop is slack unless the
    gap is stretched via ``gap_span``."""
    g_lo, g_hi = gap_span
    words = [
        {"start_ms": 0, "end_ms": 1000, "text": "Watch", "speaker": "S1"},
        {"start_ms": 1000, "end_ms": g_lo, "text": "this", "speaker": "S1"},
        {"start_ms": g_hi, "end_ms": g_hi + 1000, "text": "and", "speaker": second_speaker},
        {"start_ms": g_hi + 1000, "end_ms": g_hi + 2000, "text": "topspin", "speaker": second_speaker},
    ]
    return Lattice(file_id="f1", duration_ms=g_hi + 2000, words=words, turns=[],
                   hints=[], atoms=list(gap_atoms))


def test_bridge_absorbs_a_weldable_gap_into_one_beat():
    # One continuous action atom (no break edge, same speaker, short gap) sits
    # mid-thought. The beat [0-3] must stay ONE cut -- the atom plays inside
    # it (cuts_v4_only.plan.md: there is no video-cut-from-atoms path left to
    # verify the atom "leaving the video pool" against; V4 derives its cuts
    # from the RAW non-speech gaps this beat's own span already excludes).
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.7, coherence=0.9, is_action=True)
    lat = _bridge_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "demo beat"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 3)], fixed.speech_cuts
    assert fixed.speech_cuts[0].label == "demo beat"   # untouched, no (n/n) suffix
    print("ok  test_bridge_absorbs_a_weldable_gap_into_one_beat")


def test_bridge_splits_at_a_shot_cut_in_the_gap():
    # Two atoms in the gap with a SHOT CUT between them (a real break). Even
    # same-speaker, short gap -> hard seam -> beat splits into speech . speech.
    a0 = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2300,
              state_in="speech_edge", state_out="shot_cut", action_energy=0.5, coherence=0.9)
    a1 = Atom(atom_id=11, file_id="f1", start_ms=2300, end_ms=2600,
              state_in="shot_cut", state_out="speech_edge", action_energy=0.5, coherence=0.9)
    lat = _bridge_lattice([a0, a1])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "two shots"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 1), (2, 3)], fixed.speech_cuts
    print("ok  test_bridge_splits_at_a_shot_cut_in_the_gap")


def test_bridge_splits_at_a_speaker_change():
    # Same continuous footage, but the second half is a DIFFERENT speaker ->
    # never one beat -> hard seam -> split.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge", action_energy=0.3, coherence=0.9)
    lat = _bridge_lattice([atom], second_speaker="S2")
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "two speakers"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 1), (2, 3)], fixed.speech_cuts
    print("ok  test_bridge_splits_at_a_speaker_change")


def test_bridge_magnitude_backstop_splits_an_over_long_gap():
    # The gap (4600ms) is longer than the speech it would bridge (~4s here is
    # borderline; stretch the gap so gap_ms > left+right speech). More
    # connective tissue than speech -> not one beat -> hard, no tuned constant.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=9000,
                state_in="speech_edge", state_out="speech_edge", action_energy=0.3, coherence=0.9)
    lat = _bridge_lattice([atom], gap_span=(2000, 9000))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "long gap"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 1), (2, 3)], fixed.speech_cuts
    print("ok  test_bridge_magnitude_backstop_splits_an_over_long_gap")


def test_beat_merge_fuses_same_beat_neighbours_across_weldable_seam():
    # The model emitted TWO speech cuts but tagged them one beat_id; a
    # continuous action atom sits between them (same speaker, short gap, no
    # break) -> code merges them into ONE beat, the atom playing inside it.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.7, coherence=0.9, is_action=True)
    lat = _bridge_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 1], "label": "setup", "beat_id": "b1"},
            {"file_id": "f1", "word_span": [2, 3], "label": "payoff", "beat_id": "b1"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 3)], fixed.speech_cuts
    print("ok  test_beat_merge_fuses_same_beat_neighbours_across_weldable_seam")


def test_beat_merge_respects_the_seam_guard():
    # Same beat_id, but a SHOT CUT sits in the gap -> code refuses to merge
    # even though the model asked, and the atoms stay in the video pool.
    a0 = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2300,
              state_in="speech_edge", state_out="shot_cut", action_energy=0.5, coherence=0.9)
    a1 = Atom(atom_id=11, file_id="f1", start_ms=2300, end_ms=2600,
              state_in="shot_cut", state_out="speech_edge", action_energy=0.5, coherence=0.9)
    lat = _bridge_lattice([a0, a1])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 1], "label": "a", "beat_id": "b1"},
            {"file_id": "f1", "word_span": [2, 3], "label": "b", "beat_id": "b1"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 1), (2, 3)], fixed.speech_cuts
    print("ok  test_beat_merge_respects_the_seam_guard")


def test_beat_merge_does_not_touch_untagged_neighbours():
    # No beat_id -> even a perfectly weldable seam is left split (the model
    # never said these belong together).
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge", action_energy=0.3, coherence=0.9)
    lat = _bridge_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 1], "label": "a"},
            {"file_id": "f1", "word_span": [2, 3], "label": "b"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 1), (2, 3)], fixed.speech_cuts
    print("ok  test_beat_merge_does_not_touch_untagged_neighbours")


def _trailing_runt_lattice(gap_atoms, *, gap_span=(2000, 2600)):
    """Five real words ('Watch this trick right now') then ONE trailing runt
    word ('yeah'), split by a single wordless gap at ``gap_span`` holding
    ``gap_atoms``."""
    g_lo, g_hi = gap_span
    words = [
        {"start_ms": 0, "end_ms": 400, "text": "Watch", "speaker": "S1"},
        {"start_ms": 400, "end_ms": 800, "text": "this", "speaker": "S1"},
        {"start_ms": 800, "end_ms": 1200, "text": "trick", "speaker": "S1"},
        {"start_ms": 1200, "end_ms": 1600, "text": "right", "speaker": "S1"},
        {"start_ms": 1600, "end_ms": g_lo, "text": "now", "speaker": "S1"},
        {"start_ms": g_hi, "end_ms": g_hi + 400, "text": "yeah", "speaker": "S1"},
    ]
    return Lattice(file_id="f1", duration_ms=g_hi + 400, words=words, turns=[],
                   hints=[], atoms=list(gap_atoms))


def _leading_runt_lattice(gap_atoms, *, gap_span=(400, 1000)):
    """ONE leading runt word ('So'), a wordless gap at ``gap_span`` holding
    ``gap_atoms``, then five real words ('watch this cool trick now')."""
    g_lo, g_hi = gap_span
    words = [
        {"start_ms": 0, "end_ms": g_lo, "text": "So", "speaker": "S1"},
        {"start_ms": g_hi, "end_ms": g_hi + 400, "text": "watch", "speaker": "S1"},
        {"start_ms": g_hi + 400, "end_ms": g_hi + 800, "text": "this", "speaker": "S1"},
        {"start_ms": g_hi + 800, "end_ms": g_hi + 1200, "text": "cool", "speaker": "S1"},
        {"start_ms": g_hi + 1200, "end_ms": g_hi + 1600, "text": "trick", "speaker": "S1"},
        {"start_ms": g_hi + 1600, "end_ms": g_hi + 2000, "text": "now", "speaker": "S1"},
    ]
    return Lattice(file_id="f1", duration_ms=g_hi + 2000, words=words, turns=[],
                   hints=[], atoms=list(gap_atoms))


def test_runt_guard_absorbs_trailing_runt_into_preceding_cut():
    # A real 5-word thought followed by a 1-word trailing runt ("yeah") across
    # a weldable gap. Word count 1 is <= 3 and below this clip's own median
    # (3.0, from [5, 1]) -- a runt with no hardcoded ms threshold involved --
    # so it folds BACKWARD into the preceding cut.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.3, coherence=0.9)
    lat = _trailing_runt_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "main"},
            {"file_id": "f1", "word_span": [5, 5], "label": "trailing runt"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 5)], fixed.speech_cuts
    assert fixed.speech_cuts[0].label == "main"
    print("ok  test_runt_guard_absorbs_trailing_runt_into_preceding_cut")


def test_runt_guard_absorbs_leading_runt_into_following_cut():
    # A 1-word leading runt ("So") before a real 5-word thought, across a
    # weldable gap -> the runt folds FORWARD into the following cut.
    atom = Atom(atom_id=10, file_id="f1", start_ms=400, end_ms=1000,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.3, coherence=0.9)
    lat = _leading_runt_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 0], "label": "leading runt"},
            {"file_id": "f1", "word_span": [1, 5], "label": "main"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 5)], fixed.speech_cuts
    assert fixed.speech_cuts[0].label == "main"
    print("ok  test_runt_guard_absorbs_leading_runt_into_following_cut")


def test_runt_guard_respects_the_seam_guard():
    # Same trailing-runt shape, but a SHOT CUT sits in the gap -> hard seam ->
    # the runt is left split, same as the beat-merge guard (on any doubt,
    # leave it split).
    a0 = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2300,
              state_in="speech_edge", state_out="shot_cut", action_energy=0.5, coherence=0.9)
    a1 = Atom(atom_id=11, file_id="f1", start_ms=2300, end_ms=2600,
              state_in="shot_cut", state_out="speech_edge", action_energy=0.5, coherence=0.9)
    lat = _trailing_runt_lattice([a0, a1])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "main"},
            {"file_id": "f1", "word_span": [5, 5], "label": "trailing runt"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 4), (5, 5)], fixed.speech_cuts
    print("ok  test_runt_guard_respects_the_seam_guard")


def test_runt_guard_does_not_absorb_a_take_member():
    # The preceding cut is a take_candidate member -- its word_span identity
    # must stay exactly what pass 2a expects, so absorption is blocked even
    # though the seam is otherwise weldable.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.3, coherence=0.9)
    lat = _trailing_runt_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "main"},
            {"file_id": "f1", "word_span": [5, 5], "label": "trailing runt"},
        ],
        "take_candidates": [
            {"group_id": "t1", "members": [{"file_id": "f1", "word_span": [0, 4]}]},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 4), (5, 5)], fixed.speech_cuts
    print("ok  test_runt_guard_does_not_absorb_a_take_member")


def test_runt_guard_does_not_absorb_flagged_junk():
    # The runt itself is a flagged junk suspect -- code never silently fuses a
    # false start into the real line.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.3, coherence=0.9)
    lat = _trailing_runt_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "main"},
            {"file_id": "f1", "word_span": [5, 5], "label": "trailing runt"},
        ],
        "junk_suspects": [
            {"file_id": "f1", "word_span": [5, 5], "reason": "false start"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 4), (5, 5)], fixed.speech_cuts
    print("ok  test_runt_guard_does_not_absorb_flagged_junk")


def test_runt_guard_skips_outlook_synced_files():
    # Otherwise-weldable trailing runt, but the file belongs to an outlook
    # group -- group_outlooks (run right after enforce_lattice_partition)
    # needs BYTE-IDENTICAL word spans across every angle, so runt absorption
    # must not touch a synced file at all.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.3, coherence=0.9)
    lat = _trailing_runt_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "main"},
            {"file_id": "f1", "word_span": [5, 5], "label": "trailing runt"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat}, outlook_file_ids={"f1"})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 4), (5, 5)], fixed.speech_cuts
    print("ok  test_runt_guard_skips_outlook_synced_files")


def test_silent_trailing_word_is_folded_into_its_beat():
    # Real-data artifact: a zero-duration trailing word ("you." timed
    # end==start) left uncovered by the model, sitting flush against an atom.
    # Recovered as [2,2] it would resolve to a point (src_out==src_in) the DB
    # rejects -- enforce must fold it into the word-adjacent same-speaker beat.
    words = [
        {"start_ms": 0, "end_ms": 200, "text": "Thank", "speaker": "S1"},
        {"start_ms": 200, "end_ms": 500, "text": "you", "speaker": "S1"},
        {"start_ms": 500, "end_ms": 500, "text": "you.", "speaker": "S1"},  # zero-duration
    ]
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=500, end_ms=1500,
                  state_in="speech_edge", state_out="clip_edge",
                  action_energy=0.1, coherence=1.0)]
    lat = Lattice(file_id="f1", duration_ms=1500, words=words, turns=[], hints=[], atoms=atoms)
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 1], "label": "thanks"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    spans = sorted(tuple(sc.word_span) for sc in fixed.speech_cuts)
    assert spans == [(0, 2)], spans   # the silent [2,2] folded in, no degenerate cut
    print("ok  test_silent_trailing_word_is_folded_into_its_beat")


def test_isolated_silent_word_is_dropped():
    # A zero-duration word with no word-adjacent neighbour has no audible span
    # and nothing to fold into -> dropped (nothing lost), never a point-span cut.
    words = [
        {"start_ms": 0, "end_ms": 300, "text": "Hi", "speaker": "S1"},
        {"start_ms": 900, "end_ms": 900, "text": "x", "speaker": "S1"},  # isolated + silent
    ]
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=300, end_ms=900,
                  state_in="speech_edge", state_out="speech_edge",
                  action_energy=0.1, coherence=1.0),
             Atom(atom_id=1, file_id="f1", start_ms=900, end_ms=1500,
                  state_in="speech_edge", state_out="clip_edge",
                  action_energy=0.1, coherence=1.0)]
    lat = Lattice(file_id="f1", duration_ms=1500, words=words, turns=[], hints=[], atoms=atoms)
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 0], "label": "hi"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    spans = sorted(tuple(sc.word_span) for sc in fixed.speech_cuts)
    assert spans == [(0, 0)], spans   # the isolated silent word dropped
    print("ok  test_isolated_silent_word_is_dropped")


def test_speaker_ids_for_span_dominant_first_no_timing_falls_back_to_counts():
    # No word timings -> weight by raw word count. S1 (3 words) dominates S0
    # (1 word, 25% >= 15% -> kept). Dominant listed first.
    words = [
        {"speaker": "S1"}, {"speaker": "S1"}, {"speaker": "S0"}, {"speaker": None}, {"speaker": "S1"},
    ]
    assert pass1._speaker_ids_for_span(words, (0, 4)) == ["S1", "S0"]
    assert pass1._speaker_ids_for_span(words, (2, 2)) == ["S0"]
    assert pass1._speaker_ids_for_span(words, (3, 3)) == []
    print("ok  test_speaker_ids_for_span_dominant_first_no_timing_falls_back_to_counts")


def test_speaker_ids_for_span_orders_by_spoken_time_not_appearance():
    # S0 speaks FIRST (3s, 30%) but S1 holds the floor (7s, 70%). Both clear
    # the 15% bar, so both are kept -- but the dominant S1 must come first even
    # though S0 appeared earlier, so speaker_person credits the real speaker.
    words = [
        {"speaker": "S0", "start_ms": 0, "end_ms": 3000},
        {"speaker": "S1", "start_ms": 3000, "end_ms": 10000},
    ]
    assert pass1._speaker_ids_for_span(words, (0, 1)) == ["S1", "S0"]
    print("ok  test_speaker_ids_for_span_orders_by_spoken_time_not_appearance")


def test_speaker_ids_for_span_drops_sub_threshold_backchannel():
    # A ~49s answer by S1 with a 400ms "yeah" from S0 (~0.8% of spoken time,
    # well under 15%) -> S0 dropped, cut stays single-voice.
    words = [
        {"speaker": "S1", "start_ms": 0, "end_ms": 49000},
        {"speaker": "S0", "start_ms": 49000, "end_ms": 49400},
    ]
    assert pass1._speaker_ids_for_span(words, (0, 1)) == ["S1"]
    # A genuine ~40/60 two-person exchange keeps both (each >= 15%).
    words2 = [
        {"speaker": "S1", "start_ms": 0, "end_ms": 6000},
        {"speaker": "S0", "start_ms": 6000, "end_ms": 10000},
    ]
    assert pass1._speaker_ids_for_span(words2, (0, 1)) == ["S1", "S0"]
    print("ok  test_speaker_ids_for_span_drops_sub_threshold_backchannel")


def test_enforce_stamps_speaker_ids_from_word_diarization():
    # voice_first_identity.plan.md: the model is never asked to emit
    # speaker_ids (it always defaults to []) -- enforcement must stamp the
    # deterministic ground truth from the lattice's own word-level
    # diarization onto every FINAL speech cut.
    lat = _make_lattice()
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 2], "label": "s1 bit"},
            {"file_id": "f1", "word_span": [3, 4], "label": "s2 bit"},
        ],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    by_span = {tuple(sc.word_span): sc.speaker_ids for sc in fixed.speech_cuts}
    assert by_span[(0, 2)] == ["S1"], by_span
    assert by_span[(3, 4)] == ["S2"], by_span
    print("ok  test_enforce_stamps_speaker_ids_from_word_diarization")


def _spk_words(*pairs, wdur=1000):
    """Words from (text, speaker) pairs on a back-to-back ``wdur``-ms grid."""
    return [{"start_ms": i * wdur, "end_ms": (i + 1) * wdur, "text": t, "speaker": s}
            for i, (t, s) in enumerate(pairs)]


def test_split_at_speaker_changes_splits_a_real_turn():
    # Two real turns lumped into one span -> split at the speaker change.
    words = _spk_words(("launch", "S1"), ("today", "S1"), ("great", "S2"), ("design", "S2"))
    assert pass1._split_at_speaker_changes(words, (0, 3)) == [(0, 1), (2, 3)]
    print("ok  test_split_at_speaker_changes_splits_a_real_turn")


def test_split_at_speaker_changes_folds_an_interior_backchannel():
    # S1 holds the floor; S2 drops a "yeah" in the middle -> ONE beat, no cut.
    words = _spk_words(("launch", "S1"), ("today", "S1"), ("yeah", "S2"),
                       ("and", "S1"), ("tomorrow", "S1"))
    assert pass1._split_at_speaker_changes(words, (0, 4)) == [(0, 4)]
    print("ok  test_split_at_speaker_changes_folds_an_interior_backchannel")


def test_split_at_speaker_changes_folds_a_trailing_backchannel():
    words = _spk_words(("launch", "S1"), ("today", "S1"), ("yeah", "S2"))
    assert pass1._split_at_speaker_changes(words, (0, 2)) == [(0, 2)]
    # "right, exactly" -- multi-word, still all backchannel -> folded.
    words2 = _spk_words(("launch", "S1"), ("today", "S1"), ("right", "S2"), ("exactly", "S2"))
    assert pass1._split_at_speaker_changes(words2, (0, 3)) == [(0, 3)]
    print("ok  test_split_at_speaker_changes_folds_a_trailing_backchannel")


def test_split_at_speaker_changes_cuts_a_terse_real_answer():
    # "No" is a real (terse) answer, NOT a backchannel token -> it cuts.
    words = _spk_words(("agree", "S1"), ("no", "S2"))
    assert pass1._split_at_speaker_changes(words, (0, 1)) == [(0, 0), (1, 1)]
    print("ok  test_split_at_speaker_changes_cuts_a_terse_real_answer")


def test_split_at_speaker_changes_folds_backchannel_but_cuts_the_real_turn():
    # yeah (S2) folds into S1's beat; S2's later real turn splits off.
    words = _spk_words(("launch", "S1"), ("today", "S1"), ("yeah", "S2"),
                       ("and", "S1"), ("tomorrow", "S1"), ("actually", "S2"), ("disagree", "S2"))
    assert pass1._split_at_speaker_changes(words, (0, 6)) == [(0, 4), (5, 6)]
    print("ok  test_split_at_speaker_changes_folds_backchannel_but_cuts_the_real_turn")


def test_enforce_splits_a_single_cut_at_a_bare_speaker_change():
    # No atom in the speaker-change gap, so _seam_split can't split it -- the
    # new speaker-change pass must, and stamp each turn's own voice.
    lat = Lattice(file_id="f1", duration_ms=4000, turns=[], hints=[], atoms=[],
                  words=_spk_words(("launch", "S1"), ("today", "S1"),
                                   ("great", "S2"), ("design", "S2")))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "lumped beat"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    by_span = {tuple(sc.word_span): sc.speaker_ids for sc in fixed.speech_cuts}
    assert sorted(by_span) == [(0, 1), (2, 3)], by_span
    assert by_span[(0, 1)] == ["S1"] and by_span[(2, 3)] == ["S2"], by_span
    print("ok  test_enforce_splits_a_single_cut_at_a_bare_speaker_change")


def test_enforce_folds_a_backchannel_within_a_single_cut():
    # A "yeah" (S2, 400ms) sits inside a long S1 beat -> stays ONE cut, and the
    # sub-threshold S2 voice is dropped from attribution (_MINOR_VOICE_SHARE).
    words = _spk_words(("launch", "S1"), ("today", "S1"), ("and", "S1"), ("tomorrow", "S1"))
    words.insert(2, {"start_ms": 2000, "end_ms": 2400, "text": "yeah", "speaker": "S2"})
    for k, w in enumerate(words):  # re-grid onto contiguous times after the insert
        w["start_ms"], w["end_ms"] = k * 1000, k * 1000 + (400 if w["text"] == "yeah" else 1000)
    lat = Lattice(file_id="f1", duration_ms=6000, turns=[], hints=[], atoms=[], words=words)
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "one beat"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 4)], fixed.speech_cuts
    assert fixed.speech_cuts[0].speaker_ids == ["S1"], fixed.speech_cuts[0].speaker_ids
    print("ok  test_enforce_folds_a_backchannel_within_a_single_cut")


def test_enforce_splits_synced_file_at_speaker_change():
    # Unlike the runt guard (skipped for synced files), the speaker-change split
    # keys only on shared per-word speaker labels, so it's SAFE for outlook
    # angles and still fires -- every angle splits identically.
    lat = Lattice(file_id="f1", duration_ms=4000, turns=[], hints=[], atoms=[],
                  words=_spk_words(("launch", "S1"), ("today", "S1"),
                                   ("great", "S2"), ("design", "S2")))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "lumped"}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat}, outlook_file_ids={"f1"})
    assert sorted(tuple(sc.word_span) for sc in fixed.speech_cuts) == [(0, 1), (2, 3)], fixed.speech_cuts
    print("ok  test_enforce_splits_synced_file_at_speaker_change")


def test_no_overlapping_speech_cuts():
    bad = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "a"},
            {"file_id": "f1", "word_span": [2, 6], "label": "b"},
        ],
    })
    err = pass1._no_overlapping_speech_cuts(bad)
    assert err is not None and "overlap" in err, err

    ok = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 4], "label": "a"},
            {"file_id": "f1", "word_span": [5, 6], "label": "b"},
            {"file_id": "f2", "word_span": [0, 4], "label": "c"},   # other file: fine
        ],
    })
    assert pass1._no_overlapping_speech_cuts(ok) is None
    print("ok  test_no_overlapping_speech_cuts")


def test_pass1_output_rejects_an_unexpected_wrapper_key():
    # Observed in the wild: the model wrapped its whole real answer under a
    # spurious top-level key. Every field here has a default, so without
    # extra="forbid" this would silently "validate" as an empty result.
    wrapped = {"$PARAMETER_NAME": {
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 1], "label": "x"}],
    }}
    try:
        pass1.Pass1Output.model_validate(wrapped)
        assert False, "expected a validation error"
    except Exception:
        pass
    print("ok  test_pass1_output_rejects_an_unexpected_wrapper_key")


def main():
    test_render_clip_block_includes_transcript_and_atoms()
    test_render_clip_block_handles_no_speech()
    test_build_pass1_blocks_one_per_clip()
    test_run_pass1_empty_file_rows_raises()
    test_run_pass1_calls_complete_with_pass1_stage_and_schema()
    test_pass1_output_schema_round_trip_with_junk_suspects()
    test_pass1_output_defaults_are_empty_not_missing()
    test_enforce_splits_a_speech_cut_at_an_atom_owned_gap()
    test_enforce_leaves_a_clean_grouping_untouched()
    test_enforce_keeps_a_genuine_same_line_take_group()
    test_enforce_drops_an_over_grouped_take_member()
    test_coverage_fill_recovers_uncovered_speech_words()
    test_bridge_absorbs_a_weldable_gap_into_one_beat()
    test_bridge_splits_at_a_shot_cut_in_the_gap()
    test_bridge_splits_at_a_speaker_change()
    test_bridge_magnitude_backstop_splits_an_over_long_gap()
    test_beat_merge_fuses_same_beat_neighbours_across_weldable_seam()
    test_beat_merge_respects_the_seam_guard()
    test_beat_merge_does_not_touch_untagged_neighbours()
    test_runt_guard_absorbs_trailing_runt_into_preceding_cut()
    test_runt_guard_absorbs_leading_runt_into_following_cut()
    test_runt_guard_respects_the_seam_guard()
    test_runt_guard_does_not_absorb_a_take_member()
    test_runt_guard_does_not_absorb_flagged_junk()
    test_runt_guard_skips_outlook_synced_files()
    test_speaker_ids_for_span_dominant_first_no_timing_falls_back_to_counts()
    test_speaker_ids_for_span_orders_by_spoken_time_not_appearance()
    test_speaker_ids_for_span_drops_sub_threshold_backchannel()
    test_enforce_stamps_speaker_ids_from_word_diarization()
    test_split_at_speaker_changes_splits_a_real_turn()
    test_split_at_speaker_changes_folds_an_interior_backchannel()
    test_split_at_speaker_changes_folds_a_trailing_backchannel()
    test_split_at_speaker_changes_cuts_a_terse_real_answer()
    test_split_at_speaker_changes_folds_backchannel_but_cuts_the_real_turn()
    test_enforce_splits_a_single_cut_at_a_bare_speaker_change()
    test_enforce_folds_a_backchannel_within_a_single_cut()
    test_enforce_splits_synced_file_at_speaker_change()
    test_silent_trailing_word_is_folded_into_its_beat()
    test_isolated_silent_word_is_dropped()
    test_no_overlapping_speech_cuts()
    test_pass1_output_rejects_an_unexpected_wrapper_key()
    print("\nall pass1 tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
