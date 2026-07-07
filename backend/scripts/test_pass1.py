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


def test_no_speech_cut_swallows_atoms_flags_a_cross_pause_group():
    # Words 0-2 end at 900ms; words 3-4 run 1500-2100ms. A GROUPED atom in the
    # 900-1500 gap (it would become a video cut). A speech cut grouping words
    # [0-4] spans that atom -> a real speech/video overlap, rejected with a
    # message naming the cut and the swallowed atom.
    lat = _make_lattice()
    lat.atoms.insert(0, Atom(atom_id=99, file_id="f1", start_ms=950, end_ms=1450,
                             state_in="speech_edge", state_out="speech_edge",
                             action_energy=0.1, coherence=1.0))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "whole thing"}],
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [99]}],
    })
    err = pass1._no_speech_cut_swallows_atoms(out, {"f1": lat})
    assert err is not None and "atom 99" in err and "speech_cut[0]" in err, err
    print("ok  test_no_speech_cut_swallows_atoms_flags_a_cross_pause_group")


def test_no_speech_cut_swallows_atoms_allows_an_absorbed_ungrouped_atom():
    # The SAME geometry, but atom 99 is in NO video group -- it was absorbed
    # into the beat across a weldable seam. It plays inside the beat and
    # produces no video cut, so there is no overlap: this is legal.
    lat = _make_lattice()
    lat.atoms.insert(0, Atom(atom_id=99, file_id="f1", start_ms=950, end_ms=1450,
                             state_in="speech_edge", state_out="speech_edge",
                             action_energy=0.1, coherence=1.0))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "whole thing"}],
    })
    assert pass1._no_speech_cut_swallows_atoms(out, {"f1": lat}) is None
    print("ok  test_no_speech_cut_swallows_atoms_allows_an_absorbed_ungrouped_atom")


def test_no_speech_cut_swallows_atoms_accepts_clean_grouping():
    lat = _make_lattice()
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 2], "label": "first"},
            {"file_id": "f1", "word_span": [3, 4], "label": "second"},
        ],
    })
    assert pass1._no_speech_cut_swallows_atoms(out, {"f1": lat}) is None
    print("ok  test_no_speech_cut_swallows_atoms_accepts_clean_grouping")


def test_no_speech_cut_swallows_atoms_rejects_out_of_range_span():
    lat = _make_lattice()
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 99], "label": "oops"}],
    })
    err = pass1._no_speech_cut_swallows_atoms(out, {"f1": lat})
    assert err is not None and "out of range" in err, err
    print("ok  test_no_speech_cut_swallows_atoms_rejects_out_of_range_span")


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
    assert pass1._no_speech_cut_swallows_atoms(fixed, {"f1": lat}) is None
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


def test_enforce_remaps_take_members_onto_split_cuts():
    lat = _make_lattice()
    lat.atoms.insert(0, Atom(atom_id=99, file_id="f1", start_ms=950, end_ms=1450,
                             state_in="speech_edge", state_out="speech_edge",
                             action_energy=0.1, coherence=1.0))
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 4], "label": "whole"}],
        "take_candidates": [{"group_id": "tg1", "members": [
            {"file_id": "f1", "word_span": [0, 2]},
            {"file_id": "f1", "word_span": [3, 4]},
        ]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    tc = fixed.take_candidates[0]
    assert [tuple(m.word_span) for m in tc.members] == [(0, 2), (3, 4)], tc

    # Both members collapsing onto the SAME cut -> group degenerates and is dropped.
    out2 = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 2], "label": "only"},
                        {"file_id": "f1", "word_span": [3, 4], "label": "other"}],
        "take_candidates": [{"group_id": "tg2", "members": [
            {"file_id": "f1", "word_span": [0, 1]},
            {"file_id": "f1", "word_span": [1, 2]},
        ]}],
    })
    fixed2 = pass1.enforce_lattice_partition(out2, {"f1": lat})
    assert fixed2.take_candidates == [], fixed2.take_candidates
    print("ok  test_enforce_remaps_take_members_onto_split_cuts")


def test_enforce_splits_a_discontiguous_video_group():
    # Atoms 0 (2100-2600) and 1 (2600-3200) are contiguous in _make_lattice;
    # add atom 2 far away (5000-6000). A group [0, 1, 2] bridges a gap (in
    # real data: bridges a speech span) and must come back as [0,1] + [2].
    lat = _make_lattice()
    lat.atoms.append(Atom(atom_id=2, file_id="f1", start_ms=5000, end_ms=6000,
                          state_in="speech_edge", state_out="clip_edge",
                          action_energy=0.3, coherence=0.9))
    out = pass1.Pass1Output.model_validate({
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0, 1, 2]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    groups = [vg.atom_ids for vg in fixed.video_tentative_groups]
    assert groups == [[0, 1], [2]], groups

    # a contiguous group is left untouched (coverage-fill has nothing to add
    # here: atoms 0,1 are contiguous and every atom is now grouped).
    ok = pass1.Pass1Output.model_validate({
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0, 1]}],
    })
    fixed_ok = pass1.enforce_lattice_partition(ok, {"f1": lat})
    assert [0, 1] in [vg.atom_ids for vg in fixed_ok.video_tentative_groups]
    print("ok  test_enforce_splits_a_discontiguous_video_group")


def test_enforce_no_longer_isolates_action_atoms():
    # signal-judge: grouping/merging is the model's job now, so a contiguous
    # group containing an action atom is LEFT INTACT (no forced isolation) --
    # a swing and its follow-through can ride together if the model said so.
    lat = _make_lattice()
    lat.atoms.append(Atom(atom_id=2, file_id="f1", start_ms=3200, end_ms=3800,
                          state_in="action", state_out="action",
                          action_energy=0.7, coherence=0.9,
                          anchor_ms=[3500], is_action=True, peak_action_energy=0.9))
    lat.atoms.append(Atom(atom_id=3, file_id="f1", start_ms=3800, end_ms=4400,
                          state_in="action", state_out="clip_edge",
                          action_energy=0.2, coherence=0.9))
    out = pass1.Pass1Output.model_validate({
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0, 1, 2, 3]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    groups = [vg.atom_ids for vg in fixed.video_tentative_groups]
    assert groups == [[0, 1, 2, 3]], groups
    print("ok  test_enforce_no_longer_isolates_action_atoms")


def test_coverage_fill_readds_every_ungrouped_atom():
    # Deterministic-keep: the model dropped atoms 2 (high energy) AND 3 (low
    # energy). Coverage-fill re-adds BOTH -- no energy threshold decides who
    # survives; nothing is silently lost. Contiguous ungrouped atoms fold
    # into one recovered group.
    lat = _make_lattice()
    lat.atoms.append(Atom(atom_id=2, file_id="f1", start_ms=3200, end_ms=3800,
                          state_in="action", state_out="action",
                          action_energy=0.5, coherence=0.9,
                          anchor_ms=[3500], is_action=True, peak_action_energy=0.99))
    lat.atoms.append(Atom(atom_id=3, file_id="f1", start_ms=3800, end_ms=4400,
                          state_in="action", state_out="clip_edge",
                          action_energy=0.1, coherence=0.9, peak_action_energy=0.2))
    out = pass1.Pass1Output.model_validate({
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0, 1]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    covered = {a for vg in fixed.video_tentative_groups for a in vg.atom_ids}
    assert covered == {0, 1, 2, 3}, fixed.video_tentative_groups
    # 2 and 3 are contiguous -> folded into ONE recovered group, not two.
    assert [2, 3] in [vg.atom_ids for vg in fixed.video_tentative_groups], fixed.video_tentative_groups
    print("ok  test_coverage_fill_readds_every_ungrouped_atom")


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
    # mid-thought. The beat [0-3] must stay ONE cut, and the atom must LEAVE the
    # video pool (it plays inside the beat, produces no video cut).
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.7, coherence=0.9, is_action=True)
    lat = _bridge_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [{"file_id": "f1", "word_span": [0, 3], "label": "demo beat"}],
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [10]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 3)], fixed.speech_cuts
    assert fixed.speech_cuts[0].label == "demo beat"   # untouched, no (n/n) suffix
    grouped = {a for vg in fixed.video_tentative_groups for a in vg.atom_ids}
    assert 10 not in grouped, fixed.video_tentative_groups
    assert pass1._no_speech_cut_swallows_atoms(fixed, {"f1": lat}) is None
    print("ok  test_bridge_absorbs_a_weldable_gap_into_one_beat")


def test_bridge_splits_at_a_shot_cut_in_the_gap():
    # Two atoms in the gap with a SHOT CUT between them (a real break). Even
    # same-speaker, short gap -> hard seam -> beat splits into speech . speech,
    # and the atoms stay in the video pool.
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
    grouped = {a for vg in fixed.video_tentative_groups for a in vg.atom_ids}
    assert {10, 11} <= grouped, fixed.video_tentative_groups
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
    # break) -> code merges them into ONE beat and absorbs the atom.
    atom = Atom(atom_id=10, file_id="f1", start_ms=2000, end_ms=2600,
                state_in="speech_edge", state_out="speech_edge",
                action_energy=0.7, coherence=0.9, is_action=True)
    lat = _bridge_lattice([atom])
    out = pass1.Pass1Output.model_validate({
        "speech_cuts": [
            {"file_id": "f1", "word_span": [0, 1], "label": "setup", "beat_id": "b1"},
            {"file_id": "f1", "word_span": [2, 3], "label": "payoff", "beat_id": "b1"},
        ],
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [10]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 3)], fixed.speech_cuts
    grouped = {a for vg in fixed.video_tentative_groups for a in vg.atom_ids}
    assert 10 not in grouped, fixed.video_tentative_groups
    assert pass1._no_speech_cut_swallows_atoms(fixed, {"f1": lat}) is None
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
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [10]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    assert [tuple(sc.word_span) for sc in fixed.speech_cuts] == [(0, 1), (2, 3)], fixed.speech_cuts
    grouped = {a for vg in fixed.video_tentative_groups for a in vg.atom_ids}
    assert 10 in grouped, "untagged neighbours must not absorb the atom"
    print("ok  test_beat_merge_does_not_touch_untagged_neighbours")


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
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0]}],
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
        "video_tentative_groups": [{"file_id": "f1", "atom_ids": [0, 1]}],
    })
    fixed = pass1.enforce_lattice_partition(out, {"f1": lat})
    spans = sorted(tuple(sc.word_span) for sc in fixed.speech_cuts)
    assert spans == [(0, 0)], spans   # the isolated silent word dropped
    print("ok  test_isolated_silent_word_is_dropped")


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
    test_no_speech_cut_swallows_atoms_flags_a_cross_pause_group()
    test_no_speech_cut_swallows_atoms_allows_an_absorbed_ungrouped_atom()
    test_no_speech_cut_swallows_atoms_accepts_clean_grouping()
    test_no_speech_cut_swallows_atoms_rejects_out_of_range_span()
    test_enforce_splits_a_speech_cut_at_an_atom_owned_gap()
    test_enforce_leaves_a_clean_grouping_untouched()
    test_enforce_remaps_take_members_onto_split_cuts()
    test_enforce_splits_a_discontiguous_video_group()
    test_enforce_no_longer_isolates_action_atoms()
    test_coverage_fill_readds_every_ungrouped_atom()
    test_coverage_fill_recovers_uncovered_speech_words()
    test_bridge_absorbs_a_weldable_gap_into_one_beat()
    test_bridge_splits_at_a_shot_cut_in_the_gap()
    test_bridge_splits_at_a_speaker_change()
    test_bridge_magnitude_backstop_splits_an_over_long_gap()
    test_beat_merge_fuses_same_beat_neighbours_across_weldable_seam()
    test_beat_merge_respects_the_seam_guard()
    test_beat_merge_does_not_touch_untagged_neighbours()
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
