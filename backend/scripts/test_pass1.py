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
             camera_desc="hold", coherence=0.9, anchor_ms=[2300]),
        Atom(atom_id=1, file_id="f1", start_ms=2600, end_ms=3200,
             state_in="shot", state_out="clip_edge", action_energy=0.4,
             camera_desc="pan", coherence=0.8, anchor_ms=[]),
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
    test_pass1_output_rejects_an_unexpected_wrapper_key()
    print("\nall pass1 tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
