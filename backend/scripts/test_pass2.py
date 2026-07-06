"""
Tests for the cuts-v3 pass-2 module (``app.services.l3.pass2``) -- no DB, NO
REAL API CALLS. Reuses the fake-SDK-client pattern from
``test_ingest_client.py``.

Run:  .venv/bin/python scripts/test_pass2.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2  # noqa: E402
from app.services.l3.image_plan import PlannedFrame  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402
from app.services.l3.pass1 import (  # noqa: E402
    Pass1Output, SpeechCut, TakeCandidate, TakeMember, VideoTentativeGroup,
)
from app.services.llm import client as ic  # noqa: E402
from test_ingest_client import FakeBlock, FakeClient, FakeResponse  # noqa: E402


def _with_fake_client(responses):
    fake = FakeClient(responses)
    orig = ic._sdk_client
    ic._sdk_client = lambda: fake
    return fake, orig


def _lat(file_id):
    return Lattice(file_id=file_id, duration_ms=10000, words=[], turns=[], hints=[], atoms=[])


# --------------------------------------------------------------------------
# Pass2Cut / Pass2Output schema
# --------------------------------------------------------------------------

def test_speech_cut_requires_word_span():
    try:
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                       label="x", summary="y")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_speech_cut_requires_word_span")


def test_video_cut_requires_atom_ids():
    try:
        pass2.Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                       label="x", summary="y")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_video_cut_requires_atom_ids")


def test_invalid_take_role_rejected():
    try:
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                       word_span=(0, 1), label="x", summary="y", take_role="runner-up")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_invalid_take_role_rejected")


def test_take_role_aliases_normalize_to_take():
    # Real model output has used "alt" where the schema says "take" --
    # a reasonable synonym, normalized rather than burning a re-ask on it.
    for alias in ("alt", "ALT", "alternate", "loser", "Other"):
        cut = pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                             word_span=(0, 1), label="x", summary="y", take_role=alias)
        assert cut.take_role == "take", (alias, cut.take_role)
    print("ok  test_take_role_aliases_normalize_to_take")


def test_valid_cuts_round_trip():
    speech = pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                            word_span=(0, 3), label="intro", summary="says hello",
                            take_role="winner", take_group_id="tg1")
    video = pass2.Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                           atom_ids=[0, 1], label="pan", summary="pans across desk")
    out = pass2.Pass2Output(cuts=[speech, video])
    assert len(out.cuts) == 2
    assert out.cuts[0].take_role == "winner"
    print("ok  test_valid_cuts_round_trip")


def test_pass2_output_rejects_an_unexpected_wrapper_key():
    wrapped = {"$PARAMETER_NAME": {"cuts": [
        {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
         "word_span": [0, 1], "label": "x", "summary": "y"},
    ]}}
    try:
        pass2.Pass2Output.model_validate(wrapped)
        assert False, "expected a validation error"
    except Exception:
        pass
    print("ok  test_pass2_output_rejects_an_unexpected_wrapper_key")


def test_no_duplicate_atoms_passes_when_every_atom_is_used_once():
    out = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                       atom_ids=[0, 1], label="a", summary="a"),
        pass2.Pass2Cut(source_ref="video_group[1]", kind="video", file_id="f1",
                       atom_ids=[2, 3], label="b", summary="b"),
    ])
    assert pass2._no_duplicate_atoms(out) is None
    print("ok  test_no_duplicate_atoms_passes_when_every_atom_is_used_once")


def test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts():
    out = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                       atom_ids=[0, 1, 2], label="a", summary="a"),
        pass2.Pass2Cut(source_ref="video_group[0b]", kind="video", file_id="f1",
                       atom_ids=[2, 3], label="b", summary="b"),   # atom 2 double-counted
    ])
    err = pass2._no_duplicate_atoms(out)
    assert err is not None and "atom_id 2" in err, err
    print("ok  test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts")


def test_no_duplicate_atoms_ignores_speech_cuts():
    out = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                       word_span=(0, 1), label="a", summary="a"),
        pass2.Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                       word_span=(2, 3), label="b", summary="b"),
    ])
    assert pass2._no_duplicate_atoms(out) is None
    print("ok  test_no_duplicate_atoms_ignores_speech_cuts")


# --------------------------------------------------------------------------
# Shard building
# --------------------------------------------------------------------------

def test_single_file_one_shard():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[1]")]
    shards = pass2.build_shards(Pass1Output(), frames)
    assert shards == [["f1"]], shards
    print("ok  test_single_file_one_shard")


def test_unrelated_files_pack_into_one_shard_when_small():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f2", 100, "speech_cut", "speech_cut[0]")]
    shards = pass2.build_shards(Pass1Output(), frames)
    assert len(shards) == 1, shards
    assert set(shards[0]) == {"f1", "f2"}, shards
    print("ok  test_unrelated_files_pack_into_one_shard_when_small")


def test_take_group_forces_co_location_across_files():
    frames = [PlannedFrame("f1", 100, "take_member", "take[tg1]"),
             PlannedFrame("f2", 100, "take_member", "take[tg1]")]
    pass1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg1", members=[
        TakeMember(file_id="f1", word_span=(0, 1)),
        TakeMember(file_id="f2", word_span=(0, 1)),
    ])])
    shards = pass2.build_shards(pass1, frames)
    assert len(shards) == 1 and set(shards[0]) == {"f1", "f2"}, shards
    print("ok  test_take_group_forces_co_location_across_files")


def test_bin_packing_splits_when_over_budget():
    orig = pass2.MAX_IMAGES_PER_SHARD
    pass2.MAX_IMAGES_PER_SHARD = 2
    try:
        frames = ([PlannedFrame("f1", i, "speech_cut", f"speech_cut[{i}]") for i in range(2)] +
                 [PlannedFrame("f2", i, "speech_cut", f"speech_cut[{i}]") for i in range(2)])
        shards = pass2.build_shards(Pass1Output(), frames)
    finally:
        pass2.MAX_IMAGES_PER_SHARD = orig
    assert len(shards) == 2, shards
    all_files = {f for shard in shards for f in shard}
    assert all_files == {"f1", "f2"}
    print("ok  test_bin_packing_splits_when_over_budget")


def test_bin_packing_splits_on_cut_count_even_when_images_fit():
    # Two unrelated files, plenty of shared image budget, but together they
    # have too many cuts for one shard -- added after real ingest runs
    # showed the model getting unreliable on very large single-call outputs
    # (~40-80 cuts) independent of image count.
    orig = pass2.MAX_CUTS_PER_SHARD
    pass2.MAX_CUTS_PER_SHARD = 2
    try:
        frames = [PlannedFrame("f1", 0, "speech_cut", "speech_cut[0]"),
                 PlannedFrame("f2", 0, "speech_cut", "speech_cut[0]")]
        pass1 = Pass1Output(speech_cuts=[
            SpeechCut(file_id="f1", word_span=(0, 1), label="a"),
            SpeechCut(file_id="f1", word_span=(2, 3), label="b"),
            SpeechCut(file_id="f2", word_span=(0, 1), label="c"),
            SpeechCut(file_id="f2", word_span=(2, 3), label="d"),
        ])
        shards = pass2.build_shards(pass1, frames)
    finally:
        pass2.MAX_CUTS_PER_SHARD = orig
    assert len(shards) == 2, shards
    assert {f for shard in shards for f in shard} == {"f1", "f2"}
    print("ok  test_bin_packing_splits_on_cut_count_even_when_images_fit")


def test_oversized_take_cluster_is_not_split():
    orig = pass2.MAX_IMAGES_PER_SHARD
    pass2.MAX_IMAGES_PER_SHARD = 1
    try:
        frames = [PlannedFrame("f1", 0, "take_member", "take[tg1]"),
                 PlannedFrame("f2", 0, "take_member", "take[tg1]")]
        pass1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg1", members=[
            TakeMember(file_id="f1", word_span=(0, 1)),
            TakeMember(file_id="f2", word_span=(0, 1)),
        ])])
        shards = pass2.build_shards(pass1, frames)
    finally:
        pass2.MAX_IMAGES_PER_SHARD = orig
    assert len(shards) == 1, shards
    assert set(shards[0]) == {"f1", "f2"}
    print("ok  test_oversized_take_cluster_is_not_split")


def test_empty_frames_yield_no_shards():
    assert pass2.build_shards(Pass1Output(), []) == []
    print("ok  test_empty_frames_yield_no_shards")


# --------------------------------------------------------------------------
# Shard block rendering + orchestration
# --------------------------------------------------------------------------

def test_shard_blocks_skip_unresolved_images():
    frames = [PlannedFrame("f1", 200, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 100, "speech_cut", "speech_cut[1]")]
    images = {("f1", 100): "ZmFrZQ=="}   # only one of the two resolved
    blocks = pass2.build_pass2_shard_blocks(frames, images)
    assert len(blocks) == 2, blocks   # one [caption, image] pair only
    assert blocks[0]["type"] == "text" and "IMG 1" in blocks[0]["text"]
    assert "0.1s" in blocks[0]["text"], blocks[0]
    assert blocks[1]["type"] == "image"
    print("ok  test_shard_blocks_skip_unresolved_images")


def test_shard_blocks_ordered_by_file_then_ts():
    frames = [PlannedFrame("f2", 50, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 100, "speech_cut", "speech_cut[1]")]
    images = {("f2", 50): "a", ("f1", 200): "b", ("f1", 100): "c"}
    blocks = pass2.build_pass2_shard_blocks(frames, images)
    captions = [b["text"] for b in blocks if b["type"] == "text"]
    assert "clip f1, 0.1s" in captions[0], captions
    assert "clip f1, 0.2s" in captions[1], captions
    assert "clip f2, 0.1s" in captions[2], captions
    print("ok  test_shard_blocks_ordered_by_file_then_ts")


def test_run_pass2_shard_raises_when_no_images_resolve():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    try:
        pass2.run_pass2_shard([("f1", "a.mp4", 10000, _lat("f1"))], Pass1Output(), frames, {})
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_run_pass2_shard_raises_when_no_images_resolve")


def test_run_pass2_shard_calls_complete_with_pass2_stage_and_cached_prefix():
    good = {"cuts": [{
        "source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
        "word_span": [0, 2], "label": "intro", "summary": "hello there",
    }]}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    pass1_output = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="intro")])
    try:
        result = pass2.run_pass2_shard(
            [("f1", "a.mp4", 10000, _lat("f1"))], pass1_output, frames, {("f1", 100): "ZmFrZQ=="},
        )
    finally:
        ic._sdk_client = orig

    call = fake.messages.calls[0]
    assert call["tools"][0]["input_schema"] == pass2.Pass2Output.model_json_schema()
    content = call["messages"][0]["content"]
    # cached prefix (pass1 blocks + rendered pass1 output) ends with the cache
    # breakpoint; the image caption/image pair rides after it, uncached.
    assert content[-3]["cache_control"] == {"type": "ephemeral"}, content
    assert "cache_control" not in content[-2]
    assert "cache_control" not in content[-1]
    assert content[-2]["type"] == "text" and "IMG 1" in content[-2]["text"]
    assert content[-1]["type"] == "image"
    parsed = pass2.Pass2Output.model_validate(result.data)
    assert parsed.cuts[0].source_ref == "speech_cut[0]"
    print("ok  test_run_pass2_shard_calls_complete_with_pass2_stage_and_cached_prefix")


def main():
    test_speech_cut_requires_word_span()
    test_video_cut_requires_atom_ids()
    test_invalid_take_role_rejected()
    test_take_role_aliases_normalize_to_take()
    test_valid_cuts_round_trip()
    test_pass2_output_rejects_an_unexpected_wrapper_key()
    test_no_duplicate_atoms_passes_when_every_atom_is_used_once()
    test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts()
    test_no_duplicate_atoms_ignores_speech_cuts()
    test_single_file_one_shard()
    test_unrelated_files_pack_into_one_shard_when_small()
    test_take_group_forces_co_location_across_files()
    test_bin_packing_splits_when_over_budget()
    test_bin_packing_splits_on_cut_count_even_when_images_fit()
    test_oversized_take_cluster_is_not_split()
    test_empty_frames_yield_no_shards()
    test_shard_blocks_skip_unresolved_images()
    test_shard_blocks_ordered_by_file_then_ts()
    test_run_pass2_shard_raises_when_no_images_resolve()
    test_run_pass2_shard_calls_complete_with_pass2_stage_and_cached_prefix()
    print("\nall pass2 tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
