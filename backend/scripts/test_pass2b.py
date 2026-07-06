"""
Tests for the cuts-v3 pass-2b module (``app.services.l3.pass2b``) -- VISUAL
judgment only (framing/look/captions/taste), no cross-cut dependency. No DB,
NO REAL API CALLS. Reuses the fake-SDK-client pattern from
``test_ingest_client.py``.

Run:  .venv/bin/python scripts/test_pass2b.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2b  # noqa: E402
from app.services.l3.image_plan import PlannedFrame  # noqa: E402
from app.services.l3.pass2a import IdentityCut, IdentityOutput  # noqa: E402
from app.services.llm import client as ic  # noqa: E402
from test_ingest_client import FakeBlock, FakeClient, FakeResponse  # noqa: E402


def _with_fake_client(responses):
    fake = FakeClient(responses)
    orig = ic._sdk_client
    ic._sdk_client = lambda: fake
    return fake, orig


def _identity_output():
    return IdentityOutput(cuts=[
        IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 3),
                   label="intro", summary="says hello", junk=False),
        IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                   label="pan", summary="pans across desk", junk=True, junk_reason="blurry"),
    ])


# --------------------------------------------------------------------------
# VisualJudgment / VisualOutput schema
# --------------------------------------------------------------------------

def test_visual_judgment_defaults_are_safe():
    j = pass2b.VisualJudgment(cut_index=0)
    assert j.framing.rotation_deg == 0.0
    assert j.look.graded is False
    assert j.caption_zones == []
    assert j.taste_fences.max_tasteful_speed == 1.0
    assert j.readability_ms == 0
    print("ok  test_visual_judgment_defaults_are_safe")


def test_visual_output_rejects_an_unexpected_wrapper_key():
    wrapped = {"$PARAMETER_NAME": {"judgments": [{"cut_index": 0}]}}
    try:
        pass2b.VisualOutput.model_validate(wrapped)
        assert False, "expected a validation error"
    except Exception:
        pass
    print("ok  test_visual_output_rejects_an_unexpected_wrapper_key")


# --------------------------------------------------------------------------
# render_identity_output
# --------------------------------------------------------------------------

def test_render_identity_output_numbers_cuts_by_position():
    text = pass2b.render_identity_output(_identity_output())
    assert "CUT 0: speech file=f1 words[0-3]" in text, text
    assert "CUT 1: video file=f1 atoms[0, 1]" in text, text
    assert "junk=True" in text
    print("ok  test_render_identity_output_numbers_cuts_by_position")


# --------------------------------------------------------------------------
# _images_for_cut / batching
# --------------------------------------------------------------------------

def test_images_for_cut_matches_by_file_and_source_ref():
    planned = [
        PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
        PlannedFrame("f1", 200, "video_group", "video_group[0]"),
        PlannedFrame("f2", 300, "speech_cut", "speech_cut[0]"),   # different file, ignored
    ]
    images_b64 = {("f1", 100): "aaa", ("f1", 200): "bbb", ("f2", 300): "ccc"}
    cut = _identity_output().cuts[0]   # speech_cut[0], file_id=f1
    result = pass2b._images_for_cut(cut, planned, images_b64)
    assert result == [(100, "aaa")], result
    print("ok  test_images_for_cut_matches_by_file_and_source_ref")


def test_images_for_cut_skips_unresolved_images():
    planned = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    cut = _identity_output().cuts[0]
    result = pass2b._images_for_cut(cut, planned, {})
    assert result == []
    print("ok  test_images_for_cut_skips_unresolved_images")


def test_build_visual_batches_chunks_without_clustering():
    identity = IdentityOutput(cuts=[
        IdentityCut(source_ref=f"speech_cut[{i}]", kind="speech", file_id="f1", word_span=(i, i + 1),
                   label="x", summary="y")
        for i in range(5)
    ])
    batches = pass2b.build_visual_batches(identity, max_per_batch=2)
    assert batches == [[0, 1], [2, 3], [4]], batches
    print("ok  test_build_visual_batches_chunks_without_clustering")


def test_build_visual_batches_empty_identity_yields_no_batches():
    assert pass2b.build_visual_batches(IdentityOutput(), max_per_batch=10) == []
    print("ok  test_build_visual_batches_empty_identity_yields_no_batches")


def test_build_visual_batch_blocks_numbers_by_cut_index():
    identity = _identity_output()
    planned = [
        PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
        PlannedFrame("f1", 200, "video_group", "video_group[0]"),
    ]
    images_b64 = {("f1", 100): "aaa", ("f1", 200): "bbb"}
    blocks = pass2b.build_visual_batch_blocks(identity, [0, 1], planned, images_b64)
    assert len(blocks) == 4   # 2 caption/image pairs
    assert blocks[0]["type"] == "text" and "CUT 0" in blocks[0]["text"]
    assert blocks[1]["type"] == "image"
    assert blocks[2]["type"] == "text" and "CUT 1" in blocks[2]["text"]
    assert blocks[3]["type"] == "image"
    print("ok  test_build_visual_batch_blocks_numbers_by_cut_index")


# --------------------------------------------------------------------------
# run_visual_batch
# --------------------------------------------------------------------------

def test_run_visual_batch_raises_when_no_images_resolve():
    identity = _identity_output()
    try:
        pass2b.run_visual_batch(identity, [0, 1], [], {})
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_run_visual_batch_raises_when_no_images_resolve")


def test_run_visual_batch_calls_complete_with_pass2_stage_and_cached_prefix():
    good = {"judgments": [
        {"cut_index": 0, "framing": {"rotation_deg": 90.0}, "readability_ms": 500},
    ]}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    identity = _identity_output()
    planned = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    images_b64 = {("f1", 100): "ZmFrZQ=="}
    try:
        result = pass2b.run_visual_batch(identity, [0], planned, images_b64)
    finally:
        ic._sdk_client = orig

    call = fake.messages.calls[0]
    assert call["tools"][0]["input_schema"] == pass2b.VisualOutput.model_json_schema()
    content = call["messages"][0]["content"]
    # cached prefix (the rendered confirmed-cut list) ends with the cache
    # breakpoint; the image caption/image pair rides after it, uncached.
    assert content[-3]["cache_control"] == {"type": "ephemeral"}, content
    assert "cache_control" not in content[-2]
    assert "cache_control" not in content[-1]
    assert content[-2]["type"] == "text" and "CUT 0" in content[-2]["text"]
    assert content[-1]["type"] == "image"
    parsed = pass2b.VisualOutput.model_validate(result.data)
    assert parsed.judgments[0].cut_index == 0
    assert parsed.judgments[0].framing.rotation_deg == 90.0
    print("ok  test_run_visual_batch_calls_complete_with_pass2_stage_and_cached_prefix")


def main():
    test_visual_judgment_defaults_are_safe()
    test_visual_output_rejects_an_unexpected_wrapper_key()
    test_render_identity_output_numbers_cuts_by_position()
    test_images_for_cut_matches_by_file_and_source_ref()
    test_images_for_cut_skips_unresolved_images()
    test_build_visual_batches_chunks_without_clustering()
    test_build_visual_batches_empty_identity_yields_no_batches()
    test_build_visual_batch_blocks_numbers_by_cut_index()
    test_run_visual_batch_raises_when_no_images_resolve()
    test_run_visual_batch_calls_complete_with_pass2_stage_and_cached_prefix()
    print("\nall pass2b tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
