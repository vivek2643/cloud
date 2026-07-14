"""
Tests for the voice->face binding pass
(app.services.l3.identity.speaker_pass) -- no DB, NO REAL API CALLS. Mocks
the Gemini SDK client the same way test_ingest_gemini.py does.

Run:  .venv/bin/python scripts/test_identity_speaker_pass.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.identity import speaker_pass as sp  # noqa: E402
from app.services.l3.identity.speaker_frames import Burst  # noqa: E402
from app.services.llm import ingest_gemini as ig  # noqa: E402


def _types():
    from google.genai import types
    return types


class _FakeUsageMD:
    prompt_token_count = 100
    candidates_token_count = 20
    cached_content_token_count = 0
    thoughts_token_count = 0


class _FakeGeminiResp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = _FakeUsageMD()
        self.candidates = []


def _persons():
    return {"P0": {"display": "bald man, beard"}, "P1": {"display": "woman, long hair"}}


# --------------------------------------------------------------------------
# build_speaker_pass_blocks
# --------------------------------------------------------------------------

def test_build_speaker_pass_blocks_captions_each_candidate_with_its_display():
    bursts = [
        Burst(voice="V0", window_id="V0:w0", candidate_person="P0", file_id="f1", ts_ms=[100, 190, 280]),
    ]
    images = {("f1", 100): "b64a", ("f1", 190): "b64b", ("f1", 280): "b64c"}
    blocks = sp.build_speaker_pass_blocks(bursts, images, _persons())
    caption = blocks[0]["text"]
    assert "V0:w0" in caption and "P0" in caption and "bald man" in caption, caption
    assert len(blocks) == 1 + 3   # 1 caption + 3 frames
    print("ok  test_build_speaker_pass_blocks_captions_each_candidate_with_its_display")


def test_build_speaker_pass_blocks_skips_a_candidate_with_no_resolvable_image():
    bursts = [Burst(voice="V0", window_id="V0:w0", candidate_person="P0", file_id="f1", ts_ms=[100])]
    blocks = sp.build_speaker_pass_blocks(bursts, {}, _persons())
    assert blocks == [], blocks
    print("ok  test_build_speaker_pass_blocks_skips_a_candidate_with_no_resolvable_image")


def test_build_speaker_pass_blocks_groups_multi_candidate_window_together():
    bursts = [
        Burst(voice="V0", window_id="V0:w0", candidate_person="P0", file_id="f1", ts_ms=[100]),
        Burst(voice="V0", window_id="V0:w0", candidate_person="P1", file_id="f1", ts_ms=[100]),
    ]
    images = {("f1", 100): "b64a"}
    blocks = sp.build_speaker_pass_blocks(bursts, images, _persons())
    captions = [b["text"] for b in blocks if "text" in b]
    assert any("P0" in c for c in captions) and any("P1" in c for c in captions), captions
    print("ok  test_build_speaker_pass_blocks_groups_multi_candidate_window_together")


# --------------------------------------------------------------------------
# aggregate_votes
# --------------------------------------------------------------------------

def test_aggregate_votes_requires_majority_and_margin():
    # 3/4 votes for P0, clear majority + margin -> bound.
    votes = [sp.WindowVote(window_id=f"w{i}", speaking_person="P0") for i in range(3)]
    votes.append(sp.WindowVote(window_id="w3", speaking_person=None))
    assert sp.aggregate_votes(votes) == "P0"
    print("ok  test_aggregate_votes_requires_majority_and_margin")


def test_aggregate_votes_unbound_on_a_split_tie():
    votes = [sp.WindowVote(window_id="w0", speaking_person="P0"),
            sp.WindowVote(window_id="w1", speaking_person="P1")]
    assert sp.aggregate_votes(votes) is None
    print("ok  test_aggregate_votes_unbound_on_a_split_tie")


def test_aggregate_votes_unbound_when_nobody_ever_speaks():
    votes = [sp.WindowVote(window_id="w0", speaking_person=None),
            sp.WindowVote(window_id="w1", speaking_person=None)]
    assert sp.aggregate_votes(votes) is None
    print("ok  test_aggregate_votes_unbound_when_nobody_ever_speaks")


def test_aggregate_votes_empty_is_unbound():
    assert sp.aggregate_votes([]) is None
    print("ok  test_aggregate_votes_empty_is_unbound")


def test_aggregate_votes_unbound_below_majority_share():
    # 2/5 for the winner -- well under a majority even with a margin over 2nd.
    votes = ([sp.WindowVote(window_id="w0", speaking_person="P0")] * 2
            + [sp.WindowVote(window_id="w1", speaking_person="P1")]
            + [sp.WindowVote(window_id="w2", speaking_person=None)] * 2)
    assert sp.aggregate_votes(votes) is None
    print("ok  test_aggregate_votes_unbound_below_majority_share")


# --------------------------------------------------------------------------
# run_speaker_pass / bind_voices (mocked Gemini SDK)
# --------------------------------------------------------------------------

def test_run_speaker_pass_returns_empty_with_no_images():
    bursts = [Burst(voice="V0", window_id="V0:w0", candidate_person="P0", file_id="f1", ts_ms=[100])]
    assert sp.run_speaker_pass(bursts, {}, _persons()) == []
    print("ok  test_run_speaker_pass_returns_empty_with_no_images")


def test_run_speaker_pass_parses_the_model_votes():
    bursts = [Burst(voice="V0", window_id="V0:w0", candidate_person="P0", file_id="f1", ts_ms=[100])]
    images = {("f1", 100): "b64a"}
    payload = '{"votes": [{"window_id": "V0:w0", "speaking_person": "P0"}]}'

    def fake_generate_content(model, contents, config):
        return _FakeGeminiResp(payload)

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        votes = sp.run_speaker_pass(bursts, images, _persons())
    assert len(votes) == 1 and votes[0].window_id == "V0:w0" and votes[0].speaking_person == "P0", votes
    print("ok  test_run_speaker_pass_parses_the_model_votes")


def test_bind_voices_aggregates_per_voice_and_skips_the_rest():
    bursts = [
        Burst(voice="V0", window_id="V0:w0", candidate_person="P0", file_id="f1", ts_ms=[100]),
        Burst(voice="V1", window_id="V1:w0", candidate_person="P1", file_id="f1", ts_ms=[200]),
    ]
    images = {("f1", 100): "b64a", ("f1", 200): "b64b"}
    responses = [
        '{"votes": [{"window_id": "V0:w0", "speaking_person": "P0"}]}',
        '{"votes": [{"window_id": "V1:w0"}]}',   # nobody speaks -> unbound
    ]
    calls = []

    def fake_generate_content(model, contents, config):
        calls.append(1)
        return _FakeGeminiResp(responses[len(calls) - 1])

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        owners = sp.bind_voices(bursts, images, _persons())
    assert owners == {"V0": "P0", "V1": None}, owners
    assert len(calls) == 2   # one call per voice
    print("ok  test_bind_voices_aggregates_per_voice_and_skips_the_rest")


def main():
    test_build_speaker_pass_blocks_captions_each_candidate_with_its_display()
    test_build_speaker_pass_blocks_skips_a_candidate_with_no_resolvable_image()
    test_build_speaker_pass_blocks_groups_multi_candidate_window_together()
    test_aggregate_votes_requires_majority_and_margin()
    test_aggregate_votes_unbound_on_a_split_tie()
    test_aggregate_votes_unbound_when_nobody_ever_speaks()
    test_aggregate_votes_empty_is_unbound()
    test_aggregate_votes_unbound_below_majority_share()
    test_run_speaker_pass_returns_empty_with_no_images()
    test_run_speaker_pass_parses_the_model_votes()
    test_bind_voices_aggregates_per_voice_and_skips_the_rest()
    print("\nall identity-speaker-pass tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
