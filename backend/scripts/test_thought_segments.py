"""
Tests for the L3 thought segmenter (no DB, fake LLM).

Covers the pure logic: the LLM pass maps word indices -> ms spans, validation
clamps punch ⊆ core ⊆ thought + setup-before-core and drops junk, and the
deterministic L1 fallback derives thoughts from dialogue_segments (with the DB
query monkeypatched). Run:  .venv/bin/python scripts/test_thought_segments.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Make the module importable without a real .env (Settings has required fields).
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "x")
os.environ.setdefault("R2_ACCOUNT_ID", "x")
os.environ.setdefault("R2_ACCESS_KEY_ID", "x")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "x")

from app.services.l3 import thought_segments as ts  # noqa: E402
from app.services.llm.base import LLMResponse  # noqa: E402


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def model(self) -> str:
        return "fake"

    def run(self, *, system, messages, max_tokens=2048, **kw):
        return LLMResponse(
            text=self._text,
            assistant_message={"role": "assistant",
                               "content": [{"type": "text", "text": self._text}]},
        )


def _words():
    """11 words, all S0, 1s apart, 800ms long. Indices match the LLM bodies."""
    toks = ["we", "almost", "shut", "down", "last", "year",
            "but", "one", "customer", "changed", "everything"]
    return [{"text": t, "start_ms": i * 1000, "end_ms": i * 1000 + 800, "speaker": "S0"}
            for i, t in enumerate(toks)]


def test_llm_pass_maps_indices_to_ms():
    body = """{"thoughts":[
      {"speaker":"S0","thought":[0,5],"core":[0,3],"punch":[2,3],"setup":null,"strength":0.8},
      {"speaker":"S0","thought":[7,10],"core":[7,10],"punch":[9,10],"setup":[6,6],"strength":0.7}
    ]}"""
    out = ts.segment_with_llm(_words(), _FakeLLM(body))
    assert len(out) == 2, out
    t0 = out[0]
    assert (t0.thought.raw_in_ms, t0.thought.raw_out_ms) == (0, 5800), t0.thought
    assert (t0.core.raw_in_ms, t0.core.raw_out_ms) == (0, 3800), t0.core
    assert (t0.punch.raw_in_ms, t0.punch.raw_out_ms) == (2000, 3800), t0.punch
    assert t0.punch.text == "shut down", t0.punch.text
    assert t0.setup is None
    assert abs(t0.strength - 0.8) < 1e-6
    t1 = out[1]
    # setup sits BEFORE the thought (word 6 = "but"); thought proper is 7..10.
    assert t1.setup is not None and t1.setup.text == "but", t1.setup
    assert t1.thought.text == "one customer changed everything", t1.thought.text
    print("ok  test_llm_pass_maps_indices_to_ms")


def test_validation_clamps_and_drops():
    # punch sits OUTSIDE core -> reset to core; setup overshoots into the thought
    # and is clamped to before it; a 2-word thought is dropped; a non-dict and a
    # malformed range are skipped.
    body = """{"thoughts":[
      {"speaker":"S0","thought":[2,7],"core":[4,5],"punch":[2,2],"setup":[0,5],"strength":2.0},
      {"speaker":"S0","thought":[8,9]},
      "junk",
      {"speaker":"S0","thought":"nope"}
    ]}"""
    out = ts.segment_with_llm(_words(), _FakeLLM(body))
    assert len(out) == 1, out
    t = out[0]
    # punch was illegal (before core) -> collapses to core (4..5 -> 4000..5800)
    assert (t.punch.raw_in_ms, t.punch.raw_out_ms) == (4000, 5800), t.punch
    # setup [0,5] clamped to before the thought start (idx 2) -> [0,1]
    assert t.setup is not None and (t.setup.start_word, t.setup.end_word) == (0, 1), t.setup
    # strength clamped to [0,1]
    assert t.strength == 1.0, t.strength
    print("ok  test_validation_clamps_and_drops")


def test_empty_or_bad_json_returns_nothing():
    assert ts.segment_with_llm(_words(), _FakeLLM("not json")) == []
    assert ts.segment_with_llm(_words(), _FakeLLM('{"thoughts": "x"}')) == []
    print("ok  test_empty_or_bad_json_returns_nothing")


def test_fallback_from_dialogue(monkeypatch=None):
    dlg = {
        "topic": [{
            "seg_id": "topic-0", "speaker": "S0", "flags": [],
            "text": "we almost shut down last year but one customer changed everything",
            "raw_in_ms": 0, "raw_out_ms": 10000,
            "src_in_ms": 0, "src_out_ms": 10000,
            "child_seg_ids": ["sentence-0", "sentence-1"],
        }, {
            "seg_id": "topic-1", "speaker": "S0", "flags": ["production_cue"],
            "text": "action go", "raw_in_ms": 11000, "raw_out_ms": 12000,
            "src_in_ms": 11000, "src_out_ms": 12000, "child_seg_ids": [],
        }],
        "sentence": [
            {"seg_id": "sentence-0", "text": "we almost shut down last year",
             "raw_in_ms": 0, "raw_out_ms": 4000, "src_in_ms": 0, "src_out_ms": 4000},
            {"seg_id": "sentence-1", "text": "but one customer changed everything",
             "raw_in_ms": 5000, "raw_out_ms": 10000, "src_in_ms": 5000, "src_out_ms": 10000},
        ],
    }
    orig = ts._load_dialogue
    ts._load_dialogue = lambda fid: dlg
    try:
        out = ts.segment_fallback("fid")
    finally:
        ts._load_dialogue = orig

    assert len(out) == 1, out      # the production_cue topic is skipped
    t = out[0]
    assert (t.thought.raw_in_ms, t.thought.raw_out_ms) == (0, 10000), t.thought
    # core = last child sentence (the payoff lands last)
    assert (t.core.raw_in_ms, t.core.raw_out_ms) == (5000, 10000), t.core
    # the L1 fallback has no reliable run-up notion -> setup is null
    assert t.setup is None, t.setup
    print("ok  test_fallback_from_dialogue")


def _words_with_gap():
    """6 words in TWO runs split by a 3.2s dead gap: run A = words 0..2
    (t=0,1,2s), then silence, run B = words 3..5 (t=6,7,8s). 800ms each."""
    toks = ["ok", "so", "watch", "and", "it", "works"]
    starts = [0, 1000, 2000, 6000, 7000, 8000]
    return [{"text": t, "start_ms": s, "end_ms": s + 800, "speaker": "S0"}
            for t, s in zip(toks, starts)]


def test_silence_ceiling_clamps_thought_to_punch_run():
    # The model (mis)groups a thought ACROSS the dead gap [0..5] with the punch
    # in the SECOND run (words 4..5). The clamp must drop the pre-gap words and
    # keep only run B (words 3..5), so no level straddles the 3.2s silence.
    body = """{"thoughts":[
      {"speaker":"S0","thought":[0,5],"core":[3,5],"punch":[4,5],"setup":null,"strength":0.8}
    ]}"""
    out = ts.segment_with_llm(_words_with_gap(), _FakeLLM(body))
    assert len(out) == 1, out
    t = out[0]
    # thought clamped to run B start (word 3 @ 6000) .. word 5 end (8800).
    assert (t.thought.raw_in_ms, t.thought.raw_out_ms) == (6000, 8800), t.thought
    assert t.thought.text == "and it works", t.thought.text
    # every level sits inside the one pause-bounded run.
    assert t.core.raw_in_ms >= 6000 and t.punch.raw_in_ms >= 6000
    print("ok  test_silence_ceiling_clamps_thought_to_punch_run")


def test_no_long_gap_is_left_untouched():
    # A clean thought with only tiny inter-word gaps is not clamped at all.
    body = """{"thoughts":[
      {"speaker":"S0","thought":[0,5],"core":[0,3],"punch":[2,3],"setup":null,"strength":0.8}
    ]}"""
    out = ts.segment_with_llm(_words(), _FakeLLM(body))
    t = out[0]
    assert (t.thought.raw_in_ms, t.thought.raw_out_ms) == (0, 5800), t.thought
    print("ok  test_no_long_gap_is_left_untouched")


def test_render_words_emits_pause_markers():
    rendered = ts._render_words(_words_with_gap())
    assert "<pause 3.2s>" in rendered, rendered
    # the small 200ms gaps are below the mark threshold -> not surfaced.
    assert "<pause 0.2s>" not in rendered, rendered
    print("ok  test_render_words_emits_pause_markers")


def test_roundtrip_serialization():
    body = '{"thoughts":[{"speaker":"S0","thought":[2,5],"core":[2,3],"punch":[2,3],"setup":[0,1],"strength":0.5}]}'
    t = ts.segment_with_llm(_words(), _FakeLLM(body))[0]
    again = ts.Thought.from_dict(t.to_dict())
    assert again.to_dict() == t.to_dict()
    print("ok  test_roundtrip_serialization")


def main():
    test_llm_pass_maps_indices_to_ms()
    test_validation_clamps_and_drops()
    test_empty_or_bad_json_returns_nothing()
    test_fallback_from_dialogue()
    test_silence_ceiling_clamps_thought_to_punch_run()
    test_no_long_gap_is_left_untouched()
    test_render_words_emits_pause_markers()
    test_roundtrip_serialization()
    print("\nall thought-segment tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
