#!/usr/bin/env python3
"""Tests for per-step REASONING capture in the agentic edit tool-loop
(tools.run_edit_loop) with a scripted fake LLM -- no network, no DB.

Verifies the brain's intermediate natural-language reasoning (and extended-
thinking blocks, if a provider ever returns them) is recorded, in loop order,
into the same ``trace`` list that is persisted onto ``edit_turns.trace`` today
(store.append_turn(trace=result.trace)) -- so a future diagnosis reading a
thread's turns gets the WHY, not just the tool calls, for free.

Run:  PYTHONPATH=. python scripts/test_reasoning_capture.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing test_tools_loop applies its no-DB stub (fm._sentences_for_file) and
# gives us the shared fixture builders (_struct/_ctx/_seed_doc) -- same style,
# no duplication.
from test_tools_loop import _ctx, _seed_doc, _struct  # noqa: E402

from app.services.l3 import tools  # noqa: E402
from app.services.llm import LLMResponse, ToolCall  # noqa: E402


class _ReasoningLLM:
    """Like test_tools_loop._ScriptedLLM, but each tool-call step can also carry
    the assistant's prose TEXT and an optional THINKING block emitted alongside
    the tool_use blocks -- exactly what a real provider returns and what the
    capture must record. A script step is one of:
      - str                          -> a final prose turn (no tools)
      - (text, [ToolCall])           -> prose + tool calls
      - (text, [ToolCall], thinking) -> prose + thinking + tool calls
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    @property
    def model(self):
        return "scripted"

    def run(self, *, system, messages, tools=None, max_tokens=2048, cache_system=False):
        step = self.script[self.calls] if self.calls < len(self.script) else "Done."
        self.calls += 1
        if isinstance(step, str):
            return LLMResponse(text=step, tool_calls=[], stop_reason="end_turn",
                               assistant_message={"role": "assistant",
                                                  "content": [{"type": "text", "text": step}]})
        text, tcs = step[0], step[1]
        thinking = step[2] if len(step) > 2 else ""
        content = []
        if thinking:
            content.append({"type": "thinking", "thinking": thinking})
        if text:
            content.append({"type": "text", "text": text})
        content += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                    for tc in tcs]
        return LLMResponse(text=text, tool_calls=tcs, stop_reason="tool_use",
                           assistant_message={"role": "assistant", "content": content})


def _reasoning(trace):
    return [e for e in trace if e.get("kind") == "reasoning"]


def test_loop_records_per_step_reasoning_in_trace_in_order():
    """Every loop step's prose is captured as a kind="reasoning" trace entry,
    in true loop order, INCLUDING steps with no tool call (the finish attempts).
    A reasoning entry sits right before that step's tool entries."""
    ctx = _ctx(_struct())
    script = [
        ("First I'll read the state to see what's already placed.",
         [ToolCall(id="t1", name="read_state", input={})]),
        ("Now I'll place the strongest opening beat.",
         [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})]),
        "Placed the beat -- reads clean.",       # finish attempt (no tool) -> Stage 2 craft fires
        "Reads clean -- about 4 seconds.",       # blind verdict (no tool) -> finish
    ]
    llm = _ReasoningLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "cut the opener"}],
                              ctx=ctx, document=_seed_doc())

    reasons = [e["reasoning"] for e in _reasoning(res.trace)]
    # all four steps' prose captured, in order -- including the two no-tool steps
    assert reasons == [
        "First I'll read the state to see what's already placed.",
        "Now I'll place the strongest opening beat.",
        "Placed the beat -- reads clean.",
        "Reads clean -- about 4 seconds.",
    ], reasons

    # ordering vs actions: the read_state reasoning precedes the read_state tool
    # entry; the place reasoning precedes the place tool entry.
    kinds_names = [(e["kind"], e.get("name")) for e in res.trace]
    assert kinds_names.index(("reasoning", None)) < kinds_names.index(("tool", "read_state")), res.trace
    r_place = next(i for i, e in enumerate(res.trace)
                   if e["kind"] == "reasoning" and e["reasoning"].startswith("Now I'll place"))
    t_place = next(i for i, e in enumerate(res.trace)
                   if e["kind"] == "tool" and e.get("name") == "place")
    assert r_place < t_place, res.trace

    # tool entries still carry their existing shape (now tagged kind="tool")
    tool_entries = [e for e in res.trace if e.get("kind") == "tool"]
    assert [e["name"] for e in tool_entries] == ["read_state", "place"], tool_entries
    assert all({"turn", "name", "args", "applied", "result"} <= set(e) for e in tool_entries)
    print("ok  reasoning: every step's prose is captured in trace, in loop order (incl. no-tool steps)")


def test_loop_captures_thinking_blocks_when_present():
    """When a provider returns extended-thinking blocks alongside the text, the
    capture records them on the same reasoning entry as ``thinking`` (best-effort;
    the live Anthropic edit-loop client doesn't emit these today, but the capture
    is ready for when thinking is turned on)."""
    ctx = _ctx(_struct())
    script = [
        ("Placing the opener.",
         [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
         "The first beat has the strongest hook, so it leads."),
        "Done -- one beat on V1.",
        "Reads clean.",
    ]
    llm = _ReasoningLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "place the opener"}],
                              ctx=ctx, document=_seed_doc())
    first = _reasoning(res.trace)[0]
    assert first["reasoning"] == "Placing the opener.", first
    assert first["thinking"] == "The first beat has the strongest hook, so it leads.", first
    # a step with no thinking omits the key entirely (not an empty string)
    assert all("thinking" not in e for e in _reasoning(res.trace)[1:]), res.trace
    print("ok  reasoning: extended-thinking blocks are captured onto the reasoning entry when present")


def test_reasoning_capture_is_fail_open():
    """A capture error must never break the turn: even if reasoning extraction
    raises, the edit still applies and the reply still returns."""
    ctx = _ctx(_struct())
    script = [
        ("Placing it.", [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})]),
        "Placed.",
        # brain_accountability_architecture.plan.md PART 1: the ONE finalizer
        # now forces a wrap_up on any clean finish with an empty surface.
        ("", [ToolCall(id="t2", name="wrap_up", input={"summary": "Reads clean."})]),
        "Reads clean.",
    ]
    llm = _ReasoningLLM(script)
    orig = tools._reasoning_from_response
    tools._reasoning_from_response = lambda resp: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        res = tools.run_edit_loop(llm, system="sys",
                                  messages=[{"role": "user", "content": "place it"}],
                                  ctx=ctx, document=_seed_doc())
    finally:
        tools._reasoning_from_response = orig
    assert res.changed is True, res            # edit still applied despite capture blowing up
    assert res.reply == "Reads clean.", res.reply
    assert _reasoning(res.trace) == [], res.trace  # nothing captured, but the turn survived
    assert [e["name"] for e in res.trace if e.get("kind") == "tool"] == ["place", "wrap_up"], res.trace
    print("ok  reasoning: capture is fail-open -- a capture error leaves the edit + reply intact")


def main():
    test_loop_records_per_step_reasoning_in_trace_in_order()
    test_loop_captures_thinking_blocks_when_present()
    test_reasoning_capture_is_fail_open()
    print("\nall reasoning-capture tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
