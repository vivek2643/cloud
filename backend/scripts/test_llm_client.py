#!/usr/bin/env python3
"""Tests for app.services.llm.anthropic_client -- extended-thinking block
preservation (brain_extended_thinking.plan.md section 3.1/3.2/4.1). No
network: client.messages.create is stubbed with fake SDK-shaped objects.

Scope note: STEP ZERO's live probe (claude-opus-4-8, anthropic==0.49.0)
found the model reasons equivalently with or without a thinking parameter,
and the one shape this SDK version can send (`thinking={"type":"adaptive"}`
via a raw dict, plus `output_config` via extra_body) returns a signed
thinking block whose own `thinking` text comes back empty -- no
observability gained. Per that finding, only the PRESERVE + CAPTURE half of
the plan (3.1/3.2) is implemented; the budget-forwarding half (3.3/3.4) is
deliberately NOT -- `thinking_budget` stays accepted-and-discarded. See
`brain_extended_thinking.plan.md`'s own Results section for the full record.

Revert signature (plan section 4.3): if _block_to_anthropic's thinking case
(D1) or run()'s capture branch (D2) is reverted, the round-trip test below
fails immediately -- a real edit thread would then 400 on its second API
call inside any tool-use cycle that happens to include a thinking block,
never on the first.

Run:  .venv/bin/python scripts/test_llm_client.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.llm import anthropic_client as ac  # noqa: E402


# --------------------------------------------------------------------------
# Fake SDK completion plumbing -- mirrors just enough of the anthropic
# package's response shape (block objects with .type/.text/.thinking/
# .signature/.data/.id/.name/.input, plus .usage) for run() to walk.
# --------------------------------------------------------------------------

def _fake_usage(input_tokens=10, output_tokens=20):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0)


def _fake_completion(content, usage=None):
    return SimpleNamespace(content=content, usage=usage or _fake_usage())


def _thinking_block(thinking="reasoning...", signature="sig-abc123"):
    return SimpleNamespace(type="thinking", thinking=thinking, signature=signature)


def _redacted_thinking_block(data="opaque-bytes"):
    return SimpleNamespace(type="redacted_thinking", data=data)


def _text_block(text="hello"):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id_="t1", name="place", input_=None):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_ or {"ref": "x"})


def _stub_client(completion):
    """Patch ac._sdk_client so run() never touches the network."""
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=mock.Mock(return_value=completion)))
    return mock.patch.object(ac, "_sdk_client", return_value=fake_client)


# --------------------------------------------------------------------------
# 3.1 -- _block_to_anthropic: thinking/redacted_thinking copied whole
# --------------------------------------------------------------------------

def test_block_to_anthropic_thinking_returned_unchanged_with_signature():
    b = {"type": "thinking", "thinking": "step by step...", "signature": "sig-xyz"}
    out = ac._block_to_anthropic(b)
    assert out == b, out
    assert out is not b, "must copy, not return the same object"
    print("ok  _block_to_anthropic: a thinking block round-trips unchanged, signature intact")


def test_block_to_anthropic_redacted_thinking_returned_unchanged():
    b = {"type": "redacted_thinking", "data": "opaque"}
    out = ac._block_to_anthropic(b)
    assert out == b, out
    print("ok  _block_to_anthropic: a redacted_thinking block round-trips unchanged")


def test_block_to_anthropic_preserves_an_unknown_future_field():
    """D1's own revert signature: naming fields instead of copying the dict
    would silently drop anything the API adds later."""
    b = {"type": "thinking", "thinking": "...", "signature": "sig-1", "xyz": 1}
    out = ac._block_to_anthropic(b)
    assert out == b, out
    assert out["xyz"] == 1, out
    print("ok  _block_to_anthropic: an unknown future field on a thinking block survives")


# --------------------------------------------------------------------------
# 3.2 -- run(): capturing thinking/redacted_thinking off completion.content
# --------------------------------------------------------------------------

def test_run_captures_thinking_block_first_with_signature_preserved():
    completion = _fake_completion([
        _thinking_block(thinking="let me work through this", signature="sig-123"),
        _text_block("done"),
        _tool_use_block(),
    ])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}])
    content = resp.assistant_message["content"]
    assert content[0] == {"type": "thinking", "thinking": "let me work through this",
                          "signature": "sig-123"}, content
    types = [c["type"] for c in content]
    assert types == ["thinking", "text", "tool_use"], types
    print("ok  run(): a thinking block is captured FIRST, signature preserved")


def test_run_captures_redacted_thinking_block_with_data_not_thinking_field():
    completion = _fake_completion([_redacted_thinking_block(data="opaque-blob"), _text_block("hi")])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}])
    blk = resp.assistant_message["content"][0]
    assert blk == {"type": "redacted_thinking", "data": "opaque-blob"}, blk
    assert "thinking" not in blk and "signature" not in blk, blk
    print("ok  run(): a redacted_thinking block captures `data`, not `thinking`/`signature`")


def test_run_omits_thinking_fields_that_are_none():
    """A thinking-typed block with no signature (shouldn't happen in practice,
    but the field-presence guard must not write a literal None into the
    persisted/round-tripped shape)."""
    blk = SimpleNamespace(type="thinking", thinking="partial", signature=None)
    completion = _fake_completion([blk])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}])
    out = resp.assistant_message["content"][0]
    assert out == {"type": "thinking", "thinking": "partial"}, out
    assert "signature" not in out, out
    print("ok  run(): a None field on a thinking block is omitted, not written as null")


def test_run_thinking_block_does_not_appear_in_text_or_tool_calls():
    """A thinking block must not leak into resp.text or resp.tool_calls --
    those stay exactly what text/tool_use blocks produce."""
    completion = _fake_completion([
        _thinking_block(), _text_block("the actual reply"), _tool_use_block(name="wrap_up"),
    ])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text == "the actual reply", resp.text
    assert [tc.name for tc in resp.tool_calls] == ["wrap_up"], resp.tool_calls
    print("ok  run(): a thinking block never leaks into .text or .tool_calls")


# --------------------------------------------------------------------------
# 4.1 -- the round-trip test that actually protects the signature
# --------------------------------------------------------------------------

def test_round_trip_thinking_block_is_byte_identical_after_messages_to_anthropic():
    """The test D1 and D2 both fail: run()'s own assistant_message fed back
    through _messages_to_anthropic (as the loop does on the next turn via
    convo.append(resp.assistant_message) + the next run() call) must produce
    a thinking block byte-identical to what the SDK returned -- not
    re-derived, not missing, not silently downgraded to an empty text block."""
    completion = _fake_completion([
        _thinking_block(thinking="the reasoning", signature="sig-round-trip"),
        _tool_use_block(),
    ])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}],
                                        tools=[{"name": "place", "input_schema": {}}])
    convo = [{"role": "user", "content": "hi"}, resp.assistant_message]
    rebuilt = ac._messages_to_anthropic(convo)
    rebuilt_assistant_content = rebuilt[1]["content"]
    original_thinking_block = resp.assistant_message["content"][0]
    assert rebuilt_assistant_content[0] == original_thinking_block, \
        (rebuilt_assistant_content[0], original_thinking_block)
    assert rebuilt_assistant_content[0]["signature"] == "sig-round-trip", rebuilt_assistant_content[0]
    print("ok  round-trip: run()'s thinking block survives _messages_to_anthropic byte-identical")


def test_round_trip_fails_loud_proof_d2_would_have_broken_this():
    """Documents WHY the above test is the guard: simulating D2's bug (the
    old fallback `return {"type": "text", "text": ""}`) on the same captured
    block shows the corruption the fix prevents."""
    completion = _fake_completion([_thinking_block(thinking="x", signature="sig-9")])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}])
    captured = resp.assistant_message["content"][0]

    def _d2_buggy_block_to_anthropic(b):
        btype = b.get("type")
        if btype == "text":
            return {"type": "text", "text": b.get("text", "")}
        return {"type": "text", "text": ""}   # D2: no thinking case -> corrupted

    corrupted = _d2_buggy_block_to_anthropic(captured)
    assert corrupted != captured, "the buggy path should NOT preserve the block -- proving the guard matters"
    assert corrupted == {"type": "text", "text": ""}, corrupted
    print("ok  round-trip guard: D2's old fallback would have silently emptied the thinking block")


# --------------------------------------------------------------------------
# thinking_budget / effort stay accepted-and-discarded (deliberate, per
# STEP ZERO's finding -- not implementing 3.3/3.4)
# --------------------------------------------------------------------------

def test_thinking_budget_is_accepted_but_never_forwarded():
    completion = _fake_completion([_text_block("ok")])
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=mock.Mock(return_value=completion)))
    with mock.patch.object(ac, "_sdk_client", return_value=fake_client):
        ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}],
                                 thinking_budget=4096, effort="high")
    sent_kwargs = fake_client.messages.create.call_args.kwargs
    assert "thinking" not in sent_kwargs, sent_kwargs
    assert "effort" not in sent_kwargs, sent_kwargs
    print("ok  run(): thinking_budget/effort are accepted but never sent (deliberate, per STEP ZERO)")


def test_run_with_no_thinking_budget_arg_still_works():
    """Ordinary call shape (no thinking_budget/effort at all) is unaffected --
    the common, everyday call path this whole client exists for."""
    completion = _fake_completion([_text_block("fine"), _tool_use_block()])
    with _stub_client(completion):
        resp = ac.AnthropicClient().run(system="s", messages=[{"role": "user", "content": "hi"}],
                                        max_tokens=1024, cache_system=True)
    assert resp.text == "fine", resp.text
    assert resp.stop_reason == "tool_use", resp.stop_reason
    print("ok  run(): the ordinary call path (no thinking args) is unchanged")


def main():
    tests = [
        test_block_to_anthropic_thinking_returned_unchanged_with_signature,
        test_block_to_anthropic_redacted_thinking_returned_unchanged,
        test_block_to_anthropic_preserves_an_unknown_future_field,
        test_run_captures_thinking_block_first_with_signature_preserved,
        test_run_captures_redacted_thinking_block_with_data_not_thinking_field,
        test_run_omits_thinking_fields_that_are_none,
        test_run_thinking_block_does_not_appear_in_text_or_tool_calls,
        test_round_trip_thinking_block_is_byte_identical_after_messages_to_anthropic,
        test_round_trip_fails_loud_proof_d2_would_have_broken_this,
        test_thinking_budget_is_accepted_but_never_forwarded,
        test_run_with_no_thinking_budget_arg_still_works,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} llm-client tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
