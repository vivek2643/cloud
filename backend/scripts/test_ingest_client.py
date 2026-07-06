"""
Tests for the cuts-v3 ingest LLM wrapper (``app.services.llm.client``) -- no
DB, NO REAL API CALLS. The Anthropic SDK client is fully mocked (a fake
``.messages.create`` returning synthetic tool_use responses), so this
exercises the schema-enforcement / re-ask / cache-breakpoint logic for
$0 -- never spend real tokens just to test the wrapper.

Run:  .venv/bin/python scripts/test_ingest_client.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from pydantic import BaseModel  # noqa: E402

from app.services.llm import client as ic  # noqa: E402
from app.services.llm.base import image_block, text_block  # noqa: E402


class Foo(BaseModel):
    name: str
    count: int


class Bar(BaseModel):
    items: list[str] = []


class FakeBlock:
    def __init__(self, type_, name=None, input=None, id="fake-tool-id"):
        self.type = type_
        self.name = name
        self.input = input
        self.id = id


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50,
                cache_read_input_tokens=0, cache_creation_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class FakeResponse:
    def __init__(self, content, usage=None, stop_reason="end_turn"):
        self.content = content
        self.usage = usage or FakeUsage()
        self.stop_reason = stop_reason


class FakeStreamManager:
    """Mimics anthropic's MessageStreamManager just enough for client.py's
    ``_call``: a context manager whose ``get_final_message()`` returns the
    already-fully-formed FakeResponse (streaming isn't simulated token-by-
    token -- only the shape client.py actually consumes)."""

    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def until_done(self):
        pass

    def get_final_message(self):
        return self._response


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStreamManager(self._responses.pop(0))


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def _with_fake_client(responses):
    fake = FakeClient(responses)
    orig = ic._sdk_client
    ic._sdk_client = lambda: fake
    return fake, orig


def test_complete_success_first_try():
    good = {"name": "x", "count": 3}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo)
    finally:
        ic._sdk_client = orig
    assert result.data == good, result
    assert result.attempts == 1, result
    print("ok  test_complete_success_first_try")


def test_complete_reasks_once_then_succeeds():
    bad = {"name": "x"}          # missing `count` -- fails Foo's schema
    good = {"name": "x", "count": 3}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad)]),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)]),
    ])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 2, result
    assert result.data == good, result
    assert len(fake.messages.calls) == 2
    print("ok  test_complete_reasks_once_then_succeeds")


def test_reask_follows_tool_use_with_a_matching_tool_result_not_bare_text():
    """Anthropic's API 400s a whole request if an assistant tool_use isn't
    immediately followed by a tool_result with the same id -- a bare text
    reply (what the re-ask used to send) is a protocol violation, not just a
    style choice, and only ever surfaces against the real API since a mock
    doesn't enforce it."""
    bad = {"name": "x"}
    good = {"name": "x", "count": 3}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad, id="toolu_abc")]),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good, id="toolu_def")]),
    ])
    try:
        ic.complete("pass1", "system", [text_block("hi")], Foo)
    finally:
        ic._sdk_client = orig
    second_call_messages = fake.messages.calls[1]["messages"]
    reask_turn = second_call_messages[-1]
    assert reask_turn["role"] == "user", reask_turn
    assert reask_turn["content"][0]["type"] == "tool_result", reask_turn
    assert reask_turn["content"][0]["tool_use_id"] == "toolu_abc", reask_turn
    print("ok  test_reask_follows_tool_use_with_a_matching_tool_result_not_bare_text")


def test_complete_raises_after_two_failures():
    bad = {"name": "x"}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad)]),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad)]),
    ])
    try:
        try:
            ic.complete("pass1", "system", [text_block("hi")], Foo)
            assert False, "expected IngestFailure"
        except ic.IngestFailure as e:
            assert e.stage == "pass1", e
    finally:
        ic._sdk_client = orig
    print("ok  test_complete_raises_after_two_failures")


def test_no_tool_call_is_a_schema_violation_not_a_crash():
    """The model responding with plain text (no tool call) is treated exactly
    like any other schema violation -- re-ask once, then fail loud."""
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("text")]),
        FakeResponse([FakeBlock("text")]),
    ])
    try:
        try:
            ic.complete("pass1", "system", [text_block("hi")], Foo)
            assert False, "expected IngestFailure"
        except ic.IngestFailure:
            pass
    finally:
        ic._sdk_client = orig
    print("ok  test_no_tool_call_is_a_schema_violation_not_a_crash")


def test_answer_wrapped_under_a_spurious_single_key_is_unwrapped():
    """Observed twice against the real API: the model nests its whole real
    answer one level deeper than the schema wants, under a single spurious
    key ("$PARAMETER_NAME", or the schema's own class name). The outer shape
    fails validation; the inner dict is what the model actually meant."""
    good_inner = {"name": "x", "count": 3}
    wrapped = {"$PARAMETER_NAME": good_inner}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, wrapped)])])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 1, result
    assert result.data == good_inner, result
    print("ok  test_answer_wrapped_under_a_spurious_single_key_is_unwrapped")


def test_unwrap_is_not_attempted_when_it_would_not_help():
    # A single-key wrapper whose inner value ALSO doesn't validate must
    # still report the ORIGINAL error and go through the normal re-ask path,
    # not silently swallow a genuinely broken response.
    bad_inner = {"name": "x"}   # still missing `count`
    wrapped = {"WrapperKey": bad_inner}
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, wrapped)]),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)]),
    ])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 2, result
    assert result.data == good, result
    print("ok  test_unwrap_is_not_attempted_when_it_would_not_help")


def test_stringified_list_field_is_parsed_on_the_first_attempt():
    """Observed on nearly every real pass-2 call: a field the schema wants
    as a native list comes back as a JSON-encoded STRING instead ("cuts":
    "[...]"). This should be recovered WITHOUT a re-ask -- re-asking means
    the model regenerates the whole large payload again, which has its own
    real chance of dropping a different required field the second time."""
    stringified = {"items": '["a", "b", "c"]'}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, stringified)])])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Bar)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 1, result
    assert result.data == {"items": ["a", "b", "c"]}, result
    print("ok  test_stringified_list_field_is_parsed_on_the_first_attempt")


def test_double_wrapped_stringified_field_unwraps_the_matching_key():
    # Observed in the wild: "cuts": "{\"cuts\": [...]}" -- the string itself
    # re-wraps under the SAME key name, one level too deep.
    double_wrapped = {"items": '{"items": ["a", "b"]}'}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, double_wrapped)])])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Bar)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 1, result
    assert result.data == {"items": ["a", "b"]}, result
    print("ok  test_double_wrapped_stringified_field_unwraps_the_matching_key")


def test_stringified_field_with_a_raw_control_char_still_parses_leniently():
    # A literal (unescaped) newline embedded INSIDE a JSON string value --
    # strict json.loads rejects raw control characters there; strict=False
    # permits them. Plausible when the model hand-stringifies a pretty-
    # printed multi-line blob as a field value instead of a native list.
    raw_with_control_char = '["a\nb"]'
    stringified = {"items": raw_with_control_char}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, stringified)])])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Bar)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 1, result
    assert result.data == {"items": ["a\nb"]}, result
    print("ok  test_stringified_field_with_a_raw_control_char_still_parses_leniently")


def test_unparseable_string_field_falls_through_to_the_normal_reask():
    bad = {"items": "not json at all"}
    good = {"items": ["a"]}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad)]),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)]),
    ])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Bar)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 2, result
    assert result.data == good, result
    print("ok  test_unparseable_string_field_falls_through_to_the_normal_reask")


def test_unknown_stage_raises_before_any_call():
    try:
        ic.complete("pass3", "sys", [text_block("hi")], Foo)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_unknown_stage_raises_before_any_call")


def test_cache_breakpoint_on_last_block_and_system():
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        ic.complete("pass1", "sys", [text_block("a"), text_block("b")], Foo)
    finally:
        ic._sdk_client = orig
    call = fake.messages.calls[0]
    content = call["messages"][0]["content"]
    assert content[-1]["cache_control"] == {"type": "ephemeral"}, content
    assert "cache_control" not in content[0], content
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}, call["system"]
    print("ok  test_cache_breakpoint_on_last_block_and_system")


def test_no_cache_omits_breakpoints():
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        ic.complete("pass1", "sys", [text_block("a")], Foo, cache=False)
    finally:
        ic._sdk_client = orig
    call = fake.messages.calls[0]
    assert "cache_control" not in call["messages"][0]["content"][0]
    assert call["system"] == "sys", call["system"]
    print("ok  test_no_cache_omits_breakpoints")


def test_extra_blocks_appended_after_cached_prefix_uncached():
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        ic.complete("pass1", "sys", [text_block("prefix")], Foo,
                   extra_blocks=[image_block("ZmFrZQ==", "image/jpeg")])
    finally:
        ic._sdk_client = orig
    content = fake.messages.calls[0]["messages"][0]["content"]
    assert len(content) == 2, content
    assert content[0]["cache_control"] == {"type": "ephemeral"}, content[0]
    assert "cache_control" not in content[1], content[1]
    assert content[1]["type"] == "image", content[1]
    print("ok  test_extra_blocks_appended_after_cached_prefix_uncached")


def test_usage_sums_across_the_re_ask():
    bad = {"name": "x"}
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad)],
                    usage=FakeUsage(input_tokens=100, output_tokens=20)),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)],
                    usage=FakeUsage(input_tokens=110, output_tokens=15)),
    ])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo)
    finally:
        ic._sdk_client = orig
    assert result.usage["input_tokens"] == 210, result.usage
    assert result.usage["output_tokens"] == 35, result.usage
    print("ok  test_usage_sums_across_the_re_ask")


def test_truncated_response_is_not_accepted_even_though_schema_valid():
    """A response cut off by max_tokens mid tool-call still comes back from
    the SDK as a syntactically-valid (here: empty) tool input -- {} passes
    Foo's schema-shaped validation trivially if every field had a default,
    but truncation must be caught before that, or it's indistinguishable
    from a legitimate empty answer."""
    good = {"name": "x", "count": 3}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, {})], stop_reason="max_tokens"),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)], stop_reason="end_turn"),
    ])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo, max_tokens=100)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 2, result
    assert result.data == good, result
    print("ok  test_truncated_response_is_not_accepted_even_though_schema_valid")


def test_truncated_response_reasks_with_a_bigger_token_budget():
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, {})], stop_reason="max_tokens"),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)], stop_reason="end_turn"),
    ])
    try:
        ic.complete("pass1", "system", [text_block("hi")], Foo, max_tokens=100)
    finally:
        ic._sdk_client = orig
    assert fake.messages.calls[0]["max_tokens"] == 100
    assert fake.messages.calls[1]["max_tokens"] == 200
    print("ok  test_truncated_response_reasks_with_a_bigger_token_budget")


def test_truncated_on_both_attempts_raises_ingest_failure():
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, {})], stop_reason="max_tokens"),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, {})], stop_reason="max_tokens"),
    ])
    try:
        try:
            ic.complete("pass1", "system", [text_block("hi")], Foo, max_tokens=100)
            assert False, "expected IngestFailure"
        except ic.IngestFailure as e:
            assert "truncated" in e.reason
    finally:
        ic._sdk_client = orig
    print("ok  test_truncated_on_both_attempts_raises_ingest_failure")


def test_extra_check_rejects_a_schema_valid_but_semantically_bad_response():
    """extra_check catches invariants pydantic's type system can't express
    (e.g. cross-object uniqueness) -- a schema-valid response that fails it
    is treated exactly like a schema violation: one re-ask, fed the check's
    error text."""
    bad = {"name": "dup", "count": 1}
    good = {"name": "ok", "count": 2}
    fake, orig = _with_fake_client([
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, bad)]),
        FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)]),
    ])

    def reject_dup(parsed):
        return "name must not be 'dup'" if parsed.name == "dup" else None

    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo, extra_check=reject_dup)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 2, result
    assert result.data == good, result
    reask_content = fake.messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert "must not be 'dup'" in reask_content, reask_content
    print("ok  test_extra_check_rejects_a_schema_valid_but_semantically_bad_response")


def test_extra_check_passes_a_genuinely_good_response_straight_through():
    good = {"name": "ok", "count": 2}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        result = ic.complete("pass1", "system", [text_block("hi")], Foo, extra_check=lambda p: None)
    finally:
        ic._sdk_client = orig
    assert result.attempts == 1, result
    assert result.data == good, result
    print("ok  test_extra_check_passes_a_genuinely_good_response_straight_through")


def test_tool_choice_forces_the_schema_tool():
    good = {"name": "x", "count": 1}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    try:
        ic.complete("pass1", "sys", [text_block("a")], Foo)
    finally:
        ic._sdk_client = orig
    call = fake.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": ic._TOOL_NAME}, call["tool_choice"]
    assert call["tools"][0]["name"] == ic._TOOL_NAME
    assert call["tools"][0]["input_schema"] == Foo.model_json_schema()
    print("ok  test_tool_choice_forces_the_schema_tool")


def main():
    test_complete_success_first_try()
    test_complete_reasks_once_then_succeeds()
    test_reask_follows_tool_use_with_a_matching_tool_result_not_bare_text()
    test_complete_raises_after_two_failures()
    test_no_tool_call_is_a_schema_violation_not_a_crash()
    test_answer_wrapped_under_a_spurious_single_key_is_unwrapped()
    test_unwrap_is_not_attempted_when_it_would_not_help()
    test_extra_check_rejects_a_schema_valid_but_semantically_bad_response()
    test_extra_check_passes_a_genuinely_good_response_straight_through()
    test_stringified_list_field_is_parsed_on_the_first_attempt()
    test_double_wrapped_stringified_field_unwraps_the_matching_key()
    test_stringified_field_with_a_raw_control_char_still_parses_leniently()
    test_unparseable_string_field_falls_through_to_the_normal_reask()
    test_unknown_stage_raises_before_any_call()
    test_cache_breakpoint_on_last_block_and_system()
    test_no_cache_omits_breakpoints()
    test_extra_blocks_appended_after_cached_prefix_uncached()
    test_usage_sums_across_the_re_ask()
    test_truncated_response_is_not_accepted_even_though_schema_valid()
    test_truncated_response_reasks_with_a_bigger_token_budget()
    test_truncated_on_both_attempts_raises_ingest_failure()
    test_tool_choice_forces_the_schema_tool()
    print("\nall ingest-client tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
