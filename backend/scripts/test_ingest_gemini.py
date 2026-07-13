"""
Tests for the Gemini Pass-2 backend (``app.services.llm.ingest_gemini`` and
``client.py``'s routing guard) -- gemini_pass2.plan.md. NO real API calls
(mocked SDK client throughout); NO real DB. Covers:

  - gemini_schema(): the structural sanitizer (prefixItems -> fixed-length
    array, anyOf-null -> nullable, key stripping) + the Pass2-shaped
    reinforcements (cuts required+minItems=1, label/summary minLength,
    channel/shot_size enums) -- asserted against the REAL Pass2BatchOutput
    schema (per the plan's own P6 instruction) and against the actual SDK's
    GenerateContentConfig to confirm it's accepted without raising.
  - complete_gemini(): success / one-re-ask / two-failures-raises, usage
    mapping, bare-list normalization.
  - The P4 cache manager: create/delete (success + graceful degradation),
    and the ContextVar handle propagation through ThreadPoolExecutor
    (submit_with_cache_context is REQUIRED for this -- a bare pool.submit
    silently loses it, see the module's own note).
  - client.py's routing guard: provider="gemini" routes to complete_gemini
    (never touches the Anthropic SDK); the default "anthropic" never
    imports ingest_gemini at all.

Run:  .venv/bin/python scripts/test_ingest_gemini.py
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.config import get_settings  # noqa: E402
from app.services.l3 import pass2  # noqa: E402
from app.services.llm import client as ic  # noqa: E402
from app.services.llm import ingest_gemini as ig  # noqa: E402
from app.services.llm.base import text_block  # noqa: E402
from test_ingest_client import FakeBlock, FakeClient, FakeResponse  # noqa: E402


def _types():
    from google.genai import types
    return types


# --------------------------------------------------------------------------
# gemini_schema
# --------------------------------------------------------------------------

def _find_key(node, key, path=""):
    hits = []
    if isinstance(node, dict):
        if key in node:
            hits.append(path)
        for k, v in node.items():
            hits += _find_key(v, key, path + "." + k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += _find_key(v, key, path + f"[{i}]")
    return hits


def test_gemini_schema_strips_prefixitems_and_unsupported_keys():
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    assert _find_key(sanitized, "prefixItems") == []
    assert _find_key(sanitized, "title") == []
    assert _find_key(sanitized, "default") == []
    assert _find_key(sanitized, "additionalProperties") == []
    print("ok  test_gemini_schema_strips_prefixitems_and_unsupported_keys")


def test_gemini_schema_tuple_becomes_fixed_length_number_array():
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    word_span = sanitized["$defs"]["CutJudgment"]["properties"]["word_span"]
    assert word_span["type"] == "array"
    assert word_span["items"] == {"type": "number"}
    assert word_span["minItems"] == word_span["maxItems"] == 2
    assert word_span["nullable"] is True

    framing = sanitized["$defs"]["Framing"]["properties"]
    subject_box = framing["subject_box"]
    assert subject_box["minItems"] == subject_box["maxItems"] == 4
    assert subject_box["nullable"] is True

    cj = sanitized["$defs"]["CutJudgment"]["properties"]
    caption_zones = cj["caption_zones"]  # non-optional list-of-tuples: no anyOf at all
    assert caption_zones["type"] == "array"
    assert caption_zones["items"]["minItems"] == caption_zones["items"]["maxItems"] == 4
    print("ok  test_gemini_schema_tuple_becomes_fixed_length_number_array")


def test_gemini_schema_forces_non_empty_cuts():
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    assert sanitized["required"] == ["cuts"]
    assert sanitized["properties"]["cuts"]["minItems"] == 1
    print("ok  test_gemini_schema_forces_non_empty_cuts")


def test_gemini_schema_label_summary_minlength_and_pydantic_required():
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    cj = sanitized["$defs"]["CutJudgment"]
    assert cj["properties"]["label"]["minLength"] == 1
    assert cj["properties"]["summary"]["minLength"] == 1
    assert "label" in cj["required"] and "summary" in cj["required"]
    print("ok  test_gemini_schema_label_summary_minlength_and_pydantic_required")


def test_gemini_schema_channel_and_shot_size_enums():
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    channel = sanitized["$defs"]["CutJudgment"]["properties"]["channel"]
    assert set(channel["enum"]) == {"said", "done", "shown"}
    shot_size = sanitized["$defs"]["Framing"]["properties"]["shot_size"]
    assert set(shot_size["enum"]) == set(pass2.SHOT_SIZES)
    print("ok  test_gemini_schema_channel_and_shot_size_enums")


def test_gemini_schema_shot_quality_gets_its_enum_and_stays_optional():
    # perception_upgrade.plan.md Part C2 / Flash-Lite guardrail: shot_quality
    # gets a closed enum like shot_size, but must NEVER end up in Framing's
    # `required` -- requiring a new field is what triggered the earlier
    # subject_box runaway-thinking failure.
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    framing = sanitized["$defs"]["Framing"]
    shot_quality = framing["properties"]["shot_quality"]
    assert set(shot_quality["enum"]) == set(pass2.SHOT_QUALITY)
    assert "shot_quality" not in (framing.get("required") or [])
    print("ok  test_gemini_schema_shot_quality_gets_its_enum_and_stays_optional")


def test_gemini_schema_screen_text_stays_an_optional_plain_string():
    # perception_upgrade.plan.md Part C3: screen_text needs no special-casing
    # in gemini_schema (a plain nullable string, generic sanitizer handles
    # it) -- assert it never grows an enum/minLength and never becomes
    # required, same guardrail as shot_quality.
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    cut_judgment = sanitized["$defs"]["CutJudgment"]
    screen_text = cut_judgment["properties"]["screen_text"]
    assert screen_text.get("type") == "string", screen_text
    assert "enum" not in screen_text, screen_text
    assert "minLength" not in screen_text, screen_text
    assert "screen_text" not in (cut_judgment.get("required") or [])
    print("ok  test_gemini_schema_screen_text_stays_an_optional_plain_string")


def test_gemini_schema_accepted_by_the_real_sdk_config():
    types = _types()
    sanitized = ig.gemini_schema(pass2.Pass2BatchOutput)
    cfg = types.GenerateContentConfig(
        system_instruction="test", max_output_tokens=100, temperature=0,
        response_mime_type="application/json", response_schema=sanitized,
    )
    assert cfg.response_schema == sanitized
    print("ok  test_gemini_schema_accepted_by_the_real_sdk_config")


def test_gemini_schema_is_a_noop_on_a_schema_without_the_pass2_shape():
    from typing import Optional, Tuple

    from pydantic import BaseModel

    class Unrelated(BaseModel):
        name: str
        span: Optional[Tuple[int, int]] = None

    sanitized = ig.gemini_schema(Unrelated)
    assert _find_key(sanitized, "prefixItems") == []
    assert "required" not in sanitized or "cuts" not in (sanitized.get("required") or [])
    print("ok  test_gemini_schema_is_a_noop_on_a_schema_without_the_pass2_shape")


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------

def test_resolve_thinking_budget():
    assert ig._resolve_thinking_budget("low") == 2048
    assert ig._resolve_thinking_budget("MEDIUM") == 8192
    assert ig._resolve_thinking_budget("high") == 24576
    assert ig._resolve_thinking_budget("4096") == 4096
    assert ig._resolve_thinking_budget(None) is None
    assert ig._resolve_thinking_budget("") is None
    assert ig._resolve_thinking_budget("bogus") == 2048
    print("ok  test_resolve_thinking_budget")


def test_parse_raw_normalizes_bare_list_and_rejects_garbage():
    class R:
        def __init__(self, text):
            self.text = text
    assert ig._parse_raw(R('{"cuts": [{"a": 1}]}')) == {"cuts": [{"a": 1}]}
    assert ig._parse_raw(R('[{"a": 1}]')) == {"cuts": [{"a": 1}]}
    assert ig._parse_raw(R("not json")) is None
    assert ig._parse_raw(R("")) is None
    print("ok  test_parse_raw_normalizes_bare_list_and_rejects_garbage")


def test_usage_of_maps_gemini_fields_to_the_shared_shape():
    class Usage:
        prompt_token_count = 100
        candidates_token_count = 50
        thoughts_token_count = 20
        cached_content_token_count = 30

    class R:
        usage_metadata = Usage()

    u = ig._usage_of(R())
    assert u == {"input_tokens": 100, "output_tokens": 70,
                "cache_read_input_tokens": 30, "cache_creation_input_tokens": 0}
    assert ig._usage_of(object()) == {}
    print("ok  test_usage_of_maps_gemini_fields_to_the_shared_shape")


# --------------------------------------------------------------------------
# complete_gemini -- mocked SDK client throughout, zero real API calls
# --------------------------------------------------------------------------

class _FakeUsageMD:
    prompt_token_count = 500
    candidates_token_count = 200
    thoughts_token_count = 10
    cached_content_token_count = 0


class _FakeGeminiResp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = _FakeUsageMD()
        self.candidates = []


_GOOD_CUT = ('{"cuts": [{"source_ref": "speech_cut[0]", "kind": "speech", '
            '"file_id": "f1", "label": "x", "summary": "y"}]}')
_EMPTY_CUTS = '{"cuts": []}'


def _reject_empty(output):
    return None if output.cuts else "cuts must not be empty"


def test_complete_gemini_success_first_try():
    calls = []

    def fake_generate_content(model, contents, config):
        calls.append({"model": model, "config": config})
        return _FakeGeminiResp(_GOOD_CUT)

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        result = ig.complete_gemini("sys", [text_block("hello")], pass2.Pass2BatchOutput, max_tokens=1000)

    assert result.attempts == 1
    assert result.data["cuts"][0]["source_ref"] == "speech_cut[0]"
    assert result.usage == {"input_tokens": 500, "output_tokens": 210,
                            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert len(calls) == 1
    print("ok  test_complete_gemini_success_first_try")


def test_complete_gemini_reasks_once_then_succeeds():
    responses = [_EMPTY_CUTS, _GOOD_CUT]
    calls = []

    def fake_generate_content(model, contents, config):
        calls.append(1)
        return _FakeGeminiResp(responses.pop(0))

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        result = ig.complete_gemini("sys", [text_block("hello")], pass2.Pass2BatchOutput,
                                    max_tokens=1000, extra_check=_reject_empty)

    assert result.attempts == 2
    assert len(result.data["cuts"]) == 1
    assert len(calls) == 2
    print("ok  test_complete_gemini_reasks_once_then_succeeds")


def test_complete_gemini_raises_ingest_failure_after_two_failures():
    def fake_generate_content(model, contents, config):
        return _FakeGeminiResp(_EMPTY_CUTS)

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        try:
            ig.complete_gemini("sys", [text_block("hello")], pass2.Pass2BatchOutput,
                               max_tokens=1000, extra_check=_reject_empty)
            assert False, "expected IngestFailure"
        except ig.IngestFailure as e:
            assert e.stage == "pass2"
    print("ok  test_complete_gemini_raises_ingest_failure_after_two_failures")


def test_complete_gemini_passes_response_schema_and_thinking_config():
    captured = {}

    def fake_generate_content(model, contents, config):
        captured["config"] = config
        return _FakeGeminiResp(_GOOD_CUT)

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        ig.complete_gemini("sys", [text_block("hello")], pass2.Pass2BatchOutput,
                           max_tokens=1000, thinking="medium")

    cfg = captured["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema == ig.gemini_schema(pass2.Pass2BatchOutput)
    assert cfg.temperature == 0
    assert cfg.thinking_config.thinking_budget == 8192
    assert cfg.system_instruction == "sys"
    print("ok  test_complete_gemini_passes_response_schema_and_thinking_config")


def test_complete_gemini_omits_system_instruction_when_cached():
    captured = {}

    def fake_generate_content(model, contents, config):
        captured["config"] = config
        return _FakeGeminiResp(_GOOD_CUT)

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        ig.complete_gemini("sys", [text_block("hello")], pass2.Pass2BatchOutput,
                           max_tokens=1000, cached_content="cachedContents/abc")

    cfg = captured["config"]
    assert cfg.cached_content == "cachedContents/abc"
    assert cfg.system_instruction is None
    print("ok  test_complete_gemini_omits_system_instruction_when_cached")


# --------------------------------------------------------------------------
# P4 cache manager
# --------------------------------------------------------------------------

class _FakeCache:
    name = "cachedContents/abc123"


def test_create_pass2_cache_returns_the_resource_name():
    fake_client = mock.Mock()
    fake_client.caches.create = mock.Mock(return_value=_FakeCache())
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        name = ig.create_pass2_cache("system", [text_block("stuff")])
    assert name == "cachedContents/abc123"
    print("ok  test_create_pass2_cache_returns_the_resource_name")


def test_create_pass2_cache_degrades_to_none_on_failure():
    fake_client = mock.Mock()
    fake_client.caches.create = mock.Mock(side_effect=RuntimeError("boom"))
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        name = ig.create_pass2_cache("system", [text_block("stuff")])
    assert name is None
    print("ok  test_create_pass2_cache_degrades_to_none_on_failure")


def test_delete_pass2_cache_calls_the_sdk_and_none_is_a_noop():
    fake_client = mock.Mock()
    fake_client.caches.delete = mock.Mock()
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        ig.delete_pass2_cache("cachedContents/abc123")
    fake_client.caches.delete.assert_called_once_with(name="cachedContents/abc123")

    fake_client2 = mock.Mock()
    with mock.patch.object(ig, "_sdk", return_value=(fake_client2, _types())):
        ig.delete_pass2_cache(None)
    fake_client2.caches.delete.assert_not_called()
    print("ok  test_delete_pass2_cache_calls_the_sdk_and_none_is_a_noop")


def test_pass2_cache_scope_propagates_through_threadpoolexecutor():
    assert ig.get_pass2_cache_handle() is None

    with ig.pass2_cache_scope("cachedContents/run-A"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [ig.submit_with_cache_context(pool, ig.get_pass2_cache_handle) for _ in range(8)]
            results = [f.result() for f in futures]
        assert all(r == "cachedContents/run-A" for r in results), results

    assert ig.get_pass2_cache_handle() is None
    print("ok  test_pass2_cache_scope_propagates_through_threadpoolexecutor")


def test_bare_pool_submit_does_not_see_the_cache_scope():
    # The documented gotcha this module works around: a plain pool.submit
    # (no submit_with_cache_context) must NOT see the handle -- confirms the
    # wrapper is actually doing something, not a no-op convenience alias.
    with ig.pass2_cache_scope("cachedContents/run-B"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            r = pool.submit(ig.get_pass2_cache_handle).result()
    assert r is None, r
    print("ok  test_bare_pool_submit_does_not_see_the_cache_scope")


def test_pass2_cache_scope_sequential_scopes_do_not_leak():
    with ig.pass2_cache_scope("run-1"):
        assert ig.get_pass2_cache_handle() == "run-1"
    with ig.pass2_cache_scope("run-2"):
        assert ig.get_pass2_cache_handle() == "run-2"
    assert ig.get_pass2_cache_handle() is None
    print("ok  test_pass2_cache_scope_sequential_scopes_do_not_leak")


# --------------------------------------------------------------------------
# client.py routing guard
# --------------------------------------------------------------------------

def _with_fake_anthropic_client(responses):
    fake = FakeClient(responses)
    orig = ic._sdk_client
    ic._sdk_client = lambda: fake
    return fake, orig


def test_complete_routes_pass2_to_gemini_when_provider_is_gemini():
    settings = get_settings()
    orig = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "gemini"
    called = {}
    try:
        def fake_complete_gemini(system, blocks, schema, **kw):
            called["hit"] = True
            from app.services.llm.client import Completion
            return Completion(data={"cuts": []}, usage={}, attempts=1)

        with mock.patch.object(ig, "complete_gemini", fake_complete_gemini):
            ic.complete("pass2", "sys", [text_block("x")], pass2.Pass2BatchOutput)
    finally:
        settings.ingest_pass2_provider = orig
    assert called.get("hit") is True
    print("ok  test_complete_routes_pass2_to_gemini_when_provider_is_gemini")


def test_complete_anthropic_provider_never_touches_gemini():
    # perception_upgrade.plan.md Part A flipped the DEFAULT to "gemini" (A/B
    # verified) -- this test explicitly selects "anthropic" to verify that
    # code path stays untouched/available, rather than relying on it being
    # the ambient default.
    settings = get_settings()
    orig_provider = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "anthropic"
    good = {"cuts": []}
    fake, orig = _with_fake_anthropic_client(
        [FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    called = {"hit": False}

    def fail_if_called(*a, **k):
        called["hit"] = True
        raise AssertionError("complete_gemini must not be called on the anthropic provider")

    try:
        with mock.patch.object(ig, "complete_gemini", fail_if_called):
            ic.complete("pass2", "sys", [text_block("x")], pass2.Pass2BatchOutput)
    finally:
        ic._sdk_client = orig
        settings.ingest_pass2_provider = orig_provider
    assert called["hit"] is False
    assert len(fake.messages.calls) == 1  # went through the real Anthropic path
    print("ok  test_complete_anthropic_provider_never_touches_gemini")


def test_complete_non_pass2_stage_ignores_the_gemini_flag():
    # The routing guard is scoped to stage=="pass2" only -- pass1 must
    # never route to Gemini even if the flag is (incorrectly) set.
    settings = get_settings()
    orig = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "gemini"
    fake, orig_client = _with_fake_anthropic_client(
        [FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, {"speech_cuts": []})])])
    called = {"hit": False}

    def fail_if_called(*a, **k):
        called["hit"] = True
        raise AssertionError("complete_gemini must not be called for pass1")

    try:
        from app.services.l3.pass1 import Pass1Output
        with mock.patch.object(ig, "complete_gemini", fail_if_called):
            ic.complete("pass1", "sys", [text_block("x")], Pass1Output)
    finally:
        settings.ingest_pass2_provider = orig
        ic._sdk_client = orig_client
    assert called["hit"] is False
    print("ok  test_complete_non_pass2_stage_ignores_the_gemini_flag")


# --------------------------------------------------------------------------
# pass2.run_pass2_batch -- gated reinforcement suffix
# --------------------------------------------------------------------------

def test_run_pass2_batch_anthropic_provider_system_prompt_unchanged():
    from app.services.l3.lattice import Lattice
    from app.services.l3.pass1 import Pass1Output, SpeechCut
    from app.services.l3.image_plan import PlannedFrame

    lat = Lattice(file_id="f1", duration_ms=10000, words=[], turns=[], hints=[], atoms=[])
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    p1 = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="intro")])

    captured = {}

    def fake_complete(stage, system, blocks, schema, **kw):
        from app.services.llm.client import Completion
        captured["system"] = system
        return Completion(data={"cuts": []}, usage={}, attempts=1)

    settings = get_settings()
    orig_provider = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "anthropic"
    try:
        with mock.patch.object(ic, "complete", fake_complete):
            pass2.run_pass2_batch([("f1", "a.mp4", 10000, lat)], p1, frames, {("f1", 100): "ZmFrZQ=="})
    finally:
        settings.ingest_pass2_provider = orig_provider

    assert captured["system"] == pass2._SYSTEM
    print("ok  test_run_pass2_batch_anthropic_provider_system_prompt_unchanged")


def test_run_pass2_batch_gemini_provider_appends_reinforcement():
    from app.services.l3.lattice import Lattice
    from app.services.l3.pass1 import Pass1Output, SpeechCut
    from app.services.l3.image_plan import PlannedFrame

    lat = Lattice(file_id="f1", duration_ms=10000, words=[], turns=[], hints=[], atoms=[])
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    p1 = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="intro")])

    captured = {}

    def fake_complete(stage, system, blocks, schema, **kw):
        from app.services.llm.client import Completion
        captured["system"] = system
        return Completion(data={"cuts": []}, usage={}, attempts=1)

    settings = get_settings()
    orig = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "gemini"
    try:
        with mock.patch.object(ic, "complete", fake_complete):
            pass2.run_pass2_batch([("f1", "a.mp4", 10000, lat)], p1, frames, {("f1", 100): "ZmFrZQ=="})
    finally:
        settings.ingest_pass2_provider = orig

    assert captured["system"] == pass2._SYSTEM + pass2._GEMINI_REINFORCE
    assert captured["system"] == pass2.gemini_system_prompt()
    print("ok  test_run_pass2_batch_gemini_provider_appends_reinforcement")


def test_run_pass2_batch_omits_stable_blocks_when_a_cache_handle_is_active():
    from app.services.l3.lattice import Lattice
    from app.services.l3.pass1 import Pass1Output, SpeechCut
    from app.services.l3.image_plan import PlannedFrame

    lat = Lattice(file_id="f1", duration_ms=10000, words=[], turns=[], hints=[], atoms=[])
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    p1 = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="intro")])

    captured = {}

    def fake_complete(stage, system, blocks, schema, **kw):
        from app.services.llm.client import Completion
        captured["blocks"] = blocks
        return Completion(data={"cuts": []}, usage={}, attempts=1)

    settings = get_settings()
    orig = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "gemini"
    try:
        with mock.patch.object(ic, "complete", fake_complete), \
             ig.pass2_cache_scope("cachedContents/active"):
            pass2.run_pass2_batch([("f1", "a.mp4", 10000, lat)], p1, frames, {("f1", 100): "ZmFrZQ=="})
    finally:
        settings.ingest_pass2_provider = orig

    # only the per-batch trimmed render should ride in `blocks` -- the
    # stable build_pass1_blocks prefix is baked into the CachedContent.
    assert len(captured["blocks"]) == 1
    assert "PASS 1 RESULT" in captured["blocks"][0]["text"]
    print("ok  test_run_pass2_batch_omits_stable_blocks_when_a_cache_handle_is_active")


def main():
    test_gemini_schema_strips_prefixitems_and_unsupported_keys()
    test_gemini_schema_tuple_becomes_fixed_length_number_array()
    test_gemini_schema_forces_non_empty_cuts()
    test_gemini_schema_label_summary_minlength_and_pydantic_required()
    test_gemini_schema_channel_and_shot_size_enums()
    test_gemini_schema_shot_quality_gets_its_enum_and_stays_optional()
    test_gemini_schema_screen_text_stays_an_optional_plain_string()
    test_gemini_schema_accepted_by_the_real_sdk_config()
    test_gemini_schema_is_a_noop_on_a_schema_without_the_pass2_shape()
    test_resolve_thinking_budget()
    test_parse_raw_normalizes_bare_list_and_rejects_garbage()
    test_usage_of_maps_gemini_fields_to_the_shared_shape()
    test_complete_gemini_success_first_try()
    test_complete_gemini_reasks_once_then_succeeds()
    test_complete_gemini_raises_ingest_failure_after_two_failures()
    test_complete_gemini_passes_response_schema_and_thinking_config()
    test_complete_gemini_omits_system_instruction_when_cached()
    test_create_pass2_cache_returns_the_resource_name()
    test_create_pass2_cache_degrades_to_none_on_failure()
    test_delete_pass2_cache_calls_the_sdk_and_none_is_a_noop()
    test_pass2_cache_scope_propagates_through_threadpoolexecutor()
    test_bare_pool_submit_does_not_see_the_cache_scope()
    test_pass2_cache_scope_sequential_scopes_do_not_leak()
    test_complete_routes_pass2_to_gemini_when_provider_is_gemini()
    test_complete_anthropic_provider_never_touches_gemini()
    test_complete_non_pass2_stage_ignores_the_gemini_flag()
    test_run_pass2_batch_anthropic_provider_system_prompt_unchanged()
    test_run_pass2_batch_gemini_provider_appends_reinforcement()
    test_run_pass2_batch_omits_stable_blocks_when_a_cache_handle_is_active()
    print("\nall ingest_gemini tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
