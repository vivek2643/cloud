"""
Gemini structured backend for Cuts v3 Pass 2 (gemini_pass2.plan.md), gated
behind ``ingest_pass2_provider="gemini"``. Sibling to ``client.py`` -- mirrors
``complete()``'s contract (``Completion``, one re-ask, ``IngestFailure``) but
talks the google-genai SDK's native structured-output path instead of
Claude's forced tool-use.

Why a straight swap doesn't work (measured against the real API): Gemini's
schema converter rejects pydantic's ``Tuple[...]`` fields outright
(``prefixItems``), and an unconstrained schema lets the model satisfy
``cuts: []`` trivially instead of engaging with the batch -- there is no
Claude-style forced tool_choice to lean on. So this module has to (1) rewrite
the schema into a Gemini-shaped one, (2) force engagement (non-empty output,
required fields), (3) run the call with a real thinking budget, before cost
optimization (context caching, see ``_cache`` below) is even worth measuring.

Never imports anything from ``app.services.l3`` -- like ``client.py``, this
stays a generic provider layer; the small Pass2-shaped reinforcements in
``gemini_schema`` (the ``cuts`` field, ``label``/``summary``, the
``shot_size``/``channel`` enums) are written defensively (no-ops if the
shape isn't present) rather than importing pass2.py's constants, so this
module has zero dependency on the ingest layer that calls it.

Does NOT modify ``client.py``'s Anthropic path or ``pass2.py``'s pydantic
models -- ``main``'s default behavior (``ingest_pass2_provider="anthropic"``)
never imports this module at all (see ``client.py``'s routing guard).
"""
from __future__ import annotations

import contextvars
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from app.config import get_settings
from app.services.llm.base import Block
from app.services.llm.client import Completion, IngestFailure, _validate
from app.services.llm.gemini_client import _parts_for_content, _sdk

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Mirrors pass2.SHOT_SIZES / the "said"|"done"|"shown" channel vocabulary --
# duplicated here (not imported) to keep this module dependency-free from
# app.services.l3; both are closed, rarely-changing category lists. Only
# applied when the schema actually has these fields (defensive, not fatal
# if pass2.py's fields ever rename).
_SHOT_SIZE_ENUM = (
    "extreme_close_up", "close_up", "medium_close_up", "medium",
    "medium_wide", "wide", "extreme_wide", "unsure",
)
_CHANNEL_ENUM = ("said", "done", "shown")
# Mirrors pass2.SHOT_QUALITY (perception_upgrade.plan.md Part C2) -- OPTIONAL
# only (never added to `required`; see the Flash-Lite guardrail note below).
_SHOT_QUALITY_ENUM = (
    "stable", "shaky", "whip", "soft_focus", "racking_focus", "exposure_shift", "unsure",
)

# Keys Gemini's response_schema converter rejects outright (probed against
# the real API). "format" is stripped unconditionally -- no field in this
# ingest schema family uses one Gemini supports (e.g. date-time); revisit if
# one is ever added.
_STRIP_KEYS = {"title", "default", "additionalProperties", "$schema", "const", "format"}


# --------------------------------------------------------------------------
# 2.1 Schema sanitizer
# --------------------------------------------------------------------------

def _sanitize_node(node: Any) -> Any:
    """Recursively rewrite one JSON-schema node into Gemini's accepted
    subset: pydantic's ``Tuple[N]`` -> a fixed-length number array,
    ``anyOf: [X, {type: null}]`` -> X with ``nullable: true``, and strip
    keys the converter rejects. ``$ref``/``$defs`` are left alone (Gemini
    resolves them)."""
    if isinstance(node, list):
        return [_sanitize_node(n) for n in node]
    if not isinstance(node, dict):
        return node

    if "anyOf" in node:
        branches = node["anyOf"]
        non_null = [b for b in branches if b.get("type") != "null"]
        if len(non_null) == 1 and len(non_null) != len(branches):
            inner = dict(_sanitize_node(non_null[0]))
            inner["nullable"] = True
            if "description" in node and "description" not in inner:
                inner["description"] = node["description"]
            return {k: v for k, v in inner.items() if k not in _STRIP_KEYS}
        # Multiple non-null branches, or no null branch at all -- not a
        # pattern this schema family produces, but sanitize each branch
        # defensively rather than crash.
        out = dict(node)
        out["anyOf"] = [_sanitize_node(b) for b in branches]
        return {k: v for k, v in out.items() if k not in _STRIP_KEYS}

    out = dict(node)

    if "prefixItems" in out:
        n = len(out["prefixItems"])
        out.pop("prefixItems")
        out["type"] = "array"
        out["items"] = {"type": "number"}
        out["minItems"] = n
        out["maxItems"] = n
        return {k: v for k, v in out.items() if k not in _STRIP_KEYS}

    if "properties" in out:
        out["properties"] = {k: _sanitize_node(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _sanitize_node(out["items"])
    if "$defs" in out:
        out["$defs"] = {k: _sanitize_node(v) for k, v in out["$defs"].items()}

    return {k: v for k, v in out.items() if k not in _STRIP_KEYS}


def gemini_schema(schema: Type[BaseModel]) -> dict:
    """``schema.model_json_schema()`` rewritten for Gemini's
    ``response_schema``: structurally sanitized (see ``_sanitize_node``),
    plus the "force engagement" reinforcements that fix the observed
    0-cuts-on-every-batch failure mode -- an unconstrained schema lets the
    model satisfy an empty ``cuts: []`` trivially, since the field isn't
    ``required`` and has no ``minItems``:

      * a top-level ``cuts`` array (if present) becomes required, minItems=1.
      * a ``CutJudgment`` def's ``label``/``summary`` (if present) get
        minLength=1, on top of already being in pydantic's own ``required``.
      * ``CutJudgment.channel`` / ``Framing.shot_size`` (if present) get a
        closed ``enum`` so structured decoding can't drift onto a synonym
        pass2.py's own alias-folding would otherwise have to catch.

    Every reinforcement is defensive (checks the field exists before
    touching it) -- this stays a no-op pass-through for a schema that
    doesn't have this shape, never a KeyError."""
    sanitized = _sanitize_node(schema.model_json_schema())

    props = sanitized.get("properties") or {}
    if "cuts" in props and props["cuts"].get("type") == "array":
        required = list(sanitized.get("required") or [])
        if "cuts" not in required:
            required.append("cuts")
        sanitized["required"] = required
        props["cuts"]["minItems"] = 1

    defs = sanitized.get("$defs") or {}
    cut_def = defs.get("CutJudgment")
    if isinstance(cut_def, dict):
        cut_props = cut_def.get("properties") or {}
        for field in ("label", "summary"):
            if field in cut_props and isinstance(cut_props[field], dict):
                cut_props[field]["minLength"] = 1
        if "channel" in cut_props and isinstance(cut_props["channel"], dict):
            cut_props["channel"]["enum"] = list(_CHANNEL_ENUM)
    framing_def = defs.get("Framing")
    if isinstance(framing_def, dict):
        framing_props = framing_def.get("properties") or {}
        if "shot_size" in framing_props and isinstance(framing_props["shot_size"], dict):
            framing_props["shot_size"]["enum"] = list(_SHOT_SIZE_ENUM)
        if "shot_quality" in framing_props and isinstance(framing_props["shot_quality"], dict):
            framing_props["shot_quality"]["enum"] = list(_SHOT_QUALITY_ENUM)
        # NOTE: shot_quality (like subject_box below) stays OPTIONAL -- never
        # added to `required`. perception_upgrade.plan.md's Flash-Lite
        # guardrail: every NEW model field is additive/optional/prompt-nudged
        # only; requiring one risks the same runaway-thinking failure mode
        # subject_box hit (see below).
        # NOTE: subject_box stays OPTIONAL/nullable here. Forcing it required
        # (an earlier "knob #2" attempt) triggered runaway dynamic thinking on
        # ambiguous b-roll batches: Flash-Lite spends the ENTIRE output budget
        # reasoning about a box it can't confidently place (observed
        # think_tok >= 30k, finish=MAX_TOKENS, zero JSON emitted). subject_box
        # recovery is nudged via the prompt instead, never hard-required.

    return sanitized


# --------------------------------------------------------------------------
# 2.2 The call
# --------------------------------------------------------------------------

# "low"/"medium"/"high" -> a thinking_budget token count. This SDK version
# (google-genai) exposes ThinkingConfig.thinking_budget (an int), not a
# named "thinking_level" -- these are our own fixed mappings for the
# ingest_pass2_thinking config knob, not an SDK enum.
_THINKING_BUDGETS = {"low": 2048, "medium": 8192, "high": 24576}


def _resolve_thinking_budget(thinking: Optional[str]) -> Optional[int]:
    """A config string ("low"/"medium"/"high", a bare integer, or None/"")
    -> a thinking_budget token count, or None to leave Gemini's default.
    Never raises -- an unrecognized value logs and falls back to "low"
    rather than silently disabling thinking (the near-zero output in the
    drop-in probe correlated with no thinking budget at all)."""
    if not thinking:
        return None
    key = str(thinking).strip().lower()
    if key in _THINKING_BUDGETS:
        return _THINKING_BUDGETS[key]
    try:
        return int(key)
    except ValueError:
        logger.warning("ingest_gemini: unrecognized ingest_pass2_thinking=%r, using 'low'", thinking)
        return _THINKING_BUDGETS["low"]


def _thinking_config(types: Any, thinking: Optional[str]) -> Optional[Any]:
    """Build a ThinkingConfig, degrading gracefully (log + None, never
    raise) if this SDK version's constructor shape doesn't match -- API
    drift here should never take down an ingest run."""
    budget = _resolve_thinking_budget(thinking)
    if budget is None:
        return None
    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except Exception:
        logger.exception("ingest_gemini: failed to build ThinkingConfig (budget=%s); continuing without it", budget)
        return None


def _usage_of(resp: Any) -> Dict[str, int]:
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return {}
    output = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    return {
        "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
        "output_tokens": output,
        "cache_read_input_tokens": getattr(um, "cached_content_token_count", 0) or 0,
        "cache_creation_input_tokens": 0,
    }


def _sum_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _response_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text
    # Defensive fallback: assemble from candidate parts if the convenience
    # `.text` accessor comes back empty (e.g. thinking-only response).
    bits: List[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if t:
                bits.append(t)
    return "".join(bits)


def _diag(resp: Any) -> str:
    """One-line diagnostic for a Gemini response: finish reason, the kinds of
    content parts present, and the assembled text length -- the signal that
    tells an empty/failed pass-2 batch (finish=MAX_TOKENS, thinking-only
    parts, or a safety block) apart from a genuine answer."""
    cand = (getattr(resp, "candidates", None) or [None])[0]
    finish = getattr(cand, "finish_reason", None)
    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    kinds = [("text" if getattr(p, "text", None) else
              ("thought" if getattr(p, "thought", None) else "other")) for p in parts]
    um = getattr(resp, "usage_metadata", None)
    return (f"finish={finish} parts={kinds} text_len={len(_response_text(resp))} "
            f"in_tok={getattr(um, 'prompt_token_count', None)} "
            f"out_tok={getattr(um, 'candidates_token_count', None)} "
            f"think_tok={getattr(um, 'thoughts_token_count', None)}")


def _parse_raw(resp: Any) -> Optional[Dict[str, Any]]:
    text = _response_text(resp).strip()
    if not text:
        return None
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(raw, list):
        # Observed: Gemini sometimes drops the {"cuts": [...]} wrapper key
        # and emits the bare array the wrapper's `cuts` field would hold.
        return {"cuts": raw}
    if isinstance(raw, dict):
        return raw
    return None


def _build_config(
    types: Any, schema: Type[BaseModel], system: str, max_tokens: int,
    thinking: Optional[str], cached_content: Optional[str],
) -> Any:
    kwargs: Dict[str, Any] = {
        "max_output_tokens": max_tokens,
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": gemini_schema(schema),
    }
    if not cached_content:
        # A CachedContent already carries the system instruction; supplying
        # both raises on some SDK versions, so only set it when NOT reusing
        # a cache handle.
        kwargs["system_instruction"] = system
    else:
        kwargs["cached_content"] = cached_content
    tc = _thinking_config(types, thinking)
    if tc is not None:
        kwargs["thinking_config"] = tc
    return types.GenerateContentConfig(**kwargs)


def complete_gemini(
    system: str,
    blocks: List[Block],
    schema: Type[SchemaT],
    *,
    extra_blocks: Optional[List[Block]] = None,
    max_tokens: int = 32000,
    extra_check: Optional[Callable[[SchemaT], Optional[str]]] = None,
    model: Optional[str] = None,
    thinking: Optional[str] = None,
    cached_content: Optional[str] = None,
) -> Completion:
    """Gemini's structured-output equivalent of ``client.complete()`` --
    same contract (``Completion``, one re-ask then ``IngestFailure``, an
    ``extra_check`` semantic hook), different wire path: Gemini's native
    ``response_schema`` + ``response_mime_type="application/json"`` instead
    of Claude's forced tool_choice. ``cached_content`` is an existing
    ``CachedContent`` resource name (P4, see ``create_pass2_cache``); when
    given, ``blocks`` should be ONLY the per-call (uncached) content -- the
    cache already carries the system instruction and any content baked into
    it at creation time."""
    settings = get_settings()
    client, types = _sdk()
    resolved_model = model or settings.ingest_pass2_model

    all_blocks = list(blocks) + list(extra_blocks or [])
    parts = _parts_for_content(all_blocks, types, {})
    contents = [types.Content(role="user", parts=parts)]
    config = _build_config(types, schema, system, max_tokens, thinking, cached_content)

    def _call(cfg: Any) -> Any:
        return client.models.generate_content(model=resolved_model, contents=contents, config=cfg)

    resp = _call(config)
    logger.info("ingest_gemini: attempt 1 response -- %s", _diag(resp))
    usage = _usage_of(resp)
    raw = _parse_raw(resp)
    parsed, err = _validate(schema, raw)
    if parsed is not None and extra_check is not None:
        sem_err = extra_check(parsed)
        if sem_err:
            parsed, err = None, sem_err
    logger.info("ingest_gemini: attempt 1 ok=%s %s", parsed is not None,
               f"raw_keys={list(raw.keys())}" if isinstance(raw, dict) else f"raw={raw!r}")
    if parsed is not None:
        return Completion(data=parsed.model_dump(), usage=usage, attempts=1)

    logger.warning("ingest_gemini: schema/semantic violation, re-asking once: %s", err)
    correction = (
        f"Your previous JSON failed validation: {err}\n"
        f'Re-emit ONLY a valid JSON object {{"cuts": [...]}} for the schema; include '
        f"every required field, one cut per shown source_ref."
    )
    contents2 = contents + [
        types.Content(role="model", parts=[types.Part(text=_response_text(resp) or "{}")]),
        types.Content(role="user", parts=[types.Part(text=correction)]),
    ]

    def _call2() -> Any:
        return client.models.generate_content(model=resolved_model, contents=contents2, config=config)

    resp2 = _call2()
    logger.info("ingest_gemini: attempt 2 response -- %s", _diag(resp2))
    usage2 = _usage_of(resp2)
    total_usage = _sum_usage(usage, usage2)
    raw2 = _parse_raw(resp2)
    parsed2, err2 = _validate(schema, raw2)
    if parsed2 is not None and extra_check is not None:
        sem_err2 = extra_check(parsed2)
        if sem_err2:
            parsed2, err2 = None, sem_err2
    logger.info("ingest_gemini: attempt 2 ok=%s", parsed2 is not None)
    if parsed2 is not None:
        return Completion(data=parsed2.model_dump(), usage=total_usage, attempts=2)

    raise IngestFailure(stage="pass2", reason=f"schema validation failed twice: {err2}")


# --------------------------------------------------------------------------
# 2.2 fallback (documented, not built -- see plan SS2.2): if response_schema
# still under-produces on real footage, escalate to Gemini function-calling
# with tool_config=FunctionCallingConfig(mode="ANY"), which most closely
# mirrors Claude's forced tool_choice. Not implemented until P5 shows it's
# needed.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Phase 4 -- context caching. Optional: the run works correctly without it
# (P1-P3 correctness gate), this only realizes the cost win once correctness
# is proven. `create_pass2_cache` is called once per ingest run (the STABLE,
# project-wide prefix); `delete_pass2_cache` tears it down when the run ends
# (success or failure) so a stale handle never leaks into a later run.
# --------------------------------------------------------------------------

def create_pass2_cache(
    system: str, blocks: List[Block], *, model: Optional[str] = None, ttl_seconds: int = 900,
) -> Optional[str]:
    """Create a Gemini ``CachedContent`` from the STABLE part of the pass-2
    prefix (``build_pass1_blocks`` -- transcripts + atom tables; NOT the
    per-batch-trimmed ``render_pass1_output``, which differs batch to batch
    and must stay uncached). Returns the cache resource name to pass as
    ``cached_content`` on every batch call in this run, or None if creation
    fails/degrades (caching is a cost optimization, never a correctness
    requirement -- a caller should fall back to uncached on None)."""
    settings = get_settings()
    try:
        client, types = _sdk()
        parts = _parts_for_content(list(blocks), types, {})
        cache = client.caches.create(
            model=model or settings.ingest_pass2_model,
            config=types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=parts)],
                system_instruction=system,
                ttl=f"{ttl_seconds}s",
            ),
        )
        return cache.name
    except Exception:
        logger.exception("ingest_gemini: pass2 CachedContent creation failed -- continuing uncached")
        return None


def delete_pass2_cache(name: Optional[str]) -> None:
    """Best-effort teardown -- a failed delete just means the cache expires
    on its own TTL instead; never worth failing the ingest run over."""
    if not name:
        return
    try:
        client, _types = _sdk()
        client.caches.delete(name=name)
    except Exception:
        logger.warning("ingest_gemini: failed to delete pass2 CachedContent %s (will expire via TTL)", name)


# `run_pass2_batch` / `client.complete()` are unchanged call sites (see the
# plan's non-goals) -- so the per-run cache handle can't be threaded through
# either signature. A ContextVar instead: `ingest.py` sets it once around the
# whole batch loop (`pass2_cache_scope`), `complete()`'s routing guard reads
# it per call (`get_pass2_cache_handle`). IMPORTANT: plain
# ThreadPoolExecutor.submit() does NOT propagate context vars into the worker
# thread (each worker thread starts from the DEFAULT context, not the
# submitting thread's current one -- confirmed against the stdlib, a common
# gotcha). `submit_with_cache_context` below is the one correct way to submit
# a pass-2 batch task so the worker actually sees the handle; `ingest.py`
# must use it (not a bare `pool.submit`) for every pass2.run_pass2_batch
# task while a `pass2_cache_scope` is open. This is also what makes it safe
# under concurrent DIFFERENT runs (`ingest.run_many`): each run captures its
# OWN context at submit time, so they never see each other's handle.
_pass2_cache_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "pass2_cache_handle", default=None)


def get_pass2_cache_handle() -> Optional[str]:
    return _pass2_cache_var.get()


class pass2_cache_scope:
    """Context manager: make ``handle`` (a CachedContent resource name, or
    None to mean "no caching") the handle every ``run_pass2_batch`` call
    makes on this run reuses, for the duration of the ``with`` block."""

    def __init__(self, handle: Optional[str]):
        self._handle = handle
        self._token: Optional[Any] = None

    def __enter__(self) -> "pass2_cache_scope":
        self._token = _pass2_cache_var.set(self._handle)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _pass2_cache_var.reset(self._token)


def submit_with_cache_context(pool: Any, fn: Callable[..., Any], *args: Any) -> Any:
    """``pool.submit(fn, *args)``, but first captures the CURRENT context
    (including whatever ``pass2_cache_scope`` set) and runs ``fn`` inside
    that captured context on the worker thread -- see the note above for why
    a bare ``pool.submit`` silently loses the cache handle."""
    ctx = contextvars.copy_context()
    return pool.submit(ctx.run, fn, *args)
