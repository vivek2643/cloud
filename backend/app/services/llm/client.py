"""
Cuts v3 ingest LLM layer: a thin, STRUCTURED, schema-enforced completion
wrapper. New, sibling to the existing provider clients in this package --
v3 does not touch ``anthropic_client.py`` / ``factory.py`` / the arranger's
``get_llm()`` path; it reuses their block-translation helpers by import, not
by modification.

Why this exists (see ``cuts_v3.plan.md``, "Model layer"):

  * **Stage-based model resolution.** ``complete("pass1", ...)`` /
    ``complete("pass2", ...)`` resolve their model id from config
    (``INGEST_PASS1_MODEL`` / ``INGEST_PASS2_MODEL``) -- swapping either pass
    to a different model is an env change, never a prompt rewrite.
  * **No native JSON mode on Claude.** Structured output is enforced by
    wrapping the caller's pydantic ``schema`` as a single Anthropic TOOL and
    forcing ``tool_choice`` onto it -- the tool's ``input`` IS the response,
    already schema-shaped by the model's own constrained decoding.
  * **No fallback, per plan North Star #4.** A response that fails pydantic
    validation gets exactly ONE re-ask (the validation errors fed back
    verbatim); a second failure raises ``IngestFailure`` -- loud, for the
    caller to mark the ingest run ``failed``, never a silent degrade.
  * **Caching is the caller's to shape, this module's to apply.** The caller
    passes the STABLE, reusable content as ``blocks`` (marked with a single
    trailing cache breakpoint) and anything shard-specific/fresh as
    ``extra_blocks`` (appended after, uncached) -- e.g. pass 2's shared
    [system + transcripts + atom tables + pass-1 output] prefix in ``blocks``,
    that shard's numbered images in ``extra_blocks``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.services.llm.anthropic_client import _block_to_anthropic, _sdk_client
from app.services.llm.base import Block

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_TOOL_NAME = "emit_result"

_STAGE_MODEL_ATTR = {
    "pass1": "ingest_pass1_model",
    "pass2": "ingest_pass2_model",
}


class IngestFailure(Exception):
    """A stage's structured output never validated, even after one re-ask --
    loud failure, no fallback (plan North Star #4). The caller marks the
    owning ``ingest_runs`` row ``failed`` with ``reason`` and stops."""

    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"ingest stage {stage!r} failed: {reason}")


@dataclass
class Completion:
    """One stage call's result: the schema-validated payload (plain dict,
    JSON-ready for ``ingest_runs.pass1_output`` / downstream assembly) plus
    token/cache usage -- SUMMED across the initial attempt and the re-ask, if
    one happened, so a caller accumulating cost onto ``ingest_runs`` never
    has to know a retry occurred."""
    data: Dict[str, Any]
    usage: Dict[str, int] = field(default_factory=dict)
    attempts: int = 1


def _model_for(stage: str) -> str:
    attr = _STAGE_MODEL_ATTR.get(stage)
    if not attr:
        raise ValueError(f"unknown ingest stage {stage!r} (expected one of {list(_STAGE_MODEL_ATTR)})")
    return getattr(get_settings(), attr)


def _schema_tool(schema: Type[BaseModel]) -> Dict[str, Any]:
    return {
        "name": _TOOL_NAME,
        "description": f"Emit the {schema.__name__} result. This is the only way to respond.",
        "input_schema": schema.model_json_schema(),
    }


def _usage_of(resp: Any) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def _sum_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _extract_tool_input(resp: Any) -> Optional[Dict[str, Any]]:
    for blk in getattr(resp, "content", None) or []:
        if getattr(blk, "type", None) == "tool_use" and getattr(blk, "name", None) == _TOOL_NAME:
            return blk.input or {}
    return None


def _extract_tool_use_id(resp: Any) -> Optional[str]:
    for blk in getattr(resp, "content", None) or []:
        if getattr(blk, "type", None) == "tool_use" and getattr(blk, "name", None) == _TOOL_NAME:
            return blk.id
    return None


def _unwrap_single_key(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Observed twice in the wild: the model nests its real answer one level
    deeper than the schema calls for, under a single spurious top-level key
    -- once literally "$PARAMETER_NAME", once the schema's own class name
    ("Pass1Output"). If `raw` has exactly one key whose value is itself a
    dict, that inner dict is the more likely intended payload. A no-op
    (returns None) for any response shaped normally."""
    if len(raw) == 1:
        (only_value,) = raw.values()
        if isinstance(only_value, dict):
            return only_value
    return None


def _unstringify_field(raw: Dict[str, Any], key: str) -> Any:
    """A field the schema wants as a list/object, occasionally emitted as a
    JSON-encoded STRING instead -- observed on nearly every real pass-2
    call ("cuts": "[...]", or double-wrapped: "cuts": "{\"cuts\": [...]}").
    Parses the string and, if the result is itself a dict re-using the same
    key, unwraps that one more level too. Returns the original value
    unchanged if it isn't a string, or isn't parseable JSON even leniently.

    Tries strict JSON first, then ``strict=False`` (permits literal control
    characters -- newlines/tabs -- inside string values unescaped): when
    the model hand-stringifies a big JSON blob as a field value, the OUTER
    tool-call shape is still constrained-decoded, but that inner string's
    CONTENT is free text as far as the decoder is concerned, so a pretty-
    printed multi-line "fake JSON" string can carry a raw newline where a
    real JSON string would need ``\\n``. That's the one relaxation applied
    here -- it doesn't reinterpret or repair mismatched quotes/content."""
    v = raw.get(key)
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s[:1] not in "{[":
        return v
    parsed = None
    for strict in (True, False):
        try:
            parsed = json.loads(s, strict=strict)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if parsed is None:
        return v
    if isinstance(parsed, dict) and key in parsed:
        return parsed[key]
    return parsed


def _unstringify_json_fields(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Apply _unstringify_field to every top-level key. Returns None (no-op)
    if nothing was actually a stringified field, so the caller can tell
    "tried and it didn't apply" from "this is genuinely the normalized
    result"."""
    out: Dict[str, Any] = {}
    changed = False
    for k, v in raw.items():
        new_v = _unstringify_field(raw, k)
        out[k] = new_v
        if new_v is not v:
            changed = True
    return out if changed else None


def _validate(schema: Type[SchemaT], raw: Optional[Dict[str, Any]]) -> tuple[Optional[SchemaT], Optional[str]]:
    if raw is None:
        return None, "model did not call the required tool"
    try:
        return schema.model_validate(raw), None
    except ValidationError as e:
        unwrapped = _unwrap_single_key(raw)
        if unwrapped is not None:
            try:
                return schema.model_validate(unwrapped), None
            except ValidationError:
                pass
        destringified = _unstringify_json_fields(raw)
        if destringified is not None:
            try:
                return schema.model_validate(destringified), None
            except ValidationError:
                pass
        return None, str(e)


_MAX_TOKENS_CEILING = 64000


def _summarize(raw: Optional[Dict[str, Any]]) -> str:
    """A cheap, schema-agnostic one-liner for logs: list fields show their
    length (the thing that's actually gone missing when a response comes
    back empty), so this is useful for both Pass1Output- and
    Pass2Output-shaped payloads without hardcoding either."""
    if not isinstance(raw, dict):
        return repr(raw)
    parts = []
    for k, v in raw.items():
        if isinstance(v, list):
            parts.append(f"{k}=[{len(v)}]")
        elif isinstance(v, str):
            parts.append(f"{k}=str({len(v)})")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _truncated(resp: Any) -> bool:
    """True when the model was cut off by the token budget mid-response --
    the Anthropic SDK still returns a syntactically-valid (if empty/partial)
    tool_use.input in this case, so it can pass pydantic validation cleanly
    while silently discarding everything the model meant to say. That is
    indistinguishable from a legitimate empty answer unless checked
    explicitly -- treated as a schema violation, never a silent short
    result."""
    return getattr(resp, "stop_reason", None) == "max_tokens"


def complete(
    stage: str,
    system: str,
    blocks: List[Block],
    schema: Type[SchemaT],
    *,
    extra_blocks: Optional[List[Block]] = None,
    cache: bool = True,
    max_tokens: int = 8192,
    extra_check: Optional[Callable[[SchemaT], Optional[str]]] = None,
) -> Completion:
    """One structured ingest call. Raises ``IngestFailure`` if the response
    never validates against ``schema`` (after one re-ask) -- never returns a
    partial or best-effort result.

    ``extra_check`` is an optional SEMANTIC check pydantic's schema can't
    express on its own (e.g. "no atom_id appears in two different cuts" --
    a cross-object invariant, not a per-field type). Given the already
    schema-valid parsed object, return an error string to reject it, or
    None to accept -- a violation is folded into the exact same
    one-re-ask-then-fail-loud path as a schema violation."""
    client = _sdk_client()
    model = _model_for(stage)
    tool = _schema_tool(schema)

    a_blocks = [_block_to_anthropic(b) for b in blocks]
    if cache and a_blocks:
        a_blocks[-1] = {**a_blocks[-1], "cache_control": {"type": "ephemeral"}}
    if extra_blocks:
        a_blocks = a_blocks + [_block_to_anthropic(b) for b in extra_blocks]

    messages: List[Dict[str, Any]] = [{"role": "user", "content": a_blocks}]
    sys_block: Any = (
        [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if cache else system
    )

    def _call(tokens: int) -> Any:
        # Streamed, not create(): the SDK refuses a non-streaming call above
        # ~10 minutes of estimated max_tokens (a real risk here -- a big
        # project's pass-2 shard legitimately needs tens of thousands of
        # output tokens). The reassembled final message has the same shape
        # (.content/.usage/.stop_reason) as create()'s return either way.
        with client.messages.stream(
            model=model,
            max_tokens=tokens,
            system=sys_block,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        ) as stream:
            stream.until_done()
            return stream.get_final_message()

    resp = _call(max_tokens)
    usage = _usage_of(resp)
    if _truncated(resp):
        err = f"response truncated at max_tokens={max_tokens} before the tool call finished"
        parsed = None
    else:
        parsed, err = _validate(schema, _extract_tool_input(resp))
        if parsed is not None and extra_check is not None:
            sem_err = extra_check(parsed)
            if sem_err:
                parsed, err = None, sem_err
    logger.info("ingest stage %s attempt 1: stop_reason=%s ok=%s %s", stage,
               getattr(resp, "stop_reason", None), parsed is not None, _summarize(_extract_tool_input(resp)))
    if parsed is not None:
        return Completion(data=parsed.model_dump(), usage=usage, attempts=1)

    logger.warning("ingest stage %s: schema violation, re-asking once: %s", stage, err)
    retry_tokens = min(max_tokens, _MAX_TOKENS_CEILING)
    reask_text = f"Your last response violated the {schema.__name__} schema:\n{err}\n"
    if _truncated(resp):
        retry_tokens = min(max_tokens * 2, _MAX_TOKENS_CEILING)
        reask_text = (
            f"Your last response was truncated (ran out of output budget) before "
            f"finishing the {_TOOL_NAME} call. You have more room this time "
            f"({retry_tokens} tokens) -- "
        )
    correction = reask_text + f"Call {_TOOL_NAME} again with a CORRECTED input that satisfies it."

    # A tool_use block in the assistant turn MUST be followed by a matching
    # tool_result in the very next message, or the API rejects the whole
    # request (400) -- there is no bare "text-only" reply to a tool call.
    messages.append({"role": "assistant", "content": resp.content})
    tool_use_id = _extract_tool_use_id(resp)
    if tool_use_id:
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": correction,
                "is_error": True,
            }],
        })
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": correction}]})
    resp2 = _call(retry_tokens)
    usage2 = _usage_of(resp2)
    total_usage = _sum_usage(usage, usage2)
    if _truncated(resp2):
        err2 = f"response truncated at max_tokens={retry_tokens} before the tool call finished"
        parsed2 = None
    else:
        parsed2, err2 = _validate(schema, _extract_tool_input(resp2))
        if parsed2 is not None and extra_check is not None:
            sem_err2 = extra_check(parsed2)
            if sem_err2:
                parsed2, err2 = None, sem_err2
    logger.info("ingest stage %s attempt 2: stop_reason=%s ok=%s %s", stage,
               getattr(resp2, "stop_reason", None), parsed2 is not None, _summarize(_extract_tool_input(resp2)))
    if parsed2 is not None:
        return Completion(data=parsed2.model_dump(), usage=total_usage, attempts=2)

    raise IngestFailure(stage=stage, reason=f"schema validation failed twice: {err2}")
