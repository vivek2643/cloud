"""
Anthropic (Claude) implementation of the neutral LLMClient.

The neutral content-block schema (see ``base.py``) was modelled on Anthropic's
own Messages shape, so this translation is near-identity: text/image/tool_use/
tool_result blocks map almost one-to-one. Switching a feature to Claude is a
config change (``get_llm(provider="anthropic", ...)``), never a rewrite.

Prompt caching is first-class here (this is what ``cache_system=True`` was always
for): the stable prefix -- the system prompt, and the first user turn that
carries the big footage map -- is marked with ``cache_control`` so multi-pass
reasoning (draft -> critique) and multi-turn chat reuse it instead of re-sending
it.

``effort`` / ``thinking_budget`` are accepted for call-site compatibility but
still not forwarded (brain_extended_thinking.plan.md STEP ZERO, probed live
against ``claude-opus-4-8`` on SDK 0.49.0): a plain call and one with
``thinking={"type": "adaptive"}`` (the API's current param -- this SDK
version has no typed support for it or for ``output_config``) produced
materially the same visible step-by-step reasoning either way. The adaptive
call DID return a signed ``thinking`` block, but its own ``thinking`` text
field came back empty -- the reasoning stayed in the ordinary text block, as
in the plain call. So there is no observability prize to forward yet; the
budget stays unforwarded until a request shape that actually surfaces
content is found. ``thinking``/``redacted_thinking`` blocks ARE correctly
preserved end to end below regardless (``_block_to_anthropic``, ``run()``'s
completion loop) -- required for correctness the moment any caller (this one
or a future one) ever does send the parameter, since Anthropic rejects a
tool-use continuation whose prior thinking block was dropped or altered.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.llm.base import LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)


def _sdk_client():
    from anthropic import Anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to use Claude-backed features."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _block_to_anthropic(b: Dict[str, Any]) -> Dict[str, Any]:
    """Neutral content block -> Anthropic content block."""
    btype = b.get("type")
    if btype == "text":
        return {"type": "text", "text": b.get("text", "")}
    if btype == "image":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": b.get("media_type", "image/jpeg"),
                "data": b.get("data", ""),
            },
        }
    if btype == "tool_use":
        return {"type": "tool_use", "id": b["id"], "name": b["name"],
                "input": b.get("input", {})}
    if btype == "tool_result":
        inner = b.get("content", [])
        content = inner if isinstance(inner, str) else [
            _block_to_anthropic(x) for x in inner]
        return {"type": "tool_result", "tool_use_id": b.get("tool_use_id", ""),
                "content": content}
    if btype in ("thinking", "redacted_thinking"):
        # brain_extended_thinking.plan.md 3.1: returned to the API verbatim.
        # These carry a cryptographic `signature` attesting the block came
        # from the model; ANY mutation invalidates it and the next tool-use
        # request is rejected. Never rebuild these field by field -- copy the
        # block whole (the shape is provider-owned and versioned; naming
        # fields would silently drop one the API adds later).
        return dict(b)
    return {"type": "text", "text": ""}


def _messages_to_anthropic(messages: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        else:
            out.append({"role": m["role"],
                        "content": [_block_to_anthropic(b) for b in content]})
    return out


def _mark_cache_breakpoint(messages: List[Dict[str, Any]]) -> None:
    """Put a cache_control breakpoint on the LAST block of the FIRST user turn.

    In the arranger the big footage map lives in that first user turn, not the
    system prompt, so this is what actually caches the expensive prefix across
    the draft -> critique passes. (The system block is cached separately in
    ``run``.) Editing earlier turns would bust the cache, so we only ever mark
    the first user turn, which never changes within a thread/run.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content,
                               "cache_control": {"type": "ephemeral"}}]
        elif isinstance(content, list) and content:
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        return


def _tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("input_schema", {"type": "object"}),
        }
        for t in tools
    ]


class AnthropicClient:
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self._model = model or settings.anthropic_model

    @property
    def model(self) -> str:
        return self._model

    def run(
        self,
        *,
        system: str,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        cache_system: bool = False,
        effort: Optional[str] = None,  # noqa: ARG002 - not forwarded, see module docstring
        thinking_budget: int = 0,  # noqa: ARG002 - not forwarded, see module docstring
        **_: Any,
    ) -> LLMResponse:
        client = _sdk_client()

        a_messages = _messages_to_anthropic(messages)
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": a_messages,
        }
        if system:
            if cache_system:
                kwargs["system"] = [{"type": "text", "text": system,
                                     "cache_control": {"type": "ephemeral"}}]
            else:
                kwargs["system"] = system
        if cache_system:
            _mark_cache_breakpoint(a_messages)
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        completion = client.messages.create(**kwargs)

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        assistant_content: List[Dict[str, Any]] = []
        for blk in completion.content or []:
            btype = getattr(blk, "type", None)
            if btype == "text":
                text_parts.append(blk.text)
                assistant_content.append({"type": "text", "text": blk.text})
            elif btype == "tool_use":
                inp = blk.input or {}
                tool_calls.append(ToolCall(id=blk.id, name=blk.name, input=inp))
                assistant_content.append(
                    {"type": "tool_use", "id": blk.id, "name": blk.name, "input": inp})
            elif btype in ("thinking", "redacted_thinking"):
                # brain_extended_thinking.plan.md 3.2: converting an SDK
                # *object* (not a dict we already own) to our neutral shape,
                # so fields must be named here -- the asymmetry with
                # _block_to_anthropic's dict(b) copy above is deliberate.
                # Ordering matters: Anthropic requires a thinking block to
                # appear FIRST in assistant content, before text/tool_use --
                # appending in completion.content order preserves this
                # naturally since the API emits it first; never sort/filter.
                blk_out: Dict[str, Any] = {"type": btype}
                for f in ("thinking", "signature", "data"):
                    v = getattr(blk, f, None)
                    if v is not None:
                        blk_out[f] = v
                assistant_content.append(blk_out)

        usage: Dict[str, int] = {}
        u = getattr(completion, "usage", None)
        if u is not None:
            usage = {
                "input_tokens": getattr(u, "input_tokens", 0) or 0,
                "output_tokens": getattr(u, "output_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(
                    u, "cache_creation_input_tokens", 0) or 0,
            }

        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            assistant_message={"role": "assistant", "content": assistant_content},
            usage=usage,
            raw=completion,
        )
