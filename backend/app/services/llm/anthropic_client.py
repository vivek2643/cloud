"""
Anthropic implementation of the neutral LLMClient.

Translates neutral content blocks <-> Anthropic's Messages API shape. Because
the neutral schema mirrors Anthropic's, the message translation is mostly an
identity with two exceptions: image blocks need the `source` envelope, and we
optionally stamp `cache_control` on the stable prefix for prompt caching.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.llm.base import (
    LLMResponse,
    Message,
    ToolCall,
)

logger = logging.getLogger(__name__)


def _sdk_client():
    from anthropic import Anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to use L3 features."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _block_to_anthropic(block: Dict[str, Any]) -> Dict[str, Any]:
    btype = block.get("type")
    if btype == "text":
        return {"type": "text", "text": block["text"]}
    if btype == "image":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.get("media_type", "image/jpeg"),
                "data": block["data"],
            },
        }
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block["id"],
            "name": block["name"],
            "input": block.get("input", {}),
        }
    if btype == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block["tool_use_id"],
            "content": [_block_to_anthropic(b) for b in block.get("content", [])],
        }
    # Unknown block: pass through best-effort.
    return block


def _content_to_anthropic(content: Any) -> Any:
    if isinstance(content, str):
        return content
    return [_block_to_anthropic(b) for b in content]


def _messages_to_anthropic(messages: List[Message]) -> List[Dict[str, Any]]:
    return [
        {"role": m["role"], "content": _content_to_anthropic(m["content"])}
        for m in messages
    ]


class AnthropicClient:
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self._model = model or settings.anthropic_model
        self._caching = settings.llm_prompt_caching

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
        thinking_budget: int = 0,
        effort: Optional[str] = None,
    ) -> LLMResponse:
        """`thinking_budget` > 0 enables extended thinking. Modern Opus models
        (4.7/4.8+) only accept *adaptive* thinking -- the model decides how much
        to think -- with reasoning depth steered by `effort`
        (low|medium|high|xhigh|max) instead of a manual token budget. We map any
        positive `thinking_budget` to adaptive thinking for forward-compat.

        Thinking blocks are carried through the neutral assistant message
        verbatim (Anthropic requires them to be replayed unmodified on the next
        request of a tool-use loop)."""
        client = _sdk_client()

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": _messages_to_anthropic(messages),
        }
        if thinking_budget > 0:
            kwargs["thinking"] = {"type": "adaptive"}
        if effort:
            # `output_config` is a top-level field not yet typed by this SDK
            # version; inject it into the request body directly.
            kwargs["extra_body"] = {"output_config": {"effort": effort}}

        # System prompt, optionally cached as a stable prefix.
        if system:
            if cache_system and self._caching:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system

        if tools:
            # Neutral tool_spec already matches Anthropic's {name, description,
            # input_schema}. Cache the (stable) tool block to amortize its tokens.
            atools = [dict(t) for t in tools]
            if cache_system and self._caching and atools:
                atools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = atools

        msg = client.messages.create(**kwargs)

        text_bits: List[str] = []
        tool_calls: List[ToolCall] = []
        assistant_content: List[Dict[str, Any]] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                # Preserve verbatim (incl. signature) for replay in tool loops.
                assistant_content.append(
                    {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
                )
            elif btype == "redacted_thinking":
                assistant_content.append({"type": "redacted_thinking", "data": block.data})
            elif btype == "text":
                text_bits.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input or {})
                )
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        usage: Dict[str, int] = {}
        if getattr(msg, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(msg.usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(msg.usage, "output_tokens", 0) or 0,
            }

        return LLMResponse(
            text="".join(text_bits).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(msg, "stop_reason", "end_turn") or "end_turn",
            assistant_message={"role": "assistant", "content": assistant_content},
            usage=usage,
            raw=msg,
        )
