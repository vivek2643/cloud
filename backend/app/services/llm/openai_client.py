"""
OpenAI implementation of the neutral LLMClient.

Translates the neutral content-block schema (see ``base.py``) to/from OpenAI's
Chat Completions shape so callers depend only on the provider-agnostic surface.
Switching a feature between Anthropic / Gemini / OpenAI is therefore a config
change (``get_llm(provider=...)``), never a rewrite.

The neutral schema mirrors Anthropic's, so the lift here is: split a neutral
turn's ``tool_use`` / ``tool_result`` blocks out into OpenAI's separate
``tool_calls`` field and ``role: "tool"`` messages, and wrap images as
``image_url`` parts. Anthropic-only kwargs (``cache_system``,
``thinking_budget``, ``effort``) are accepted and ignored so this class is a
drop-in for the same call sites.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.llm.base import LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)


def _sdk_client():
    from openai import OpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env to use OpenAI-backed features."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _content_parts(content: Any) -> Any:
    """Neutral content -> OpenAI message content (str or list of parts).

    Only text/image parts live in ``content``; tool_use/tool_result are handled
    at the message level by the caller.
    """
    if isinstance(content, str):
        return content
    parts: List[Dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            media = block.get("media_type", "image/jpeg")
            data = block.get("data", "")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{data}"},
                }
            )
    # Collapse a lone text part back to a plain string (cheaper, identical).
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


def _messages_to_openai(messages: List[Message]) -> List[Dict[str, Any]]:
    """Translate neutral messages to OpenAI's list, splitting tool blocks out."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")

        if isinstance(content, list):
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            other = [b for b in content if b.get("type") in ("text", "image")]

            if role == "assistant" and tool_uses:
                msg: Dict[str, Any] = {"role": "assistant"}
                text_parts = _content_parts(other) if other else ""
                msg["content"] = text_parts or None
                msg["tool_calls"] = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    }
                    for b in tool_uses
                ]
                out.append(msg)
                continue

            if tool_results:
                # Each tool_result becomes its own role:"tool" message.
                for b in tool_results:
                    inner = b.get("content", [])
                    text = inner if isinstance(inner, str) else " ".join(
                        x.get("text", "") for x in inner if x.get("type") == "text"
                    )
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": text,
                        }
                    )
                if other:
                    out.append({"role": role, "content": _content_parts(other)})
                continue

        out.append({"role": role, "content": _content_parts(content)})
    return out


# Neutral "effort" (shared with the Anthropic client) -> OpenAI reasoning_effort.
# Lets latency-sensitive callers (e.g. sentence classification) skip the deep,
# slow reasoning a GPT-5 model does by default.
_EFFORT_MAP = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools
    ]


class OpenAIClient:
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self._model = model or settings.openai_model

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
        cache_system: bool = False,  # noqa: ARG002 - Anthropic-only; ignored
        thinking_budget: int = 0,  # noqa: ARG002 - Anthropic-only; ignored
        effort: Optional[str] = None,
        **_: Any,
    ) -> LLMResponse:
        client = _sdk_client()

        oai_messages: List[Dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(_messages_to_openai(messages))

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            # GPT-5-class models use max_completion_tokens (max_tokens is rejected).
            "max_completion_tokens": max_tokens,
        }
        if effort:
            # Ignored by non-reasoning models; GPT-5 uses it to bound latency.
            kwargs["reasoning_effort"] = _EFFORT_MAP.get(effort.lower(), effort.lower())
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)

        completion = client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        text = msg.content or ""
        tool_calls: List[ToolCall] = []
        assistant_content: List[Dict[str, Any]] = []
        if text:
            assistant_content.append({"type": "text", "text": text})
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
            assistant_content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args}
            )

        usage: Dict[str, int] = {}
        u = getattr(completion, "usage", None)
        if u is not None:
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(u, "completion_tokens", 0) or 0,
            }

        return LLMResponse(
            text=text.strip(),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            assistant_message={"role": "assistant", "content": assistant_content},
            usage=usage,
            raw=completion,
        )
