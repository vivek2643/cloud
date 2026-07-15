"""
Neutral, provider-agnostic types and content-block builders for the LLM
adapter. Both the Anthropic and Gemini clients translate to/from these.

Neutral content blocks (plain dicts) used inside message `content` lists:
  {"type": "text", "text": str}
  {"type": "image", "data": <base64 str>, "media_type": "image/jpeg"}
  {"type": "media", "data": <base64 str>, "media_type": "video/mp4"}
  {"type": "tool_use", "id": str, "name": str, "input": dict}   (assistant)
  {"type": "tool_result", "tool_use_id": str, "content": [blocks]} (user)

A neutral message is {"role": "user"|"assistant", "content": str | [blocks]}.
Keeping these aligned with Anthropic's schema (the current backbone) keeps the
Anthropic translation near-identity; the Gemini client does the heavier lift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

Block = Dict[str, Any]
Content = Union[str, List[Block]]
Message = Dict[str, Any]


@dataclass
class ToolCall:
    """A normalized tool/function invocation requested by the model."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized model response.

    `assistant_message` is the neutral assistant turn (text + tool_use blocks)
    ready to append back onto the running `messages` list for the next round,
    so callers never touch provider-native shapes.
    """

    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    assistant_message: Message = field(default_factory=lambda: {"role": "assistant", "content": []})
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None


# ---------------------------------------------------------------------------
# Neutral content-block builders
# ---------------------------------------------------------------------------

def text_block(text: str) -> Block:
    return {"type": "text", "text": text}


def image_block(data_b64: str, media_type: str = "image/jpeg") -> Block:
    return {"type": "image", "data": data_b64, "media_type": media_type}


def media_block(data_b64: str, media_type: str) -> Block:
    """Inline video (or other non-image) media, e.g. a short clip for the
    voice-ID pass (identity/voice_id.py) to judge lip-sync against heard
    audio -- Gemini-only today (gemini_client._parts_for_content)."""
    return {"type": "media", "data": data_b64, "media_type": media_type}


def tool_use_block(id: str, name: str, input: Dict[str, Any]) -> Block:
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def tool_result_block(tool_use_id: str, content: Union[str, List[Block]]) -> Block:
    blocks = [text_block(content)] if isinstance(content, str) else content
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": blocks}


def user_message(content: Content) -> Message:
    return {"role": "user", "content": content}


def assistant_message(content: Content) -> Message:
    return {"role": "assistant", "content": content}


def tool_spec(name: str, description: str, input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """A neutral tool declaration. Matches Anthropic's tool shape; the Gemini
    client maps it to a functionDeclaration."""
    return {"name": name, "description": description, "input_schema": input_schema}


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """A provider client. Stateless per call; the caller owns the message list."""

    @property
    def model(self) -> str:  # pragma: no cover - trivial
        ...

    def run(
        self,
        *,
        system: str,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        cache_system: bool = False,
    ) -> LLMResponse:
        ...
