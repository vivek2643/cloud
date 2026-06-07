"""
Gemini implementation of the neutral LLMClient (built, off by default).

Maps neutral blocks to the google-genai SDK:
  text         -> {"text": ...}
  image        -> inline_data part (decoded bytes)
  tool_use     -> function_call part (assistant/"model" turn)
  tool_result  -> function_response part (correlated by tool name, since Gemini
                  keys results by function name rather than an id)

Image-bearing tool results are best-effort: the function_response carries any
text payload, and the images are appended as inline_data parts on the same user
turn. Validated/benchmarked in Phase 4 before flipping `llm_provider=gemini`.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.llm.base import (
    LLMResponse,
    Message,
    ToolCall,
)

logger = logging.getLogger(__name__)


def _sdk():
    from google import genai  # google-genai
    from google.genai import types

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env to use llm_provider=gemini."
        )
    return genai.Client(api_key=settings.gemini_api_key), types


def _id_to_name_map(messages: List[Message]) -> Dict[str, str]:
    """tool_use_id -> tool name, harvested from assistant tool_use blocks so we
    can label the matching function_response (Gemini correlates by name)."""
    out: Dict[str, str] = {}
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") == "tool_use":
                out[b["id"]] = b["name"]
    return out


def _parts_for_content(content: Any, types, id_to_name: Dict[str, str]) -> List[Any]:
    if isinstance(content, str):
        return [types.Part(text=content)]
    parts: List[Any] = []
    for b in content:
        btype = b.get("type")
        if btype == "text":
            parts.append(types.Part(text=b["text"]))
        elif btype == "image":
            raw = base64.b64decode(b["data"])
            parts.append(
                types.Part.from_bytes(data=raw, mime_type=b.get("media_type", "image/jpeg"))
            )
        elif btype == "tool_use":
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(name=b["name"], args=b.get("input", {}))
                )
            )
        elif btype == "tool_result":
            name = id_to_name.get(b.get("tool_use_id", ""), "tool")
            text_bits: List[str] = []
            image_parts: List[Any] = []
            for inner in b.get("content", []):
                if inner.get("type") == "text":
                    text_bits.append(inner["text"])
                elif inner.get("type") == "image":
                    raw = base64.b64decode(inner["data"])
                    image_parts.append(
                        types.Part.from_bytes(
                            data=raw, mime_type=inner.get("media_type", "image/jpeg")
                        )
                    )
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response={"text": "\n".join(text_bits) or "see attached frames"},
                    )
                )
            )
            parts.extend(image_parts)
    return parts


def _messages_to_contents(messages: List[Message], types) -> List[Any]:
    id_to_name = _id_to_name_map(messages)
    contents: List[Any] = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        parts = _parts_for_content(m["content"], types, id_to_name)
        contents.append(types.Content(role=role, parts=parts))
    return contents


def _tools_to_gemini(tools: List[Dict[str, Any]], types) -> List[Any]:
    decls = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("input_schema", {"type": "object", "properties": {}}),
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=decls)]


class GeminiClient:
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self._model = model or settings.gemini_model

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
    ) -> LLMResponse:
        client, types = _sdk()
        contents = _messages_to_contents(messages, types)

        config_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = _tools_to_gemini(tools, types)

        resp = client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text_bits: List[str] = []
        tool_calls: List[ToolCall] = []
        assistant_content: List[Dict[str, Any]] = []

        candidate = (resp.candidates or [None])[0]
        parts = []
        if candidate is not None and candidate.content is not None:
            parts = candidate.content.parts or []
        for idx, part in enumerate(parts):
            if getattr(part, "text", None):
                text_bits.append(part.text)
                assistant_content.append({"type": "text", "text": part.text})
            fc = getattr(part, "function_call", None)
            if fc is not None:
                call_id = f"call_{fc.name}_{idx}"
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCall(id=call_id, name=fc.name, input=args))
                assistant_content.append(
                    {"type": "tool_use", "id": call_id, "name": fc.name, "input": args}
                )

        usage: Dict[str, int] = {}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
            }

        return LLMResponse(
            text="".join(text_bits).strip(),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            assistant_message={"role": "assistant", "content": assistant_content},
            usage=usage,
            raw=resp,
        )
