"""
Thin wrapper around the Anthropic SDK with one helper: get a JSON response
constrained to a schema using Claude's strict JSON output mode.

Keeping this in its own file so swapping providers (Bedrock, Vertex, etc.)
is a one-file change.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def _client():
    from anthropic import Anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to use L3 features."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def call_json(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> dict[str, Any]:
    """
    Call Claude and return a parsed JSON dict.

    Claude is instructed via the system prompt to emit JSON only. We then
    strip code fences if present and parse. Raises ValueError if the result
    can't be parsed.
    """
    settings = get_settings()
    client = _client()
    msg = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    ).strip()
    if not text:
        raise ValueError("Empty response from Claude")

    # Strip markdown fences if Claude wrapped JSON in them
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return json.loads(text)
