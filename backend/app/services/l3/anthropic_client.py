"""
JSON helper on top of the provider-agnostic LLM adapter: get a JSON dict back
from a single system+user call. Provider selection lives in
`app.services.llm` (keyed on `llm_provider`), so this file no longer talks to
any vendor SDK directly.

`_client()` is retained as a thin shim for any legacy import; new code should
use `app.services.llm.get_llm()`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.services.llm import get_llm, text_block, user_message

logger = logging.getLogger(__name__)


def _client():
    """Deprecated: returns the raw Anthropic SDK client. Prefer get_llm()."""
    from anthropic import Anthropic

    from app.config import get_settings

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to use L3 features."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def call_json(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> dict[str, Any]:
    """
    Call the configured LLM and return a parsed JSON dict.

    The model is instructed via the system prompt to emit JSON only. We then
    strip code fences if present and parse. Raises ValueError if the result
    can't be parsed.
    """
    llm = get_llm()
    resp = llm.run(
        system=system_prompt,
        messages=[user_message([text_block(user_prompt)])],
        max_tokens=max_tokens,
    )
    text = resp.text.strip()
    if not text:
        raise ValueError("Empty response from LLM")

    # Strip markdown fences if the model wrapped JSON in them
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return json.loads(text)
