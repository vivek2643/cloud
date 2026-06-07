"""
LLM provider factory. `get_llm()` returns a client for the configured
`llm_provider` (default "anthropic"). Callers depend only on the neutral
LLMClient surface, so flipping providers is a config change.
"""
from __future__ import annotations

from typing import Optional

from app.config import get_settings
from app.services.llm.base import LLMClient


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    settings = get_settings()
    name = (provider or settings.llm_provider or "anthropic").lower()

    if name == "anthropic":
        from app.services.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model)
    if name == "gemini":
        from app.services.llm.gemini_client import GeminiClient

        return GeminiClient(model=model)

    raise ValueError(
        f"Unknown llm_provider {name!r}. Expected 'anthropic' or 'gemini'."
    )
