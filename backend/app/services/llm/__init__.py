"""
Provider-agnostic LLM adapter.

Every model call (the auto-editor's director/editor passes, JSON helpers)
routes through this package so switching providers is a config change
(`llm_provider`) rather than a rewrite.

Public surface:
  get_llm()            -> an LLMClient for the configured provider
  LLMClient / LLMResponse / ToolCall
  neutral block builders: text_block, image_block, tool_use_block,
    tool_result_block, user_message, assistant_message, tool_spec
"""
from app.services.llm.base import (
    LLMClient,
    LLMResponse,
    ToolCall,
    assistant_message,
    image_block,
    text_block,
    tool_result_block,
    tool_spec,
    tool_use_block,
    user_message,
)
from app.services.llm.factory import get_llm

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ToolCall",
    "get_llm",
    "text_block",
    "image_block",
    "tool_use_block",
    "tool_result_block",
    "user_message",
    "assistant_message",
    "tool_spec",
]
