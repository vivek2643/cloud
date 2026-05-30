"""
Filesystem-backed prompt loader.

Every prompt sent to Claude lives in `backend/app/prompts/<name>.md`.
`load(name, **subs)` reads the file fresh on every call so an operator can
edit a prompt and see the change on the next request without restarting
the worker / API.

Read cost is negligible (~0.1ms per call) compared to the Claude round-trip
(~1-3s), so we deliberately do NOT cache.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# `backend/app/services/prompts.py` -> parent.parent = `backend/app`
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptNotFoundError(FileNotFoundError):
    pass


def path_for(name: str) -> Path:
    """Resolve a prompt name (e.g. 'query_parser') to its file path."""
    return PROMPTS_DIR / f"{name}.md"


def load(name: str, **subs: Any) -> str:
    """
    Load a prompt by name and apply `{placeholder}` substitutions.

    Args:
        name: filename stem under `backend/app/prompts/` (without `.md`).
        **subs: keyword args fed to `str.format_map`. Missing placeholders
            raise KeyError so a malformed template fails loudly.

    Returns:
        The fully substituted prompt string.
    """
    p = path_for(name)
    if not p.exists():
        raise PromptNotFoundError(f"Prompt file not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    # ALWAYS run format_map: this collapses `{{` -> `{` / `}}` -> `}` and
    # raises if any unescaped placeholder lacks a substitution.
    return raw.format_map(_StrictDict(subs))


class _StrictDict(dict):
    """Raise on missing keys so an unsubstituted `{foo}` becomes a loud error."""
    def __missing__(self, key):
        raise KeyError(f"Prompt placeholder {{{key}}} has no value supplied")
