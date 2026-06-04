"""
L2 Stage D: Per-shot narrative analysis via Qwen2.5-VL.

VLM selection order (see `analyze_shot`):
  1. Hosted Qwen endpoint  -- if QWEN_VL_ENDPOINT_URL is set (explicit override)
  2. Self-hosted Qwen2.5-VL on the worker GPU -- default; see services/l2/qwen_vl.py
  3. Claude vision (Anthropic) -- fallback when no GPU / Qwen fails to load

The model returns JSON of: { "description": str, "role": str, "valence": float }.

Self-hosting Qwen on the GPU replaces ~N per-shot Claude API calls (network
latency + per-call cost) with local inference, which is the dominant cost of
L2. Claude remains a zero-config fallback for CPU-only environments.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import httpx

from app.config import get_settings
from app.services import prompts as prompts_mod
from app.services.l3.anthropic_client import call_json

logger = logging.getLogger(__name__)

# The Claude vision prompt template lives in backend/app/prompts/narrative_stage.md.
# Edit the file directly; the next L2 run picks up the change without restart.
PROMPT_NAME = "narrative_stage"


@dataclass
class NarrativeResult:
    description: str
    role: str           # 'setup' | 'payoff' | 'aside' | 'reaction' | 'transition'
    valence: float      # -1..1


ROLE_VALUES = {"setup", "payoff", "aside", "reaction", "transition"}


def _encode_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        logger.exception("Could not read keyframe %s", path)
        return None


def analyze_shot(
    keyframe_paths: List[str],
    transcript_segment: Optional[str],
) -> Optional[NarrativeResult]:
    """
    Run VLM analysis. Tries self-hosted Qwen on the GPU first, then a hosted
    Qwen endpoint, then Claude. Returns None when nothing is available.
    """
    settings = get_settings()
    valid_paths = [p for p in keyframe_paths if p and os.path.exists(p)]
    if not valid_paths:
        return None

    # 1. Explicit hosted Qwen endpoint override.
    if settings.qwen_vl_endpoint_url:
        imgs_b64 = [b for b in (_encode_b64(p) for p in valid_paths) if b]
        if imgs_b64:
            return _call_qwen(settings.qwen_vl_endpoint_url, settings.qwen_vl_api_key, imgs_b64, transcript_segment)

    # 2. Self-hosted Qwen2.5-VL on the worker GPU (the default fast path).
    if settings.qwen_vl_local:
        res = _call_qwen_local(valid_paths, transcript_segment)
        if res is not None:
            return res
        # else fall through to Claude (no GPU, or Qwen failed to load)

    # 3. Claude vision fallback.
    if settings.anthropic_api_key:
        imgs_b64 = [b for b in (_encode_b64(p) for p in valid_paths) if b]
        if imgs_b64:
            return _call_claude_vision(imgs_b64, transcript_segment)

    logger.info("Stage D skipped: no Qwen GPU/endpoint and no ANTHROPIC_API_KEY.")
    return None


def _call_qwen_local(
    image_paths: List[str],
    transcript: Optional[str],
) -> Optional[NarrativeResult]:
    """Self-hosted Qwen2.5-VL on the GPU. Returns None if unavailable/failed so
    the caller can fall back to Claude."""
    try:
        from app.services.l2.qwen_vl import _QwenVLEngine

        if not _QwenVLEngine.available():
            return None
        settings = get_settings()
        prompt_text = prompts_mod.load(PROMPT_NAME, transcript=transcript or "(none)")
        data = _QwenVLEngine.infer(image_paths, prompt_text, settings.qwen_vl_max_tokens)
        if not data:
            return None
        return _coerce(data)
    except Exception:
        logger.exception("Local Qwen narrative call failed")
        return None


def _call_qwen(
    url: str,
    api_key: str,
    imgs_b64: List[str],
    transcript: Optional[str],
) -> Optional[NarrativeResult]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json={
                "images_b64": imgs_b64,
                "transcript": transcript or "",
            })
            r.raise_for_status()
            data = r.json()
    except Exception:
        logger.exception("Qwen endpoint call failed")
        return None
    return _coerce(data)


def _call_claude_vision(
    imgs_b64: List[str],
    transcript: Optional[str],
) -> Optional[NarrativeResult]:
    """
    Fallback: use Claude (multimodal) when Qwen isn't configured.
    Slightly more expensive per call but works out of the box.
    """
    settings = get_settings()
    from anthropic import Anthropic
    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        content: list = []
        for b64 in imgs_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
        prompt_text = prompts_mod.load(PROMPT_NAME, transcript=transcript or "(none)")
        content.append({"type": "text", "text": prompt_text})
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=400,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        return _coerce(json.loads(text))
    except Exception:
        logger.exception("Claude vision narrative call failed")
        return None


def _coerce(data: dict) -> Optional[NarrativeResult]:
    try:
        role = str(data.get("role", "")).strip().lower()
        if role not in ROLE_VALUES:
            role = "aside"
        return NarrativeResult(
            description=str(data.get("description", "")).strip()[:500],
            role=role,
            valence=max(-1.0, min(1.0, float(data.get("valence", 0.0)))),
        )
    except Exception:
        return None
