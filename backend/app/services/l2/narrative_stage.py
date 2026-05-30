"""
L2 Stage D: Per-shot narrative analysis via a hosted Qwen2.5-VL endpoint.

For Phase 2 v1 we call a hosted multimodal endpoint (Replicate / Anyscale /
Cloudflare AI Gateway) configured via QWEN_VL_ENDPOINT_URL + QWEN_VL_API_KEY.
The endpoint is expected to accept a JSON payload of:
  {
    "images_b64": [<3 base64 strings, anchor/motion/variance>],
    "transcript": "..."
  }
and return JSON of:
  { "description": str, "role": str, "valence": float }

If QWEN_VL_ENDPOINT_URL is empty, this stage is a no-op and the columns stay
null. That keeps Phase 2 useful without forcing a paid VLM dependency from
day one.

A future "Stage D v2" can self-host Qwen2.5-VL-3B quantized when traffic
justifies it; only this file changes.
"""
from __future__ import annotations

import base64
import json
import logging
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
    Run VLM analysis. Returns None when no endpoint is configured AND no
    Anthropic fallback is available (so the caller can decide to skip).
    """
    settings = get_settings()
    imgs_b64 = [b for b in (_encode_b64(p) for p in keyframe_paths) if b]
    if not imgs_b64:
        return None

    if settings.qwen_vl_endpoint_url:
        return _call_qwen(settings.qwen_vl_endpoint_url, settings.qwen_vl_api_key, imgs_b64, transcript_segment)

    if settings.anthropic_api_key:
        return _call_claude_vision(imgs_b64, transcript_segment)

    logger.info("Stage D skipped: no QWEN_VL_ENDPOINT_URL and no ANTHROPIC_API_KEY configured.")
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
