"""
Thin Gemini wrapper for whole-video perception.

Unlike app.services.llm.gemini_client (text/tool-call chat for the old editor),
this sends an entire video file to Gemini and asks for one structured-JSON
answer. Videos are pushed through the Files API (not inline) because clips can
be tens of MB and the Files API is the supported path for video; the upload is
deleted again as soon as the call returns.

Kept dependency-light and lazily imported so importing this module never pulls
in google-genai on hosts that don't run L2.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type

from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)


def _sdk():
    from google import genai
    from google.genai import types

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env to run the L2 perception pass."
        )
    return genai.Client(api_key=settings.gemini_api_key), types


def _media_resolution(types, value: str):
    """Map the config string to the SDK enum; None lets the model use its default."""
    table = {
        "low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
    }
    name = table.get((value or "").lower())
    if not name:
        return None
    return getattr(types.MediaResolution, name, None)


def _state_name(file_obj) -> str:
    state = getattr(file_obj, "state", None)
    # SDK returns an enum (with .name) in newer versions, a plain string in others.
    return getattr(state, "name", None) or str(state or "")


@dataclass
class VideoAnalysisResult:
    parsed: Optional[BaseModel]
    raw_text: str
    usage: dict
    model: str


def analyze_video(
    *,
    video_path: str,
    mime_type: str,
    system_instruction: str,
    prompt: str,
    response_schema: Type[BaseModel],
) -> VideoAnalysisResult:
    """Upload `video_path`, run one structured-JSON perception call, clean up.

    Returns the parsed pydantic instance (best effort) plus the raw text and
    token usage. Raises on hard SDK/transport failures so procrastinate retries.
    """
    settings = get_settings()
    client, types = _sdk()

    model = settings.l2_gemini_model
    uploaded = None
    try:
        logger.info("L2: uploading %s to Gemini Files API", video_path)
        uploaded = client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(mime_type=mime_type),
        )

        # The Files API ingests video asynchronously; we can't reference it until
        # it flips ACTIVE. Poll with a hard ceiling so a stuck upload fails the
        # job instead of hanging the worker.
        deadline = time.monotonic() + settings.l2_file_active_timeout_seconds
        while _state_name(uploaded) == "PROCESSING":
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Gemini file stayed PROCESSING > {settings.l2_file_active_timeout_seconds}s"
                )
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if _state_name(uploaded) != "ACTIVE":
            raise RuntimeError(f"Gemini file upload failed, state={_state_name(uploaded)}")

        video_part = types.Part(
            file_data=types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            video_metadata=types.VideoMetadata(fps=settings.l2_video_fps),
        )

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "max_output_tokens": settings.l2_max_output_tokens,
            "temperature": 0.2,
        }
        media_res = _media_resolution(types, settings.l2_media_resolution)
        if media_res is not None:
            config_kwargs["media_resolution"] = media_res

        logger.info("L2: requesting perception from %s", model)
        resp = client.models.generate_content(
            model=model,
            contents=[video_part, types.Part(text=prompt)],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        parsed = _extract_parsed(resp, response_schema)
        raw_text = getattr(resp, "text", "") or ""

        usage: dict = {}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
            }

        return VideoAnalysisResult(parsed=parsed, raw_text=raw_text, usage=usage, model=model)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                logger.warning("L2: failed to delete uploaded Gemini file %s", getattr(uploaded, "name", "?"))


def _extract_parsed(resp, response_schema: Type[BaseModel]) -> Optional[BaseModel]:
    """Prefer the SDK's auto-parsed instance; fall back to parsing the raw text."""
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, response_schema):
        return parsed
    text = getattr(resp, "text", None)
    if not text:
        return None
    try:
        return response_schema.model_validate_json(text)
    except Exception:
        logger.exception("L2: could not validate Gemini JSON against the schema")
        return None
