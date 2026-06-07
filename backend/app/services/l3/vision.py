"""
Turn selected keyframes into Anthropic multimodal content blocks.

The editor is otherwise blind: with local L2 VLM captioning deprecated, the
text catalog has no visual description. This module fetches the Layer-B-selected
224x224 keyframe JPEGs from R2 and interleaves them with labels so Claude can
actually SEE each shot and map the image back to its shot_id.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List

from app.services.l3.keyframe_select import SelectedFrame
from app.services.processing import _download_from_r2

logger = logging.getLogger(__name__)


def _fmt_tc(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


def _fetch_b64(r2_key: str) -> str:
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = tmp.name
    try:
        _download_from_r2(r2_key, path)
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("ascii")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_image_blocks(frames: List[SelectedFrame]) -> List[Dict[str, Any]]:
    """Build an interleaved [label-text, image, label-text, image, ...] content
    list. Frames that fail to fetch are skipped. Returns [] if none succeed."""
    blocks: List[Dict[str, Any]] = []
    for idx, fr in enumerate(frames):
        try:
            b64 = _fetch_b64(fr.r2_key)
        except Exception:
            logger.warning("Could not fetch keyframe %s for vision", fr.r2_key)
            continue
        blocks.append({
            "type": "text",
            "text": f"[FRAME {idx} | shot_id={fr.shot_id} | t={_fmt_tc(fr.ts_ms)} | {fr.kind}]",
        })
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    return blocks


def vision_preamble(n_images: int) -> str:
    """A short text block that tells Claude how to read the attached frames."""
    if n_images <= 0:
        return ""
    return (
        f"\nVISUAL FRAMES: {n_images} keyframe image(s) from the candidate shots "
        "are attached below, each preceded by a label of the form "
        "[FRAME i | shot_id=... | t=m:ss | kind]. Use them as your eyes -- judge "
        "composition, subject, motion and continuity from the actual pixels, and "
        "map each image to its shot_id via the label. Not every shot has a frame; "
        "absence of an image is not evidence about a shot.\n"
    )
