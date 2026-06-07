"""
On-demand keyframe service for the agentic (blind) director.

The director sees only text up front and pulls keyframes via the `view_frames`
tool when it needs to look. This module is the supply side:

  available_frames(shot_ids) -> {shot_id: [FrameRef]}   (what CAN be seen)
  fetch_images(refs, max_total, caption_for) -> [neutral blocks]  (fetch+caption)
  run_view_frames(...)        -> (blocks, n_images, note)  (tool execution)
  VIEW_FRAMES_TOOL            -> neutral tool spec

Frames come from `shot_keyframes` (<=8 adaptive frames/shot, migration 008)
with a legacy `shots.keyframe_r2_key` fallback. Downloads hit a small
process-local cache and run in parallel to hide R2 latency.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import psycopg

from app.config import get_settings
from app.services.llm import image_block, text_block, tool_spec
from app.services.processing import _download_from_r2

logger = logging.getLogger(__name__)


@dataclass
class FrameRef:
    shot_id: str
    kind: str
    ts_ms: int
    r2_key: str
    frame_index: int


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _fmt_tc(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def available_frames(shot_ids: List[str]) -> Dict[str, List[FrameRef]]:
    """Map each shot_id to its available keyframes (adaptive set preferred,
    legacy anchor as fallback). Shots with no stored frame are absent."""
    out: Dict[str, List[FrameRef]] = {}
    if not shot_ids:
        return out
    with _pg() as conn:
        try:
            rows = conn.execute(
                """
                select sk.shot_id, sk.kind, sk.ts_ms, sk.r2_key, sk.frame_index
                  from shot_keyframes sk
                 where sk.shot_id = any(%s::uuid[]) and sk.r2_key is not null
                 order by sk.shot_id, sk.frame_index
                """,
                (shot_ids,),
            ).fetchall()
        except psycopg.errors.UndefinedTable:
            rows = []
        for sid, kind, ts, key, fidx in rows:
            out.setdefault(str(sid), []).append(
                FrameRef(str(sid), kind or "frame", int(ts or 0), key, int(fidx or 0))
            )

        missing = [s for s in shot_ids if s not in out]
        if missing:
            legacy = conn.execute(
                """
                select s.id, s.keyframe_r2_key, s.start_ms, s.end_ms
                  from shots s
                 where s.id = any(%s::uuid[]) and s.keyframe_r2_key is not null
                """,
                (missing,),
            ).fetchall()
            for sid, key, sms, ems in legacy:
                mid = (int(sms or 0) + int(ems or 0)) // 2
                out[str(sid)] = [FrameRef(str(sid), "anchor", mid, key, 0)]
    return out


# ---------------------------------------------------------------------------
# Fetch (cached, parallel)
# ---------------------------------------------------------------------------

_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 1024


def _cache_get(key: str) -> Optional[str]:
    with _CACHE_LOCK:
        return _CACHE.get(key)


def _cache_put(key: str, b64: str) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            # Cheap eviction: drop an arbitrary entry. Frames are re-fetchable.
            _CACHE.pop(next(iter(_CACHE)), None)
        _CACHE[key] = b64


def _fetch_one_b64(r2_key: str) -> Optional[str]:
    cached = _cache_get(r2_key)
    if cached is not None:
        return cached
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
        _download_from_r2(r2_key, path)
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        _cache_put(r2_key, b64)
        return b64
    except Exception:
        logger.warning("frame_service: failed to fetch %s", r2_key)
        return None
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def _default_caption(ref: FrameRef) -> str:
    return f"[shot={ref.shot_id} | t={_fmt_tc(ref.ts_ms)} | {ref.kind}]"


def fetch_images(
    refs: List[FrameRef],
    max_total: int,
    caption_for: Optional[Callable[[FrameRef], str]] = None,
) -> List[Dict]:
    """Fetch up to `max_total` frames in parallel and interleave
    [caption, image] neutral blocks. Failed fetches are skipped."""
    refs = refs[: max(0, max_total)]
    if not refs:
        return []
    caption_for = caption_for or _default_caption
    with ThreadPoolExecutor(max_workers=8) as ex:
        b64s = list(ex.map(lambda r: _fetch_one_b64(r.r2_key), refs))
    blocks: List[Dict] = []
    for ref, b64 in zip(refs, b64s):
        if not b64:
            continue
        blocks.append(text_block(caption_for(ref)))
        blocks.append(image_block(b64))
    return blocks


# ---------------------------------------------------------------------------
# view_frames tool
# ---------------------------------------------------------------------------

VIEW_FRAMES_TOOL = tool_spec(
    name="view_frames",
    description=(
        "Look at actual keyframe images for one or more targets. Targets are "
        "unit labels (e.g. 'U7') or shot ids. Use this whenever pixels would "
        "change your decision: to verify framing/sharpness of a clean line, to "
        "understand a busy or ambiguous moment, to check visual continuity "
        "between candidate cuts, or to confirm an assembled clip. Be frugal but "
        "look enough to be sure. Returns the images inline."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Unit labels (U0, U7, ...) or shot ids to view.",
            },
            "max_per_shot": {
                "type": "integer",
                "description": "Optional cap on frames per shot (default: all available).",
            },
            "reason": {
                "type": "string",
                "description": "One short phrase: why you need to look now.",
            },
        },
        "required": ["targets"],
    },
)


def run_view_frames(
    *,
    targets: List[str],
    resolve_target: Callable[[str], List[str]],
    caption_for: Optional[Callable[[FrameRef], str]] = None,
    per_shot_max: int,
    max_total: int,
) -> tuple[List[Dict], int, str]:
    """Execute a view_frames call.

    resolve_target(target) -> list of shot_ids the target refers to (a unit may
    span multiple shots; a shot id resolves to itself). Returns
    (image_blocks, n_images, note). Enforces per_shot_max and max_total.
    """
    # Resolve targets -> ordered unique shot_ids.
    shot_ids: List[str] = []
    seen = set()
    unresolved: List[str] = []
    for t in targets or []:
        resolved = resolve_target(str(t)) or []
        if not resolved:
            unresolved.append(str(t))
        for sid in resolved:
            if sid not in seen:
                seen.add(sid)
                shot_ids.append(sid)

    avail = available_frames(shot_ids)

    refs: List[FrameRef] = []
    no_frames: List[str] = []
    for sid in shot_ids:
        frames = avail.get(sid) or []
        if not frames:
            no_frames.append(sid)
            continue
        refs.extend(frames[: max(1, per_shot_max)])

    blocks = fetch_images(refs, max_total=max_total, caption_for=caption_for)
    n_images = sum(1 for b in blocks if b.get("type") == "image")

    note_bits = [f"Showing {n_images} frame(s) across {len(shot_ids)} shot(s)."]
    if unresolved:
        note_bits.append(f"Could not resolve: {', '.join(unresolved[:6])}.")
    if no_frames:
        note_bits.append(f"No stored frames for {len(no_frames)} shot(s).")
    if n_images == 0:
        note_bits.append("No images available; decide from the text catalog.")
    return blocks, n_images, " ".join(note_bits)
