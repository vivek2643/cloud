"""
Render a timeline to a flat preview MP4 using FFmpeg.

Strategy: for each clip, download the proxy (or raw) once, cut it with
`-c copy` for zero-reencode speed, then concat-demux the pieces. Upload the
final MP4 to R2 under `previews/{request_id}.mp4` and return a presigned URL.

This is best-effort. `-c copy` requires consistent codec/container across
sources and can leave keyframe drift; if it fails on any segment we fall
back to a slow re-encode for that segment.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from typing import List

from app.services.processing import _download_from_r2, _upload_to_r2
from app.services.r2 import generate_presigned_get
from app.services.l3.edit_logic_basic import TimelineClip

logger = logging.getLogger(__name__)

PREVIEW_PREFIX = "previews"


def render_preview(clips: List[TimelineClip]) -> str:
    """Render the given timeline and return a presigned MP4 GET URL."""
    if not clips:
        raise ValueError("Empty timeline")

    request_id = uuid.uuid4().hex
    out_key = f"{PREVIEW_PREFIX}/{request_id}.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        # Group clips by source so we download each file at most once.
        downloads: dict[str, str] = {}
        cut_paths: List[str] = []
        for idx, clip in enumerate(clips):
            src_key = clip.file_r2_proxy_key or clip.file_r2_key
            if src_key not in downloads:
                local = os.path.join(tmp, f"src_{len(downloads):04d}.mp4")
                _download_from_r2(src_key, local)
                downloads[src_key] = local
            src_local = downloads[src_key]

            cut_path = os.path.join(tmp, f"cut_{idx:04d}.mp4")
            ok = _cut(src_local, cut_path, clip.source_in_ms, clip.source_out_ms, fast=True)
            if not ok:
                logger.info("Fast cut failed for clip %d; falling back to re-encode", idx)
                ok = _cut(src_local, cut_path, clip.source_in_ms, clip.source_out_ms, fast=False)
            if not ok:
                logger.warning("Skipping unreadable clip %d", idx)
                continue
            cut_paths.append(cut_path)

        if not cut_paths:
            raise RuntimeError("No clips could be cut from source media")

        concat_path = os.path.join(tmp, "concat.txt")
        with open(concat_path, "w") as f:
            for p in cut_paths:
                f.write(f"file '{p}'\n")

        out_path = os.path.join(tmp, "preview.mp4")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_path, "-c", "copy", out_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            # Concat-demux failed (codec mismatch?). Fall back to re-encode merge.
            logger.info("Concat-copy failed; falling back to re-encode merge.")
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    out_path,
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg concat failed: {result.stderr.decode('utf-8', errors='ignore')[-400:]}"
                )

        _upload_to_r2(out_path, out_key, "video/mp4")

    return generate_presigned_get(out_key, expires_in=86400)


def _cut(src: str, dst: str, in_ms: int, out_ms: int, fast: bool) -> bool:
    in_s = max(in_ms / 1000.0, 0.0)
    dur_s = max((out_ms - in_ms) / 1000.0, 0.1)
    if fast:
        cmd = [
            "ffmpeg", "-y", "-ss", str(in_s), "-i", src,
            "-t", str(dur_s),
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            dst,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", str(in_s), "-i", src,
            "-t", str(dur_s),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            dst,
        ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
