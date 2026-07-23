"""
Shared media I/O helpers: R2 download/upload + ffprobe metadata.

The actual video analysis lives in `app.services.l1.pipeline` (the staged L1
orchestrator). This module is just the thin filesystem/probe layer underneath
it, reused by L1 ingest, L2 perception, the render compositor, and the
maintenance scripts.
"""

from __future__ import annotations
import json
import subprocess

from app.services import limits
from app.services.r2 import _get_client
from app.config import get_settings


def _download_from_r2(r2_key: str, dest_path: str) -> None:
    settings = get_settings()
    client = _get_client()
    with limits.r2_slot():
        client.download_file(settings.r2_bucket_name, r2_key, dest_path)


def _upload_to_r2(local_path: str, r2_key: str, content_type: str = "application/octet-stream") -> None:
    settings = get_settings()
    client = _get_client()
    with limits.r2_slot():
        client.upload_file(
            local_path,
            settings.r2_bucket_name,
            r2_key,
            ExtraArgs={"ContentType": content_type},
        )


def _probe_video(path: str) -> dict:
    with limits.ffmpeg_slot():
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                path,
            ],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)
