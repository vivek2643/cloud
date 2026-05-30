"""
Video processing via FFmpeg, run as a FastAPI background task.

Handles:
  1. Metadata extraction (duration, resolution) via ffprobe
  2. 1080p proxy generation for smooth browser playback
  3. Thumbnail extraction (single JPEG at ~25% mark)
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
import tempfile

from app.services.r2 import generate_presigned_get, _get_client
from app.services.supabase_client import get_supabase
from app.config import get_settings

logger = logging.getLogger(__name__)


def _download_from_r2(r2_key: str, dest_path: str) -> None:
    settings = get_settings()
    client = _get_client()
    client.download_file(settings.r2_bucket_name, r2_key, dest_path)


def _upload_to_r2(local_path: str, r2_key: str, content_type: str = "application/octet-stream") -> None:
    settings = get_settings()
    client = _get_client()
    client.upload_file(
        local_path,
        settings.r2_bucket_name,
        r2_key,
        ExtraArgs={"ContentType": content_type},
    )


def _probe_video(path: str) -> dict:
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


def process_video(file_id: str, r2_key: str) -> None:
    """
    Full processing pipeline for a single uploaded video.
    Designed to be called from BackgroundTasks.
    """
    sb = get_supabase()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw_video")
            proxy_path = os.path.join(tmpdir, "proxy.mp4")
            thumb_path = os.path.join(tmpdir, "thumb.jpg")

            # 1. Download raw file from R2
            logger.info("Downloading %s for file %s", r2_key, file_id)
            _download_from_r2(r2_key, raw_path)

            # 2. Probe metadata
            logger.info("Probing metadata for %s", file_id)
            probe = _probe_video(raw_path)
            duration = float(probe.get("format", {}).get("duration", 0))
            width, height = 0, 0
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    width = int(stream.get("width", 0))
                    height = int(stream.get("height", 0))
                    break

            # 3. Generate 1080p proxy
            logger.info("Generating proxy for %s", file_id)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", raw_path,
                    "-vf", "scale=-2:1080",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    proxy_path,
                ],
                check=True, capture_output=True,
            )

            proxy_key = f"proxies/{file_id}/proxy.mp4"
            logger.info("Uploading proxy to %s", proxy_key)
            _upload_to_r2(proxy_path, proxy_key, "video/mp4")

            # 4. Extract thumbnail at 25% mark
            logger.info("Extracting thumbnail for %s", file_id)
            thumb_time = max(duration * 0.25, 1.0)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(thumb_time),
                    "-i", raw_path,
                    "-vframes", "1",
                    "-vf", "scale=640:-2",
                    "-q:v", "3",
                    thumb_path,
                ],
                check=True, capture_output=True,
            )

            thumb_key = f"thumbnails/{file_id}/thumb.jpg"
            logger.info("Uploading thumbnail to %s", thumb_key)
            _upload_to_r2(thumb_path, thumb_key, "image/jpeg")

            # 5. Update database -> ready
            sb.table("files").update({
                "r2_proxy_key": proxy_key,
                "r2_thumbnail_key": thumb_key,
                "duration_seconds": duration,
                "width": width,
                "height": height,
                "status": "ready",
            }).eq("id", file_id).execute()

            logger.info("Processing complete for %s", file_id)

    except Exception:
        logger.exception("Processing failed for %s", file_id)
        sb.table("files").update({"status": "failed"}).eq("id", file_id).execute()
