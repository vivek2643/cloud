"""
Short video+audio CLIP extraction for the voice-ID pass (identity/voice_id.py,
voice_id_pass.plan.md) -- mirrors frames.py's R2-download-once + thread-pool
pattern, but emits short MP4 clips (video WITH audio) instead of a silent
still, so the model can judge lip-sync against the heard audio directly
instead of guessing whose mouth moved from a burst of stills.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from app.services.processing import _download_from_r2

FFMPEG_TIMEOUT_S = 30
# Video+audio encoding is heavier than a single still grab -- keep this
# pool's R2/ffmpeg footprint small, since it runs CONCURRENTLY with
# frames.py's own pool (Pass 2's image extraction) during ingest.
MAX_PARALLEL_FILES = 3


def _run_ffmpeg_clip(video_path: str, start_ms: int, end_ms: int, out_path: str, width: int) -> bool:
    """One ffmpeg clip attempt. Returns True iff a non-empty MP4 was written."""
    start_s = max(0.0, start_ms / 1000.0)
    dur_s = max(0.05, (end_ms - start_ms) / 1000.0)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(start_s), "-i", video_path, "-t", str(dur_s),
                "-vf", f"scale={width}:-2,format=yuv420p",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-ac", "1",
                "-movflags", "+faststart",
                out_path,
            ],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.CalledProcessError:
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def extract_clip(video_path: str, start_ms: int, end_ms: int, out_path: str, width: int = 512) -> bool:
    """Pull one short video+audio clip from `video_path` covering
    [start_ms, end_ms), scaled to `width`px wide. Returns False (never
    raises) when the window can't be decoded -- a single bad clip must not
    fail the whole voice-ID pass, unlike a hero still where the caller has
    no fallback."""
    return _run_ffmpeg_clip(video_path, start_ms, end_ms, out_path, width)


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def extract_clips(video_path: str, requests: List[Any], width: int = 512) -> Dict[str, str]:
    """One clip per request (duck-typed: `.clip_id`/`.start_ms`/`.end_ms`),
    from an already-local video file. Returns {clip_id: base64 mp4}. A
    request whose window can't be decoded is silently skipped -- never a
    fabricated/blank clip."""
    out: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, req in enumerate(requests):
            clip_path = os.path.join(tmp, f"clip_{i}.mp4")
            if extract_clip(video_path, req.start_ms, req.end_ms, clip_path, width=width):
                out[req.clip_id] = file_to_b64(clip_path)
    return out


def extract_clips_from_r2(proxy_r2_key: str, requests: List[Any], width: int = 512) -> Dict[str, str]:
    """Download the proxy once, then pull every requested clip from the
    local copy -- one R2 GET total per file, however many clips it needs."""
    with tempfile.TemporaryDirectory() as tmp:
        proxy_path = os.path.join(tmp, "proxy.mp4")
        _download_from_r2(proxy_r2_key, proxy_path)
        return extract_clips(proxy_path, requests, width=width)


def extract_for_clip_requests(
    requests: List[Any], proxy_key_by_file: Dict[str, str], width: int = 512,
) -> Dict[str, str]:
    """{clip_id: base64 mp4} for every request (duck-typed: `.file_id`/
    `.clip_id`/`.start_ms`/`.end_ms`), grouped by file so each proxy is
    downloaded exactly once. A file missing from `proxy_key_by_file` is
    silently skipped -- its clips simply won't appear, same fail-open
    contract as `frames.extract_for_planned_frames`."""
    by_file: Dict[str, List[Any]] = {}
    for r in requests:
        by_file.setdefault(r.file_id, []).append(r)

    jobs = [(fid, reqs, proxy_key_by_file[fid])
           for fid, reqs in by_file.items() if proxy_key_by_file.get(fid)]
    if not jobs:
        return {}

    out: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_FILES, len(jobs))) as pool:
        futures = [pool.submit(extract_clips_from_r2, proxy_key, reqs, width)
                  for _fid, reqs, proxy_key in jobs]
        for future in futures:
            out.update(future.result())
    return out
