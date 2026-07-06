"""
Cuts v3: JPEG still extraction for ingest -- pass-2 planning images and final
hero frames alike. Pure ffmpeg off the proxy, no model cost. Mirrors the
single-frame extraction already used for file thumbnails in
``l1/pipeline.py`` (seek with ``-ss``, ``-vframes 1``), scaled to a chosen
width.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Tuple

from app.services.l3.image_plan import PlannedFrame
from app.services.processing import _download_from_r2

FFMPEG_TIMEOUT_S = 30
MAX_PARALLEL_FILES = 8


def extract_still(video_path: str, ts_ms: int, out_path: str, width: int = 768) -> None:
    """Pull one JPEG frame from ``video_path`` at ``ts_ms``, scaled to
    ``width``px wide (height auto, even, yuv420p for safe mjpeg encode).
    Raises on ffmpeg failure -- no silent fallback."""
    ts_s = max(0.0, ts_ms / 1000.0)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(ts_s),
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={width}:-2,format=yuv420p",
            "-q:v", "3",
            out_path,
        ],
        check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
    )


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def extract_stills(video_path: str, ts_list: Iterable[int], width: int = 768) -> Dict[int, str]:
    """One still per distinct ``ts_ms`` in ``ts_list``, from an already-local
    video file. Returns {ts_ms: base64 jpeg}."""
    unique = sorted(set(ts_list))
    out: Dict[int, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, ts_ms in enumerate(unique):
            still_path = os.path.join(tmp, f"still_{i}.jpg")
            extract_still(video_path, ts_ms, still_path, width=width)
            out[ts_ms] = file_to_b64(still_path)
    return out


def extract_stills_from_r2(proxy_r2_key: str, ts_list: Iterable[int], width: int = 768) -> Dict[int, str]:
    """Download the proxy once, then pull every requested still from the
    local copy -- one R2 GET total per clip, however many frames it needs."""
    with tempfile.TemporaryDirectory() as tmp:
        proxy_path = os.path.join(tmp, "proxy.mp4")
        _download_from_r2(proxy_r2_key, proxy_path)
        return extract_stills(proxy_path, ts_list, width=width)


def extract_for_planned_frames(
    planned_frames: List[PlannedFrame], proxy_key_by_file: Dict[str, str], width: int = 768,
) -> Dict[Tuple[str, int], str]:
    """images_b64 for ``pass2.build_pass2_shard_blocks``: groups frames by
    file so each clip's proxy is downloaded exactly once, however many
    stills it needs. A file missing from ``proxy_key_by_file`` (not yet
    proxied) is silently skipped -- its frames simply won't appear in the
    prompt, same as any other not-yet-resolved image.

    One file's extraction (an R2 download + N ffmpeg subprocess calls) has
    zero shared state with another's, so different files run concurrently
    -- this is pure I/O + subprocess wait, not CPU work competing for the
    GIL, so a thread pool is a straightforward win."""
    by_file: Dict[str, List[int]] = {}
    for f in planned_frames:
        by_file.setdefault(f.file_id, []).append(f.ts_ms)

    jobs = [(fid, ts_list, proxy_key_by_file[fid])
           for fid, ts_list in by_file.items() if proxy_key_by_file.get(fid)]
    if not jobs:
        return {}

    out: Dict[Tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_FILES, len(jobs))) as pool:
        futures = {
            pool.submit(extract_stills_from_r2, proxy_key, ts_list, width): file_id
            for file_id, ts_list, proxy_key in jobs
        }
        for future in futures:
            file_id = futures[future]
            stills = future.result()
            for ts_ms, b64 in stills.items():
                out[(file_id, ts_ms)] = b64
    return out
