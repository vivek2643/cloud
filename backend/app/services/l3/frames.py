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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

from app.services import limits
from app.services.l3.image_plan import PlannedFrame
from app.services.processing import _download_from_r2

FFMPEG_TIMEOUT_S = 30
MAX_PARALLEL_FILES = 8

# Seek fallbacks (ms) when the requested timestamp lands past the last decodable
# frame. A planned/hero frame can sit within a few ms of a clip's end (e.g. an
# action anchor on the final beat); the proxy's real duration is often a hair
# shorter than the source, so `-ss` there yields no frame and ffmpeg errors.
# Nudging earlier grabs the nearest real frame -- content-equivalent for a still.
_SEEK_BACKOFF_MS = (0, 150, 400, 1000, 2000)


def _run_ffmpeg_still(video_path: str, ts_s: float, out_path: str, width: int) -> bool:
    """One ffmpeg still attempt. Returns True iff a non-empty JPEG was written.
    Swallows the CalledProcessError (returns False) so the caller can retry an
    earlier seek; other failures (timeout, missing ffmpeg) still raise."""
    try:
        with limits.ffmpeg_slot():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(max(0.0, ts_s)),
                    "-i", video_path,
                    "-vframes", "1",
                    "-vf", f"scale={width}:-2,format=yuv420p",
                    "-q:v", "3",
                    out_path,
                ],
                check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
            )
    except subprocess.CalledProcessError:
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def extract_still(video_path: str, ts_ms: int, out_path: str, width: int = 768) -> None:
    """Pull one JPEG frame from ``video_path`` at ``ts_ms``, scaled to
    ``width``px wide (height auto, even, yuv420p for safe mjpeg encode).
    Seeks progressively earlier if the requested point is past the last
    decodable frame; raises only if no frame can be pulled at all."""
    ts_s = max(0.0, ts_ms / 1000.0)
    for back_ms in _SEEK_BACKOFF_MS:
        if _run_ffmpeg_still(video_path, ts_s - back_ms / 1000.0, out_path, width):
            return
    raise RuntimeError(
        f"extract_still: no decodable frame at or before {ts_ms}ms in {video_path}"
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


class ProxyCache:
    """Run-scoped, lazy per-file proxy download cache. Within one ingest run
    the SAME clip's proxy is needed twice -- once for pass-2 planning stills
    (``extract_for_planned_frames``) and again for hero frames
    (``ingest._extract_and_upload_heroes``) -- separated by the pass-2 LLM
    calls. This downloads each clip's proxy from R2 at most once and hands
    back the local path for reuse across both phases; the caller still runs
    its own ffmpeg still-extraction per phase (different widths/timestamps),
    so only the R2 GET is deduped -- output is unchanged.

    Files live under ``tmp_dir`` (a caller-owned ``TemporaryDirectory`` scoped
    to the download-reuse span). A file with no proxy key returns ``None`` --
    the same silent skip both call sites already do today. Thread-safe:
    ``extract_for_planned_frames`` pulls stills across files from a
    ``ThreadPoolExecutor`` (one distinct file per thread), so the dict
    check/insert is guarded by a lock; the download itself runs outside the
    lock so distinct files still download concurrently, exactly as the
    per-file ``extract_stills_from_r2`` thread pool did before."""

    def __init__(self, tmp_dir: str, proxy_key_by_file: Dict[str, str]):
        self._tmp_dir = tmp_dir
        self._proxy_key_by_file = proxy_key_by_file
        self._local_by_file: Dict[str, str] = {}
        self._lock = threading.Lock()

    def local_path(self, file_id: str) -> Optional[str]:
        proxy_key = self._proxy_key_by_file.get(file_id)
        if not proxy_key:
            return None
        with self._lock:
            path = self._local_by_file.get(file_id)
        if path is not None:
            return path
        # Distinct file_ids across threads -> at most one thread ever downloads
        # a given file, so the GET runs outside the lock (concurrent across
        # files) and the lock only guards the shared dict.
        local = os.path.join(self._tmp_dir, f"{file_id}.mp4")
        _download_from_r2(proxy_key, local)
        with self._lock:
            self._local_by_file[file_id] = local
        return local


def extract_for_planned_frames(
    planned_frames: List[PlannedFrame], proxy_key_by_file: Dict[str, str], width: int = 768,
    cache: Optional[ProxyCache] = None,
) -> Dict[Tuple[str, int], str]:
    """images_b64 for ``pass2.build_pass2_shard_blocks``: groups frames by
    file so each clip's proxy is downloaded exactly once, however many
    stills it needs. A file missing from ``proxy_key_by_file`` (not yet
    proxied) is silently skipped -- its frames simply won't appear in the
    prompt, same as any other not-yet-resolved image.

    One file's extraction (an R2 download + N ffmpeg subprocess calls) has
    zero shared state with another's, so different files run concurrently
    -- this is pure I/O + subprocess wait, not CPU work competing for the
    GIL, so a thread pool is a straightforward win.

    When ``cache`` is given the R2 download is served by the run-scoped
    ``ProxyCache`` (so hero extraction later reuses the same local proxy
    instead of re-downloading); the ffmpeg still-extraction is unchanged --
    same width, same timestamps -- so the returned base64 stills are
    byte-identical to the no-cache path."""
    by_file: Dict[str, List[int]] = {}
    for f in planned_frames:
        by_file.setdefault(f.file_id, []).append(f.ts_ms)

    jobs = [(fid, ts_list) for fid, ts_list in by_file.items() if proxy_key_by_file.get(fid)]
    if not jobs:
        return {}

    def _stills_for(file_id: str, ts_list: List[int]) -> Dict[int, str]:
        if cache is not None:
            local = cache.local_path(file_id)
            if local is None:
                return {}
            return extract_stills(local, ts_list, width=width)
        return extract_stills_from_r2(proxy_key_by_file[file_id], ts_list, width=width)

    out: Dict[Tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_FILES, len(jobs))) as pool:
        futures = {
            pool.submit(_stills_for, file_id, ts_list): file_id
            for file_id, ts_list in jobs
        }
        for future in futures:
            file_id = futures[future]
            stills = future.result()
            for ts_ms, b64 in stills.items():
                out[(file_id, ts_ms)] = b64
    return out
