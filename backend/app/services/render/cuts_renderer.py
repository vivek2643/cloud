"""
Cut-only renderer.

Takes an EDL JSON dict + a file lookup table and produces a normalized MP4.

Strategy (Phase 1):
  1. For each clip in the EDL, produce a normalized cached segment via ffmpeg.
     - Cache key includes file_id + source_in/out + target resolution + fps,
       so re-renders of the same EDL slice are nearly free.
     - Normalization: scale to target resolution, force fps, h264, aac, faststart.
  2. Concat the normalized segments via ffmpeg's concat demuxer (one ffmpeg
     invocation, no re-encode -- the cached segments are already aligned).

Hard cuts only. No xfade, no acrossfade, no fade. If cut artifacts show up
in real footage we'll add a 5ms afade per boundary; not yet.

This is intentionally simpler than a single-pass `filter_complex` graph
because the cut cache lets warm renders be near-free. A cold-cache N-clip
render still costs O(N) ffmpeg passes, but each pass is a short trim.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.processing import _download_from_r2, _upload_to_r2
from app.services.r2 import generate_presigned_get

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-clip cache lives outside any individual job tmpdir so it persists across
# renders. /tmp gets cleaned by the OS eventually; that's fine for v1.
CACHE_ROOT = os.environ.get("EDSO_RENDER_CACHE", "/tmp/edso_render_cache")
RENDER_PREFIX = "renders"

# Render presets (cut-only versions; pro-grade presets land in future phases).
PRESETS: Dict[str, Dict[str, Any]] = {
    "preview": {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "video_codec": "libx264",
        "video_preset": "veryfast",
        "video_crf": 24,
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "use_proxy": True,
    },
    "export_landscape": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "video_codec": "libx264",
        "video_preset": "medium",
        "video_crf": 20,
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "use_proxy": False,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """File-level info the renderer needs to fetch source media."""
    file_id: str
    r2_key: str
    r2_proxy_key: Optional[str]


def render_edl(
    edl: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    preset: str = "preview",
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, int]:
    """
    Render the given EDL to MP4 and upload it to R2.

    Args:
        edl: EDL JSON dict (must validate against edl.store.validate_edl).
        file_lookup: shot_file_id -> FileEntry. Each clip's shot_id is mapped
            to a file_id by the caller (we need shots metadata to do that
            join, which lives in claude_editor / query_executor).
        preset: name in PRESETS.
        progress_cb: optional callable(percent_int, label) for progress logging.

    Returns:
        (output_r2_key, duration_ms)
    """
    if not edl.get("clips"):
        raise ValueError("EDL has no clips to render.")
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset!r}")
    cfg = PRESETS[preset]

    os.makedirs(CACHE_ROOT, exist_ok=True)

    def report(pct: int, label: str) -> None:
        logger.info("render: %d%% %s", pct, label)
        if progress_cb is not None:
            try:
                progress_cb(pct, label)
            except Exception:
                logger.exception("progress_cb raised; ignoring")

    report(2, "starting")

    # ---- Phase 1: ensure each clip has a cached normalized segment --------
    clip_segments: List[str] = []
    total_dur_ms = 0
    n = len(edl["clips"])
    src_cache: Dict[str, str] = {}  # r2_key -> local path within this render

    with tempfile.TemporaryDirectory(prefix="edso_render_") as tmp:
        for i, clip in enumerate(edl["clips"]):
            file_id = _resolve_file_id_for_clip(clip, file_lookup)
            entry = file_lookup.get(file_id)
            if entry is None:
                raise RuntimeError(
                    f"clips[{i}] references file_id={file_id} but no FileEntry was provided"
                )

            in_ms = int(clip["source_in_ms"])
            out_ms = int(clip["source_out_ms"])
            dur_ms = max(0, out_ms - in_ms)
            total_dur_ms += dur_ms

            cache_path = _segment_cache_path(
                file_id=file_id,
                in_ms=in_ms,
                out_ms=out_ms,
                width=cfg["width"],
                height=cfg["height"],
                fps=cfg["fps"],
            )
            if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
                # Cache miss -- need source media on local disk first.
                src_key = (entry.r2_proxy_key if cfg["use_proxy"] and entry.r2_proxy_key
                           else entry.r2_key)
                if src_key not in src_cache:
                    label = f"download {os.path.basename(src_key)}"
                    report(_pct(i, n, 5, 35), label)
                    src_local = os.path.join(tmp, f"src_{len(src_cache):04d}.mp4")
                    _download_from_r2(src_key, src_local)
                    src_cache[src_key] = src_local
                src_local = src_cache[src_key]

                report(_pct(i, n, 35, 75), f"cut clip {i + 1}/{n}")
                _produce_segment(
                    src=src_local,
                    dst=cache_path,
                    in_ms=in_ms,
                    out_ms=out_ms,
                    cfg=cfg,
                )
            else:
                report(_pct(i, n, 35, 75), f"clip {i + 1}/{n} from cache")
            clip_segments.append(cache_path)

        report(78, "concatenating")
        out_local = os.path.join(tmp, "out.mp4")
        _concat_segments(clip_segments, out_local, cfg)

        report(90, "uploading")
        out_key = f"{RENDER_PREFIX}/{uuid.uuid4().hex}.mp4"
        _upload_to_r2(out_local, out_key, "video/mp4")

    report(100, "done")
    return out_key, total_dur_ms


def presigned_url_for(out_key: str, expires_in: int = 86400) -> str:
    return generate_presigned_get(out_key, expires_in=expires_in)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_file_id_for_clip(clip: Dict[str, Any], file_lookup: Dict[str, FileEntry]) -> str:
    """
    A clip in the EDL points at a shot_id, but the renderer works on file_ids.
    For ergonomics we accept either:
      - clip["file_id"] is set (denormalized cache), OR
      - the caller has already pre-resolved shot_id->file_id and stored it
        in file_lookup keyed on the shot_id (less common)
    The standard caller (edl_runner) builds file_lookup keyed on file_id and
    sets clip["file_id"] before calling. We honor that here.
    """
    fid = clip.get("file_id")
    if fid:
        return str(fid)
    sid = clip.get("shot_id")
    if sid and sid in file_lookup:
        return sid
    raise RuntimeError(
        f"Clip is missing a resolvable file reference: id={clip.get('id')!r}"
    )


def _segment_cache_path(*, file_id: str, in_ms: int, out_ms: int,
                        width: int, height: int, fps: int) -> str:
    key = f"{file_id}|{in_ms}|{out_ms}|{width}x{height}|{fps}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_ROOT, f"{digest}.mp4")


def _produce_segment(*, src: str, dst: str, in_ms: int, out_ms: int, cfg: Dict[str, Any]) -> None:
    """
    Produce one normalized clip via ffmpeg.

    We use input-side -ss for fast seek, then re-encode video so the segment
    is keyframe-aligned and the concat step doesn't need to re-encode. Audio
    is also re-encoded for sample-rate normalization.
    """
    in_s = max(in_ms / 1000.0, 0.0)
    dur_s = max((out_ms - in_ms) / 1000.0, 0.05)

    # Use a sibling .part.mp4 file so ffmpeg can infer the muxer from the
    # extension. We rename atomically into the cache slot on success.
    tmp_path = dst.replace(".mp4", ".part.mp4") if dst.endswith(".mp4") else (dst + ".part.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(in_s),
        "-i", src,
        "-t", str(dur_s),
        # Video: scale to target while preserving aspect via padding (letterbox);
        # SAR=1 + setdar so concat demuxer doesn't reject mismatched aspects.
        "-vf",
        f"scale=w={cfg['width']}:h={cfg['height']}:force_original_aspect_ratio=decrease,"
        f"pad={cfg['width']}:{cfg['height']}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={cfg['fps']}",
        "-c:v", cfg["video_codec"],
        "-preset", cfg["video_preset"],
        "-crf", str(cfg["video_crf"]),
        "-pix_fmt", "yuv420p",
        # Audio: normalize codec/sample rate/channel count.
        "-c:a", cfg["audio_codec"],
        "-b:a", cfg["audio_bitrate"],
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        # Avoid stale negative timestamps from input-side seek.
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        tmp_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        tail = proc.stderr.decode("utf-8", errors="ignore")[-800:]
        raise RuntimeError(f"ffmpeg cut failed for {os.path.basename(src)} {in_ms}-{out_ms}ms:\n{tail}")
    # Atomic move into cache.
    shutil.move(tmp_path, dst)


def _concat_segments(segments: List[str], dst: str, cfg: Dict[str, Any]) -> None:
    """
    Concat segments via the demuxer. Because every segment is normalized to
    the same codec/fps/resolution/sample rate, -c copy works reliably.
    """
    list_path = dst + ".list"
    with open(list_path, "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            dst,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            tail = proc.stderr.decode("utf-8", errors="ignore")[-800:]
            # Fallback: full re-encode merge (slower, but always works if
            # segments somehow desynced -- shouldn't happen since we
            # normalized them, but defense in depth).
            logger.warning("concat -c copy failed; falling back to re-encode merge.\n%s", tail)
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c:v", cfg["video_codec"],
                "-preset", cfg["video_preset"],
                "-crf", str(cfg["video_crf"]),
                "-pix_fmt", "yuv420p",
                "-c:a", cfg["audio_codec"],
                "-b:a", cfg["audio_bitrate"],
                "-ar", "48000",
                "-ac", "2",
                "-movflags", "+faststart",
                dst,
            ]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                tail = proc.stderr.decode("utf-8", errors="ignore")[-800:]
                raise RuntimeError(f"ffmpeg concat re-encode failed:\n{tail}")
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def _pct(i: int, n: int, lo: int, hi: int) -> int:
    if n <= 0:
        return lo
    return int(lo + (hi - lo) * (i / n))
