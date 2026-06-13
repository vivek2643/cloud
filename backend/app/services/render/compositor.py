"""
Render the RESOLVED layer set to an MP4.

Two paths, picked automatically:

  * Pure spine (no operations -> every layer is `kind="spine"`): normalize each
    spine segment with ffmpeg (cached per file+range+preset) and concat with the
    demuxer. Warm renders are near-free. This is the cut-only fast path.

  * Layered (coverage / beds / split edits / ducking): one `filter_complex`
    graph. Video layers are composited bottom->top by z over a black canvas
    (each shifted to its program start; opacity via alpha); audio layers are
    delayed to their program start, gained (gain+duck dB), and summed with amix.

Both consume `layers.ResolvedTimeline.to_dict()` so the render and the preview
can never disagree about what plays.

Hard cuts only -- no transitions/speed/text yet (those are later phases and
slot in as extra filter nodes without changing this contract).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.processing import _download_from_r2, _upload_to_r2
from app.services.r2 import generate_presigned_get

logger = logging.getLogger(__name__)

CACHE_ROOT = os.environ.get("EDSO_RENDER_CACHE", "/tmp/edso_render_cache")
RENDER_PREFIX = "renders"

PRESETS: Dict[str, Dict[str, Any]] = {
    "preview": {
        "width": 1280, "height": 720, "fps": 30,
        "video_codec": "libx264", "video_preset": "veryfast", "video_crf": 24,
        "audio_codec": "aac", "audio_bitrate": "128k", "use_proxy": True,
    },
    "export": {
        "width": 1920, "height": 1080, "fps": 30,
        "video_codec": "libx264", "video_preset": "medium", "video_crf": 20,
        "audio_codec": "aac", "audio_bitrate": "192k", "use_proxy": False,
    },
}

ProgressCb = Optional[Callable[[int, str], None]]


@dataclass
class FileEntry:
    file_id: str
    r2_key: str
    r2_proxy_key: Optional[str]
    has_video: bool = True


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def resolved_hash(resolved: Dict[str, Any], preset: str) -> str:
    """Stable fingerprint of (timeline, preset) for render de-dup / caching."""
    payload = json.dumps(resolved, sort_keys=True) + "|" + preset
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def render_resolved(
    resolved: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    preset: str = "preview",
    progress_cb: ProgressCb = None,
) -> Tuple[str, int]:
    """Render `resolved` to MP4, upload to R2, return (output_r2_key, duration_ms)."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; known: {list(PRESETS)}")
    cfg = PRESETS[preset]
    os.makedirs(CACHE_ROOT, exist_ok=True)

    video = sorted(resolved.get("video_layers") or [], key=lambda v: (v["prog_start_ms"], v["z"]))
    audio = sorted(resolved.get("audio_layers") or [], key=lambda a: a["prog_start_ms"])
    total_ms = int(resolved.get("duration_ms") or 0)
    if not video and not audio:
        raise ValueError("Nothing to render: resolved timeline is empty.")
    if total_ms <= 0:
        total_ms = max([v["prog_end_ms"] for v in video] + [a["prog_end_ms"] for a in audio] + [0])
    if total_ms <= 0:
        raise ValueError("Resolved timeline has zero duration.")

    def report(pct: int, label: str) -> None:
        logger.info("render: %d%% %s", pct, label)
        if progress_cb:
            try:
                progress_cb(pct, label)
            except Exception:
                logger.exception("render progress_cb raised; ignoring")

    pure_spine = (
        all(v.get("kind") == "spine" for v in video)
        and all(a.get("kind") == "spine" for a in audio)
        and len(audio) <= len(video)  # spine: one dialogue layer per picture
    )

    with tempfile.TemporaryDirectory(prefix="edso_render_") as tmp:
        if pure_spine and video:
            out_key = _render_spine_concat(video, cfg, file_lookup, tmp, report)
        else:
            out_key = _render_layers(video, audio, total_ms, cfg, file_lookup, tmp, report)
    report(100, "done")
    return out_key, total_ms


def presigned_url_for(out_key: str, expires_in: int = 86400) -> str:
    return generate_presigned_get(out_key, expires_in=expires_in)


# --------------------------------------------------------------------------
# Source media
# --------------------------------------------------------------------------

def _source_path(
    entry: FileEntry, cfg: Dict[str, Any], tmp: str, cache: Dict[str, str]
) -> str:
    key = (entry.r2_proxy_key if cfg["use_proxy"] and entry.r2_proxy_key else entry.r2_key)
    if key not in cache:
        ext = os.path.splitext(key)[1] or ".mp4"
        local = os.path.join(tmp, f"src_{len(cache):04d}{ext}")
        _download_from_r2(key, local)
        cache[key] = local
    return cache[key]


# --------------------------------------------------------------------------
# Pure-spine fast path: per-segment normalized cache + concat demuxer
# --------------------------------------------------------------------------

def _render_spine_concat(
    video: List[dict],
    cfg: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    tmp: str,
    report: Callable[[int, str], None],
) -> str:
    report(4, "starting (spine)")
    src_cache: Dict[str, str] = {}
    segments: List[str] = []
    n = len(video)
    for i, v in enumerate(video):
        entry = file_lookup.get(v["source_file_id"])
        if entry is None:
            raise RuntimeError(f"no FileEntry for {v['source_file_id']}")
        in_ms, out_ms = int(v["src_in_ms"]), int(v["src_out_ms"])
        cache_path = _segment_cache_path(v["source_file_id"], in_ms, out_ms, cfg)
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            src = _source_path(entry, cfg, tmp, src_cache)
            report(_pct(i, n, 8, 78), f"cut {i + 1}/{n}")
            _produce_segment(src=src, dst=cache_path, in_ms=in_ms, out_ms=out_ms, cfg=cfg)
        else:
            report(_pct(i, n, 8, 78), f"cut {i + 1}/{n} (cache)")
        segments.append(cache_path)

    report(82, "concatenating")
    out_local = os.path.join(tmp, "out.mp4")
    _concat_segments(segments, out_local, cfg)
    report(92, "uploading")
    out_key = f"{RENDER_PREFIX}/{uuid.uuid4().hex}.mp4"
    _upload_to_r2(out_local, out_key, "video/mp4")
    return out_key


def _segment_cache_path(file_id: str, in_ms: int, out_ms: int, cfg: Dict[str, Any]) -> str:
    key = f"{file_id}|{in_ms}|{out_ms}|{cfg['width']}x{cfg['height']}|{cfg['fps']}|{cfg['video_crf']}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_ROOT, f"{digest}.mp4")


def _vf_normalize(cfg: Dict[str, Any]) -> str:
    return (
        f"scale=w={cfg['width']}:h={cfg['height']}:force_original_aspect_ratio=decrease,"
        f"pad={cfg['width']}:{cfg['height']}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={cfg['fps']}"
    )


def _produce_segment(*, src: str, dst: str, in_ms: int, out_ms: int, cfg: Dict[str, Any]) -> None:
    in_s = max(in_ms / 1000.0, 0.0)
    dur_s = max((out_ms - in_ms) / 1000.0, 0.05)
    tmp_path = dst.replace(".mp4", ".part.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{in_s:.3f}", "-i", src, "-t", f"{dur_s:.3f}",
        "-vf", _vf_normalize(cfg),
        "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
        "-pix_fmt", "yuv420p",
        "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-avoid_negative_ts", "make_zero", "-fflags", "+genpts",
        tmp_path,
    ]
    _run_ffmpeg(cmd, tmp_path, f"cut {os.path.basename(src)} {in_ms}-{out_ms}ms")
    shutil.move(tmp_path, dst)


def _concat_segments(segments: List[str], dst: str, cfg: Dict[str, Any]) -> None:
    list_path = dst + ".list"
    with open(list_path, "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
               "-c", "copy", "-movflags", "+faststart", dst]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not _ok(dst):
            logger.warning("concat copy failed; re-encoding merge.")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
                   "-pix_fmt", "yuv420p", "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"],
                   "-ar", "48000", "-ac", "2", "-movflags", "+faststart", dst]
            _run_ffmpeg(cmd, dst, "concat re-encode")
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Layered path: one filter_complex graph (overlay stack + audio mix)
# --------------------------------------------------------------------------

def _render_layers(
    video: List[dict],
    audio: List[dict],
    total_ms: int,
    cfg: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    tmp: str,
    report: Callable[[int, str], None],
) -> str:
    report(6, "preparing sources")
    src_cache: Dict[str, str] = {}
    total_s = total_ms / 1000.0
    W, H, FPS = cfg["width"], cfg["height"], cfg["fps"]

    inputs: List[str] = []          # ffmpeg -i paths, one per layer
    filt: List[str] = []
    # Black base canvas for the full program duration.
    filt.append(f"color=c=black:s={W}x{H}:r={FPS}:d={total_s:.3f},format=yuv420p[base]")

    # --- video layers: trim -> normalize -> alpha -> shift to program start ---
    cur = "[base]"
    for i, v in enumerate(video):
        entry = file_lookup.get(v["source_file_id"])
        if entry is None:
            raise RuntimeError(f"no FileEntry for {v['source_file_id']}")
        idx = len(inputs)
        inputs.append(_source_path(entry, cfg, tmp, src_cache))
        in_s = int(v["src_in_ms"]) / 1000.0
        out_s = int(v["src_out_ms"]) / 1000.0
        ps = int(v["prog_start_ms"]) / 1000.0
        opacity = float(v.get("opacity", 1.0))
        chain = (
            f"[{idx}:v]trim=start={in_s:.3f}:end={out_s:.3f},setpts=PTS-STARTPTS,"
            f"{_vf_normalize(cfg)},format=yuva420p"
        )
        if opacity < 0.999:
            chain += f",colorchannelmixer=aa={max(0.0, min(1.0, opacity)):.3f}"
        # Shift the layer's timeline so it composites at its program start.
        chain += f",setpts=PTS+{ps:.3f}/TB[v{i}]"
        filt.append(chain)
        out = f"[vt{i}]"
        filt.append(f"{cur}[v{i}]overlay=eof_action=pass:format=auto{out}")
        cur = out
    video_out = cur

    # --- audio layers: trim -> delay to program start -> gain -> mix ---
    audio_labels: List[str] = []
    for j, a in enumerate(audio):
        entry = file_lookup.get(a["source_file_id"])
        if entry is None:
            raise RuntimeError(f"no FileEntry for {a['source_file_id']}")
        idx = len(inputs)
        inputs.append(_source_path(entry, cfg, tmp, src_cache))
        in_s = int(a["src_in_ms"]) / 1000.0
        out_s = int(a["src_out_ms"]) / 1000.0
        delay_ms = max(0, int(a["prog_start_ms"]))
        gain_db = float(a.get("gain_db", 0.0)) + float(a.get("duck_db", 0.0))
        chain = (
            f"[{idx}:a]atrim=start={in_s:.3f}:end={out_s:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}"
        )
        if gain_db <= -119:
            chain += ",volume=0"
        elif abs(gain_db) > 0.01:
            chain += f",volume={gain_db:.2f}dB"
        chain += f"[a{j}]"
        filt.append(chain)
        audio_labels.append(f"[a{j}]")

    has_audio = bool(audio_labels)
    if has_audio:
        if len(audio_labels) == 1:
            # Single bus: still clamp to the program duration.
            filt.append(f"{audio_labels[0]}apad,atrim=0:{total_s:.3f}[aout]")
        else:
            filt.append(
                "".join(audio_labels)
                + f"amix=inputs={len(audio_labels)}:normalize=0:dropout_transition=0,"
                + f"atrim=0:{total_s:.3f}[aout]"
            )

    # Trim the composited video to the exact program duration.
    filt.append(f"{video_out}trim=0:{total_s:.3f},setpts=PTS-STARTPTS[vout]")

    report(20, "compositing")
    out_local = os.path.join(tmp, "out.mp4")
    cmd: List[str] = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(filt), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]"]
    cmd += [
        "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", "48000", "-ac", "2"]
    cmd += ["-movflags", "+faststart", "-t", f"{total_s:.3f}", out_local]

    _run_ffmpeg(cmd, out_local, "filter_complex composite")
    report(92, "uploading")
    out_key = f"{RENDER_PREFIX}/{uuid.uuid4().hex}.mp4"
    _upload_to_r2(out_local, out_key, "video/mp4")
    return out_key


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _run_ffmpeg(cmd: List[str], out_path: str, what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not _ok(out_path):
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        tail = proc.stderr.decode("utf-8", errors="ignore")[-1200:]
        raise RuntimeError(f"ffmpeg failed ({what}):\n{tail}")


def _pct(i: int, n: int, lo: int, hi: int) -> int:
    if n <= 0:
        return lo
    return int(lo + (hi - lo) * (i / n))
