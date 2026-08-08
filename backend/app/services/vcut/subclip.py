"""
pass1_video_input.plan.md section 2/3 -- cut a file's non-speech spans into
ONE low-res, target-fps, NO-audio sub-clip via ffmpeg trim+concat, plus the
pure time-map back to original-file ms. This is Pass 1's video INPUT prep
(orchestrate.py calls this, per file, then hands the result + an uploaded
Files API handle to pass1.py); nothing here calls any LLM.

Reuses the generic R2->local proxy download primitive
(app.services.processing._download_from_r2, the same one app.services.l3.
frames/vcut.sampling already ride) rather than re-implementing R2 auth --
principle 7's isolation constraint (generic I/O primitives are fair game,
L3 business logic is not).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.services import limits
from app.services.vcut.params import SUBCLIP_MAX_MS, SUBCLIP_MIN_MS

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT_S = 120


@dataclass
class SubClip:
    path: str                              # local temp mp4 -- caller uploads, then cleanup_subclip()
    segments: List[Tuple[int, int, int]]   # (orig_start_ms, orig_end_ms, sub_start_ms), in sub-clip order
    duration_ms: int                       # total sub-clip duration (== sum of segment lengths)


# --------------------------------------------------------------------------
# Pure time-map helpers (section 3) -- unit-tested with synthetic segments,
# no I/O.
# --------------------------------------------------------------------------

def map_sub_to_orig(sub_ms: int, segments: List[Tuple[int, int, int]]) -> Optional[int]:
    """Sub-clip ms -> original-file ms. None if it lands in no segment
    (shouldn't happen for an in-range t from a well-formed sub-clip)."""
    for orig_s, orig_e, sub_s in segments:
        seg_len = orig_e - orig_s
        if sub_s <= sub_ms < sub_s + seg_len:
            return orig_s + (sub_ms - sub_s)
    return None


def is_near_join(sub_ms: int, segments: List[Tuple[int, int, int]], guard_ms: int) -> bool:
    """True if sub_ms is within guard_ms of an INTERNAL concat boundary --
    the hard cut where two spliced non-speech spans meet is a stitching
    ARTIFACT, not a real moment. The sub-clip's own start (segments[0]'s
    sub_start, always 0) and end are real footage boundaries, not splices,
    so only segments[1:]'s own sub_start values count as joins -- a
    single-segment clip (no concatenation happened at all) has none."""
    joins = [sub_s for _orig_s, _orig_e, sub_s in segments[1:]]
    return any(abs(sub_ms - j) <= guard_ms for j in joins)


def _build_segments(spans_ms: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    segments: List[Tuple[int, int, int]] = []
    sub_start = 0
    for s, e in spans_ms:
        segments.append((s, e, sub_start))
        sub_start += e - s
    return segments


def _apply_max_cap(spans_ms: List[Tuple[int, int]], max_ms: int) -> List[Tuple[int, int]]:
    """Keep spans in order until the cumulative total would exceed
    ``max_ms``, truncating the boundary span to land exactly on the cap
    rather than dropping it wholesale (SUBCLIP_MAX_MS is a rare safety net
    against a pathological all-non-speech file, not a normal-path knob --
    fps is the main cost lever)."""
    capped: List[Tuple[int, int]] = []
    running = 0
    for s, e in spans_ms:
        if running >= max_ms:
            break
        if running + (e - s) > max_ms:
            e = s + (max_ms - running)
        if e <= s:
            break
        capped.append((s, e))
        running += e - s
    return capped


# --------------------------------------------------------------------------
# ffmpeg trim + concat (section 2) -- I/O, no LLM.
# --------------------------------------------------------------------------

def _build_filter_complex(spans_ms: List[Tuple[int, int]], width_px: int, fps: float) -> str:
    trims = []
    for i, (s, e) in enumerate(spans_ms):
        trims.append(f"[0:v]trim=start={s / 1000.0:.3f}:end={e / 1000.0:.3f},setpts=PTS-STARTPTS[v{i}]")
    concat_inputs = "".join(f"[v{i}]" for i in range(len(spans_ms)))
    parts = trims + [
        f"{concat_inputs}concat=n={len(spans_ms)}:v=1:a=0[c]",
        f"[c]scale={width_px}:-2,fps={fps},format=yuv420p[outv]",
    ]
    return ";".join(parts)


def _run_ffmpeg_subclip(proxy_path: str, spans_ms: List[Tuple[int, int]], out_path: str,
                        width_px: int, fps: float) -> bool:
    """One ffmpeg attempt (no retry -- a failure here just means this file
    falls back to frames, section 9). Returns True iff a non-empty mp4 was
    written."""
    filter_complex = _build_filter_complex(spans_ms, width_px, fps)
    try:
        with limits.ffmpeg_slot():
            subprocess.run(
                ["ffmpeg", "-y", "-i", proxy_path, "-filter_complex", filter_complex,
                 "-map", "[outv]", "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-movflags", "+faststart", out_path],
                check=True, capture_output=True, timeout=_FFMPEG_TIMEOUT_S,
            )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", "replace")[-2000:]
        logger.warning("vcut subclip: ffmpeg trim+concat failed (rc=%s): %s", e.returncode, stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("vcut subclip: ffmpeg trim+concat timed out after %ss", _FFMPEG_TIMEOUT_S)
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def cut_non_speech_subclip(
    proxy_key: str, non_speech_spans: List[Tuple[int, int]], *, fps: float, width_px: int,
) -> Optional[SubClip]:
    """Fetch the proxy, ffmpeg trim+concat its non-speech spans into one
    low-res/target-fps/no-audio clip. None (caller falls back to frames,
    section 9) when: there are no usable spans, the total non-speech is
    below SUBCLIP_MIN_MS (too little to bother cutting/uploading), the R2
    fetch fails, or ffmpeg fails. Always cleans up the downloaded proxy;
    the returned SubClip's own local file is the CALLER's to clean up
    (via cleanup_subclip) once it's been uploaded."""
    from app.services.processing import _download_from_r2

    spans = [(s, e) for s, e in non_speech_spans if e > s]
    if not spans:
        return None
    total_ms = sum(e - s for s, e in spans)
    if total_ms < SUBCLIP_MIN_MS:
        return None
    spans = _apply_max_cap(spans, SUBCLIP_MAX_MS)
    total_ms = sum(e - s for s, e in spans)

    tmp_dir = tempfile.mkdtemp(prefix="vcut_subclip_")
    proxy_path = os.path.join(tmp_dir, "proxy.mp4")
    out_path = os.path.join(tmp_dir, "subclip.mp4")
    try:
        _download_from_r2(proxy_key, proxy_path)
    except Exception:
        logger.exception("vcut subclip: failed to download proxy %s", proxy_key)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    ok = _run_ffmpeg_subclip(proxy_path, spans, out_path, width_px, fps)
    try:
        os.remove(proxy_path)
    except OSError:
        pass
    if not ok:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    return SubClip(path=out_path, segments=_build_segments(spans), duration_ms=total_ms)


def probe_proxy_dims(proxy_key: str) -> Optional[Tuple[int, int]]:
    """(width, height) of the proxy's own video stream, via a presigned-URL
    ffprobe -- no local download (a faststart mp4, which both the L1 proxy
    encode and this module's own subclip output use, only needs its header
    read, and R2/S3 support HTTP range requests either way).

    reframe_vcut_geometry.plan.md section 2: this is deliberately the
    PROXY's own dims, not ``files.width``/``files.height`` (probed from the
    RAW upload before L1's proxy encode runs, so they can be pre-rotation
    and mismatched with the proxy's actual pixel layout). The proxy is also
    already upright -- l1.pipeline._encode_proxy auto-applies rotation
    during transcode -- so no rotation reading is needed here; see store.py
    for where rotation_deg is set (always 0.0, with that reasoning spelled
    out there too).

    None on any probe failure (framing then falls back to a centered crop
    for that file -- never a hard failure, matching the rest of this
    module's fail-open-to-frames contract)."""
    from app.services.processing import _probe_video
    from app.services.r2 import generate_presigned_get

    try:
        url = generate_presigned_get(proxy_key)
        probe = _probe_video(url)
    except Exception:
        logger.exception("vcut subclip: failed to probe proxy dims for %s", proxy_key)
        return None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            w, h = int(stream.get("width", 0) or 0), int(stream.get("height", 0) or 0)
            if w > 0 and h > 0:
                return w, h
    return None


def cleanup_subclip(sub: Optional[SubClip]) -> None:
    """Best-effort local temp-dir cleanup, called once the clip has been
    uploaded (or the upload attempt has already failed) -- never worth
    raising over."""
    if sub is None:
        return
    try:
        shutil.rmtree(os.path.dirname(sub.path), ignore_errors=True)
    except Exception:
        logger.warning("vcut subclip: failed to clean up temp dir for %s", sub.path)
