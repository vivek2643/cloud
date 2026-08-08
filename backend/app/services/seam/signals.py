"""
seam_function.plan.md §3/§6 -- acquiring the six S(t) input signals.

Reads persisted L1 where available (direct SELECTs, same tables/fields
build_l1_snapshot itself reads -- see module docstring notes below on why
this queries the tables directly rather than calling build_l1_snapshot:
that helper trims `audio_features.onsets_ms` down to a bare count for the
debug log viewer, but this module needs the raw timestamps). Recomputes
`frame_diff` and non-speech onsets from the R2 proxy where L1 has nothing
at all (see the plan's "Honest risks" -- frame_diff isn't in the
motion_dynamics DB schema yet, and there is no separate onset detector
anywhere in L1, only beat-track times gated on `is_musical`). No L1 writes,
ever -- this module is read-only against Postgres and read-only against R2
(one proxy GET).
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services import limits
from app.services.l1.cut_grid_common import normalize_pctl
from app.services.seam.params import NORM_PCTL

logger = logging.getLogger(__name__)


@dataclass
class SeamSignals:
    file_id: str
    hop_ms: int                 # common motion grid step
    n: int                      # number of grid points
    blur_n: List[float] = field(default_factory=list)          # 1 = blurred (g_sharp source)
    act_n: List[float] = field(default_factory=list)            # subject motion (g_gest source)
    cam_n: List[float] = field(default_factory=list)            # camera-vector magnitude, pctl-normalized
    fd_n: List[float] = field(default_factory=list)             # frame_diff, pctl-normalized
    beats_ms: List[int] = field(default_factory=list)           # musical beats (empty if not musical)
    onsets_ms: List[int] = field(default_factory=list)          # non-speech onsets (recomputed)
    onset_strength: List[float] = field(default_factory=list)   # per-onset strength in [0,1]
    is_musical: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)          # provenance: "persisted" | "recomputed"


# --------------------------------------------------------------------------
# Postgres reads (direct, minimal SELECTs -- see module docstring)
# --------------------------------------------------------------------------

def _pg_conn():
    from app.services import db
    return db.connection_dict_row()


def _file_row(file_id: str) -> Optional[dict]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select r2_proxy_key, r2_key, duration_seconds from files where id = %s",
            (file_id,),
        ).fetchone()
    return dict(row) if row else None


def _persisted_motion(file_id: str) -> Optional[dict]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select hop_ms, action_energy, blur, camera_dx, camera_dy, camera_zoom "
            "from motion_dynamics where file_id = %s",
            (file_id,),
        ).fetchone()
    if not row or not row["action_energy"]:
        return None
    return dict(row)


def _persisted_audio(file_id: str) -> Optional[dict]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select is_musical, onsets_ms from audio_features where file_id = %s",
            (file_id,),
        ).fetchone()
    return dict(row) if row else None


def _speech_spans(file_id: str) -> List[Tuple[int, int]]:
    """(start_ms, end_ms) per transcript segment -- non-speech onsets
    falling inside one of these are dropped (§3). No transcript -> no
    spans -> every onset survives (§3's explicit note for non-musical
    clips with no transcript)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from transcripts where file_id = %s",
            (file_id,),
        ).fetchone()
    if not row or not row["segments"]:
        return []
    return [
        (int(seg["start_ms"]), int(seg["end_ms"]))
        for seg in row["segments"]
        if isinstance(seg, dict) and "start_ms" in seg and "end_ms" in seg
    ]


# --------------------------------------------------------------------------
# Proxy acquisition + recompute
# --------------------------------------------------------------------------

def _download_proxy(proxy_key: str, dest_path: str) -> None:
    from app.services.processing import _download_from_r2
    _download_from_r2(proxy_key, dest_path)


def _recompute_motion_bundle(proxy_path: str, duration_ms: int) -> dict:
    """The full motion_dynamics bundle, freshly computed on the proxy --
    blur/action_energy/frame_diff are ALREADY clip-normalize_pctl'd inside
    compute_motion_dynamics itself (same as the persisted fields), so
    callers use them as-is; only the signed camera_dx/dy/zoom need a
    post-hoc magnitude + renormalize (§4)."""
    from app.services.l1.motion_dynamics import compute_motion_dynamics

    md = compute_motion_dynamics(proxy_path, duration_ms=duration_ms or 60000)
    if not md.has_motion:
        raise ValueError("motion recompute produced no signal (decode/opencv failure)")
    return md.to_dict()


def _camera_magnitude_n(dx: List[float], dy: List[float], zoom: List[float]) -> List[float]:
    """Per-hop |dx,dy,zoom| vector magnitude from the signed, un-normalized
    camera vector, then clip-relative normalize_pctl (§4) -- a true renorm
    of the vector, distinct from the persisted `camera_motion` scalar
    (fitted-displacement pctl-normalized, kept as the documented fallback
    but not used here since dx/dy/zoom are always available alongside it)."""
    n = max(len(dx), len(dy), len(zoom))
    mags = []
    for i in range(n):
        d = dx[i] if i < len(dx) else 0.0
        e = dy[i] if i < len(dy) else 0.0
        z = zoom[i] if i < len(zoom) else 0.0
        mags.append(math.sqrt(d * d + e * e + z * z))
    return normalize_pctl(mags, NORM_PCTL)


def _align_to_grid(values: List[float], n: int) -> List[float]:
    """Truncate/zero-pad to exactly ``n`` -- guards the persisted-motion
    fast path, where a freshly recomputed frame_diff's own hop count can
    differ by a frame or two from the persisted grid's length (§9 grid-
    consistency risk)."""
    if len(values) >= n:
        return values[:n]
    return list(values) + [0.0] * (n - len(values))


def _has_audio_stream(video_path: str) -> bool:
    """Cheap ffprobe check -- DJI/aerial-style silent footage has no audio
    stream at all, which makes ffmpeg's audio-only extraction fail with a
    non-zero exit (expected, not an error). Probing first lets that case
    log a clean one-liner instead of a full extraction traceback."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30,
    )
    return bool(out.stdout.strip())


def _extract_wav(video_path: str, wav_path: str) -> None:
    """Mono 16kHz PCM WAV -- the same flags l1/pipeline.py's own
    ``_demux_wav`` uses, duplicated here (not imported) rather than reused
    across modules, so this package stays fully separate from the L1
    pipeline per the plan's §5 namespace isolation."""
    with limits.ffmpeg_slot():
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", wav_path],
            check=True, capture_output=True, timeout=120,
        )


def _in_speech(t_ms: int, speech_spans_ms: List[Tuple[int, int]]) -> bool:
    return any(s <= t_ms <= e for s, e in speech_spans_ms)


def _recompute_onsets(
    wav_path: str, speech_spans_ms: List[Tuple[int, int]],
) -> Tuple[List[int], List[float]]:
    """Non-speech onset timestamps + strength (§3): librosa onset detection
    on a 16kHz mono load of the proxy's audio (mirrors
    audio_features._detect_musicality's own load/onset_strength calls),
    strengths normalize_pctl'd clip-relative (§4), any onset landing inside
    a transcript speech segment dropped (busy dialogue reduces the audio
    attractor -- speech is a separate channel, §3's own note)."""
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        return [], []
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0:
        return [], []
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    if len(onset_frames) == 0:
        return [], []
    onset_times_ms = librosa.frames_to_time(onset_frames, sr=sr) * 1000.0
    raw_strengths = [float(onset_env[f]) for f in onset_frames]
    norm_strengths = normalize_pctl(raw_strengths, NORM_PCTL)

    onsets_ms: List[int] = []
    strengths: List[float] = []
    for t_ms, s in zip(onset_times_ms, norm_strengths):
        t_int = int(round(t_ms))
        if _in_speech(t_int, speech_spans_ms):
            continue
        onsets_ms.append(t_int)
        strengths.append(s)
    return onsets_ms, strengths


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def build_seam_signals(
    file_id: str,
    *,
    proxy_path: Optional[str] = None,
    force_recompute_motion: bool = True,
) -> SeamSignals:
    """Assemble all six S(t) inputs on a common grid: read persisted L1
    where available, recompute frame_diff and non-speech onsets from the
    proxy where L1 has nothing (§3). No L1 writes.

    ``proxy_path``: pass an already-downloaded local proxy (e.g. cutviz.py,
    which already has one) to skip the R2 GET; otherwise fetched here.

    ``force_recompute_motion`` (default True, and the driver's own policy,
    §6): recompute the WHOLE motion bundle (blur/action/camera/frame_diff)
    from the proxy in one compute_motion_dynamics pass, so every motion
    term lives on the SAME grid -- avoids the persisted-vs-recomputed grid
    mismatch risk (§9). Set False for a faster persisted-motion path (skips
    the blur/action/camera recompute, keeping the proxy pass only for
    frame_diff, length-aligned onto the persisted grid) -- useful once
    frame_diff eventually lands in the L1 schema; frame_diff itself is
    recomputed either way, since it is never persisted today.
    """
    file_row = _file_row(file_id)
    if file_row is None:
        raise ValueError(f"{file_id}: file not found")
    duration_ms = int(round((file_row["duration_seconds"] or 0) * 1000))
    proxy_key = file_row["r2_proxy_key"] or file_row["r2_key"]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_proxy = proxy_path
        if local_proxy is None:
            if not proxy_key:
                raise ValueError(f"{file_id}: no proxy/original media available")
            local_proxy = os.path.join(tmpdir, "proxy.mp4")
            _download_proxy(proxy_key, local_proxy)

        meta: Dict[str, Any] = {}
        persisted_motion = _persisted_motion(file_id)

        if force_recompute_motion or not persisted_motion:
            bundle = _recompute_motion_bundle(local_proxy, duration_ms)
            hop_ms = int(bundle["hop_ms"])
            act_n = list(bundle["action_energy"])
            blur_n = list(bundle["blur"])
            cam_n = _camera_magnitude_n(bundle["camera_dx"], bundle["camera_dy"], bundle["camera_zoom"])
            fd_n = list(bundle["frame_diff"])
            n = len(act_n)
            meta.update(blur="recomputed", action_energy="recomputed", camera="recomputed",
                       frame_diff="recomputed")
        else:
            hop_ms = int(persisted_motion["hop_ms"])
            act_n = list(persisted_motion["action_energy"])
            blur_n = list(persisted_motion["blur"])
            cam_n = _camera_magnitude_n(persisted_motion["camera_dx"], persisted_motion["camera_dy"],
                                        persisted_motion["camera_zoom"])
            n = len(act_n)
            bundle = _recompute_motion_bundle(local_proxy, duration_ms)
            fd_n = _align_to_grid(list(bundle["frame_diff"]), n)
            meta.update(blur="persisted", action_energy="persisted", camera="persisted",
                       frame_diff="recomputed (aligned to persisted grid)")

        audio_row = _persisted_audio(file_id)
        is_musical = bool(audio_row and audio_row["is_musical"])
        beats_ms = [int(t) for t in (audio_row["onsets_ms"] or [])] if (is_musical and audio_row) else []
        meta["beats"] = "persisted" if is_musical else "none (not musical)"

        speech_spans_ms = _speech_spans(file_id)
        if not _has_audio_stream(local_proxy):
            logger.info("%s: no audio stream in proxy, skipping onset recompute", file_id)
            onsets_ms, onset_strength = [], []
            meta["onsets"] = "none (no audio track)"
        else:
            wav_path = os.path.join(tmpdir, "audio.wav")
            try:
                _extract_wav(local_proxy, wav_path)
                onsets_ms, onset_strength = _recompute_onsets(wav_path, speech_spans_ms)
                meta["onsets"] = "recomputed"
            except Exception:
                logger.exception("%s: non-speech onset recompute failed (continuing with none)", file_id)
                onsets_ms, onset_strength = [], []
                meta["onsets"] = "failed"

    return SeamSignals(
        file_id=file_id, hop_ms=hop_ms, n=n,
        blur_n=blur_n, act_n=act_n, cam_n=cam_n, fd_n=fd_n,
        beats_ms=beats_ms, onsets_ms=onsets_ms, onset_strength=onset_strength,
        is_musical=is_musical, meta=meta,
    )
