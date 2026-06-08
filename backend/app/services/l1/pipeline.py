"""
L1 orchestrator (procrastinate task).

Downloads the raw video once, runs all five stages, persists results to
Supabase Postgres. Each stage is idempotent: on retry it checks
processing_jobs and skips stages already marked done. The single
declared task name `l1_orchestrate` is what `/upload/.../complete` enqueues.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import traceback
from typing import List, Optional

import numpy as np
import psycopg
from procrastinate import RetryStrategy

from app.config import get_settings
from app.services import audit_log
from app.services.jobs import app
from app.services.l1 import audio_features as af_mod
from app.services.l1 import diarization as diar_mod
from app.services.l1 import embeddings as emb_mod
from app.services.l1 import keyframes as kf_mod
from app.services.l1 import shots as shots_mod
from app.services.l1 import transcript as tr_mod
from app.services.l1.snapshot import build_l1_snapshot
from app.services.processing import _download_from_r2, _probe_video, _upload_to_r2
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

STAGES = ("proxy", "transcript", "shots", "embeddings", "audio_features", "diarization")


def _pg_conn() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


# Cached per-process: whether the Layer A shot_keyframes table exists. Lets L1
# run cleanly before migration 008 is applied (writes just no-op).
_SHOT_KEYFRAMES_OK: Optional[bool] = None


def _shot_keyframes_available(conn: psycopg.Connection) -> bool:
    global _SHOT_KEYFRAMES_OK
    if _SHOT_KEYFRAMES_OK is None:
        try:
            row = conn.execute("select to_regclass('public.shot_keyframes')").fetchone()
            _SHOT_KEYFRAMES_OK = bool(row and row[0] is not None)
        except Exception:
            _SHOT_KEYFRAMES_OK = False
        if not _SHOT_KEYFRAMES_OK:
            logger.warning(
                "shot_keyframes table missing; Layer A adaptive keyframes disabled "
                "(apply migration 008_adaptive_keyframes.sql)."
            )
    return _SHOT_KEYFRAMES_OK


def _vec_to_pg(v: np.ndarray) -> str:
    """Format a numpy vector as pgvector text literal `[v1,v2,...]`.
    Works for both `vector` and `halfvec` via SQL `::halfvec` cast.
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


# --- Per-stage job tracking ----------------------------------------------

def _stage_status(conn: psycopg.Connection, file_id: str, stage: str) -> Optional[str]:
    cur = conn.execute(
        "select status from processing_jobs where file_id = %s and stage = %s",
        (file_id, stage),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _stage_begin(conn: psycopg.Connection, file_id: str, stage: str) -> None:
    conn.execute(
        """
        insert into processing_jobs (file_id, stage, status, started_at, attempts)
        values (%s, %s, 'running', now(), 1)
        on conflict (file_id, stage) do update set
            status = 'running',
            started_at = now(),
            attempts = processing_jobs.attempts + 1,
            error = null
        """,
        (file_id, stage),
    )


def _stage_done(conn: psycopg.Connection, file_id: str, stage: str) -> None:
    conn.execute(
        """
        update processing_jobs
           set status = 'done', finished_at = now(), error = null
         where file_id = %s and stage = %s
        """,
        (file_id, stage),
    )


def _stage_fail(conn: psycopg.Connection, file_id: str, stage: str, err: str) -> None:
    conn.execute(
        """
        update processing_jobs
           set status = 'failed', finished_at = now(), error = %s
         where file_id = %s and stage = %s
        """,
        (err[:8000], file_id, stage),
    )


def _set_l1_status(file_id: str, status: str) -> None:
    sb = get_supabase()
    sb.table("files").update({"l1_status": status}).eq("id", file_id).execute()


def _file_exists(file_id: str) -> bool:
    """True if the files row still exists. A user can delete a file (or abort a
    multipart upload) mid-processing; without this guard the shots/embeddings
    inserts hit a foreign-key violation and retry forever."""
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select 1 from files where id = %s", (file_id,)
            ).fetchone()
        return row is not None
    except Exception:
        # If we can't tell, assume it exists and let normal error handling run.
        return True


def _enqueue_l2_if_needed(file_id: str) -> None:
    """Chain deep L2 enrichment after a successful L1 run so every video gets
    enriched in the background without any manual trigger.

    Best-effort: uses a short-lived App/connector so it never collides with the
    worker's own (already-open, async) procrastinate app, and is skipped when
    L2 is already running/ready (L2 itself is idempotent, but this avoids
    redundant queue churn on an L1 re-run).
    """
    if not get_settings().enable_l2_vlm:
        # Deep L2 (Qwen VLM / faces / dinov2) is disabled by default -- edit-time
        # managed multimodal vision replaces per-shot pre-captioning.
        return
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select l2_status from files where id = %s", (file_id,)
            ).fetchone()
        l2 = row[0] if row else None
        if l2 in ("running", "ready"):
            logger.info("L2 already %s for %s; not re-enqueuing.", l2, file_id)
            return

        from procrastinate import App, PsycopgConnector

        # Tiny pool: this runs inside an L1 worker that already holds a pool, and
        # Supabase's session pooler caps total clients (see jobs.DB_POOL_MAX).
        enqueue_app = App(connector=PsycopgConnector(
            conninfo=get_settings().database_url, min_size=1, max_size=2))
        with enqueue_app.open():
            # queue="gpu" so the GPU fleet (not CPU render workers) picks it up.
            enqueue_app.configure_task("l2_enrich_file", queue="gpu").defer(file_id=file_id)
        logger.info("Auto-enqueued L2 enrichment for %s.", file_id)
    except Exception:
        logger.exception("L1 done but auto-enqueue of L2 failed for %s.", file_id)


# --- Stage 1: proxy + thumb (refactor of existing process_video) ---------

def _gpu_available() -> bool:
    try:
        from app.services.ml_device import torch_device
        return torch_device() == "cuda"
    except Exception:
        return False


def _has_audio_stream(raw_path: str) -> bool:
    """True if the file carries at least one audio stream. Silent footage
    (screen recordings, some drone/action clips) skips demux + transcript +
    audio-features instead of crashing."""
    try:
        probe = _probe_video(raw_path)
        return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    except Exception:
        return False


# Bound the proxy to fit inside this box, preserving aspect (any orientation),
# never upscaling small sources, and forcing even dimensions.
_PROXY_SCALE = (
    "scale='min(1920,iw)':'min(1080,ih)':"
    "force_original_aspect_ratio=decrease:force_divisible_by=2"
)
# HDR (PQ/HLG, bt2020) -> SDR bt709. Needs an ffmpeg built with zscale (libzimg);
# if absent the attempt fails and we retry without it.
_TONEMAP = (
    "zscale=transfer=linear:npl=100,tonemap=hable:desat=0,"
    "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
)


def _normalize_vf(is_hdr: bool, tonemap: bool) -> str:
    """Build the normalization filter chain. Always lands on 8-bit yuv420p at
    <=1080p; optionally tonemaps HDR first."""
    parts = []
    if is_hdr and tonemap:
        parts.append(_TONEMAP)
    parts.append(_PROXY_SCALE)
    parts.append("format=yuv420p")
    return ",".join(parts)


# Upper bound on any single ffmpeg invocation so a wedged encode kills the job
# (procrastinate then retries) instead of hanging a worker indefinitely.
FFMPEG_TIMEOUT_S = 2 * 60 * 60


def _run_ffmpeg(cmd: list) -> tuple[bool, bytes]:
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
        return True, b""
    except subprocess.TimeoutExpired:
        return False, b"ffmpeg timed out"
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or b"")[-400:]


def _encode_proxy(raw_path: str, proxy_path: str, is_hdr: bool = False) -> None:
    """Normalize ANY input into a uniform, edit-safe 1080p H.264 proxy:
    8-bit yuv420p, constant frame rate, bounded resolution (any orientation),
    rotation auto-applied, HDR tonemapped to SDR. Tries NVENC then libx264, and
    drops HDR tonemapping if this ffmpeg build can't do it -- so exotic uploads
    (10-bit, HDR, VFR, rotated) still produce a working proxy instead of failing."""
    common_out = [
        "-vsync", "cfr",  # normalize VFR (phone footage) to constant frame rate
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        proxy_path,
    ]

    # (use_nvenc, tonemap) attempts, most-preferred first. We degrade encoder
    # (NVENC -> CPU) and, for HDR, degrade the filter (tonemap -> plain) so a
    # missing zimg build never blocks the proxy.
    attempts: list[tuple[bool, bool]] = []
    if _gpu_available():
        attempts.append((True, True))
        attempts.append((True, False))
    attempts.append((False, True))
    attempts.append((False, False))

    last_err = b""
    seen: set[tuple[bool, bool]] = set()
    for use_nvenc, tonemap in attempts:
        key = (use_nvenc, tonemap and is_hdr)
        if key in seen:
            continue
        seen.add(key)

        vf = _normalize_vf(is_hdr, tonemap)
        cmd = ["ffmpeg", "-y"]
        if use_nvenc:
            cmd += ["-hwaccel", "auto"]
        cmd += ["-i", raw_path, "-vf", vf]
        if use_nvenc:
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "25"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        cmd += common_out

        ok, err = _run_ffmpeg(cmd)
        if ok:
            return
        last_err = err
        logger.warning(
            "Proxy encode attempt failed (nvenc=%s tonemap=%s): %s",
            use_nvenc, tonemap and is_hdr, err,
        )

    raise RuntimeError(f"All proxy encode attempts failed: {last_err!r}")


def _is_hdr_stream(stream: dict) -> bool:
    """Detect HDR/wide-gamut transfer so we know to tonemap."""
    transfer = (stream.get("color_transfer") or "").lower()
    primaries = (stream.get("color_primaries") or "").lower()
    return transfer in {"smpte2084", "arib-std-b67"} or primaries == "bt2020"


def _stage1_proxy(file_id: str, raw_path: str, tmpdir: str) -> tuple[float, int, int]:
    """Probe + 1080p proxy + thumbnail. Returns (duration, w, h)."""
    sb = get_supabase()
    proxy_path = os.path.join(tmpdir, "proxy.mp4")
    thumb_path = os.path.join(tmpdir, "thumb.jpg")

    probe = _probe_video(raw_path)
    duration = float(probe.get("format", {}).get("duration", 0))
    width, height = 0, 0
    is_hdr = False
    has_video = False
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            has_video = True
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            is_hdr = _is_hdr_stream(stream)
            break

    if not has_video:
        # Audio-only file or a corrupt/non-video upload -- fail fast and clearly
        # rather than letting ffmpeg die deep in the encode with a cryptic error.
        raise ValueError("Uploaded file contains no decodable video stream")

    _encode_proxy(raw_path, proxy_path, is_hdr=is_hdr)
    proxy_key = f"proxies/{file_id}/proxy.mp4"
    _upload_to_r2(proxy_path, proxy_key, "video/mp4")

    # Thumbnail straight off the proxy: it's already normalized (8-bit, SDR,
    # rotation baked), so mjpeg never trips over exotic source pixel formats.
    thumb_time = max(duration * 0.25, 1.0)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(thumb_time),
            "-i", proxy_path,
            "-vframes", "1",
            "-vf", "scale=640:-2,format=yuv420p",
            "-q:v", "3",
            thumb_path,
        ],
        check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
    )
    thumb_key = f"thumbnails/{file_id}/thumb.jpg"
    _upload_to_r2(thumb_path, thumb_key, "image/jpeg")

    sb.table("files").update({
        "r2_proxy_key": proxy_key,
        "r2_thumbnail_key": thumb_key,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "status": "ready",
    }).eq("id", file_id).execute()

    return duration, width, height


# --- WAV demux for stages 2 & 5 ------------------------------------------

def _demux_wav(raw_path: str, out_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", raw_path,
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            out_path,
        ],
        check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
    )


# --- Stage 2: transcript -------------------------------------------------

def _stage2_transcript(file_id: str, wav_path: str, conn: psycopg.Connection) -> None:
    result = tr_mod.transcribe(wav_path)
    conn.execute(
        """
        insert into transcripts (file_id, language, text, segments, fillers)
        values (%s, %s, %s, %s::jsonb, %s::jsonb)
        on conflict (file_id) do update set
            language = excluded.language,
            text = excluded.text,
            segments = excluded.segments,
            fillers = excluded.fillers
        """,
        (
            file_id,
            result.language,
            result.text,
            json.dumps(tr_mod.serialize_segments(result.segments)),
            json.dumps(tr_mod.serialize_fillers(result.fillers)),
        ),
    )


# --- Stage 3: shots ------------------------------------------------------

def _stage3_shots(
    file_id: str,
    video_path: str,
    duration_s: float,
    tmpdir: str,
    conn: psycopg.Connection,
) -> List[shots_mod.Shot]:
    """
    Detect shots + extract 3 keyframes per shot. For each keyframe:
      - keep the JPEG locally (used by SigLIP embedding stage)
      - downscale to 224x224 and upload that compact copy to R2

    `video_path` is the 1080p proxy when available (set by the orchestrator),
    which makes detection/keyframe/telemetry decode a small file instead of a
    full-res raw. Shot boundaries are in milliseconds, so they map cleanly back
    onto the raw video for later L2 keyframe extraction.
    """
    shots_dir = os.path.join(tmpdir, "keyframes")
    r2_dir = os.path.join(tmpdir, "keyframes_r2")
    os.makedirs(shots_dir, exist_ok=True)
    os.makedirs(r2_dir, exist_ok=True)

    shots = shots_mod.detect_shots(video_path, duration_s, shots_dir)

    conn.execute("delete from shots where file_id = %s", (file_id,))
    sk_ok = _shot_keyframes_available(conn)

    for shot in shots:
        # Upload the base triple (anchor / motion / variance) at 224x224.
        base_keys: dict[str, str] = {}
        for kind, src in (
            ("anchor", shot.anchor_local_path),
            ("motion", shot.motion_local_path),
            ("variance", shot.variance_local_path),
        ):
            if not src or not os.path.exists(src):
                continue
            small = os.path.join(r2_dir, f"shot_{shot.index:05d}_{kind}.jpg")
            if not kf_mod.downscale_for_storage(src, small, target_size=224, quality=85):
                continue
            key = f"keyframes/{file_id}/{shot.index:05d}_{kind}.jpg"
            _upload_to_r2(small, key, "image/jpeg")
            base_keys[kind] = key
        shot.keyframe_r2_key = base_keys.get("anchor")
        shot.r2_keyframe_motion_key = base_keys.get("motion")
        shot.r2_keyframe_variance_key = base_keys.get("variance")

        row = conn.execute(
            """
            insert into shots (
                file_id, shot_index, start_ms, end_ms,
                keyframe_r2_key, r2_keyframe_motion_key, r2_keyframe_variance_key,
                focus_score, brightness, motion_magnitude,
                blur_min, peak_motion_ms, peak_variance_ms,
                motion_dx, motion_dy
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            (
                file_id, shot.index, shot.start_ms, shot.end_ms,
                shot.keyframe_r2_key,
                shot.r2_keyframe_motion_key,
                shot.r2_keyframe_variance_key,
                shot.focus_score, shot.brightness, shot.motion_magnitude,
                shot.blur_min, shot.motion_ts_ms, shot.variance_ts_ms,
                shot.motion_dx, shot.motion_dy,
            ),
        ).fetchone()
        shot_id = row[0]

        if not sk_ok:
            continue

        # Layer A: persist the full adaptive keyframe set. Base frames reuse the
        # JPEGs uploaded above; coverage frames are uploaded under their own keys.
        for frame_index, kf in enumerate(shot.keyframes):
            if kf.kind in base_keys:
                kf.r2_key = base_keys[kf.kind]
            else:
                if not kf.local_path or not os.path.exists(kf.local_path):
                    continue
                small = os.path.join(r2_dir, f"shot_{shot.index:05d}_{frame_index:02d}_cov.jpg")
                if not kf_mod.downscale_for_storage(kf.local_path, small, target_size=224, quality=85):
                    continue
                key = f"keyframes/{file_id}/{shot.index:05d}_{frame_index:02d}_cov.jpg"
                _upload_to_r2(small, key, "image/jpeg")
                kf.r2_key = key
            if not kf.r2_key:
                continue
            conn.execute(
                """
                insert into shot_keyframes (shot_id, frame_index, kind, ts_ms, r2_key, blur)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (shot_id, frame_index) do update set
                    kind = excluded.kind, ts_ms = excluded.ts_ms,
                    r2_key = excluded.r2_key, blur = excluded.blur
                """,
                (shot_id, frame_index, kf.kind, kf.ts_ms, kf.r2_key, kf.blur),
            )
    return shots


# --- Stage 4: SigLIP 2 embeddings ----------------------------------------

def _stage4_embeddings(
    file_id: str,
    shots: List[shots_mod.Shot],
    conn: psycopg.Connection,
) -> None:
    """
    Compute SigLIP embeddings on all 3 keyframes per shot (anchor, motion,
    variance) and store them. Also computes:
      - intra_shot_variance = 1 - cosine(anchor, motion)  -> shots.intra_shot_variance
    Inserts/upserts shot_embeddings; idempotent.

    Reads keyframes from the in-memory shot list (full-res local paths
    written during _stage3_shots). If those paths are gone (rare retry
    case), falls back to skipping the shot.
    """
    cur = conn.execute(
        """
        select s.id, s.shot_index
          from shots s
          left join shot_embeddings se on se.shot_id = s.id
         where s.file_id = %s and (se.shot_id is null or se.embedding_motion is null)
         order by s.shot_index
        """,
        (file_id,),
    )
    pending_rows = cur.fetchall()
    if not pending_rows:
        return

    by_idx = {s.index: s for s in shots}
    # Build a flat list of (shot_id, kind, path) so we batch-encode all 3 frames
    # of all pending shots in one SigLIP forward pass.
    plan: List[tuple[str, int, str, str]] = []  # (shot_id, shot_index, kind, path)
    for sid, sidx in pending_rows:
        s = by_idx.get(sidx)
        if not s:
            continue
        for kind, path in (
            ("anchor", s.anchor_local_path),
            ("motion", s.motion_local_path),
            ("variance", s.variance_local_path),
        ):
            if path and os.path.exists(path):
                plan.append((str(sid), sidx, kind, path))

    if not plan:
        return

    vecs = emb_mod.embed_images([p for _, _, _, p in plan])
    if vecs.shape[0] == 0:
        return

    # Index vectors back by (shot_id, kind)
    by_shot: dict[str, dict[str, np.ndarray]] = {}
    for (sid, _sidx, kind, _path), vec in zip(plan, vecs):
        by_shot.setdefault(sid, {})[kind] = vec

    for sid, vmap in by_shot.items():
        anchor = vmap.get("anchor")
        motion = vmap.get("motion")
        variance = vmap.get("variance")
        if anchor is None:
            continue

        conn.execute(
            """
            insert into shot_embeddings (shot_id, embedding, embedding_motion, embedding_variance)
            values (%s, %s::halfvec, %s::halfvec, %s::halfvec)
            on conflict (shot_id) do update set
                embedding = excluded.embedding,
                embedding_motion = excluded.embedding_motion,
                embedding_variance = excluded.embedding_variance
            """,
            (
                sid,
                _vec_to_pg(anchor),
                _vec_to_pg(motion) if motion is not None else None,
                _vec_to_pg(variance) if variance is not None else None,
            ),
        )

        # intra_shot_variance = 1 - cosine(anchor, motion). SigLIP outputs are
        # already L2-normalized so we can compute cosine via a single dot product.
        if motion is not None:
            cos = float(np.clip(np.dot(anchor, motion), -1.0, 1.0))
            intra_var = 1.0 - cos
            conn.execute(
                "update shots set intra_shot_variance = %s where id = %s",
                (intra_var, sid),
            )

    _embed_shot_keyframes(file_id, shots, conn)


def _embed_shot_keyframes(
    file_id: str,
    shots: List[shots_mod.Shot],
    conn: psycopg.Connection,
) -> None:
    """Layer A: SigLIP-embed every shot_keyframes row that lacks an embedding.

    Robust to retries: prefers the in-memory full-res frame, but falls back to
    downloading the 224px JPEG from R2 by its stored key, so coverage frames get
    embedded even when the indexing tmpdir is gone."""
    if not _shot_keyframes_available(conn):
        return
    rows = conn.execute(
        """
        select sk.id, sk.r2_key
          from shot_keyframes sk
          join shots s on s.id = sk.shot_id
         where s.file_id = %s and sk.embedding is null
        """,
        (file_id,),
    ).fetchall()
    if not rows:
        return

    local_by_key: dict[str, str] = {}
    for s in shots:
        for kf in s.keyframes:
            if kf.r2_key and kf.local_path and os.path.exists(kf.local_path):
                local_by_key[kf.r2_key] = kf.local_path

    with tempfile.TemporaryDirectory() as dl_dir:
        plan: List[tuple] = []  # (sk_id, path)
        for sk_id, r2_key in rows:
            path = local_by_key.get(r2_key)
            if not path or not os.path.exists(path):
                path = os.path.join(dl_dir, f"{sk_id}.jpg")
                try:
                    _download_from_r2(r2_key, path)
                except Exception:
                    logger.warning("Could not fetch keyframe %s for embedding", r2_key)
                    continue
            plan.append((sk_id, path))

        if not plan:
            return
        vecs = emb_mod.embed_images([p for _, p in plan])
        if vecs.shape[0] == 0:
            return
        for (sk_id, _path), vec in zip(plan, vecs):
            conn.execute(
                "update shot_keyframes set embedding = %s::halfvec where id = %s",
                (_vec_to_pg(vec), sk_id),
            )


# --- Stage 5: audio features ---------------------------------------------

def _stage5_audio(file_id: str, wav_path: str, conn: psycopg.Connection) -> None:
    af = af_mod.compute_audio_features(wav_path)
    conn.execute(
        """
        insert into audio_features (
            file_id, integrated_lufs, true_peak_db,
            is_musical, bpm, onsets_ms, silence_intervals,
            energy_peaks_ms, pause_map, rms_db, pitch_hz, prosody_hop_ms
        ) values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                  %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
        on conflict (file_id) do update set
            integrated_lufs = excluded.integrated_lufs,
            true_peak_db = excluded.true_peak_db,
            is_musical = excluded.is_musical,
            bpm = excluded.bpm,
            onsets_ms = excluded.onsets_ms,
            silence_intervals = excluded.silence_intervals,
            energy_peaks_ms = excluded.energy_peaks_ms,
            pause_map = excluded.pause_map,
            rms_db = excluded.rms_db,
            pitch_hz = excluded.pitch_hz,
            prosody_hop_ms = excluded.prosody_hop_ms
        """,
        (
            file_id,
            af.integrated_lufs, af.true_peak_db,
            af.is_musical, af.bpm,
            json.dumps(af.onsets_ms),
            json.dumps(af.silence_intervals),
            json.dumps(af.energy_peaks_ms),
            json.dumps(af.pause_map),
            json.dumps(af.rms_db),
            json.dumps(af.pitch_hz),
            af.prosody_hop_ms,
        ),
    )


# --- Stage 6: diarization (who-says-what) --------------------------------

def _stage6_diarization(file_id: str, wav_path: str, conn: psycopg.Connection) -> None:
    """Label every transcript word with a per-file speaker id.

    Best-effort: reuses Whisper's word timings (read back from the transcript
    row), runs CPU diarization, and writes a `speaker` key into each word in
    `transcripts.segments`. Any failure leaves speakers unset without breaking
    the pipeline -- diarization is a soft signal, not a gate.
    """
    settings = get_settings()
    if not settings.enable_diarization:
        return

    row = conn.execute(
        "select segments from transcripts where file_id = %s",
        (file_id,),
    ).fetchone()
    if not row or not row[0]:
        return
    segments = row[0]

    # Flatten words with a back-reference so we can write speakers in place.
    flat: List[dict] = []
    refs: List[tuple] = []  # (segment, word_dict)
    for seg in segments:
        for w in seg.get("words") or []:
            flat.append(w)
            refs.append((seg, w))
    if not flat:
        return

    result = diar_mod.diarize(
        wav_path,
        flat,
        backend=settings.diarization_backend,
        max_speakers=settings.diarization_max_speakers,
    )
    speakers = result.speaker_by_word
    if not speakers or len(speakers) != len(refs):
        return

    for (_seg, w), spk in zip(refs, speakers):
        if spk is not None:
            w["speaker"] = spk

    conn.execute(
        "update transcripts set segments = %s::jsonb where file_id = %s",
        (json.dumps(segments), file_id),
    )
    logger.info("Diarization: %s -> %d speaker(s)", file_id, result.num_speakers)


# --- Helpers -------------------------------------------------------------

def _run_stage(
    conn: psycopg.Connection,
    file_id: str,
    stage: str,
    fn,
    *args,
    **kwargs,
):
    if _stage_status(conn, file_id, stage) == "done":
        logger.info("Stage %s already done for %s; skipping", stage, file_id)
        return None
    _stage_begin(conn, file_id, stage)
    try:
        result = fn(*args, **kwargs)
        _stage_done(conn, file_id, stage)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        _stage_fail(conn, file_id, stage, f"{type(e).__name__}: {e}\n{tb}")
        raise


# --- Top-level procrastinate task ----------------------------------------

@app.task(name="l1_orchestrate", queue="gpu", retry=RetryStrategy(max_attempts=3, exponential_wait=4))
def l1_orchestrate(file_id: str, r2_key: str) -> None:
    """
    Single procrastinate task that downloads the raw video once and runs all
    five L1 stages. Idempotent: each stage checks processing_jobs first.
    """
    settings = get_settings()

    if not _file_exists(file_id):
        logger.info("File %s no longer exists; skipping L1 (likely deleted/aborted).", file_id)
        return

    _set_l1_status(file_id, "running")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw")
            wav_path = os.path.join(tmpdir, "audio.wav")
            logger.info("L1: downloading %s for file %s", r2_key, file_id)
            _download_from_r2(r2_key, raw_path)

            has_audio = _has_audio_stream(raw_path)
            if not has_audio:
                logger.info("File %s has no audio stream; skipping transcript/audio.", file_id)

            # Audio demux is a pure subprocess->file step with no DB access, so
            # we overlap it with the (CPU/NVENC-heavy) proxy encode to take it
            # off the critical path. Transcript/audio stages join on it later.
            demux_err: dict = {}

            def _demux_bg() -> None:
                try:
                    _demux_wav(raw_path, wav_path)
                except Exception as e:  # noqa: BLE001 - surfaced after join
                    demux_err["e"] = e

            demux_thread: Optional[threading.Thread] = None
            if has_audio:
                demux_thread = threading.Thread(target=_demux_bg, daemon=True)
                demux_thread.start()

            with _pg_conn() as conn:
                duration, _w, _h = _run_stage(
                    conn, file_id, "proxy",
                    _stage1_proxy, file_id, raw_path, tmpdir,
                ) or (0.0, 0, 0)

                cur = conn.execute(
                    "select duration_seconds from files where id = %s",
                    (file_id,),
                )
                row = cur.fetchone()
                duration_s = float(row[0]) if row and row[0] else (duration or 0.0)

                if duration_s > settings.max_l1_duration_seconds:
                    logger.info(
                        "Duration %.1fs > guardrail %ds; skipping deep L1 stages",
                        duration_s, settings.max_l1_duration_seconds,
                    )
                    if demux_thread is not None:
                        demux_thread.join()
                    _set_l1_status(file_id, "skipped")
                    return

                # Ensure the demux finished (it almost always has, in parallel
                # with the proxy) before the transcript/audio stages need it.
                if demux_thread is not None:
                    demux_thread.join()
                    if "e" in demux_err:
                        raise demux_err["e"]

                # Prefer the 1080p proxy for shot detection / keyframes /
                # telemetry (much cheaper to decode than a 4K raw). It exists
                # locally whenever the proxy stage ran in this invocation; on a
                # rare retry where it doesn't, fall back to the raw file.
                proxy_local = os.path.join(tmpdir, "proxy.mp4")
                shot_source = proxy_local if os.path.exists(proxy_local) else raw_path

                shots_result = _run_stage(
                    conn, file_id, "shots",
                    _stage3_shots, file_id, shot_source, duration_s, tmpdir, conn,
                )

                # Re-hydrate shot list from disk so the embeddings stage can
                # read keyframe paths even on retry. If the shots stage just
                # ran in this invocation, prefer its in-memory list (paths
                # are guaranteed valid); otherwise reconstruct from disk.
                if shots_result is not None:
                    shots_for_emb = shots_result
                else:
                    shots_for_emb = []
                    kf_dir = os.path.join(tmpdir, "keyframes")
                    cur = conn.execute(
                        """
                        select shot_index, start_ms, end_ms,
                               keyframe_r2_key, r2_keyframe_motion_key, r2_keyframe_variance_key,
                               peak_motion_ms, peak_variance_ms
                          from shots where file_id = %s order by shot_index
                        """,
                        (file_id,),
                    )
                    for sidx, sms, ems, kkey_a, kkey_m, kkey_v, m_ts, v_ts in cur.fetchall():
                        s = shots_mod.Shot(index=sidx, start_ms=sms, end_ms=ems)
                        s.keyframe_r2_key = kkey_a
                        s.r2_keyframe_motion_key = kkey_m
                        s.r2_keyframe_variance_key = kkey_v
                        s.motion_ts_ms = m_ts
                        s.variance_ts_ms = v_ts
                        s.anchor_ts_ms = (sms + ems) // 2  # best-effort fallback
                        for kind, attr in (
                            ("anchor", "anchor_local_path"),
                            ("motion", "motion_local_path"),
                            ("variance", "variance_local_path"),
                        ):
                            p = os.path.join(kf_dir, f"shot_{sidx:05d}_{kind}.jpg")
                            if os.path.exists(p):
                                setattr(s, attr, p)
                        shots_for_emb.append(s)

                _run_stage(conn, file_id, "embeddings",
                           _stage4_embeddings, file_id, shots_for_emb, conn)

                if has_audio:
                    _run_stage(conn, file_id, "transcript",
                               _stage2_transcript, file_id, wav_path, conn)

                    _run_stage(conn, file_id, "audio_features",
                               _stage5_audio, file_id, wav_path, conn)

                    # Diarization depends on the transcript words written above.
                    _run_stage(conn, file_id, "diarization",
                               _stage6_diarization, file_id, wav_path, conn)

        _set_l1_status(file_id, "ready")
        logger.info("L1 complete for %s", file_id)
        try:
            snapshot = build_l1_snapshot(file_id)
            log_path = audit_log.write_l1_analysis(file_id, snapshot)
            logger.info("L1 analysis written to %s", log_path)
        except Exception:
            logger.exception("L1 succeeded but writing the audit log failed for %s", file_id)

        # Every video flows L1 -> L2 automatically, in the background.
        _enqueue_l2_if_needed(file_id)
    except psycopg.errors.ForeignKeyViolation:
        # The files row was deleted while we were processing it -- the inserts
        # have nowhere to land. Stop cleanly instead of failing + retrying.
        logger.info("File %s deleted mid-L1; abandoning cleanly.", file_id)
        return
    except Exception:
        if not _file_exists(file_id):
            logger.info("File %s deleted mid-L1; abandoning cleanly.", file_id)
            return
        logger.exception("L1 failed for %s", file_id)
        _set_l1_status(file_id, "failed")
        raise
