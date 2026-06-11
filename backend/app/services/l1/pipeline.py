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

import psycopg
from procrastinate import RetryStrategy

from app.config import get_settings
from app.services import audit_log
from app.services.jobs import app
from app.services.l1 import audio_features as af_mod
from app.services.l1 import beat_cost as beat_mod
from app.services.l1 import cut_cost as cutcost_mod
from app.services.l1 import diarization as diar_mod
from app.services.l1 import motion_dynamics as motion_mod
from app.services.l1 import transcript as tr_mod
from app.services.l1.snapshot import build_l1_snapshot
from app.services.processing import _download_from_r2, _probe_video, _upload_to_r2
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

STAGES = ("proxy", "transcript", "audio_features", "diarization", "dialogue_cut", "beat_cut", "motion_dynamics")


def _pg_conn() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


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
    multipart upload) mid-processing; without this guard the L1 inserts hit a
    foreign-key violation and retry forever."""
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select 1 from files where id = %s", (file_id,)
            ).fetchone()
        return row is not None
    except Exception:
        # If we can't tell, assume it exists and let normal error handling run.
        return True


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


# --- Stage 5: audio features ---------------------------------------------

def _stage5_audio(file_id: str, wav_path: str, conn: psycopg.Connection) -> None:
    af = af_mod.compute_audio_features(wav_path)
    conn.execute(
        """
        insert into audio_features (
            file_id, integrated_lufs, true_peak_db,
            is_musical, bpm, onsets_ms, silence_intervals,
            energy_peaks_ms, pause_map, rms_db, pitch_hz, prosody_hop_ms,
            sync_env, sync_hop_ms
        ) values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                  %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s)
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
            prosody_hop_ms = excluded.prosody_hop_ms,
            sync_env = excluded.sync_env,
            sync_hop_ms = excluded.sync_hop_ms
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
            json.dumps(af.sync_env),
            af.sync_hop_ms,
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


# --- Stage 7: dialogue cut-cost grid (derived; cheap, CPU-only) -----------

def _flatten_words(segments: list) -> List[dict]:
    """Chronological word list with the diarization speaker carried through."""
    words: List[dict] = []
    for seg in segments or []:
        for w in seg.get("words") or []:
            words.append({
                "start_ms": w.get("start_ms", 0),
                "end_ms": w.get("end_ms", 0),
                "text": w.get("text", ""),
                "is_filler": w.get("is_filler", False),
                "speaker": w.get("speaker"),
            })
    return words


def _stage7_dialogue_cut(file_id: str, duration_s: float, conn: psycopg.Connection) -> None:
    """Derive the dialogue cut-cost grid from the transcript words + audio
    pause/energy signals written by the earlier stages, and persist it onto the
    file's audio_features row. Pure arithmetic -- no model, no GPU.

    Soft signal: a missing transcript / audio_features row just no-ops.
    """
    row = conn.execute(
        "select segments from transcripts where file_id = %s", (file_id,)
    ).fetchone()
    if not row or not row[0]:
        return
    words = _flatten_words(row[0])
    if not words:
        return

    af = conn.execute(
        "select rms_db, prosody_hop_ms from audio_features where file_id = %s",
        (file_id,),
    ).fetchone()
    rms_db = (af[0] if af else None) or []
    prosody_hop_ms = (af[1] if af else 0) or 0

    grid = cutcost_mod.compute_dialogue_cut_grid(
        words=words,
        rms_db=rms_db,
        prosody_hop_ms=prosody_hop_ms,
        duration_ms=int((duration_s or 0) * 1000),
    )
    if not grid.has_dialogue:
        return

    conn.execute(
        """
        update audio_features
           set dialogue_cut_cost   = %s::jsonb,
               dialogue_cut_hop_ms = %s,
               dialogue_cut_points = %s::jsonb
         where file_id = %s
        """,
        (
            json.dumps(grid.cost_payload()),
            grid.hop_ms,
            json.dumps(grid.points_payload()),
            file_id,
        ),
    )
    logger.info(
        "Dialogue cut grid: %s -> %d hops, %d cut points",
        file_id, len(grid.cut_cost), len(grid.cut_points),
    )


def _stage8_beat_cut(file_id: str, duration_s: float, conn: psycopg.Connection) -> None:
    """Derive the beat/music cut grid from the librosa onsets/bpm already on the
    file's audio_features row. Free -- pure arithmetic, no new decode.

    Non-musical files leave the columns empty.
    """
    af = conn.execute(
        "select is_musical, bpm, onsets_ms from audio_features where file_id = %s",
        (file_id,),
    ).fetchone()
    if not af:
        return

    grid = beat_mod.compute_beat_grid(
        is_musical=bool(af[0]),
        bpm=float(af[1] or 0.0),
        onsets_ms=af[2] or [],
        duration_ms=int((duration_s or 0) * 1000),
    )
    if not grid.has_beat:
        return

    conn.execute(
        """
        update audio_features
           set beat_cut_cost   = %s::jsonb,
               beat_cut_hop_ms = %s,
               beat_cut_points = %s::jsonb
         where file_id = %s
        """,
        (json.dumps(grid.cost), grid.hop_ms, json.dumps(grid.points), file_id),
    )
    logger.info(
        "Beat cut grid: %s -> %d hops, %d beats (bpm=%.1f)",
        file_id, len(grid.cost), len(grid.points), grid.bpm,
    )


def _stage9_motion_dynamics(
    file_id: str, video_path: str, duration_s: float, conn: psycopg.Connection
) -> None:
    """One optical-flow pass over the proxy -> action + camera/distortion cut
    grids. Best-effort: a flow/decode failure no-ops without failing L1.
    """
    md = motion_mod.compute_motion_dynamics(
        video_path, duration_ms=int((duration_s or 0) * 1000)
    )
    if not md.has_motion:
        return

    conn.execute(
        """
        insert into motion_dynamics
            (file_id, hop_ms, action_energy, camera_motion, camera_coherence,
             camera_stability, blur, action_cut_cost, camera_cut_cost, action_points)
        values (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb, %s::jsonb)
        on conflict (file_id) do update set
            hop_ms           = excluded.hop_ms,
            action_energy    = excluded.action_energy,
            camera_motion    = excluded.camera_motion,
            camera_coherence = excluded.camera_coherence,
            camera_stability = excluded.camera_stability,
            blur             = excluded.blur,
            action_cut_cost  = excluded.action_cut_cost,
            camera_cut_cost  = excluded.camera_cut_cost,
            action_points    = excluded.action_points
        """,
        (
            file_id, md.hop_ms,
            json.dumps(md.action_energy), json.dumps(md.camera_motion),
            json.dumps(md.camera_coherence), json.dumps(md.camera_stability),
            json.dumps(md.blur), json.dumps(md.action_cut_cost),
            json.dumps(md.camera_cut_cost), json.dumps(md.action_points),
        ),
    )
    logger.info(
        "Motion dynamics: %s -> %d hops, %d action impacts",
        file_id, len(md.action_cut_cost), len(md.action_points),
    )


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

                # Prefer the 1080p proxy for the motion pass (much cheaper to
                # decode than a 4K raw). It exists locally whenever the proxy
                # stage ran in this invocation; on a rare retry where it
                # doesn't, fall back to the raw file.
                proxy_local = os.path.join(tmpdir, "proxy.mp4")
                video_source = proxy_local if os.path.exists(proxy_local) else raw_path

                if has_audio:
                    _run_stage(conn, file_id, "transcript",
                               _stage2_transcript, file_id, wav_path, conn)

                    _run_stage(conn, file_id, "audio_features",
                               _stage5_audio, file_id, wav_path, conn)

                    # Diarization depends on the transcript words written above.
                    _run_stage(conn, file_id, "diarization",
                               _stage6_diarization, file_id, wav_path, conn)

                    # Dialogue cut-cost grid: pure derivation over the transcript
                    # words + speakers (post-diarization) + audio pause/energy.
                    _run_stage(conn, file_id, "dialogue_cut",
                               _stage7_dialogue_cut, file_id, duration_s, conn)

                    # Beat/music cut grid: free derivation over the librosa
                    # onsets/bpm written by the audio_features stage.
                    _run_stage(conn, file_id, "beat_cut",
                               _stage8_beat_cut, file_id, duration_s, conn)

                # Motion dynamics (action + camera/distortion): one optical-flow
                # pass over the proxy. Video-only -- runs even on silent files.
                _run_stage(conn, file_id, "motion_dynamics",
                           _stage9_motion_dynamics, file_id, video_source, duration_s, conn)

        _set_l1_status(file_id, "ready")
        logger.info("L1 complete for %s", file_id)
        try:
            snapshot = build_l1_snapshot(file_id)
            log_path = audit_log.write_l1_analysis(file_id, snapshot)
            logger.info("L1 analysis written to %s", log_path)
        except Exception:
            logger.exception("L1 succeeded but writing the audit log failed for %s", file_id)

        # L2 (Gemini perception) runs as its own task so a VLM hiccup retries
        # independently of the L1 index. Gated by duration inside the enqueue.
        try:
            from app.services.l2.perception import enqueue_l2_if_eligible
            enqueue_l2_if_eligible(file_id, duration_s)
        except Exception:
            logger.exception("L1 done but enqueuing L2 failed for %s", file_id)
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
