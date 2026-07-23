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
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional

import psycopg
from procrastinate import RetryStrategy

from app.config import get_settings
from app.services import audit_log
from app.services import correlation
from app.services import limits
from app.services.jobs import app
from app.services.l1 import active_speaker as asd_mod
from app.services.l1 import audio_features as af_mod
from app.services.l1 import color_stats as color_stats_mod
from app.services.l1 import diarization as diar_mod
from app.services.l1 import dialogue_segments as dlg_mod
from app.services.l1 import motion_dynamics as motion_mod
from app.services.l1 import scene_cuts as scene_mod
from app.services.l1 import transcript as tr_mod
from app.services.l1.snapshot import build_l1_snapshot
from app.services.processing import _download_from_r2, _probe_video, _upload_to_r2
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

STAGES = ("proxy", "transcript", "audio_features", "diarization", "motion_dynamics", "dialogue_segments")
# cuts-v2: additive on top of STAGES (old tuple kept as-is; nothing that reads
# STAGES needs to change). Runs in parallel with the v1 pipeline until v2 is
# validated -- see cuts_v2.plan.md.
STAGES_V2 = STAGES + ("scene_detect",)
# color grading: additive, independent of the cuts-v2 versioning above.
STAGES_COLOR = STAGES_V2 + ("color_stats",)
# Audio-only uploads run a different, video-free set of stages. `transcript`/
# `diarization` are CONDITIONAL (audio_sync.plan.md SS4.1): they only run when
# `audio_features.is_musical` comes back false (a spoken external-mic/audio
# file, not music/SFX) -- see `_orchestrate_audio`.
AUDIO_STAGES = ("audio_proxy", "audio_features", "transcript", "diarization")


def _pg_conn():
    from app.services import db
    return db.connection()


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


def _file_type(file_id: str) -> Optional[str]:
    """Read the file's classified type ('video' | 'audio' | ...) so the
    orchestrator can pick the right (video vs audio-only) pipeline."""
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select file_type from files where id = %s", (file_id,)
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _user_id_for_file(file_id: str) -> Optional[str]:
    """Pillar 7: for correlation-scope logging only -- best-effort, never
    raises (an unknown owner just logs as "-")."""
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select user_id::text from files where id = %s", (file_id,)
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


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
        with limits.ffmpeg_slot():
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
    (10-bit, HDR, VFR, rotated) still produce a working proxy instead of failing.

    The proxy is also SEEK-OPTIMIZED for interactive editing: a keyframe is
    forced every second (``-force_key_frames``, fps-agnostic so it holds for
    24/25/30/60fps sources alike) and the moov atom is moved to the front
    (``+faststart``). Frequent keyframes mean a scrub/cut lands near-instantly
    (the decoder only walks <=1s to the target) instead of grinding through
    libx264's default ~250-frame GOP -- the root of the preview seek stalls.
    Denser keyframes also speed the perception pipeline's random-access decodes,
    so the single proxy serves both consumers; the only cost is a modestly larger
    file (more I-frames compress worse)."""
    common_out = [
        "-vsync", "cfr",  # normalize VFR (phone footage) to constant frame rate
        # ~1s keyframe cadence, independent of source fps, so seeks are cheap.
        "-force_key_frames", "expr:gte(t,n_forced*1)",
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
    with limits.ffmpeg_slot():
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
    with limits.ffmpeg_slot():
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_path,
                "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                out_path,
            ],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
        )


# --- Audio-only (music) stages -------------------------------------------

# Waveform thumbnail dimensions (also stored as the file's width/height so the
# grid has something to lay out).
_WAVEFORM_W = 1280
_WAVEFORM_H = 320


def _stage_audio_proxy(file_id: str, raw_path: str, tmpdir: str) -> float:
    """Music upload: normalized AAC proxy for browser playback + a waveform PNG
    thumbnail. Flips the file to 'ready'. Returns duration (s)."""
    sb = get_supabase()
    proxy_path = os.path.join(tmpdir, "proxy.m4a")
    thumb_path = os.path.join(tmpdir, "waveform.png")

    probe = _probe_video(raw_path)
    duration = float(probe.get("format", {}).get("duration", 0) or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    if not has_audio:
        raise ValueError("Uploaded audio file contains no decodable audio stream")

    ok, err = _run_ffmpeg([
        "ffmpeg", "-y", "-i", raw_path,
        "-vn", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        proxy_path,
    ])
    if not ok:
        raise RuntimeError(f"Audio proxy encode failed: {err!r}")
    proxy_key = f"proxies/{file_id}/proxy.m4a"
    _upload_to_r2(proxy_path, proxy_key, "audio/mp4")

    # Waveform image as the thumbnail (best-effort -- a missing thumb shouldn't
    # fail ingest).
    thumb_key = None
    wf_ok, _ = _run_ffmpeg([
        "ffmpeg", "-y", "-i", raw_path,
        "-filter_complex",
        f"aformat=channel_layouts=mono,showwavespic=s={_WAVEFORM_W}x{_WAVEFORM_H}:colors=#7c5cff",
        "-frames:v", "1", thumb_path,
    ])
    if wf_ok and os.path.exists(thumb_path):
        thumb_key = f"thumbnails/{file_id}/waveform.png"
        _upload_to_r2(thumb_path, thumb_key, "image/png")

    update = {
        "r2_proxy_key": proxy_key,
        "duration_seconds": duration,
        "width": _WAVEFORM_W,
        "height": _WAVEFORM_H,
        "status": "ready",
    }
    if thumb_key:
        update["r2_thumbnail_key"] = thumb_key
    sb.table("files").update(update).eq("id", file_id).execute()
    return duration


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
            rms_db, prosody_hop_ms, sections, drop_ms
        ) values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s)
        on conflict (file_id) do update set
            integrated_lufs = excluded.integrated_lufs,
            true_peak_db = excluded.true_peak_db,
            is_musical = excluded.is_musical,
            bpm = excluded.bpm,
            onsets_ms = excluded.onsets_ms,
            silence_intervals = excluded.silence_intervals,
            rms_db = excluded.rms_db,
            prosody_hop_ms = excluded.prosody_hop_ms,
            sections = excluded.sections,
            drop_ms = excluded.drop_ms
        """,
        (
            file_id,
            af.integrated_lufs, af.true_peak_db,
            af.is_musical, af.bpm,
            json.dumps(af.onsets_ms),
            json.dumps(af.silence_intervals),
            json.dumps(af.rms_db),
            af.prosody_hop_ms,
            json.dumps(af.sections),
            af.drop_ms,
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
        max_speakers=settings.diarization_max_speakers,
    )
    speakers = result.speaker_by_word
    if not speakers or len(speakers) != len(refs):
        return

    for (_seg, w), spk in zip(refs, speakers):
        if spk is not None:
            w["speaker"] = spk

    conn.execute(
        "update transcripts set segments = %s::jsonb, speaker_embeddings = %s::jsonb where file_id = %s",
        (json.dumps(segments), json.dumps(result.embedding_by_speaker), file_id),
    )
    logger.info("Diarization: %s -> %d speaker(s), %d voiceprint(s)",
               file_id, result.num_speakers, len(result.embedding_by_speaker))


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


def _stage_dialogue_segments(file_id: str, wav_path: str, conn: psycopg.Connection) -> None:
    """Build the Dialogues-lens selects (sentence + topic) from the diarized
    transcript, with audio-snapped cut points, and upsert them. Best-effort: a
    missing transcript just no-ops; any failure leaves the lens uncomputed."""
    row = conn.execute(
        "select segments from transcripts where file_id = %s", (file_id,)
    ).fetchone()
    if not row or not row[0]:
        return
    words = _flatten_words(row[0])
    if not words:
        return
    result = dlg_mod.build_dialogue_segments(words, wav_path)
    conn.execute(
        """
        insert into dialogue_segments (file_id, schema_version, segments)
        values (%s, %s, %s::jsonb)
        on conflict (file_id) do update set
            schema_version = excluded.schema_version,
            segments = excluded.segments,
            created_at = now()
        """,
        (file_id, dlg_mod.SCHEMA_VERSION, json.dumps(result)),
    )
    logger.info(
        "Dialogue segments: %s -> %d sentence, %d topic",
        file_id, len(result.get("sentence", [])), len(result.get("topic", [])),
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
             camera_stability, blur, action_cut_cost, camera_cut_cost, action_points,
             transition_points, camera_dx, camera_dy, camera_zoom)
        values (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
        on conflict (file_id) do update set
            hop_ms             = excluded.hop_ms,
            action_energy      = excluded.action_energy,
            camera_motion      = excluded.camera_motion,
            camera_coherence   = excluded.camera_coherence,
            camera_stability   = excluded.camera_stability,
            blur               = excluded.blur,
            action_cut_cost    = excluded.action_cut_cost,
            camera_cut_cost    = excluded.camera_cut_cost,
            action_points      = excluded.action_points,
            transition_points  = excluded.transition_points,
            camera_dx          = excluded.camera_dx,
            camera_dy          = excluded.camera_dy,
            camera_zoom        = excluded.camera_zoom
        """,
        (
            file_id, md.hop_ms,
            json.dumps(md.action_energy), json.dumps(md.camera_motion),
            json.dumps(md.camera_coherence), json.dumps(md.camera_stability),
            json.dumps(md.blur), json.dumps(md.action_cut_cost),
            json.dumps(md.camera_cut_cost), json.dumps(md.action_points),
            json.dumps(md.transition_points),
            json.dumps(md.camera_dx), json.dumps(md.camera_dy), json.dumps(md.camera_zoom),
        ),
    )
    logger.info(
        "Motion dynamics: %s -> %d hops, %d action impacts, %d transition points",
        file_id, len(md.action_cut_cost), len(md.action_points), len(md.transition_points),
    )


def _stage_scene_detect(
    file_id: str, video_path: str, duration_s: float, conn: psycopg.Connection
) -> None:
    """cuts-v2: one histogram-drift pass over the proxy -> shot/composition
    boundaries. Best-effort: a decode failure no-ops without failing L1.
    """
    sc = scene_mod.compute_scene_cuts(
        video_path, duration_ms=int((duration_s or 0) * 1000)
    )
    if not sc.has_scenes:
        return

    conn.execute(
        """
        insert into scene_cuts (file_id, hop_ms, shot_points, composition_points, schema_version)
        values (%s, %s, %s::jsonb, %s::jsonb, %s)
        on conflict (file_id) do update set
            hop_ms             = excluded.hop_ms,
            shot_points        = excluded.shot_points,
            composition_points = excluded.composition_points,
            schema_version     = excluded.schema_version
        """,
        (
            file_id, sc.hop_ms,
            json.dumps(sc.shot_points), json.dumps(sc.composition_points),
            scene_mod.SCHEMA_VERSION,
        ),
    )
    logger.info(
        "Scene detect: %s -> %d shot cuts, %d composition points",
        file_id, len(sc.shot_points), len(sc.composition_points),
    )


def _stage_color_stats(
    file_id: str, video_path: str, duration_s: float, conn: psycopg.Connection
) -> None:
    """color_grading.plan.md SS2.2: one sampled-frame pass over the proxy ->
    deterministic per-file color measurement. Best-effort: a decode failure
    no-ops without failing L1.
    """
    cs = color_stats_mod.compute_color_stats(
        video_path, duration_ms=int((duration_s or 0) * 1000)
    )
    if not cs.has_stats:
        return

    conn.execute(
        """
        insert into color_stats
            (file_id, schema_version, frames_sampled, luma_hist, black_point,
             white_point, mid_gray, rgb_mean, rgb_median, rgb_std, lab_ab_cast,
             wb_gray_world, wb_white_patch, clip_shadow_pct, clip_highlight_pct,
             is_log_flat, skin_lab, palette, chroma_mean)
        values (%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
        on conflict (file_id) do update set
            schema_version     = excluded.schema_version,
            frames_sampled     = excluded.frames_sampled,
            luma_hist          = excluded.luma_hist,
            black_point        = excluded.black_point,
            white_point        = excluded.white_point,
            mid_gray           = excluded.mid_gray,
            rgb_mean           = excluded.rgb_mean,
            rgb_median         = excluded.rgb_median,
            rgb_std            = excluded.rgb_std,
            lab_ab_cast        = excluded.lab_ab_cast,
            wb_gray_world      = excluded.wb_gray_world,
            wb_white_patch     = excluded.wb_white_patch,
            clip_shadow_pct    = excluded.clip_shadow_pct,
            clip_highlight_pct = excluded.clip_highlight_pct,
            is_log_flat        = excluded.is_log_flat,
            skin_lab           = excluded.skin_lab,
            palette            = excluded.palette,
            chroma_mean        = excluded.chroma_mean
        """,
        (
            file_id, color_stats_mod.SCHEMA_VERSION, cs.frames_sampled,
            json.dumps(cs.luma_hist), cs.black_point, cs.white_point, cs.mid_gray,
            json.dumps(cs.rgb_mean), json.dumps(cs.rgb_median), json.dumps(cs.rgb_std),
            json.dumps(cs.lab_ab_cast),
            json.dumps(cs.wb_gray_world), json.dumps(cs.wb_white_patch),
            cs.clip_shadow_pct, cs.clip_highlight_pct, cs.is_log_flat,
            json.dumps(cs.skin_lab) if cs.skin_lab is not None else None,
            json.dumps(cs.palette),
            cs.chroma_mean,
        ),
    )
    logger.info(
        "Color stats: %s -> %d frames sampled, log_flat=%s, clip shadow/highlight=%.3f/%.3f",
        file_id, cs.frames_sampled, cs.is_log_flat, cs.clip_shadow_pct, cs.clip_highlight_pct,
    )


def _stage_active_speaker(file_id: str, proxy_path: str, conn: psycopg.Connection) -> None:
    """asd_identity.plan.md: face detect+track+embed+ASD off the canonical
    1080p editing proxy. Best-effort -- compute_face_tracks never raises
    (see active_speaker.py's own fail-open contract); an empty result is
    still persisted (an explicit "no faces here" is as real a fact as a
    populated track list, not a reason to skip the write)."""
    tracks = asd_mod.compute_face_tracks(proxy_path)
    conn.execute(
        """
        insert into face_tracks (file_id, schema_version, tracks)
        values (%s, %s, %s::jsonb)
        on conflict (file_id) do update set
            schema_version = excluded.schema_version,
            tracks = excluded.tracks
        """,
        (file_id, asd_mod.SCHEMA_VERSION, json.dumps([t.to_dict() for t in tracks])),
    )
    logger.info("Active speaker: %s -> %d face track(s)", file_id, len(tracks))


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


# --- Parallel deep-stage tracks (video pipeline) -------------------------
#
# The deep L1 stages split into three INDEPENDENT tracks that run concurrently,
# each on its OWN psycopg connection (psycopg connections are not safe to share
# across threads). The heavy CPU work -- Whisper (CTranslate2), optical flow
# (OpenCV) and librosa -- all release the GIL, so threads give real parallelism.
# No two tracks write the same row, so their autocommit writes don't collide.
# `dialogue_cut` is NOT in a track: it needs transcript + diarization + audio
# features, so it runs after the tracks join.

def _track_speech(file_id: str, wav_path: str, duration_s: float) -> None:
    """transcript (Whisper) -> diarization -> dialogue_segments. The heaviest
    track."""
    with _pg_conn() as conn:
        _run_stage(conn, file_id, "transcript", _stage2_transcript, file_id, wav_path, conn)
        _run_stage(conn, file_id, "diarization", _stage6_diarization, file_id, wav_path, conn)
        # Dialogues lens: needs the diarized words + the WAV (for silence-snapped
        # cuts), both in hand here, so it rides the speech track after diarization.
        _run_stage(conn, file_id, "dialogue_segments", _stage_dialogue_segments, file_id, wav_path, conn)


def _track_audio(file_id: str, wav_path: str, duration_s: float) -> None:
    """audio_features (librosa)."""
    with _pg_conn() as conn:
        _run_stage(conn, file_id, "audio_features", _stage5_audio, file_id, wav_path, conn)


def _track_motion(file_id: str, video_source: str, duration_s: float) -> None:
    """motion_dynamics (optical flow) -> scene_detect (histogram drift) ->
    color_stats (sampled-frame color measurement). Independent of all audio
    stages; scene_detect/color_stats share this track's proxy (additive --
    see STAGES_V2/STAGES_COLOR)."""
    with _pg_conn() as conn:
        _run_stage(conn, file_id, "motion_dynamics",
                   _stage9_motion_dynamics, file_id, video_source, duration_s, conn)
        _run_stage(conn, file_id, "scene_detect",
                   _stage_scene_detect, file_id, video_source, duration_s, conn)
        _run_stage(conn, file_id, "color_stats",
                   _stage_color_stats, file_id, video_source, duration_s, conn)


def _run_deep_stages_parallel(
    file_id: str, wav_path: str, video_source: str, duration_s: float, has_audio: bool
) -> None:
    """Run the speech / audio / video tracks concurrently.

    Waits for ALL tracks before returning (so no orphan thread keeps writing
    after the task ends), then re-raises the first track error. Idempotency and
    best-effort semantics are unchanged -- each stage still records
    processing_jobs and a retry skips finished stages.
    """
    tracks = [(_track_motion, (file_id, video_source, duration_s))]
    if has_audio:
        tracks.append((_track_speech, (file_id, wav_path, duration_s)))
        tracks.append((_track_audio, (file_id, wav_path, duration_s)))

    with ThreadPoolExecutor(max_workers=len(tracks)) as ex:
        # correlation.run_with_scope: plain ex.submit wouldn't propagate this
        # file's correlation.scope() into these worker threads, and their log
        # lines would show "-" for file_id/user_id.
        futs = [correlation.run_with_scope(ex, fn, *args) for fn, args in tracks]
        wait(futs)
    for f in futs:
        exc = f.exception()
        if exc is not None:
            raise exc


# --- Audio-only (music) orchestrator -------------------------------------

def _orchestrate_audio(file_id: str, r2_key: str, settings) -> None:
    """L1 for a standalone music/audio upload. No video proxy, no motion, no
    speech tools. Runs: playable proxy + waveform thumb, audio_features
    (loudness/BPM/onsets). Stages are idempotent via processing_jobs."""
    _set_l1_status(file_id, "running")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw")
            wav_path = os.path.join(tmpdir, "audio.wav")        # 16k for features
            logger.info("L1(audio): downloading %s for file %s", r2_key, file_id)
            _download_from_r2(r2_key, raw_path)

            with _pg_conn() as conn:
                duration = _run_stage(
                    conn, file_id, "audio_proxy",
                    _stage_audio_proxy, file_id, raw_path, tmpdir,
                )
                cur = conn.execute(
                    "select duration_seconds from files where id = %s", (file_id,)
                )
                row = cur.fetchone()
                duration_s = float(row[0]) if row and row[0] else float(duration or 0.0)

                if duration_s > settings.max_l1_duration_seconds:
                    logger.info(
                        "Audio %.1fs > guardrail %ds; proxy kept, deep stages skipped",
                        duration_s, settings.max_l1_duration_seconds,
                    )
                    _set_l1_status(file_id, "skipped")
                    return

                _demux_wav(raw_path, wav_path)

                _run_stage(conn, file_id, "audio_features",
                           _stage5_audio, file_id, wav_path, conn)

                # audio_sync.plan.md SS4.1: a spoken external-mic/audio upload
                # needs a transcript + diarization to ever be an authoritative
                # sync source. Gate on `is_musical` (the only "is this speech"
                # signal that exists today -- see audio_features._detect_musicality)
                # so a pure music/SFX file skips the heavy speech stages
                # entirely, same as before this change.
                row = conn.execute(
                    "select is_musical from audio_features where file_id = %s", (file_id,)
                ).fetchone()
                is_musical = bool(row[0]) if row else False
                if not is_musical:
                    _run_stage(conn, file_id, "transcript", _stage2_transcript, file_id, wav_path, conn)
                    _run_stage(conn, file_id, "diarization", _stage6_diarization, file_id, wav_path, conn)

        _set_l1_status(file_id, "ready")
        logger.info("L1(audio) complete for %s", file_id)
        try:
            snapshot = build_l1_snapshot(file_id)
            log_path = audit_log.write_l1_analysis(file_id, snapshot)
            logger.info("L1(audio) analysis written to %s", log_path)
        except Exception:
            logger.exception("L1(audio) done but writing the audit log failed for %s", file_id)
    except psycopg.errors.ForeignKeyViolation:
        logger.info("File %s deleted mid-L1(audio); abandoning cleanly.", file_id)
        return
    except Exception:
        if not _file_exists(file_id):
            logger.info("File %s deleted mid-L1(audio); abandoning cleanly.", file_id)
            return
        logger.exception("L1(audio) failed for %s", file_id)
        _set_l1_status(file_id, "failed")
        raise


# --- Top-level procrastinate task ----------------------------------------

def _analysis_proxy_keys(file_id: str) -> tuple[Optional[str], Optional[str]]:
    """(r2_proxy_a_key, r2_proxy_b_key) -- the client-generated analysis proxies,
    or (None, None) when the client couldn't produce them (fallback to raw)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select r2_proxy_a_key, r2_proxy_b_key from files where id = %s",
            (file_id,),
        ).fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def _prepare_from_raw(
    file_id: str, r2_key: str, tmpdir: str, wav_path: str
) -> tuple[float, bool, str]:
    """Fallback ingest: download the raw once and derive every analysis input
    from it -- the 1080p editing proxy + thumbnail, the demuxed WAV, and the
    motion source. This is the pre-client-proxy behavior, unchanged, so ingest
    still works for any upload that arrives without client proxies. Returns
    (duration_s, has_audio, motion_source)."""
    raw_path = os.path.join(tmpdir, "raw")
    logger.info("L1: downloading raw %s for file %s (no client proxies)", r2_key, file_id)
    _download_from_r2(r2_key, raw_path)

    has_audio = _has_audio_stream(raw_path)
    if not has_audio:
        logger.info("File %s has no audio stream; skipping transcript/audio.", file_id)

    # Audio demux is a pure subprocess->file step with no DB access, so we
    # overlap it with the (CPU/NVENC-heavy) proxy encode to take it off the
    # critical path. Transcript/audio stages join on it below.
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
            "select duration_seconds from files where id = %s", (file_id,)
        )
        row = cur.fetchone()
        duration_s = float(row[0]) if row and row[0] else (duration or 0.0)

    # This fallback path builds the editing proxy itself (l1_editing_proxy's
    # OWN chain-enqueue never fires here, since the client-proxy fast path is
    # what usually triggers that task) -- so the active-speaker pass is
    # chained here too, exactly the same way.
    try:
        app.configure_task("l1_active_speaker", queue="gpu").defer(file_id=file_id)
    except Exception:
        logger.exception("Could not enqueue active-speaker pass for %s.", file_id)

    if demux_thread is not None:
        demux_thread.join()
        if "e" in demux_err:
            raise demux_err["e"]

    # Prefer the 1080p proxy for the motion pass (much cheaper to decode than a
    # 4K raw). It exists locally whenever the proxy stage ran in this
    # invocation; on a rare retry where it doesn't, fall back to the raw.
    proxy_local = os.path.join(tmpdir, "proxy.mp4")
    motion_source = proxy_local if os.path.exists(proxy_local) else raw_path
    return duration_s, has_audio, motion_source


def _prepare_from_client_proxies(
    file_id: str, proxy_a_key: str, proxy_b_key: str, tmpdir: str, wav_path: str
) -> tuple[float, bool, str]:
    """Fast path: analysis runs entirely off the two tiny client proxies and
    never touches the multi-GB raw. Proxy A (480p@1fps + audio) feeds the whole
    speech/audio stack (WAV demux); proxy B (160x90@10fps) feeds motion.
    The 1080p editing proxy + real width/height/thumbnail are produced
    separately by `l1_editing_proxy` when the raw finishes uploading. Returns
    (duration_s, has_audio, motion_source)."""
    proxy_a_path = os.path.join(tmpdir, "proxy_a.mp4")
    motion_source = os.path.join(tmpdir, "proxy_b.mp4")
    logger.info("L1: downloading client proxies A/B for file %s", file_id)
    _download_from_r2(proxy_a_key, proxy_a_path)
    _download_from_r2(proxy_b_key, motion_source)

    probe = _probe_video(proxy_a_path)
    duration_s = float(probe.get("format", {}).get("duration", 0) or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    if not has_audio:
        logger.info("File %s proxy A has no audio; skipping transcript/audio.", file_id)

    # Record duration now so the deep-stage guardrail doesn't wait on the
    # editing-proxy task. width/height/thumbnail/status come from the raw in
    # l1_editing_proxy.
    try:
        get_supabase().table("files").update(
            {"duration_seconds": duration_s}
        ).eq("id", file_id).execute()
    except Exception:
        logger.exception("L1: failed to record duration for %s", file_id)

    if has_audio:
        _demux_wav(proxy_a_path, wav_path)
    return duration_s, has_audio, motion_source


@app.task(name="l1_editing_proxy", queue="gpu", retry=RetryStrategy(max_attempts=3, exponential_wait=4))
def l1_editing_proxy(file_id: str, r2_key: str) -> None:
    """Build the 1080p editing proxy + thumbnail from the raw once it finishes
    uploading. Split out of l1_orchestrate so analysis can run off the client
    proxies without blocking on the multi-GB raw. Idempotent via the shared
    'proxy' stage, so it's a harmless no-op when the raw-fallback path already
    generated it."""
    if not _file_exists(file_id):
        logger.info("File %s gone; skipping editing proxy.", file_id)
        return
    if _file_type(file_id) == "audio":
        return  # audio's playable proxy is made inside _orchestrate_audio
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw")
            logger.info("Editing proxy: downloading %s for file %s", r2_key, file_id)
            _download_from_r2(r2_key, raw_path)
            with _pg_conn() as conn:
                _run_stage(conn, file_id, "proxy", _stage1_proxy, file_id, raw_path, tmpdir)
    except psycopg.errors.ForeignKeyViolation:
        logger.info("File %s deleted mid editing-proxy; abandoning cleanly.", file_id)
        return
    except Exception:
        if not _file_exists(file_id):
            logger.info("File %s deleted mid editing-proxy; abandoning cleanly.", file_id)
            return
        logger.exception("Editing proxy failed for %s", file_id)
        raise

    # asd_identity.plan.md: the active-speaker pass needs the 1080p proxy
    # (real video+audio+fps -- the client A/B analysis proxies are unusable
    # for lip-sync, see active_speaker.py's own docstring), which the stage
    # just above guarantees is present (freshly built or already done).
    # Chaining here -- rather than enqueuing alongside from upload.py --
    # means l1_active_speaker never has to poll/retry waiting on the proxy.
    try:
        app.configure_task("l1_active_speaker", queue="gpu").defer(file_id=file_id)
    except Exception:
        logger.exception("Could not enqueue active-speaker pass for %s.", file_id)


@app.task(name="l1_active_speaker", queue="gpu", retry=RetryStrategy(max_attempts=3, exponential_wait=4))
def l1_active_speaker(file_id: str) -> None:
    """asd_identity.plan.md: detect+track+embed+ASD-score faces off the
    canonical editing proxy. Idempotent via the 'active_speaker'
    processing_jobs stage; chained off l1_editing_proxy above so the proxy
    is guaranteed to exist by the time this runs. Best-effort throughout --
    a failure here degrades identity to id-less PIC/SND for this file,
    never fails the file's own L1/ingest."""
    if not _file_exists(file_id):
        logger.info("File %s gone; skipping active-speaker pass.", file_id)
        return
    if _file_type(file_id) == "audio":
        return  # no video to detect faces in
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select r2_proxy_key from files where id = %s", (file_id,)
            ).fetchone()
        proxy_key = row[0] if row else None
        if not proxy_key:
            logger.warning(
                "File %s has no editing proxy yet; active-speaker pass has nothing to read.", file_id)
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            proxy_path = os.path.join(tmpdir, "proxy.mp4")
            _download_from_r2(proxy_key, proxy_path)
            with _pg_conn() as conn:
                _run_stage(conn, file_id, "active_speaker", _stage_active_speaker, file_id, proxy_path, conn)
    except psycopg.errors.ForeignKeyViolation:
        logger.info("File %s deleted mid active-speaker pass; abandoning cleanly.", file_id)
        return
    except Exception:
        if not _file_exists(file_id):
            logger.info("File %s deleted mid active-speaker pass; abandoning cleanly.", file_id)
            return
        logger.exception("Active-speaker pass failed for %s", file_id)
        raise


@app.task(name="l1_orchestrate", queue="gpu", retry=RetryStrategy(max_attempts=3, exponential_wait=4))
def l1_orchestrate(file_id: str, r2_key: str) -> None:
    """
    Runs the L1 analysis stages. When the client uploaded the two analysis
    proxies (see client_proxy.plan.md) analysis runs off them and never touches
    the raw; otherwise every input is regenerated from the raw (fallback).
    Branches by file_type: audio uploads run the video-free music path.
    Idempotent: each stage checks processing_jobs first.
    """
    # scale_architecture.plan.md Pillar 7: every log line for this file's L1
    # run carries file_id/user_id from here on (correlation.scope, not
    # threaded through every logger.info call by hand).
    with correlation.scope(file_id=file_id, user_id=_user_id_for_file(file_id)):
        _l1_orchestrate(file_id, r2_key)


def _l1_orchestrate(file_id: str, r2_key: str) -> None:
    settings = get_settings()

    if not _file_exists(file_id):
        logger.info("File %s no longer exists; skipping L1 (likely deleted/aborted).", file_id)
        return

    if _file_type(file_id) == "audio":
        _orchestrate_audio(file_id, r2_key, settings)
        return

    _set_l1_status(file_id, "running")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "audio.wav")
            proxy_a_key, proxy_b_key = _analysis_proxy_keys(file_id)

            if proxy_a_key and proxy_b_key:
                duration_s, has_audio, video_source = _prepare_from_client_proxies(
                    file_id, proxy_a_key, proxy_b_key, tmpdir, wav_path,
                )
            else:
                duration_s, has_audio, video_source = _prepare_from_raw(
                    file_id, r2_key, tmpdir, wav_path,
                )

            if duration_s > settings.max_l1_duration_seconds:
                logger.info(
                    "Duration %.1fs > guardrail %ds; skipping deep L1 stages",
                    duration_s, settings.max_l1_duration_seconds,
                )
                _set_l1_status(file_id, "skipped")
                return

            # Deep stages run as three concurrent tracks (speech / audio / video),
            # overlapping the two heaviest stages (Whisper + optical flow).
            _run_deep_stages_parallel(
                file_id, wav_path, video_source, duration_s, has_audio,
            )

        _set_l1_status(file_id, "ready")
        logger.info("L1 complete for %s", file_id)
        try:
            snapshot = build_l1_snapshot(file_id)
            log_path = audit_log.write_l1_analysis(file_id, snapshot)
            logger.info("L1 analysis written to %s", log_path)
        except Exception:
            logger.exception("L1 succeeded but writing the audit log failed for %s", file_id)
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
