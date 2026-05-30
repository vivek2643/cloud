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
import traceback
from typing import List, Optional

import numpy as np
import psycopg

from app.config import get_settings
from app.services import audit_log
from app.services.jobs import app
from app.services.l1 import audio_features as af_mod
from app.services.l1 import embeddings as emb_mod
from app.services.l1 import keyframes as kf_mod
from app.services.l1 import shots as shots_mod
from app.services.l1 import transcript as tr_mod
from app.services.l1.snapshot import build_l1_snapshot
from app.services.processing import _download_from_r2, _probe_video, _upload_to_r2
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

STAGES = ("proxy", "transcript", "shots", "embeddings", "audio_features")


def _pg_conn() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


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


# --- Stage 1: proxy + thumb (refactor of existing process_video) ---------

def _stage1_proxy(file_id: str, raw_path: str, tmpdir: str) -> tuple[float, int, int]:
    """Probe + 1080p proxy + thumbnail. Returns (duration, w, h)."""
    sb = get_supabase()
    proxy_path = os.path.join(tmpdir, "proxy.mp4")
    thumb_path = os.path.join(tmpdir, "thumb.jpg")

    probe = _probe_video(raw_path)
    duration = float(probe.get("format", {}).get("duration", 0))
    width, height = 0, 0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            break

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", raw_path,
            "-vf", "scale=-2:1080",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            proxy_path,
        ],
        check=True, capture_output=True,
    )
    proxy_key = f"proxies/{file_id}/proxy.mp4"
    _upload_to_r2(proxy_path, proxy_key, "video/mp4")

    thumb_time = max(duration * 0.25, 1.0)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(thumb_time),
            "-i", raw_path,
            "-vframes", "1",
            "-vf", "scale=640:-2",
            "-q:v", "3",
            thumb_path,
        ],
        check=True, capture_output=True,
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
        check=True, capture_output=True,
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
    raw_path: str,
    duration_s: float,
    tmpdir: str,
    conn: psycopg.Connection,
) -> List[shots_mod.Shot]:
    """
    Detect shots + extract 3 keyframes per shot. For each keyframe:
      - keep the full-res JPEG locally (used by SigLIP embedding stage)
      - downscale to 224x224 and upload that compact copy to R2
    """
    shots_dir = os.path.join(tmpdir, "keyframes")
    r2_dir = os.path.join(tmpdir, "keyframes_r2")
    os.makedirs(shots_dir, exist_ok=True)
    os.makedirs(r2_dir, exist_ok=True)

    shots = shots_mod.detect_shots(raw_path, duration_s, shots_dir)

    conn.execute("delete from shots where file_id = %s", (file_id,))

    for shot in shots:
        # Upload all 3 keyframes (anchor / motion / variance) at 224x224.
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
            if kind == "anchor":
                shot.keyframe_r2_key = key
            elif kind == "motion":
                shot.r2_keyframe_motion_key = key
            else:
                shot.r2_keyframe_variance_key = key

        conn.execute(
            """
            insert into shots (
                file_id, shot_index, start_ms, end_ms,
                keyframe_r2_key, r2_keyframe_motion_key, r2_keyframe_variance_key,
                focus_score, brightness, motion_magnitude,
                blur_min, peak_motion_ms, peak_variance_ms
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                file_id, shot.index, shot.start_ms, shot.end_ms,
                shot.keyframe_r2_key,
                shot.r2_keyframe_motion_key,
                shot.r2_keyframe_variance_key,
                shot.focus_score, shot.brightness, shot.motion_magnitude,
                shot.blur_min, shot.motion_ts_ms, shot.variance_ts_ms,
            ),
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


# --- Stage 5: audio features ---------------------------------------------

def _stage5_audio(file_id: str, wav_path: str, conn: psycopg.Connection) -> None:
    af = af_mod.compute_audio_features(wav_path)
    conn.execute(
        """
        insert into audio_features (
            file_id, integrated_lufs, true_peak_db,
            is_musical, bpm, onsets_ms, silence_intervals
        ) values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        on conflict (file_id) do update set
            integrated_lufs = excluded.integrated_lufs,
            true_peak_db = excluded.true_peak_db,
            is_musical = excluded.is_musical,
            bpm = excluded.bpm,
            onsets_ms = excluded.onsets_ms,
            silence_intervals = excluded.silence_intervals
        """,
        (
            file_id,
            af.integrated_lufs, af.true_peak_db,
            af.is_musical, af.bpm,
            json.dumps(af.onsets_ms),
            json.dumps(af.silence_intervals),
        ),
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

@app.task(name="l1_orchestrate", retry={"max_attempts": 3, "wait": "exponential"})
def l1_orchestrate(file_id: str, r2_key: str) -> None:
    """
    Single procrastinate task that downloads the raw video once and runs all
    five L1 stages. Idempotent: each stage checks processing_jobs first.
    """
    settings = get_settings()
    _set_l1_status(file_id, "running")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw")
            wav_path = os.path.join(tmpdir, "audio.wav")
            logger.info("L1: downloading %s for file %s", r2_key, file_id)
            _download_from_r2(r2_key, raw_path)

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
                    _set_l1_status(file_id, "skipped")
                    return

                _demux_wav(raw_path, wav_path)

                shots_result = _run_stage(
                    conn, file_id, "shots",
                    _stage3_shots, file_id, raw_path, duration_s, tmpdir, conn,
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

                _run_stage(conn, file_id, "transcript",
                           _stage2_transcript, file_id, wav_path, conn)

                _run_stage(conn, file_id, "audio_features",
                           _stage5_audio, file_id, wav_path, conn)

        _set_l1_status(file_id, "ready")
        logger.info("L1 complete for %s", file_id)
        try:
            snapshot = build_l1_snapshot(file_id)
            log_path = audit_log.write_l1_analysis(file_id, snapshot)
            logger.info("L1 analysis written to %s", log_path)
        except Exception:
            logger.exception("L1 succeeded but writing the audit log failed for %s", file_id)
    except Exception:
        logger.exception("L1 failed for %s", file_id)
        _set_l1_status(file_id, "failed")
        raise
