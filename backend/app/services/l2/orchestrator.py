"""
L2 orchestrator: runs the four L2 stages against a target set of shots and
writes back into the nullable columns added by migration 003.

Two entry points:
  - enrich_file(file_id):      run all stages on every shot in one file
  - enrich_shots(shot_ids):    run all stages on just these shots
                               (used by the edit-request candidate enricher)

We download the raw video once per file in either case so the 3-keyframe
extractor and audio-event stage can read pixels and waveforms.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services import audit_log
from app.services.jobs import app as proc_app
from app.services.l1 import keyframes as kf_mod
from app.services.l1.pipeline import _vec_to_pg
from app.services.l1.snapshot import build_l1_snapshot
from app.services.l2 import audio_events_stage as audio_mod
from app.services.l2 import dinov2_stage as dino_mod
from app.services.l2 import faces_stage as faces_mod
from app.services.l2 import narrative_stage as narrative_mod
from app.services.processing import _download_from_r2
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

# Number of concurrent Claude Vision calls in Stage D. Anthropic's published
# rate limits comfortably handle 5; raise carefully if you observe throttling.
NARRATIVE_PARALLEL = 5


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _set_l2_status(file_id: str, status: Optional[str]) -> None:
    sb = get_supabase()
    sb.table("files").update({"l2_status": status}).eq("id", file_id).execute()


# --- L2 wall-clock timing -------------------------------------------------
# We reuse the same processing_jobs ledger the L1 pipeline uses, under a
# synthetic stage name 'l2', so the logs UI can show how long enrichment took
# (finished_at - started_at). One row per file; re-running resets the clock.

def _l2_timing_begin(conn: psycopg.Connection, file_id: str) -> None:
    conn.execute(
        """
        insert into processing_jobs (file_id, stage, status, started_at, attempts)
        values (%s, 'l2', 'running', now(), 1)
        on conflict (file_id, stage) do update set
            status = 'running',
            started_at = now(),
            finished_at = null,
            attempts = processing_jobs.attempts + 1,
            error = null
        """,
        (file_id,),
    )


def _l2_timing_done(conn: psycopg.Connection, file_id: str) -> None:
    conn.execute(
        """
        update processing_jobs
           set status = 'done', finished_at = now(), error = null
         where file_id = %s and stage = 'l2'
        """,
        (file_id,),
    )


def _l2_timing_fail(conn: psycopg.Connection, file_id: str, err: str) -> None:
    conn.execute(
        """
        update processing_jobs
           set status = 'failed', finished_at = now(), error = %s
         where file_id = %s and stage = 'l2'
        """,
        (err[:8000], file_id),
    )


def _fetch_target_shots(
    conn: psycopg.Connection,
    file_id: Optional[str],
    shot_ids: Optional[Iterable[str]],
):
    common_cols = """
        s.id, s.file_id, s.shot_index, s.start_ms, s.end_ms,
        s.focus_score, s.motion_magnitude,
        s.keyframe_r2_key, s.r2_keyframe_motion_key, s.r2_keyframe_variance_key,
        s.dinov2_embedding is not null as has_dinov2,
        s.tracked_character_ids is not null as has_faces,
        s.narrative_description is not null as has_narrative
    """
    if shot_ids:
        cur = conn.execute(
            f"select {common_cols} from shots s "
            "where s.id = any(%s::uuid[]) order by s.file_id, s.shot_index",
            (list(shot_ids),),
        )
    else:
        cur = conn.execute(
            f"select {common_cols} from shots s "
            "where s.file_id = %s order by s.shot_index",
            (file_id,),
        )
    return cur.fetchall()


def _fetch_file(conn: psycopg.Connection, file_id: str):
    cur = conn.execute(
        "select id, user_id, r2_key from files where id = %s",
        (file_id,),
    )
    return cur.fetchone()


def _stage_a_dinov2(conn, shot_row, anchor_path: str):
    """Embed anchor frame + write heuristics."""
    vecs = dino_mod.embed_images([anchor_path])
    if vecs.shape[0] == 0:
        return
    framing = dino_mod.infer_framing_from_focus(
        shot_row.get("focus_score"), shot_row.get("motion_magnitude"),
    )
    dynamics = dino_mod.infer_camera_dynamics(shot_row.get("motion_magnitude"))
    conn.execute(
        """
        update shots
           set dinov2_embedding = %s::halfvec,
               framing_scale    = %s,
               camera_dynamics  = %s
         where id = %s
        """,
        (_vec_to_pg(vecs[0]), framing, dynamics, shot_row["id"]),
    )


def _stage_b_faces(conn, user_id: str, shot_id: str, paths: List[str]):
    faces_mod.enrich_shot_with_faces(user_id, shot_id, paths)


def _stage_c_audio_events(file_id: str, wav_path: str):
    tags, segments = audio_mod.analyze(wav_path)
    sb = get_supabase()
    sb.table("audio_features").upsert({
        "file_id": file_id,
        "acoustic_tags": tags,
        "event_segments": audio_mod.serialize_segments(segments),
    }).execute()


def _write_narrative(conn, shot_id: str, result) -> None:
    """Single-threaded writeback. Called from the main thread after the
    parallel narrative batch completes."""
    if not result:
        return
    conn.execute(
        """
        update shots
           set narrative_description = %s,
               narrative_role        = %s,
               emotional_valence     = %s
         where id = %s
        """,
        (result.description, result.role, result.valence, shot_id),
    )


def _fetch_transcript_segments(conn, file_id: str) -> list:
    cur = conn.execute(
        "select segments from transcripts where file_id = %s",
        (file_id,),
    )
    row = cur.fetchone()
    if not row:
        return []
    return row["segments"] or []


def _slice_transcript(segments: list, start_ms: int, end_ms: int) -> Optional[str]:
    """Pure in-memory slice; safe to call from worker threads."""
    parts: List[str] = []
    for seg in segments:
        if seg.get("end_ms", 0) < start_ms or seg.get("start_ms", 0) > end_ms:
            continue
        parts.append(seg.get("text", ""))
    return " ".join(parts).strip() or None


def _run_narratives_parallel(jobs: List[dict]) -> List[tuple]:
    """
    Fire `narrative_mod.analyze_shot` on all jobs concurrently.
    `jobs` is a list of `{"shot_id": str, "paths": [str], "text": str|None}`.
    Returns a list of `(shot_id, NarrativeResult|None)`.

    The Anthropic SDK is thread-safe; each thread holds its own HTTP
    connection. We deliberately avoid asyncio here because the rest of
    the orchestrator (psycopg, ffmpeg subprocess) is sync.
    """
    if not jobs:
        return []
    results: List[tuple] = []

    def _do_one(job: dict):
        return job["shot_id"], narrative_mod.analyze_shot(job["paths"], job["text"])

    logger.info("Stage D: dispatching %d narrative calls (concurrency=%d)", len(jobs), NARRATIVE_PARALLEL)
    with ThreadPoolExecutor(max_workers=NARRATIVE_PARALLEL) as ex:
        futures = {ex.submit(_do_one, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception("Stage D narrative failed for shot %s", j["shot_id"])
                results.append((j["shot_id"], None))
    return results


def _demux_wav(raw_path: str, out_path: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", out_path],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.exists(out_path)


def _materialize_keyframes(
    row: dict,
    raw_path: str,
    tmp: str,
) -> List[Optional[str]]:
    """
    Get the 3 keyframe paths for a shot, preferring R2 (downloaded by L1)
    and falling back to fresh extraction for legacy rows that pre-date
    migration 005.

    Returns a list of 3 local paths in [anchor, motion, variance] order;
    individual entries may be None if a particular keyframe is unavailable.
    """
    kf_dir = os.path.join(tmp, f"shot_{row['shot_index']:05d}")
    os.makedirs(kf_dir, exist_ok=True)

    keys = (
        ("anchor",   row.get("keyframe_r2_key")),
        ("motion",   row.get("r2_keyframe_motion_key")),
        ("variance", row.get("r2_keyframe_variance_key")),
    )

    if all(k for _, k in keys):
        # Fast path: every keyframe exists on R2; just download.
        out: List[Optional[str]] = []
        for kind, k in keys:
            local = os.path.join(kf_dir, f"{kind}.jpg")
            try:
                _download_from_r2(k, local)
                out.append(local if os.path.exists(local) else None)
            except Exception:
                logger.exception("Failed to download keyframe %s; will re-extract", k)
                out.append(None)
        if all(out):
            return out
        # Partial download failure -> fall through to extract.

    # Legacy / fallback path: extract from the raw video.
    kfs = kf_mod.extract_three(
        raw_path, row["start_ms"], row["end_ms"], kf_dir,
        f"shot_{row['shot_index']:05d}",
    )
    return [
        kfs.anchor_path,
        kfs.motion_path or kfs.anchor_path,
        kfs.variance_path or kfs.anchor_path,
    ]


def enrich(
    *,
    file_id: Optional[str] = None,
    shot_ids: Optional[Iterable[str]] = None,
    run_dinov2: bool = True,
    run_faces: bool = True,
    run_audio_events: bool = True,
    run_narrative: bool = True,
) -> dict:
    """
    Run L2 stages on the target shots. Either file_id (whole file) or
    shot_ids (subset) must be supplied. shot_ids may span multiple files;
    we group by file to avoid re-downloading the raw video.
    """
    if not file_id and not shot_ids:
        raise ValueError("Pass either file_id or shot_ids")

    with _pg() as conn:
        shots = _fetch_target_shots(conn, file_id, shot_ids)
        if not shots:
            return {"shots_processed": 0, "files_processed": 0}

        by_file: dict[str, list] = {}
        for row in shots:
            by_file.setdefault(str(row["file_id"]), []).append(row)

        files_processed = 0
        shots_processed = 0
        for fid, rows in by_file.items():
            file_row = _fetch_file(conn, fid)
            if not file_row:
                continue
            _set_l2_status(fid, "running")
            _l2_timing_begin(conn, fid)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    raw_path = os.path.join(tmp, "raw")
                    _download_from_r2(file_row["r2_key"], raw_path)
                    wav_path = os.path.join(tmp, "audio.wav")
                    have_wav = _demux_wav(raw_path, wav_path)

                    if run_audio_events and have_wav:
                        try:
                            _stage_c_audio_events(fid, wav_path)
                        except Exception:
                            logger.exception("Stage C failed for file %s", fid)

                    # Pre-fetch transcript once per file (Stage D needs it
                    # but psycopg connections aren't thread-safe).
                    segments = _fetch_transcript_segments(conn, fid)

                    # Phase 1: per-shot, sequential -- keyframes, DINOv2, faces.
                    # We collect Stage D inputs here but DEFER the API calls
                    # to Phase 2 so they can run concurrently.
                    narrative_jobs: List[dict] = []
                    for row in rows:
                        triple = _materialize_keyframes(row, raw_path, tmp)
                        anchor, motion, variance = triple
                        paths = [p for p in (anchor, motion, variance) if p]

                        if run_dinov2 and anchor and not row["has_dinov2"]:
                            try:
                                _stage_a_dinov2(conn, row, anchor)
                            except Exception:
                                logger.exception("Stage A failed for shot %s", row["id"])

                        if run_faces and paths and not row["has_faces"]:
                            try:
                                _stage_b_faces(conn, str(file_row["user_id"]), str(row["id"]), paths)
                            except Exception:
                                logger.exception("Stage B failed for shot %s", row["id"])

                        if run_narrative and paths and not row["has_narrative"]:
                            narrative_jobs.append({
                                "shot_id": str(row["id"]),
                                "paths": paths,
                                "text": _slice_transcript(segments, row["start_ms"], row["end_ms"]),
                            })

                        shots_processed += 1

                    # Phase 2: parallel Stage D Claude vision calls.
                    # Phase 3: single-threaded writeback (psycopg).
                    for shot_id, result in _run_narratives_parallel(narrative_jobs):
                        try:
                            _write_narrative(conn, shot_id, result)
                        except Exception:
                            logger.exception("Stage D writeback failed for shot %s", shot_id)

                _l2_timing_done(conn, fid)
                _set_l2_status(fid, "ready")
                files_processed += 1
                try:
                    snap = build_l1_snapshot(fid)
                    audit_log.write_l1_analysis(fid, snap)
                except Exception:
                    logger.exception("L2 succeeded but writing the audit log failed for %s", fid)
            except Exception as exc:
                logger.exception("L2 enrich failed for file %s", fid)
                try:
                    _l2_timing_fail(conn, fid, str(exc))
                except Exception:
                    logger.exception("Failed to record L2 timing failure for %s", fid)
                _set_l2_status(fid, "failed")

    return {"shots_processed": shots_processed, "files_processed": files_processed}


# --- Procrastinate task wrappers -----------------------------------------

@proc_app.task(name="l2_enrich_file", queue="gpu", retry={"max_attempts": 2, "wait": "exponential"})
def l2_enrich_file_task(file_id: str) -> None:
    enrich(file_id=file_id)


@proc_app.task(name="l2_enrich_shots", retry={"max_attempts": 2, "wait": "exponential"})
def l2_enrich_shots_task(shot_ids: list[str]) -> None:
    enrich(shot_ids=shot_ids)
