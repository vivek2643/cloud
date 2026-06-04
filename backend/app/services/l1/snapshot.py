"""
Build a serializable, human-readable L1 snapshot for a single file from
whatever's already persisted in Postgres. Used by:
  - the L1 pipeline (writes logs/l1/<file_id>.json on success)
  - a CLI backfill script (regenerates the log for an already-indexed file)
  - the /api/files/{id}/l1 endpoint (returns this same JSON)

Anything we can't compute is left as null instead of crashing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings

logger = logging.getLogger(__name__)


def _row_to_dict(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {k: v for k, v in row.items()}


_L1_STAGES = ("proxy", "transcript", "shots", "embeddings", "audio_features")


def _iso(v) -> Optional[str]:
    return v.isoformat(timespec="seconds") if v is not None else None


def list_l1_analyses(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """List every file that has at least one L1 stage recorded, with the
    L1 wall-clock duration (span of all L1 processing_jobs) in seconds.

    Sourced entirely from Postgres so it works no matter which machine
    (local or a remote GPU worker) actually ran the analysis.
    """
    settings = get_settings()
    sql = """
        select
            f.id            as file_id,
            f.name          as name,
            f.l1_status     as l1_status,
            f.l2_status     as l2_status,
            f.duration_seconds as duration_seconds,
            f.created_at    as created_at,
            (select count(*) from shots s where s.file_id = f.id) as shot_count,
            -- Sum of per-stage durations (not the start->end span) so a
            -- failed-then-retried stage doesn't inflate the number with the
            -- idle gap between attempts. The serial L1 pipeline runs stages
            -- back-to-back, so this equals real compute time.
            (select extract(epoch from sum(pj.finished_at - pj.started_at))
               from processing_jobs pj
              where pj.file_id = f.id
                and pj.stage = any(%s)
                and pj.started_at is not null
                and pj.finished_at is not null) as l1_seconds,
            (select max(pj.finished_at)
               from processing_jobs pj
              where pj.file_id = f.id and pj.stage = any(%s)) as l1_finished_at
        from files f
        where f.user_id = %s
          and exists (
              select 1 from processing_jobs pj2
               where pj2.file_id = f.id and pj2.stage = any(%s)
          )
        order by f.created_at desc
        limit %s
    """
    stages = list(_L1_STAGES)
    out: List[Dict[str, Any]] = []
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        cur = conn.execute(sql, (stages, stages, user_id, stages, limit))
        for r in cur.fetchall():
            secs = r["l1_seconds"]
            out.append({
                "file_id": str(r["file_id"]),
                "name": r["name"],
                "l1_status": r["l1_status"],
                "l2_status": r["l2_status"],
                "duration_seconds": r["duration_seconds"],
                "shot_count": r["shot_count"],
                "l1_seconds": round(float(secs), 1) if secs is not None else None,
                "analyzed_at": _iso(r["l1_finished_at"]) or _iso(r["created_at"]),
            })
    return out


def list_l2_analyses(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """List every file that has had L2 enrichment started/finished, with the
    L2 wall-clock duration (stage='l2' processing_jobs row) in seconds."""
    settings = get_settings()
    sql = """
        select
            f.id            as file_id,
            f.name          as name,
            f.l2_status     as l2_status,
            f.created_at    as created_at,
            (select count(*) from shots s where s.file_id = f.id) as shot_count,
            (select count(*) from shots s
              where s.file_id = f.id and s.dinov2_embedding is not null) as enriched_shots,
            (select extract(epoch from (pj.finished_at - pj.started_at))
               from processing_jobs pj
              where pj.file_id = f.id and pj.stage = 'l2'
                and pj.started_at is not null and pj.finished_at is not null) as l2_seconds,
            (select pj.finished_at
               from processing_jobs pj
              where pj.file_id = f.id and pj.stage = 'l2') as l2_finished_at
        from files f
        where f.user_id = %s and f.l2_status is not null
        order by f.created_at desc
        limit %s
    """
    out: List[Dict[str, Any]] = []
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        cur = conn.execute(sql, (user_id, limit))
        for r in cur.fetchall():
            secs = r["l2_seconds"]
            out.append({
                "file_id": str(r["file_id"]),
                "name": r["name"],
                "l2_status": r["l2_status"],
                "shot_count": r["shot_count"],
                "enriched_shots": r["enriched_shots"],
                "l2_seconds": round(float(secs), 1) if secs is not None else None,
                "analyzed_at": _iso(r["l2_finished_at"]) or _iso(r["created_at"]),
            })
    return out


def build_l1_snapshot(file_id: str) -> Dict[str, Any]:
    """Read every L1-relevant row for `file_id` and return a clean JSON-able dict.

    Schema is stable: callers (frontend, log viewer) can rely on these keys.
    """
    settings = get_settings()
    out: Dict[str, Any] = {
        "file_id": file_id,
        "file": None,
        "shots": [],
        "transcript": None,
        "audio_features": None,
        "processing_jobs": [],
        "summary": {},
    }

    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        # File row (only the fields we care about)
        cur = conn.execute(
            """
            select id, name, filename, mime_type, file_size, file_type,
                   r2_key, r2_proxy_key, r2_thumbnail_key,
                   duration_seconds, width, height,
                   status, l1_status, l2_status,
                   created_at, updated_at
              from files where id = %s
            """,
            (file_id,),
        )
        out["file"] = _row_to_dict(cur.fetchone())

        # Shots
        cur = conn.execute(
            """
            select shot_index, start_ms, end_ms,
                   keyframe_r2_key, r2_keyframe_motion_key, r2_keyframe_variance_key,
                   peak_motion_ms, peak_variance_ms,
                   focus_score, brightness, motion_magnitude,
                   intra_shot_variance, blur_min,
                   framing_scale, camera_dynamics,
                   narrative_description, narrative_role, emotional_valence
              from shots where file_id = %s order by shot_index
            """,
            (file_id,),
        )
        out["shots"] = [
            {**r, "duration_ms": r["end_ms"] - r["start_ms"]}
            for r in cur.fetchall()
        ]

        # Transcript (no embedding bytes; just the human-readable bits)
        cur = conn.execute(
            """
            select language, text, segments, fillers
              from transcripts where file_id = %s
            """,
            (file_id,),
        )
        tr = cur.fetchone()
        if tr:
            out["transcript"] = {
                "language": tr["language"],
                "text": tr["text"],
                "segment_count": len(tr["segments"] or []),
                "filler_count": len(tr["fillers"] or []),
                "segments": tr["segments"],
                "fillers": tr["fillers"],
            }

        # Audio features
        cur = conn.execute(
            """
            select integrated_lufs, true_peak_db,
                   is_musical, bpm,
                   onsets_ms, silence_intervals,
                   acoustic_tags, event_segments
              from audio_features where file_id = %s
            """,
            (file_id,),
        )
        af = cur.fetchone()
        if af:
            out["audio_features"] = {
                "integrated_lufs": af["integrated_lufs"],
                "true_peak_db": af["true_peak_db"],
                "is_musical": af["is_musical"],
                "bpm": af["bpm"],
                "onset_count": len(af["onsets_ms"] or []),
                "silence_interval_count": len(af["silence_intervals"] or []),
                "silence_intervals": af["silence_intervals"],
                "acoustic_tags": af["acoustic_tags"],
                "event_segments": af["event_segments"],
            }

        # Per-stage processing job rows
        cur = conn.execute(
            """
            select stage, status, attempts, error, started_at, finished_at
              from processing_jobs where file_id = %s order by stage
            """,
            (file_id,),
        )
        out["processing_jobs"] = [_row_to_dict(r) for r in cur.fetchall()]

        # Embedding count (for sanity)
        cur = conn.execute(
            """
            select count(*) as n
              from shot_embeddings se
              join shots s on s.id = se.shot_id
             where s.file_id = %s
            """,
            (file_id,),
        )
        out["summary"]["shot_embeddings_stored"] = cur.fetchone()["n"]

    # High-level summary numbers
    shots = out["shots"]
    out["summary"]["shot_count"] = len(shots)
    if shots:
        durations = [s["duration_ms"] for s in shots]
        out["summary"]["avg_shot_duration_ms"] = int(sum(durations) / len(durations))
        out["summary"]["min_shot_duration_ms"] = min(durations)
        out["summary"]["max_shot_duration_ms"] = max(durations)
        # New L1.C metrics: how many shots have multi-keyframe data and
        # how often we'd default-trim them.
        with_var = [s["intra_shot_variance"] for s in shots if s.get("intra_shot_variance") is not None]
        if with_var:
            out["summary"]["avg_intra_shot_variance"] = round(sum(with_var) / len(with_var), 4)
            out["summary"]["max_intra_shot_variance"] = round(max(with_var), 4)
            out["summary"]["chaotic_shots"] = sum(1 for v in with_var if v >= 0.25)
        blurs = [s["blur_min"] for s in shots if s.get("blur_min") is not None]
        if blurs:
            out["summary"]["min_blur_score"] = round(min(blurs), 2)
            out["summary"]["avg_blur_score"] = round(sum(blurs) / len(blurs), 2)
    if out["transcript"]:
        out["summary"]["transcript_chars"] = len(out["transcript"]["text"] or "")
        out["summary"]["transcript_segment_count"] = out["transcript"]["segment_count"]
        out["summary"]["filler_count"] = out["transcript"]["filler_count"]
    return out
