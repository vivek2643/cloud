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


_L1_STAGES = ("proxy", "transcript", "audio_features", "diarization",
              "motion_dynamics", "dialogue_segments", "audio_proxy",
              "scene_detect")  # cuts-v2, additive -- see STAGES_V2


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
            f.duration_seconds as duration_seconds,
            f.created_at    as created_at,
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
                "duration_seconds": r["duration_seconds"],
                "l1_seconds": round(float(secs), 1) if secs is not None else None,
                "analyzed_at": _iso(r["l1_finished_at"]) or _iso(r["created_at"]),
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
                   status, l1_status,
                   created_at, updated_at
              from files where id = %s
            """,
            (file_id,),
        )
        out["file"] = _row_to_dict(cur.fetchone())

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
                   prosody_hop_ms, rms_db
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
                "prosody_hop_ms": af["prosody_hop_ms"],
                "rms_db": af["rms_db"],
            }

        # Motion dynamics (action + camera/distortion) -- video-derived, own table.
        cur = conn.execute(
            """
            select hop_ms, action_energy, camera_motion, camera_coherence,
                   camera_stability, blur, action_cut_cost, camera_cut_cost,
                   action_points, transition_points
              from motion_dynamics where file_id = %s
            """,
            (file_id,),
        )
        md = cur.fetchone()
        if md:
            out["motion_dynamics"] = {
                "hop_ms": md["hop_ms"],
                "action_energy": md["action_energy"] or [],
                "camera_motion": md["camera_motion"] or [],
                # Camera-motion QUALITY: coherence (rigid global move) + stability
                # (sustained vs jerky). High+high => a deliberate move (cuttable).
                "camera_coherence": md["camera_coherence"] or [],
                "camera_stability": md["camera_stability"] or [],
                "blur": md["blur"] or [],
                # Derived cut-cost channels (0=ideal seam .. 1=avoid).
                "action_cut_cost": md["action_cut_cost"] or [],
                "camera_cut_cost": md["camera_cut_cost"] or [],
                "action_points": md["action_points"] or [],
                "action_point_count": len(md["action_points"] or []),
                # cuts-v3 premium natural cut instants (occlusion wipe / degenerate span).
                "transition_points": md["transition_points"] or [],
                "transition_point_count": len(md["transition_points"] or []),
            }

        # Scene/shot detection (cuts-v2) -- video-derived, own table.
        cur = conn.execute(
            """
            select hop_ms, shot_points, composition_points
              from scene_cuts where file_id = %s
            """,
            (file_id,),
        )
        sc = cur.fetchone()
        if sc:
            out["scene_cuts"] = {
                "hop_ms": sc["hop_ms"],
                "shot_points": sc["shot_points"] or [],
                "shot_point_count": len(sc["shot_points"] or []),
                "composition_points": sc["composition_points"] or [],
                "composition_point_count": len(sc["composition_points"] or []),
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

    # High-level summary numbers
    if out["transcript"]:
        out["summary"]["transcript_chars"] = len(out["transcript"]["text"] or "")
        out["summary"]["transcript_segment_count"] = out["transcript"]["segment_count"]
        out["summary"]["filler_count"] = out["transcript"]["filler_count"]
    return out
