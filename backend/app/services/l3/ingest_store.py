"""
Cuts: DB persistence for ``ingest_runs`` + ``cut_records`` (migration
024_cuts_v3.sql). Plain psycopg, autocommit -- the orchestrator (``ingest.py``)
calls these at each stage transition so a crash mid-run leaves an
inspectable, re-runnable row instead of losing all progress silently.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from app.services.l3.post import CutRecord


def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def create_ingest_run(project_id: str, pass1_model: str, pass2_model: str) -> str:
    with _pg_conn() as conn:
        row = conn.execute(
            """
            insert into ingest_runs (project_id, status, pass1_model, pass2_model)
            values (%s, 'pending', %s, %s)
            returning id::text
            """,
            (project_id, pass1_model, pass2_model),
        ).fetchone()
    return row[0]


def set_status(ingest_run_id: str, status: str, error: Optional[str] = None) -> None:
    with _pg_conn() as conn:
        conn.execute(
            "update ingest_runs set status = %s, error = %s where id = %s",
            (status, error, ingest_run_id),
        )


def record_pass1_result(ingest_run_id: str, pass1_output: Dict[str, Any],
                        usage: Dict[str, int], project_summary: str) -> None:
    with _pg_conn() as conn:
        conn.execute(
            """
            update ingest_runs
               set status = 'pass1', pass1_output = %s, project_summary = %s,
                   input_tokens = input_tokens + %s, output_tokens = output_tokens + %s,
                   cache_read_tokens = cache_read_tokens + %s,
                   cache_write_tokens = cache_write_tokens + %s
             where id = %s
            """,
            (json.dumps(pass1_output), project_summary,
             usage.get("input_tokens", 0), usage.get("output_tokens", 0),
             usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0),
             ingest_run_id),
        )


def set_identity_map(ingest_run_id: str, identity_map: Dict[str, Any]) -> None:
    """Persist this run's reconciled-cast payload (identity_map.plan.md
    Phase 3b) -- `footage_map.assemble_map` loads it back by `run_id` to
    fill the `oncam`/`alias` maps."""
    with _pg_conn() as conn:
        conn.execute(
            "update ingest_runs set identity_map = %s where id = %s",
            (json.dumps(identity_map), ingest_run_id),
        )


def get_identity_map(ingest_run_id: str) -> Optional[Dict[str, Any]]:
    """None for an older run, a run identity reconciliation found nothing
    to bind/cluster for, or an unknown run id -- callers treat that as
    "no reconciled cast," never an error (fail-open, same contract as
    every other identity_map fallback)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select identity_map from ingest_runs where id = %s", (ingest_run_id,)
        ).fetchone()
    if not row or not row[0]:
        return None
    return row[0] if isinstance(row[0], dict) else json.loads(row[0])


def accumulate_pass2_usage(ingest_run_id: str, usage: Dict[str, int]) -> None:
    with _pg_conn() as conn:
        conn.execute(
            """
            update ingest_runs
               set input_tokens = input_tokens + %s, output_tokens = output_tokens + %s,
                   cache_read_tokens = cache_read_tokens + %s,
                   cache_write_tokens = cache_write_tokens + %s
             where id = %s
            """,
            (usage.get("input_tokens", 0), usage.get("output_tokens", 0),
             usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0),
             ingest_run_id),
        )


def take_group_uuid_map(records: List[CutRecord]) -> Dict[str, str]:
    """The LLM emits arbitrary string take-group ids (e.g. "tg1"); the
    ``cut_records.take_group_id`` column is a real uuid, so map each
    distinct string to one fresh uuid, stable across every record in this
    run that shares it."""
    ids = sorted({r.take_group_id for r in records if r.take_group_id})
    return {tg: str(uuid.uuid4()) for tg in ids}


def delete_cut_records_for_run(ingest_run_id: str) -> None:
    with _pg_conn() as conn:
        conn.execute("delete from cut_records where ingest_run_id = %s", (ingest_run_id,))


def insert_cut_records(ingest_run_id: str, records: List[CutRecord]) -> List[str]:
    """Insert every assembled CutRecord, return their new ids in the same
    order. A run is re-run in full (never patched) -- callers should
    ``delete_cut_records_for_run`` first when re-inserting for an existing
    ingest_run_id."""
    tg_map = take_group_uuid_map(records)
    ids: List[str] = []
    with _pg_conn() as conn:
        for r in records:
            row = conn.execute(
                """
                insert into cut_records (
                    ingest_run_id, file_id, src_in_ms, src_out_ms, kind,
                    word_span, atom_ids, label, summary, on_camera,
                    take_group_id, take_role, junk, junk_reason,
                    framing, look, caption_zones, pace, hero_ts_ms, channel, continuity,
                    speech_quality, total_quality, characteristics, camera, sync_group_id,
                    screen_text, salience, voice_ids, speaker_person, visible_persons,
                    audio_file_id, audio_offset_ms, audio_align_confidence
                ) values (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                returning id::text
                """,
                (
                    ingest_run_id, r.file_id, r.src_in_ms, r.src_out_ms, r.kind,
                    json.dumps(list(r.word_span)) if r.word_span else None,
                    json.dumps(r.atom_ids) if r.atom_ids is not None else None,
                    r.label, r.summary, r.on_camera,
                    tg_map.get(r.take_group_id) if r.take_group_id else None, r.take_role,
                    r.junk, r.junk_reason,
                    json.dumps(r.framing), json.dumps(r.look),
                    json.dumps([list(z) for z in r.caption_zones]),
                    json.dumps(r.pace.to_dict()), r.hero_ts_ms, r.channel,
                    json.dumps(r.continuity),
                    r.speech_quality, r.total_quality, json.dumps(r.characteristics),
                    r.camera, r.sync_group_id,
                    r.screen_text, json.dumps(r.salience),
                    json.dumps(r.voice_ids), r.speaker_person, json.dumps(r.visible_persons),
                    r.audio_file_id or None, r.audio_offset_ms, r.audio_align_confidence,
                ),
            ).fetchone()
            ids.append(row[0])
    return ids


def set_hero_key(cut_record_id: str, hero_key: str) -> None:
    with _pg_conn() as conn:
        conn.execute("update cut_records set hero_key = %s where id = %s", (hero_key, cut_record_id))
