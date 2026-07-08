"""
Cuts v3 read path: ``GET /api/projects/{id}/cuts-v3``. Pure DB read, zero
model calls -- the latest ``ingest_run`` for a project plus every
``cut_record`` it produced. See cuts_v3.plan.md section 7.

Also the resolver + shared row fetch the agentic editor's footage map
(``cutrecord_map.py``) uses to read the SAME ``cut_records`` a thread's files
were ingested into -- see cuts_v3_to_brain.plan.md.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


def _pg_conn():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _iso(v) -> Optional[str]:
    return v.isoformat(timespec="seconds") if v is not None else None


# --------------------------------------------------------------------------
# Thread (file-scoped) <-> project/run (ingest-scoped) resolver
# --------------------------------------------------------------------------

def latest_run_for_files(file_ids: List[str]) -> Optional[str]:
    """The ``ingest_run_id`` with the best ``cut_records`` coverage of
    ``file_ids`` -- most distinct files covered, ties broken by most recent.
    None when no run has any cut_records for these files yet (not ingested via
    Cuts v3). Edit threads are file-scoped while cut_records are ingest-run
    -scoped; this is the join that lets the brain find "the run the editor was
    looking at" from a thread's ``file_ids`` alone."""
    if not file_ids:
        return None
    with _pg_conn() as conn:
        row = conn.execute(
            """
            select cr.ingest_run_id::text as run_id
              from cut_records cr
              join ingest_runs ir on ir.id = cr.ingest_run_id
             where cr.file_id = any(%s::uuid[])
             group by cr.ingest_run_id, ir.created_at
             order by count(distinct cr.file_id) desc, ir.created_at desc
             limit 1
            """,
            (file_ids,),
        ).fetchone()
    return row["run_id"] if row else None


def rows_for_run(run_id: str, file_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every ``cut_record`` row for one ingest run, optionally scoped to a set
    of files. The reusable row fetch behind both the UI read path
    (``load_cuts_v3``) and the brain's projection (``cutrecord_map``)."""
    query = (
        "select id::text, file_id::text, src_in_ms, src_out_ms, kind, word_span, atom_ids,\n"
        "       label, summary, speaker, on_camera, take_group_id::text, take_role, channel,\n"
        "       junk, junk_reason, junk_confidence, framing, look, caption_zones, pace,\n"
        "       hero_ts_ms, hero_key, transition_in, transition_out\n"
        "  from cut_records where ingest_run_id = %s"
    )
    params: List[Any] = [run_id]
    if file_ids:
        query += " and file_id = any(%s::uuid[])"
        params.append(file_ids)
    query += " order by file_id, src_in_ms"
    with _pg_conn() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def load_cuts_v3(project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """None when the project doesn't exist or isn't owned by ``user_id``."""
    with _pg_conn() as conn:
        proj = conn.execute(
            "select id::text, name from projects where id = %s and user_id = %s",
            (project_id, user_id),
        ).fetchone()
        if not proj:
            return None

        run = conn.execute(
            """
            select id::text, status, pass1_model, pass2_model,
                   input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                   cost_usd, project_summary, error, created_at, updated_at
              from ingest_runs where project_id = %s
             order by created_at desc limit 1
            """,
            (project_id,),
        ).fetchone()

        if run is None:
            return {"project_id": project_id, "name": proj["name"], "ingest_run": None, "cuts": []}

    return {
        "project_id": project_id,
        "name": proj["name"],
        "ingest_run": {
            "id": run["id"], "status": run["status"],
            "pass1_model": run["pass1_model"], "pass2_model": run["pass2_model"],
            "input_tokens": run["input_tokens"], "output_tokens": run["output_tokens"],
            "cache_read_tokens": run["cache_read_tokens"], "cache_write_tokens": run["cache_write_tokens"],
            "cost_usd": float(run["cost_usd"]) if run["cost_usd"] is not None else None,
            "project_summary": run["project_summary"], "error": run["error"],
            "created_at": _iso(run["created_at"]), "updated_at": _iso(run["updated_at"]),
        },
        "cuts": rows_for_run(run["id"]),
    }
