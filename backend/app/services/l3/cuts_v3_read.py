"""
Cuts v3 read path: ``GET /api/projects/{id}/cuts-v3``. Pure DB read, zero
model calls -- the latest ``ingest_run`` for a project plus every
``cut_record`` it produced. See cuts_v3.plan.md section 7.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


def _pg_conn():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _iso(v) -> Optional[str]:
    return v.isoformat(timespec="seconds") if v is not None else None


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

        cuts = conn.execute(
            """
            select id::text, file_id::text, src_in_ms, src_out_ms, kind, word_span, atom_ids,
                   label, summary, speaker, on_camera, take_group_id::text, take_role, channel,
                   junk, junk_reason, framing, look, caption_zones, pace,
                   hero_ts_ms, hero_key, transition_in, transition_out
              from cut_records where ingest_run_id = %s
             order by file_id, src_in_ms
            """,
            (run["id"],),
        ).fetchall()

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
        "cuts": [dict(c) for c in cuts],
    }
