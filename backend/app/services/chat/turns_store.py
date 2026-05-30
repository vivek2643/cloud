"""
chat_turns persistence.

The durable record of a conversational-editor round-trip. Live progress is
streamed from the in-memory broker; this table holds the latest snapshot
(phase/progress), the final result, lineage, and the cancel flag.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings

logger = logging.getLogger(__name__)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def create_turn(user_id: str, request_json: Dict[str, Any]) -> Dict[str, Any]:
    with _pg() as conn:
        cur = conn.execute(
            """
            insert into public.chat_turns (user_id, status, phase, progress_pct, request_json)
            values (%s, 'queued', 'queued', 0, %s::jsonb)
            returning id, user_id, project_id, status, phase, progress_pct,
                      request_json, result_json, error, cancel_requested,
                      edl_version_id, render_id, created_at, updated_at
            """,
            (user_id, json.dumps(request_json)),
        )
        return _row(cur.fetchone())


def get_turn(turn_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, user_id, project_id, status, phase, progress_pct,
                   request_json, result_json, error, cancel_requested,
                   edl_version_id, render_id, created_at, updated_at
            from public.chat_turns where id = %s
            """,
            (turn_id,),
        )
        row = cur.fetchone()
        return _row(row) if row else None


def list_turns(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, user_id, project_id, status, phase, progress_pct,
                   request_json, result_json, error, cancel_requested,
                   edl_version_id, render_id, created_at, updated_at
            from public.chat_turns
            where user_id = %s
            order by created_at desc
            limit %s
            """,
            (user_id, limit),
        )
        return [_row(r) for r in cur.fetchall()]


def update_turn(
    turn_id: str,
    *,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    progress_pct: Optional[int] = None,
    result_json: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    project_id: Optional[str] = None,
    edl_version_id: Optional[str] = None,
    render_id: Optional[str] = None,
) -> None:
    sets: List[str] = []
    params: List[Any] = []
    if status is not None:
        sets.append("status = %s"); params.append(status)
    if phase is not None:
        sets.append("phase = %s"); params.append(phase)
    if progress_pct is not None:
        sets.append("progress_pct = %s"); params.append(int(progress_pct))
    if result_json is not None:
        sets.append("result_json = %s::jsonb"); params.append(json.dumps(result_json))
    if error is not None:
        sets.append("error = %s"); params.append(error)
    if project_id is not None:
        sets.append("project_id = %s"); params.append(project_id)
    if edl_version_id is not None:
        sets.append("edl_version_id = %s"); params.append(edl_version_id)
    if render_id is not None:
        sets.append("render_id = %s"); params.append(render_id)
    if not sets:
        return
    params.append(turn_id)
    with _pg() as conn:
        conn.execute(
            f"update public.chat_turns set {', '.join(sets)} where id = %s",
            tuple(params),
        )


def request_cancel(turn_id: str) -> bool:
    """Set the durable cancel flag. Returns True if the row existed and was
    not already terminal."""
    with _pg() as conn:
        cur = conn.execute(
            """
            update public.chat_turns
            set cancel_requested = true
            where id = %s and status in ('queued', 'running')
            returning id
            """,
            (turn_id,),
        )
        return cur.fetchone() is not None


def is_cancel_requested(turn_id: str) -> bool:
    with _pg() as conn:
        cur = conn.execute(
            "select cancel_requested from public.chat_turns where id = %s",
            (turn_id,),
        )
        row = cur.fetchone()
        return bool(row and row["cancel_requested"])


def _row(row: Dict[str, Any]) -> Dict[str, Any]:
    def _maybe_json(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v

    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "project_id": str(row["project_id"]) if row.get("project_id") else None,
        "status": row["status"],
        "phase": row.get("phase"),
        "progress_pct": row["progress_pct"],
        "request_json": _maybe_json(row.get("request_json")),
        "result_json": _maybe_json(row.get("result_json")),
        "error": row.get("error"),
        "cancel_requested": bool(row.get("cancel_requested")),
        "edl_version_id": str(row["edl_version_id"]) if row.get("edl_version_id") else None,
        "render_id": str(row["render_id"]) if row.get("render_id") else None,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }
