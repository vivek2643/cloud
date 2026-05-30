"""
Render persistence: rows in the `renders` table.

Lifecycle:
  1. API creates a row (status='queued') and enqueues a procrastinate task.
  2. Worker flips it to 'running', updates progress_pct as it goes,
     finally writes output_r2_key + duration_ms + status='done'.
  3. On failure: status='failed' + error message.
  4. Frontend polls GET /api/renders/:id until status is terminal.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings

logger = logging.getLogger(__name__)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def create_render(edl_version_id: str, preset: str = "preview") -> Dict[str, Any]:
    with _pg() as conn:
        cur = conn.execute(
            """
            insert into public.renders (edl_version_id, preset, status, progress_pct)
            values (%s, %s, 'queued', 0)
            returning id, edl_version_id, preset, status, progress_pct,
                      output_r2_key, duration_ms, error, created_at, updated_at
            """,
            (edl_version_id, preset),
        )
        return _row(cur.fetchone())


def get_render(render_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, edl_version_id, preset, status, progress_pct,
                   output_r2_key, duration_ms, error, created_at, updated_at
            from public.renders where id = %s
            """,
            (render_id,),
        )
        row = cur.fetchone()
        return _row(row) if row else None


def get_renders_for_version(edl_version_id: str) -> List[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, edl_version_id, preset, status, progress_pct,
                   output_r2_key, duration_ms, error, created_at, updated_at
            from public.renders
            where edl_version_id = %s
            order by created_at desc
            """,
            (edl_version_id,),
        )
        return [_row(r) for r in cur.fetchall()]


def update_status(
    render_id: str,
    *,
    status: Optional[str] = None,
    progress_pct: Optional[int] = None,
    output_r2_key: Optional[str] = None,
    duration_ms: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    sets: List[str] = []
    params: List[Any] = []
    if status is not None:
        sets.append("status = %s")
        params.append(status)
    if progress_pct is not None:
        sets.append("progress_pct = %s")
        params.append(int(progress_pct))
    if output_r2_key is not None:
        sets.append("output_r2_key = %s")
        params.append(output_r2_key)
    if duration_ms is not None:
        sets.append("duration_ms = %s")
        params.append(int(duration_ms))
    if error is not None:
        sets.append("error = %s")
        params.append(error)
    if not sets:
        return
    params.append(render_id)
    with _pg() as conn:
        conn.execute(
            f"update public.renders set {', '.join(sets)} where id = %s",
            tuple(params),
        )


def _row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "edl_version_id": str(row["edl_version_id"]),
        "preset": row["preset"],
        "status": row["status"],
        "progress_pct": row["progress_pct"],
        "output_r2_key": row.get("output_r2_key"),
        "duration_ms": row.get("duration_ms"),
        "error": row.get("error"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }
