"""Render row persistence (the `renders` table, migration 016)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _pg():
    from app.services import db
    return db.connection()


_COLS = (
    "id::text, thread_id::text, document_version, preset, status, progress_pct, "
    "resolved_hash, output_r2_key, duration_ms, error, created_at, updated_at"
)


def create_render(
    thread_id: str, document_version: int, preset: str, resolved_hash: Optional[str]
) -> Dict[str, Any]:
    with _pg() as conn:
        row = conn.execute(
            f"""
            insert into renders (thread_id, document_version, preset, status, progress_pct, resolved_hash)
            values (%s, %s, %s, 'queued', 0, %s)
            returning {_COLS}
            """,
            (thread_id, document_version, preset, resolved_hash),
        ).fetchone()
    return _row(row)


def get_render(render_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        row = conn.execute(
            f"select {_COLS} from renders where id = %s", (render_id,)
        ).fetchone()
    return _row(row) if row else None


def find_done(
    thread_id: str, document_version: int, preset: str, resolved_hash: str
) -> Optional[Dict[str, Any]]:
    """An existing successful render of the identical timeline+preset, if any."""
    with _pg() as conn:
        row = conn.execute(
            f"""
            select {_COLS} from renders
             where thread_id = %s and document_version = %s and preset = %s
               and resolved_hash = %s and status = 'done' and output_r2_key is not null
             order by created_at desc limit 1
            """,
            (thread_id, document_version, preset, resolved_hash),
        ).fetchone()
    return _row(row) if row else None


def list_for_thread(thread_id: str) -> List[Dict[str, Any]]:
    with _pg() as conn:
        rows = conn.execute(
            f"select {_COLS} from renders where thread_id = %s order by created_at desc",
            (thread_id,),
        ).fetchall()
    return [_row(r) for r in rows]


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
    for col, val in (
        ("status", status), ("progress_pct", progress_pct),
        ("output_r2_key", output_r2_key), ("duration_ms", duration_ms),
        ("error", error),
    ):
        if val is not None:
            sets.append(f"{col} = %s")
            params.append(val)
    if not sets:
        return
    params.append(render_id)
    with _pg() as conn:
        conn.execute(f"update renders set {', '.join(sets)} where id = %s", tuple(params))


def _row(row: tuple) -> Dict[str, Any]:
    return {
        "id": row[0],
        "thread_id": row[1],
        "document_version": row[2],
        "preset": row[3],
        "status": row[4],
        "progress_pct": row[5],
        "resolved_hash": row[6],
        "output_r2_key": row[7],
        "duration_ms": row[8],
        "error": row[9],
        "created_at": row[10].isoformat() if row[10] else None,
        "updated_at": row[11].isoformat() if row[11] else None,
    }
