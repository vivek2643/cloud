"""Export row persistence (the `exports` table, migration 047). Mirrors
`render/store.py`'s shape (same lifecycle: queued -> running -> done/failed)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg

from app.config import get_settings


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


_COLS = (
    "id::text, thread_id::text, document_version, kind, quality, include_media, "
    "status, output_r2_key, error, created_at, updated_at"
)


def create_export(
    thread_id: str, document_version: int, kind: str, quality: str, include_media: bool
) -> Dict[str, Any]:
    with _pg() as conn:
        row = conn.execute(
            f"""
            insert into exports (thread_id, document_version, kind, quality, include_media, status)
            values (%s, %s, %s, %s, %s, 'queued')
            returning {_COLS}
            """,
            (thread_id, document_version, kind, quality, include_media),
        ).fetchone()
    return _row(row)


def get_export(export_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        row = conn.execute(
            f"select {_COLS} from exports where id = %s", (export_id,)
        ).fetchone()
    return _row(row) if row else None


def list_for_thread(thread_id: str) -> List[Dict[str, Any]]:
    with _pg() as conn:
        rows = conn.execute(
            f"select {_COLS} from exports where thread_id = %s order by created_at desc",
            (thread_id,),
        ).fetchall()
    return [_row(r) for r in rows]


def update_status(
    export_id: str,
    *,
    status: Optional[str] = None,
    output_r2_key: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    sets: List[str] = []
    params: List[Any] = []
    for col, val in (
        ("status", status), ("output_r2_key", output_r2_key), ("error", error),
    ):
        if val is not None:
            sets.append(f"{col} = %s")
            params.append(val)
    if not sets:
        return
    params.append(export_id)
    with _pg() as conn:
        conn.execute(f"update exports set {', '.join(sets)} where id = %s", tuple(params))


def _row(row: tuple) -> Dict[str, Any]:
    return {
        "id": row[0],
        "thread_id": row[1],
        "document_version": row[2],
        "kind": row[3],
        "quality": row[4],
        "include_media": row[5],
        "status": row[6],
        "output_r2_key": row[7],
        "error": row[8],
        "created_at": row[9].isoformat() if row[9] else None,
        "updated_at": row[10].isoformat() if row[10] else None,
    }
