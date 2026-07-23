"""
Find-or-create a ``projects`` row for a set of files. The ``projects`` table
(migration 006) was built for exactly this ("we create one 'default' project
per (user, source_file_ids set)") but nothing in the live app has needed a
durable project identity until cuts-v3's ``ingest_runs``/``cut_records``
(migration 024), which are keyed on ``project_id``. This is the missing
find-or-create half of that original design -- the rest of the app (edit
threads, hero cuts, etc.) still addresses clips by ``file_ids`` directly and
is unaffected.
"""
from __future__ import annotations

from typing import List


def _pg_conn():
    from app.services import db
    return db.connection()


def find_or_create_project(user_id: str, file_ids: List[str]) -> str:
    """Return the id of the project owning exactly this (user, file set),
    creating one if it doesn't exist yet. ``source_file_ids`` is stored
    sorted so array-equality lookups are order-independent."""
    sorted_ids = sorted(file_ids)
    with _pg_conn() as conn:
        row = conn.execute(
            "select id::text from projects where user_id = %s and source_file_ids = %s::uuid[]",
            (user_id, sorted_ids),
        ).fetchone()
        if row:
            return row[0]
        row = conn.execute(
            "insert into projects (user_id, source_file_ids) values (%s, %s::uuid[]) returning id::text",
            (user_id, sorted_ids),
        ).fetchone()
    return row[0]
