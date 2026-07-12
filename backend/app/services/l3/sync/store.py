"""
Persistence for `sync_groups`/`sync_group_members` (audio_sync.plan.md SS6).
Plain psycopg CRUD, same shape as `cuts_v3_read.py` -- no ORM in this
codebase (see that module's own note on the point).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


def _pg_conn():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def create_sync_group(
    project_id: str,
    members: List[Dict[str, Any]],  # [{file_id, offset_ms, role, confidence, aligned_by}, ...]
    authoritative_audio_file_id: Optional[str],
    created_by: str = "user",
) -> str:
    """Persist a new sync group + its members. Returns the new group id.
    Members list must be non-empty (a sync group with < 2 files is
    meaningless -- callers enforce that; this layer just persists whatever
    it's given)."""
    with _pg_conn() as conn:
        row = conn.execute(
            """
            insert into sync_groups (project_id, authoritative_audio_file_id, created_by)
            values (%s, %s, %s)
            returning id::text
            """,
            (project_id, authoritative_audio_file_id, created_by),
        ).fetchone()
        group_id = row["id"]
        for m in members:
            conn.execute(
                """
                insert into sync_group_members (group_id, file_id, offset_ms, role, confidence, aligned_by)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (group_id, file_id) do update set
                    offset_ms = excluded.offset_ms, role = excluded.role,
                    confidence = excluded.confidence, aligned_by = excluded.aligned_by
                """,
                (group_id, m["file_id"], int(m["offset_ms"]), m["role"], m.get("confidence"), m["aligned_by"]),
            )
    return group_id


def get_sync_group(group_id: str) -> Optional[Dict[str, Any]]:
    with _pg_conn() as conn:
        group = conn.execute(
            "select id::text, project_id::text, authoritative_audio_file_id::text, created_by, created_at"
            " from sync_groups where id = %s",
            (group_id,),
        ).fetchone()
        if not group:
            return None
        members = conn.execute(
            "select file_id::text, offset_ms, role, confidence, aligned_by"
            " from sync_group_members where group_id = %s",
            (group_id,),
        ).fetchall()
    return {**group, "members": members}


def list_sync_groups_for_project(project_id: str) -> List[Dict[str, Any]]:
    with _pg_conn() as conn:
        groups = conn.execute(
            "select id::text, project_id::text, authoritative_audio_file_id::text, created_by, created_at"
            " from sync_groups where project_id = %s order by created_at desc",
            (project_id,),
        ).fetchall()
        if not groups:
            return []
        group_ids = [g["id"] for g in groups]
        members = conn.execute(
            "select group_id::text, file_id::text, offset_ms, role, confidence, aligned_by"
            " from sync_group_members where group_id = any(%s::uuid[])",
            (group_ids,),
        ).fetchall()
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for m in members:
        by_group.setdefault(m["group_id"], []).append(m)
    return [{**g, "members": by_group.get(g["id"], [])} for g in groups]


def sync_groups_for_files(file_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """file_id -> its sync group (with members + offsets), for every file in
    `file_ids` that belongs to one. A file in no group is simply absent --
    callers treat that as "not synced, behave exactly as today" (SS2's
    no-op guarantee for non-multicam projects)."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select sg.id::text as group_id, sg.authoritative_audio_file_id::text,
                   sgm.file_id::text, sgm.offset_ms, sgm.role
              from sync_group_members sgm
              join sync_groups sg on sg.id = sgm.group_id
             where sgm.group_id in (
                 select group_id from sync_group_members where file_id = any(%s::uuid[])
             )
            """,
            (file_ids,),
        ).fetchall()
    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        g = groups.setdefault(r["group_id"], {
            "group_id": r["group_id"],
            "authoritative_audio_file_id": r["authoritative_audio_file_id"],
            "members": {},
        })
        g["members"][r["file_id"]] = {"offset_ms": r["offset_ms"], "role": r["role"]}
    out: Dict[str, Dict[str, Any]] = {}
    for g in groups.values():
        for fid in g["members"]:
            out[fid] = g
    return out


def set_authoritative(group_id: str, file_id: str) -> None:
    """Manual override of the code-picked authoritative source (SS10
    "Authoritative source picker ... with a manual override")."""
    with _pg_conn() as conn:
        conn.execute(
            "update sync_groups set authoritative_audio_file_id = %s where id = %s",
            (file_id, group_id),
        )


def set_member_offset(group_id: str, file_id: str, offset_ms: int) -> None:
    """Manual nudge commit (SS10): `aligned_by` flips to 'manual', confidence
    cleared (a manual value has no correlation-peak confidence)."""
    with _pg_conn() as conn:
        conn.execute(
            "update sync_group_members set offset_ms = %s, aligned_by = 'manual', confidence = null"
            " where group_id = %s and file_id = %s",
            (int(offset_ms), group_id, file_id),
        )


def delete_sync_group(group_id: str) -> None:
    with _pg_conn() as conn:
        conn.execute("delete from sync_groups where id = %s", (group_id,))
