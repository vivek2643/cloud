"""
Persistence for L3 edit threads: threads, the agent message log (turns), and
versioned Edit Document snapshots.

Turns store the *neutral* message dicts used by the LLM adapter, so resuming a
paused thread is literally `messages = load_messages(thread_id)` -- the agent
continues with byte-identical context (including preserved thinking blocks).
Documents are append-only versions; the latest version is the live plan.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg

from app.config import get_settings

logger = logging.getLogger(__name__)


def _pg_conn() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


# --- Threads ---------------------------------------------------------------

def create_thread(user_id: str, file_ids: List[str], brief: str, title: Optional[str] = None) -> str:
    with _pg_conn() as conn:
        row = conn.execute(
            """
            insert into edit_threads (user_id, file_ids, brief, title, status)
            values (%s, %s::uuid[], %s, %s, 'drafting')
            returning id::text
            """,
            (user_id, file_ids, brief, title or (brief[:80] if brief else "Untitled edit")),
        ).fetchone()
    return row[0]


def get_thread(thread_id: str) -> Optional[dict]:
    with _pg_conn() as conn:
        row = conn.execute(
            """
            select id::text, user_id::text, title, file_ids::text[], brief, status,
                   created_at, updated_at
              from edit_threads where id = %s
            """,
            (thread_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "user_id": row[1], "title": row[2], "file_ids": row[3],
        "brief": row[4], "status": row[5],
        "created_at": row[6].isoformat(), "updated_at": row[7].isoformat(),
    }


def list_threads(user_id: str) -> List[dict]:
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select t.id::text, t.title, t.status, t.created_at,
                   coalesce(array_length(t.file_ids, 1), 0),
                   (select max(version) from edit_documents d where d.thread_id = t.id)
              from edit_threads t
             where t.user_id = %s
             order by t.updated_at desc
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "status": r[2],
            "created_at": r[3].isoformat(), "clip_count": r[4],
            "latest_version": r[5],
        }
        for r in rows
    ]


def set_thread_status(thread_id: str, status: str) -> None:
    with _pg_conn() as conn:
        conn.execute(
            "update edit_threads set status = %s where id = %s",
            (status, thread_id),
        )


# --- Turns (the agent message log) ------------------------------------------

def append_turn(thread_id: str, role: str, content: Any, usage: Optional[dict] = None) -> None:
    with _pg_conn() as conn:
        conn.execute(
            """
            insert into edit_turns (thread_id, seq, role, content, usage)
            values (
                %s,
                coalesce((select max(seq) from edit_turns where thread_id = %s), 0) + 1,
                %s, %s::jsonb, %s::jsonb
            )
            """,
            (thread_id, thread_id, role, json.dumps(content), json.dumps(usage) if usage else None),
        )


def load_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Rebuild the neutral message list for the LLM (user/assistant roles only;
    'tool' rows are stored user-role tool_result messages already)."""
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select role, content from edit_turns
             where thread_id = %s and role in ('user', 'assistant')
             order by seq
            """,
            (thread_id,),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for role, content in rows:
        # psycopg returns jsonb already parsed: a plain user brief comes back as
        # a str (the literal message text), assistant/tool turns as a list of
        # blocks. Only re-parse if a driver handed us serialized JSON text for a
        # container; never json.loads a bare string (it's the message itself).
        if isinstance(content, str) and content[:1] in ("[", "{"):
            try:
                c: Any = json.loads(content)
            except json.JSONDecodeError:
                c = content
        else:
            c = content
        out.append({"role": role, "content": c})
    return out


def total_usage(thread_id: str) -> Dict[str, int]:
    with _pg_conn() as conn:
        rows = conn.execute(
            "select usage from edit_turns where thread_id = %s and usage is not null",
            (thread_id,),
        ).fetchall()
    totals: Dict[str, int] = {}
    for (u,) in rows:
        u = u if isinstance(u, dict) else json.loads(u)
        for k, v in u.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
    return totals


# --- Documents (versioned snapshots) -----------------------------------------

def save_document(thread_id: str, document: dict, created_by: str = "agent") -> int:
    with _pg_conn() as conn:
        row = conn.execute(
            """
            insert into edit_documents (thread_id, version, document, created_by)
            values (
                %s,
                coalesce((select max(version) from edit_documents where thread_id = %s), 0) + 1,
                %s::jsonb, %s
            )
            returning version
            """,
            (thread_id, thread_id, json.dumps(document), created_by),
        ).fetchone()
    return int(row[0])


def latest_document(thread_id: str) -> Tuple[Optional[dict], int]:
    """(document, version); (None, 0) when no snapshot exists yet."""
    with _pg_conn() as conn:
        row = conn.execute(
            """
            select document, version from edit_documents
             where thread_id = %s order by version desc limit 1
            """,
            (thread_id,),
        ).fetchone()
    if not row:
        return None, 0
    doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return doc, int(row[1])


def list_document_versions(thread_id: str) -> List[dict]:
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select version, created_by, created_at from edit_documents
             where thread_id = %s order by version
            """,
            (thread_id,),
        ).fetchall()
    return [
        {"version": r[0], "created_by": r[1], "created_at": r[2].isoformat()}
        for r in rows
    ]
