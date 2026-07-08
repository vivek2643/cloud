"""
L3 thread + clip-context helpers.

The live brain is the AGENTIC chat assistant (``converse`` + ``tools``): it sees
the whole footage map and edits the document DIRECTLY with tools (observe/act)
during a turn -- there is no separate compile/arrange step anymore. What survives
here is only what the rest of the system still needs:

  * ``_clip_cards`` -- per-clip summary cards from L2 perception (the footage map
    uses these as its clip headers).
  * ``start_thread`` -- create a chat-first edit thread (drafts nothing on
    creation; the first turn's tool loop does the editing).

The old Director -> Editor -> Coverage arranger AND the later prose-harvest
compile path (``compile_chat_edit`` / ``arrange.compile_placements``) have both
been removed -- the agentic loop is the single path.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import psycopg

from app.config import get_settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Loading per-clip context (used by footage_map)
# --------------------------------------------------------------------------

def _pg_conn() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _as_doc(v) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def _clip_cards(file_ids: List[str]) -> Dict[str, dict]:
    """Per-clip context cards: the editorially-useful summary fields from L2
    perception + duration. Keyed by file_id, order-stable."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, f.name, coalesce(f.duration_seconds, 0), cp.perception
              from files f
              left join clip_perception cp on cp.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    cards: Dict[str, dict] = {}
    for fid, name, dur_s, perception in rows:
        p = _as_doc(perception) or {}
        edit = p.get("editability") or {}
        setting = p.get("setting") or {}
        look = p.get("look") or {}
        persons = p.get("persons") or []
        roles = [pp.get("role") for pp in persons if pp.get("role")]
        cards[fid] = {
            "file_id": fid,
            "name": name or fid,
            "duration_ms": int(float(dur_s) * 1000),
            "content_type": p.get("content_type"),
            "primary_axis": edit.get("primary_axis"),
            "cut_sensitivity": edit.get("cut_sensitivity"),
            "best_use": edit.get("best_use") or [],
            "logline": p.get("logline"),
            "synopsis": p.get("synopsis"),
            "topics": p.get("topics") or [],
            "location": setting.get("location"),
            "mood": look.get("mood"),
            "people": roles,
            "notes": p.get("notes"),
        }
    # Preserve the caller's order so prompts are deterministic.
    return {fid: cards[fid] for fid in file_ids if fid in cards}


# --------------------------------------------------------------------------
# Thread lifecycle
# --------------------------------------------------------------------------

def start_thread(user_id: str, file_ids: List[str], brief: str) -> str:
    """Create a CHAT-FIRST edit thread. Nothing is drafted on creation: the
    assistant converses and edits directly with tools during a turn (see
    ``converse.respond`` + the /messages route). A non-empty ``brief`` is seeded
    as the user's opening message so the conversation has a start.

    In Cuts v3 mode we PIN the thread to the covering ingest run at creation
    (migration 028) so a re-ingest mid-thread can't swap the beat universe under
    an active edit. Resolve fails open: no covering run yet => null => the turn
    falls back to live "latest run" resolution."""
    from app.services.l3 import store

    ingest_run_id: Optional[str] = None
    if get_settings().footage_source == "cut_records":
        try:
            from app.services.l3 import cuts_v3_read
            ingest_run_id = cuts_v3_read.latest_run_for_files(file_ids)
        except Exception:
            logger.exception("start_thread: run pin resolve failed (continuing unpinned)")

    thread_id = store.create_thread(user_id, file_ids, brief, ingest_run_id=ingest_run_id)
    if brief.strip():
        store.append_turn(thread_id, "user", brief.strip())
    store.set_thread_status(thread_id, "ready")
    return thread_id
