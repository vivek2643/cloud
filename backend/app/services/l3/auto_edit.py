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

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Loading per-clip context (used by footage_map)
# --------------------------------------------------------------------------

def _pg_conn():
    from app.services import db
    return db.connection()


def _clip_cards(file_ids: List[str]) -> Dict[str, dict]:
    """Per-clip context cards for the footage-map clip headers: file name +
    duration. Keyed by file_id, order-stable. (Editorial header fields once came
    from the L2 VLM; that layer is gone, so headers now carry only what the file
    row knows -- the Cuts v3 per-cut labels/summaries carry the meaning.)"""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select id::text, name, coalesce(duration_seconds, 0) "
            "from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()

    cards: Dict[str, dict] = {
        fid: {"file_id": fid, "name": name or fid, "duration_ms": int(float(dur_s) * 1000)}
        for fid, name, dur_s in rows
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

    We PIN the thread to the covering ingest run at creation (migration 028)
    so a re-ingest mid-thread can't swap the beat universe under an active
    edit. Resolve fails open: no covering run yet => null => the turn falls
    back to live "latest run" resolution."""
    from app.services.l3 import store

    ingest_run_id: Optional[str] = None
    try:
        from app.services.l3 import cuts_read
        ingest_run_id = cuts_read.latest_run_for_files(file_ids)
    except Exception:
        logger.exception("start_thread: run pin resolve failed (continuing unpinned)")

    thread_id = store.create_thread(user_id, file_ids, brief, ingest_run_id=ingest_run_id)
    if brief.strip():
        store.append_turn(thread_id, "user", brief.strip())
    store.set_thread_status(thread_id, "ready")
    return thread_id
