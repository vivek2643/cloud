"""Render API for L3 edit documents.

  POST /api/edit/threads/{thread_id}/render   create+enqueue a render of a version
  GET  /api/edit/threads/{thread_id}/renders  list renders for a thread
  GET  /api/renders/{render_id}               poll one render (presigned URL when done)

Work runs on the `render` procrastinate queue (see services/render/tasks.py).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user_id
from app.config import get_settings
from app.services.l3 import store as l3_store
from app.services.render import compositor, store as render_store
from app.services.render.tasks import resolve_document

logger = logging.getLogger(__name__)
router = APIRouter(tags=["renders"])


class CreateRenderBody(BaseModel):
    preset: str = "preview"
    version: Optional[int] = None  # default: latest


def _owned_thread(thread_id: str, user_id: str) -> dict:
    thread = l3_store.get_thread(thread_id)
    if thread is None or thread["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


def _enqueue(render_id: str) -> bool:
    try:
        from procrastinate import App, PsycopgConnector

        enqueue_app = App(connector=PsycopgConnector(
            conninfo=get_settings().database_url, min_size=1, max_size=2))
        with enqueue_app.open():
            enqueue_app.configure_task("render_edit", queue="render").defer(render_id=render_id)
        return True
    except Exception:
        logger.exception("Could not enqueue render %s", render_id)
        return False


def _to_response(row: dict) -> dict:
    out = dict(row)
    out["output_url"] = None
    if row.get("status") == "done" and row.get("output_r2_key"):
        try:
            out["output_url"] = compositor.presigned_url_for(row["output_r2_key"])
        except Exception:
            logger.exception("presign failed for render %s", row.get("id"))
    return out


@router.post("/api/edit/threads/{thread_id}/render")
def create_render(
    thread_id: str, body: CreateRenderBody, user_id: str = Depends(get_current_user_id)
):
    _owned_thread(thread_id, user_id)
    if body.preset not in compositor.PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset {body.preset!r}; known: {list(compositor.PRESETS)}",
        )
    document, latest = l3_store.latest_document(thread_id)
    version = body.version if body.version is not None else latest
    if document is None or version == 0:
        raise HTTPException(status_code=409, detail="No edit document to render yet")
    if version != latest:
        document = _load_version(thread_id, version)
        if document is None:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
    if not (document.get("timeline") or document.get("resolved")):
        raise HTTPException(status_code=409, detail="Edit document has no timeline to render")

    resolved = resolve_document(document)
    rhash = compositor.resolved_hash(resolved, body.preset)

    # Short-circuit to an identical successful render if one exists.
    existing = render_store.find_done(thread_id, version, body.preset, rhash)
    if existing:
        return _to_response(existing)

    row = render_store.create_render(thread_id, version, body.preset, rhash)
    if not _enqueue(row["id"]):
        render_store.update_status(row["id"], status="failed", error="Worker unavailable.")
        row = render_store.get_render(row["id"]) or row
    return _to_response(row)


@router.get("/api/edit/threads/{thread_id}/renders")
def list_renders(thread_id: str, user_id: str = Depends(get_current_user_id)):
    _owned_thread(thread_id, user_id)
    return {"renders": [_to_response(r) for r in render_store.list_for_thread(thread_id)]}


@router.get("/api/renders/{render_id}")
def get_render(render_id: str, user_id: str = Depends(get_current_user_id)):
    row = render_store.get_render(render_id)
    if not row:
        raise HTTPException(status_code=404, detail="Render not found")
    _owned_thread(row["thread_id"], user_id)
    return _to_response(row)


def _load_version(thread_id: str, version: int) -> Optional[dict]:
    import json

    from app.services.l3.store import _pg_conn

    with _pg_conn() as conn:
        row = conn.execute(
            "select document from edit_documents where thread_id = %s and version = %s",
            (thread_id, version),
        ).fetchone()
    if not row:
        return None
    return row[0] if isinstance(row[0], dict) else json.loads(row[0])
