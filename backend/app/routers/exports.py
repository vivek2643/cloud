"""Export API for L3 edit documents (export_options.plan.md).

  POST /api/edit/threads/{thread_id}/export   create+enqueue an export of a version
  GET  /api/edit/threads/{thread_id}/exports  list exports for a thread
  GET  /api/exports/{export_id}               poll one export (presigned URL when done)

Mirrors routers/renders.py exactly -- see export_options.plan.md's own
"Note on document source" for why this is thread-scoped, not project-scoped
(there is no project_id -> thread_id link anywhere in this schema, and every
sibling feature -- renders, captions, color grade -- is already thread-scoped
the same way). Work runs on the `export` procrastinate queue (see
services/export/tasks.py).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user_id
from app.config import get_settings
from app.services.export import bundle, store as export_store
from app.services.l3 import store as l3_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["exports"])

_KINDS = ("mp4", "rough_cut", "srt")


class CreateExportBody(BaseModel):
    kind: str
    quality: str = "1080"
    include_media: bool = False
    version: Optional[int] = None  # default: latest


def _owned_thread(thread_id: str, user_id: str) -> dict:
    thread = l3_store.get_thread(thread_id)
    if thread is None or thread["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


def _enqueue(export_id: str) -> bool:
    try:
        from procrastinate import App, PsycopgConnector

        enqueue_app = App(connector=PsycopgConnector(
            conninfo=get_settings().database_url, min_size=1, max_size=2))
        with enqueue_app.open():
            enqueue_app.configure_task("build_export", queue="export").defer(export_id=export_id)
        return True
    except Exception:
        logger.exception("Could not enqueue export %s", export_id)
        return False


def _to_response(row: dict) -> dict:
    out = dict(row)
    out["output_url"] = None
    if row.get("status") == "done" and row.get("output_r2_key"):
        try:
            out["output_url"] = bundle.presigned_url_for(row["output_r2_key"])
        except Exception:
            logger.exception("presign failed for export %s", row.get("id"))
    return out


@router.post("/api/edit/threads/{thread_id}/export")
def create_export(
    thread_id: str, body: CreateExportBody, user_id: str = Depends(get_current_user_id)
):
    _owned_thread(thread_id, user_id)
    if body.kind not in _KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown kind {body.kind!r}; known: {list(_KINDS)}")
    # quality is only meaningful for 'mp4' (and included media in 'rough_cut'),
    # but validate it unconditionally -- a bad value is a bad request either way.
    from app.services.render import compositor
    if body.quality not in compositor.PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown quality {body.quality!r}; known: {list(compositor.PRESETS)}",
        )

    document, latest = l3_store.latest_document(thread_id)
    version = body.version if body.version is not None else latest
    if document is None or version == 0:
        raise HTTPException(status_code=409, detail="No edit document to export yet")
    if version != latest:
        from app.services.render.tasks import load_document_version
        document = load_document_version(thread_id, version)
        if document is None:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
    if not (document.get("timeline") or document.get("resolved")):
        raise HTTPException(status_code=409, detail="Edit document has no timeline to export")

    row = export_store.create_export(thread_id, version, body.kind, body.quality, body.include_media)
    if not _enqueue(row["id"]):
        export_store.update_status(row["id"], status="failed", error="Worker unavailable.")
        row = export_store.get_export(row["id"]) or row
    return _to_response(row)


@router.get("/api/edit/threads/{thread_id}/exports")
def list_exports(thread_id: str, user_id: str = Depends(get_current_user_id)):
    _owned_thread(thread_id, user_id)
    return {"exports": [_to_response(r) for r in export_store.list_for_thread(thread_id)]}


@router.get("/api/exports/{export_id}")
def get_export(export_id: str, user_id: str = Depends(get_current_user_id)):
    row = export_store.get_export(export_id)
    if not row:
        raise HTTPException(status_code=404, detail="Export not found")
    _owned_thread(row["thread_id"], user_id)
    return _to_response(row)
