"""
L3 edit-thread API.

Endpoints:
  POST /api/edit/threads                     start a thread (clips + brief)
  GET  /api/edit/threads                     list my threads
  GET  /api/edit/threads/{id}                thread + latest document + questions
  POST /api/edit/threads/{id}/message        answer questions / give feedback
  GET  /api/edit/threads/{id}/versions       document version history
  GET  /api/edit/threads/{id}/versions/{v}   one specific document version

The agent runs asynchronously on the worker; clients poll GET {id} (status
moves drafting -> awaiting_user|ready). Streaming can layer on later without
changing this surface.
"""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.services.l3 import auto_edit
from app.services.l3 import store
from app.services.l3.orchestrator import send_user_message, start_thread

router = APIRouter(prefix="/api/edit/threads", tags=["edit"])


class CreateThreadBody(BaseModel):
    file_ids: List[str] = Field(min_length=1)
    brief: str = ""
    # "agent" = the Claude agentic loop (default); "auto" = the L3 v2 one-shot
    # prompt-driven auto-editor (OpenAI).
    mode: str = "agent"


class MessageBody(BaseModel):
    text: Optional[str] = None
    # Structured answers to open questions: {q_id: chosen answer}. Folded into
    # one user turn alongside any free text.
    answers: Optional[dict] = None


class EditDocumentBody(BaseModel):
    # The version the edit is based on; rejected (409) if the head has moved
    # (the agent or another tab wrote a newer version meanwhile).
    base_version: int
    timeline: List[dict]
    operations: List[dict] = []
    summary: Optional[str] = None
    notes: Optional[List[str]] = None


_ALLOWED_OP_TYPES = {"place_video", "place_audio", "split_edit", "level"}


def _sanitize_timeline(timeline: List[dict], durations: dict) -> List[dict]:
    out: List[dict] = []
    for i, seg in enumerate(timeline):
        try:
            seg_id = str(seg["seg_id"])
            file_id = str(seg["file_id"])
            in_ms = max(0, int(seg["in_ms"]))
            out_ms = int(seg["out_ms"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"timeline[{i}] missing seg_id/file_id/in_ms/out_ms")
        dur = durations.get(file_id)
        if dur:
            out_ms = min(out_ms, dur)
            in_ms = min(in_ms, max(0, dur - 1))
        if out_ms <= in_ms:
            raise HTTPException(status_code=422, detail=f"timeline[{i}] out_ms must exceed in_ms")
        clean = {**seg, "seg_id": seg_id, "file_id": file_id, "in_ms": in_ms, "out_ms": out_ms}
        out.append(clean)
    return out


def _sanitize_operations(operations: List[dict]) -> List[dict]:
    out: List[dict] = []
    for i, op in enumerate(operations):
        t = op.get("type")
        if t not in _ALLOWED_OP_TYPES:
            raise HTTPException(status_code=422, detail=f"operations[{i}] invalid type {t!r}")
        if not op.get("op_id"):
            raise HTTPException(status_code=422, detail=f"operations[{i}] missing op_id")
        out.append(op)
    return out


@router.post("")
def create_thread(body: CreateThreadBody, user_id: str = Depends(get_current_user_id)):
    if body.mode == "auto":
        thread_id = auto_edit.start_thread(user_id, body.file_ids, body.brief)
    else:
        thread_id = start_thread(user_id, body.file_ids, body.brief)
    return {"thread_id": thread_id, "status": "drafting", "mode": body.mode}


@router.get("")
def list_threads(user_id: str = Depends(get_current_user_id)):
    return {"threads": store.list_threads(user_id)}


def _owned_thread(thread_id: str, user_id: str) -> dict:
    thread = store.get_thread(thread_id)
    if thread is None or thread["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.get("/{thread_id}")
def get_thread(thread_id: str, user_id: str = Depends(get_current_user_id)):
    thread = _owned_thread(thread_id, user_id)
    document, version = store.latest_document(thread_id)
    return {
        **thread,
        "document": document,
        "document_version": version,
        "open_questions": (document or {}).get("open_questions", []),
        "usage": store.total_usage(thread_id),
    }


@router.post("/{thread_id}/message")
def post_message(
    thread_id: str, body: MessageBody, user_id: str = Depends(get_current_user_id)
):
    _owned_thread(thread_id, user_id)
    parts = []
    if body.answers:
        parts.append("Answers to your questions:\n" + json.dumps(body.answers, indent=2))
    if body.text:
        parts.append(body.text)
    if not parts:
        raise HTTPException(status_code=422, detail="Provide text and/or answers")
    send_user_message(thread_id, "\n\n".join(parts))
    return {"ok": True, "status": "drafting"}


@router.put("/{thread_id}/document")
def put_document(
    thread_id: str, body: EditDocumentBody, user_id: str = Depends(get_current_user_id)
):
    """Save a human edit of the timeline as a new (created_by='user') version.

    The whole edited timeline + operations come from the client; we re-resolve
    the layer set server-side so `resolved` stays authoritative and the render
    matches the preview. Optimistic concurrency on `base_version`."""
    _owned_thread(thread_id, user_id)
    from app.services.render.tasks import _durations, resolve_document

    doc, head = store.latest_document(thread_id)
    if doc is None:
        raise HTTPException(status_code=409, detail="No edit document to edit yet")
    if body.base_version != head:
        raise HTTPException(
            status_code=409,
            detail={"error": "stale_base_version", "latest_version": head},
        )

    fids = list({str(s.get("file_id")) for s in body.timeline if s.get("file_id")})
    durations = _durations(fids)
    timeline = _sanitize_timeline(body.timeline, durations)
    operations = _sanitize_operations(body.operations)

    new_doc = {**doc, "timeline": timeline, "operations": operations}
    if body.summary is not None:
        new_doc["summary"] = body.summary
    if body.notes is not None:
        new_doc["notes"] = body.notes
    new_doc.pop("resolved", None)  # force a fresh recompute below
    new_doc["resolved"] = resolve_document(new_doc)

    version = store.save_document(thread_id, new_doc, created_by="user")
    return {"version": version, "document": new_doc}


@router.get("/{thread_id}/versions")
def list_versions(thread_id: str, user_id: str = Depends(get_current_user_id)):
    _owned_thread(thread_id, user_id)
    return {"versions": store.list_document_versions(thread_id)}


@router.get("/{thread_id}/versions/{version}")
def get_version(
    thread_id: str, version: int, user_id: str = Depends(get_current_user_id)
):
    _owned_thread(thread_id, user_id)
    import psycopg  # local: matches store's lazy driver use

    from app.services.l3.store import _pg_conn

    with _pg_conn() as conn:
        row = conn.execute(
            "select document from edit_documents where thread_id = %s and version = %s",
            (thread_id, version),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Version not found")
    doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return {"version": version, "document": doc}
