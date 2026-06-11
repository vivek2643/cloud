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
from app.services.l3 import store
from app.services.l3.orchestrator import send_user_message, start_thread

router = APIRouter(prefix="/api/edit/threads", tags=["edit"])


class CreateThreadBody(BaseModel):
    file_ids: List[str] = Field(min_length=1)
    brief: str = ""


class MessageBody(BaseModel):
    text: Optional[str] = None
    # Structured answers to open questions: {q_id: chosen answer}. Folded into
    # one user turn alongside any free text.
    answers: Optional[dict] = None


@router.post("")
def create_thread(body: CreateThreadBody, user_id: str = Depends(get_current_user_id)):
    thread_id = start_thread(user_id, body.file_ids, body.brief)
    return {"thread_id": thread_id, "status": "drafting"}


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
