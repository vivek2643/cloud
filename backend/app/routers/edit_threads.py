"""
L3 edit-thread API.

Endpoints:
  POST /api/edit/threads                     start a (chat-first) thread
  POST /api/edit/threads/{id}/messages       a conversational, AGENTIC turn
  GET  /api/edit/threads                     list my threads
  GET  /api/edit/threads/{id}                thread + latest document
  PUT  /api/edit/threads/{id}/document       save a human edit of the timeline
  GET  /api/edit/threads/{id}/versions       document version history
  GET  /api/edit/threads/{id}/versions/{v}   one specific document version

Chat-first + agentic: a thread is a conversation. The assistant talks, answers,
and EDITS directly with tools (observe/act) during a turn -- no confirm round
trip. When a turn changes the edit, /messages persists a new document version
and returns it; clients refresh GET {id}. The human can still edit the timeline
by hand (PUT /document) and undo via version history.
"""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.services.l3 import auto_edit
from app.services.l3 import store
from app.services.l3.grade.cdl import Grade

router = APIRouter(prefix="/api/edit/threads", tags=["edit"])


class CreateThreadBody(BaseModel):
    file_ids: List[str] = Field(min_length=1)
    brief: str = ""


class MessageBody(BaseModel):
    text: str = Field(min_length=1)


class EditDocumentBody(BaseModel):
    # The version the edit is based on; rejected (409) if the head has moved
    # (the auto-editor or another tab wrote a newer version meanwhile).
    base_version: int
    timeline: List[dict]
    operations: List[dict] = []
    summary: Optional[str] = None
    notes: Optional[List[str]] = None
    # Sequence-level color grade selection (color_grading.plan.md SS2.4/SS7):
    # {mode, preset_id, reference_image_ref, reference_stats, lut_ref,
    # match_strength, arc_intensity}. Optional so a plain timeline-edit save
    # never has to know/care about grading.
    look: Optional[dict] = None
    # Caption style selection (captions.plan.md SS3): {style_id, enabled,
    # base_style, overrides}. Optional + starts unset so a document is never
    # captioned until the user explicitly picks a style (SS1.3 "no auto-apply").
    captions: Optional[dict] = None


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
    """Start a CHAT-FIRST edit thread. No edit is drafted on creation -- the
    assistant converses and only edits once the user confirms one."""
    thread_id = auto_edit.start_thread(user_id, body.file_ids, body.brief)
    return {"thread_id": thread_id, "status": "ready", "mode": "chat"}


@router.post("/{thread_id}/messages")
def post_message(
    thread_id: str, body: MessageBody, user_id: str = Depends(get_current_user_id)
):
    """A conversational turn -- now AGENTIC. Appends the user's message and runs
    the editor's tool loop (observe -> act -> re-observe). If the turn changed the
    edit, we persist it directly as a new (created_by='auto') document version --
    no confirm round-trip -- and return the new version so the client can refresh
    the timeline. Pure-chat turns just return the reply."""
    _owned_thread(thread_id, user_id)
    from app.services.l3 import converse

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Empty message")

    store.append_turn(thread_id, "user", text)
    result = converse.respond(thread_id)
    store.append_turn(thread_id, "assistant", result.reply, trace=result.trace or None)

    version: Optional[int] = None
    if result.changed and result.document is not None:
        version = store.save_document(thread_id, result.document, created_by="auto")
    store.set_thread_status(thread_id, "awaiting_user" if result.awaiting_user else "ready")
    return {
        "reply": result.reply,
        "changed": bool(result.changed),
        "document_version": version,
        "awaiting_user": bool(result.awaiting_user),
        "questions": result.questions,
    }


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
    if body.look is not None:
        new_doc["look"] = body.look
    if body.captions is not None:
        new_doc["captions"] = body.captions
    new_doc.pop("resolved", None)  # force a fresh recompute below
    new_doc["resolved"] = resolve_document(new_doc)

    version = store.save_document(thread_id, new_doc, created_by="user")
    return {"version": version, "document": new_doc}


@router.get("/{thread_id}/grade-export")
def grade_export(
    thread_id: str,
    export_format: str = "ccc",
    user_id: str = Depends(get_current_user_id),
):
    """Export bundle (color_grading.plan.md SS11): the professional
    round-trip out of EDSO's grade -- `.cdl`/`.ccc` (ASC CDL XML) or a
    grade-carrying EDL (`*ASC_SOP`/`*ASC_SAT` comment convention), one entry
    per currently-resolved graded video clip, in timeline order. `.cube`
    export doesn't need a new endpoint -- GET /api/grade/cube already serves
    the baked LUT for any clip's exact resolved CDL."""
    from fastapi import Response

    from app.services.l3.grade.export_bundle import ccc_xml, cdl_xml, edl_bundle

    _owned_thread(thread_id, user_id)
    doc, _head = store.latest_document(thread_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="No edit document yet")

    resolved = (doc.get("resolved") or {})
    video_layers = resolved.get("video_layers") or []
    entries = [
        {"id": v.get("layer_id"), "cdl": (v.get("grade") or {}).get("cdl")}
        for v in video_layers
        if (v.get("grade") or {}).get("cdl")
    ]
    working_space = next(
        (v["grade"]["working_space"] for v in video_layers if v.get("grade")), "rec709"
    )

    if export_format == "cdl":
        if not entries:
            raise HTTPException(status_code=404, detail="No graded clips to export")
        text = cdl_xml(Grade.from_dict(entries[0]["cdl"]), working_space=working_space)
        media_type, filename = "application/xml", "grade.cdl"
    elif export_format == "ccc":
        text = ccc_xml(entries, working_space=working_space)
        media_type, filename = "application/xml", "grade.ccc"
    elif export_format == "edl":
        text = edl_bundle(entries, title=f"EDSO_{thread_id[:8]}")
        media_type, filename = "text/plain", "grade.edl"
    else:
        raise HTTPException(status_code=400, detail="format must be one of: cdl, ccc, edl")

    return Response(
        content=text, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
