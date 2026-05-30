"""
L3 edit-request endpoint.

POST /api/edit-request
    body: { prompt: str, folder_id?: str, candidate_limit?: int, fps?: int }
    -> returns:
        - the parsed structured query
        - candidate shots (top N for transparency)
        - timeline clips (final picks)
        - an XML download URL

GET /api/edit-request/{request_id}.xml
    -> returns the compiled FCP7 XML body for download

We don't yet persist edit requests; the GET path expects the timeline to be
re-derived from the URL params (kept simple for Phase 3a). We'll persist in
Phase 3b when the UI needs request history.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue as _queue
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.auth import get_current_user_id
from app.services import audit_log
from app.services.l2 import orchestrator as l2_orch
from app.services.edl import renders_store
from app.services.edl import store as edl_store
from app.services.chat import turns_store
from app.services.chat.broker import broker
from app.services.chat.runner import ChatTurnInput, TurnCancelled, run_chat_turn
from app.services.l3 import claude_editor
from app.services.l3.edit_logic import build_timeline_full
from app.services.l3.edit_logic_basic import TimelineClip, build_timeline
from app.services.l3.preview_render import render_preview
from app.services.l3.query_executor import (
    fetch_candidates_by_shot_ids,
    retrieve_top_k,
    run_query,
)
from app.services.l3.query_parser import parse_prompt
from app.services.l3.xml_builder import build_fcp7_xml
from app.services.r2 import generate_presigned_get

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/edit-request", tags=["edit"])


class EditRequestBody(BaseModel):
    prompt: str
    folder_id: Optional[str] = None
    candidate_limit: int = 200
    fps: int = 24
    sequence_name: str = "AI Rough Cut"
    # When true (default), the parser's needs_l2 hint triggers synchronous L2
    # enrichment of the candidate set before edit logic runs. Set to false for
    # a fast L1-only preview.
    enrich_l2: bool = True
    # "smart" (default) -> Claude reasons over the catalog and returns a timeline.
    # "fast"            -> deterministic SQL filter + score sort + duration fill.
    # The smart path is meaningfully better for compound / narrative prompts;
    # the fast path is ~50x faster for simple "top-N by similarity" cases.
    mode: str = "smart"
    # Optional duration target for the smart path. If null, Claude picks.
    duration_target_s: Optional[int] = None
    # How many SigLIP-ranked shots to send to Claude as the catalog. The
    # smart path narrows down via SigLIP first so we don't blow Claude's
    # context window on long videos.
    catalog_size: int = 50


class TimelineClipOut(BaseModel):
    file_id: str
    file_name: str
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    timeline_end_ms: int
    score: float


class CandidateShotOut(BaseModel):
    shot_id: str
    file_id: str
    file_name: str
    shot_index: int
    start_ms: int
    end_ms: int
    score: float
    keyframe_r2_key: Optional[str]


class EditRequestResponse(BaseModel):
    query: dict
    candidates: list[CandidateShotOut]
    timeline: list[TimelineClipOut]
    fcp7_xml: str
    total_duration_ms: int
    # Smart-mode only: Claude's editorial reasoning, surfaced so the user
    # can see WHY each shot was chosen. Empty string in fast mode.
    reasoning: str = ""
    warnings: list[str] = []
    mode: str = "smart"


def _pathurl_for(clip):
    """Return a presigned HTTPS GET URL for the source media (DaVinci-friendly).
    Premiere users can re-link to local files after import."""
    key = clip.file_r2_proxy_key or clip.file_r2_key
    return generate_presigned_get(key, expires_in=86400)


@router.post("", response_model=EditRequestResponse)
def create_edit_request(
    body: EditRequestBody,
    user_id: str = Depends(get_current_user_id),
):
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")

    audit = audit_log.open_edit_log(body.prompt)
    audit.set("user_id", user_id)
    audit.set("folder_id", body.folder_id)
    audit.set("candidate_limit", body.candidate_limit)
    audit.set("fps", body.fps)
    audit.set("enrich_l2_requested", body.enrich_l2)
    audit.set("mode", body.mode)

    try:
        if body.mode == "fast":
            return _run_fast_path(body, user_id, audit)
        # Default: smart path
        return _run_smart_path(body, user_id, audit)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Edit request failed")
        audit.fail(f"{type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# SMART PATH (default): Claude reasons over the catalog
# ---------------------------------------------------------------------------

def _run_smart_path(body: EditRequestBody, user_id: str, audit) -> EditRequestResponse:
    # 1. SigLIP retrieval narrows the haystack to the K most-similar shots so
    #    Claude doesn't have to read every shot the user owns.
    candidates = retrieve_top_k(
        user_id=user_id,
        prompt=body.prompt,
        folder_id=body.folder_id,
        k=body.catalog_size,
    )
    audit.stage("candidates_l1_only", _serialize_candidates(candidates))
    audit.set("catalog_size", len(candidates))

    if not candidates:
        audit.succeed()
        return EditRequestResponse(
            query={"mode": "smart"},
            candidates=[],
            timeline=[],
            fcp7_xml=build_fcp7_xml(body.sequence_name, [], _pathurl_for, body.fps),
            total_duration_ms=0,
            reasoning="No indexed shots are available for this user/folder.",
            warnings=["Empty catalog -- upload and index footage first."],
            mode="smart",
        )

    # 2. Hand the catalog to Claude for actual editorial reasoning.
    result = claude_editor.compile_timeline(
        brief=body.prompt,
        candidates=candidates,
        duration_target_s=body.duration_target_s,
    )
    audit.stage("editor_user_message", result.user_text)
    audit.stage("editor_raw_response", result.raw_response)
    audit.stage("editor_reasoning", result.reasoning)
    audit.stage("editor_warnings", result.warnings)
    audit.stage("editor_post_processing", result.post_processing)

    timeline = result.timeline
    actual_ms = timeline[-1].timeline_end_ms if timeline else 0
    target_s = body.duration_target_s
    target_ms = int(target_s) * 1000 if target_s else None
    audit.stage("timeline", [_serialize_clip(t) for t in timeline])
    audit.stage("timeline_summary", {
        "clip_count": len(timeline),
        "actual_duration_ms": actual_ms,
        "actual_duration_s": round(actual_ms / 1000.0, 2),
        "target_duration_s": target_s,
        "delta_ms_vs_target": (actual_ms - target_ms) if target_ms else None,
        "delta_pct_vs_target": (
            round((actual_ms - target_ms) / target_ms * 100, 1) if target_ms else None
        ),
    })

    fcp7 = build_fcp7_xml(body.sequence_name, timeline, _pathurl_for, body.fps)
    audit.set("fcp7_xml_chars", len(fcp7))
    audit.succeed()

    return EditRequestResponse(
        query={
            "mode": "smart",
            "duration_target_s": target_s,
            "catalog_size": len(candidates),
            "post_processing": result.post_processing,
        },
        candidates=[
            CandidateShotOut(
                shot_id=c.shot_id,
                file_id=c.file_id,
                file_name=c.file_name,
                shot_index=c.shot_index,
                start_ms=c.start_ms,
                end_ms=c.end_ms,
                score=c.score,
                keyframe_r2_key=c.keyframe_r2_key,
            )
            for c in candidates[:50]
        ],
        timeline=[
            TimelineClipOut(
                file_id=t.file_id,
                file_name=t.file_name,
                source_in_ms=t.source_in_ms,
                source_out_ms=t.source_out_ms,
                timeline_start_ms=t.timeline_start_ms,
                timeline_end_ms=t.timeline_end_ms,
                score=t.score,
            )
            for t in timeline
        ],
        fcp7_xml=fcp7,
        total_duration_ms=actual_ms,
        reasoning=result.reasoning,
        warnings=result.warnings,
        mode="smart",
    )


# ---------------------------------------------------------------------------
# FAST PATH (legacy / opt-in): SQL filter + score sort + duration fill
# ---------------------------------------------------------------------------

def _run_fast_path(body: EditRequestBody, user_id: str, audit) -> EditRequestResponse:
    try:
        query = parse_prompt(body.prompt)
    except Exception as e:
        logger.exception("Prompt parsing failed")
        audit.fail(f"prompt_parse: {e}")
        raise HTTPException(status_code=502, detail=f"Prompt parsing failed: {e}") from e
    audit.stage("query", query)

    candidates = run_query(
        user_id=user_id,
        query=query,
        folder_id=body.folder_id,
        limit=body.candidate_limit,
        raw_prompt=body.prompt,
    )
    audit.stage("candidates_l1_only", _serialize_candidates(candidates))

    needs_l2 = bool(query.get("needs_l2")) and body.enrich_l2
    audit.set("needs_l2_resolved", needs_l2)
    if needs_l2 and candidates:
        try:
            l2_orch.enrich(shot_ids=[c.shot_id for c in candidates[:50]])
            candidates = run_query(
                user_id=user_id,
                query=query,
                folder_id=body.folder_id,
                limit=body.candidate_limit,
                raw_prompt=body.prompt,
            )
            audit.stage("candidates_after_l2", _serialize_candidates(candidates))
        except Exception as e:
            logger.exception("Inline L2 enrichment failed; continuing with L1 data only.")
            audit.set("l2_enrichment_error", str(e))

    if needs_l2:
        timeline = build_timeline_full(candidates, query)
    else:
        timeline = build_timeline(candidates, query)

    actual_ms = timeline[-1].timeline_end_ms if timeline else 0
    target_s = query.get("duration_target_s")
    target_ms = int(target_s) * 1000 if target_s else None
    audit.stage("timeline", [_serialize_clip(t) for t in timeline])
    audit.stage("timeline_summary", {
        "clip_count": len(timeline),
        "actual_duration_ms": actual_ms,
        "actual_duration_s": round(actual_ms / 1000.0, 2),
        "target_duration_s": target_s,
        "delta_ms_vs_target": (actual_ms - target_ms) if target_ms is not None else None,
        "delta_pct_vs_target": (
            round((actual_ms - target_ms) / target_ms * 100, 1) if target_ms else None
        ),
    })

    fcp7 = build_fcp7_xml(body.sequence_name, timeline, _pathurl_for, body.fps)
    audit.set("fcp7_xml_chars", len(fcp7))
    audit.succeed()

    return EditRequestResponse(
        query=query,
        candidates=[
            CandidateShotOut(
                shot_id=c.shot_id,
                file_id=c.file_id,
                file_name=c.file_name,
                shot_index=c.shot_index,
                start_ms=c.start_ms,
                end_ms=c.end_ms,
                score=c.score,
                keyframe_r2_key=c.keyframe_r2_key,
            )
            for c in candidates[:50]
        ],
        timeline=[
            TimelineClipOut(
                file_id=t.file_id,
                file_name=t.file_name,
                source_in_ms=t.source_in_ms,
                source_out_ms=t.source_out_ms,
                timeline_start_ms=t.timeline_start_ms,
                timeline_end_ms=t.timeline_end_ms,
                score=t.score,
            )
            for t in timeline
        ],
        fcp7_xml=fcp7,
        total_duration_ms=actual_ms,
        reasoning="",
        warnings=[],
        mode="fast",
    )


def _serialize_candidates(cands: list) -> list[dict]:
    return [
        {
            "shot_id": c.shot_id,
            "file_id": c.file_id,
            "file_name": c.file_name,
            "shot_index": c.shot_index,
            "start_ms": c.start_ms,
            "end_ms": c.end_ms,
            "duration_ms": c.end_ms - c.start_ms,
            "score": c.score,
            "intra_shot_variance": getattr(c, "intra_shot_variance", None),
            "peak_motion_ms": getattr(c, "peak_motion_ms", None),
            "blur_min": getattr(c, "blur_min", None),
        }
        for c in cands[:50]
    ]


def _serialize_clip(t) -> dict:
    return {
        "file_name": t.file_name,
        "source_in_ms": t.source_in_ms,
        "source_out_ms": t.source_out_ms,
        "duration_ms": t.source_out_ms - t.source_in_ms,
        "timeline_start_ms": t.timeline_start_ms,
        "timeline_end_ms": t.timeline_end_ms,
        "score": t.score,
        "trimmed_around_ms": getattr(t, "trimmed_around_ms", None),
    }


# ---------------------------------------------------------------------------
# CHAT (multi-turn smart editor)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    # User messages: plain text in `content`.
    content: Optional[str] = None
    # Assistant messages: editor's reasoning + the timeline it returned that turn.
    reasoning: Optional[str] = None
    timeline: Optional[list[dict]] = None  # [{shot_id, source_in_ms, source_out_ms, role_in_edit, why}]


class ChatRequestBody(BaseModel):
    messages: list[ChatMessage]
    # Explicit file selection (takes precedence over folder_id when both are
    # set). When None or empty, the editor draws from every file the user owns
    # (subject to folder_id, if any).
    file_ids: Optional[list[str]] = None
    folder_id: Optional[str] = None
    sequence_name: str = "AI Rough Cut"
    fps: int = 24
    catalog_size: int = 50
    duration_target_s: Optional[int] = None


class ChatTimelineClipOut(BaseModel):
    """Chat-mode timeline clip: file-level fields the UI needs PLUS the editor-level
    fields (shot_id/role_in_edit/why) so the next turn's history retains them."""
    file_id: str
    file_name: str
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    timeline_end_ms: int
    score: float
    shot_id: Optional[str] = None
    role_in_edit: Optional[str] = None
    why: Optional[str] = None


class ChatResponse(BaseModel):
    timeline: list[ChatTimelineClipOut]
    fcp7_xml: str
    total_duration_ms: int
    reasoning: str
    warnings: list[str] = []
    catalog_size: int
    # Phase 1: every chat turn writes a new EDL version and auto-enqueues a
    # render. The frontend polls /api/renders/:id to swap in the rendered MP4.
    project_id: Optional[str] = None
    edl_version_id: Optional[str] = None
    render_id: Optional[str] = None


def _body_to_turn_input(body: ChatRequestBody) -> ChatTurnInput:
    return ChatTurnInput(
        messages=[m.model_dump() for m in body.messages],
        file_ids=body.file_ids,
        folder_id=body.folder_id,
        sequence_name=body.sequence_name,
        fps=body.fps,
        catalog_size=body.catalog_size,
        duration_target_s=body.duration_target_s,
    )


def _payload_to_chat_response(payload: dict) -> ChatResponse:
    return ChatResponse(
        timeline=[ChatTimelineClipOut(**c) for c in payload.get("timeline", [])],
        fcp7_xml=payload.get("fcp7_xml", ""),
        total_duration_ms=payload.get("total_duration_ms", 0),
        reasoning=payload.get("reasoning", ""),
        warnings=payload.get("warnings", []),
        catalog_size=payload.get("catalog_size", 0),
        project_id=payload.get("project_id"),
        edl_version_id=payload.get("edl_version_id"),
        render_id=payload.get("render_id"),
    )


def _validate_chat_body(body: ChatRequestBody) -> None:
    if not body.messages or body.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="The last message in the history must be a user message.",
        )
    if not (body.messages[-1].content or "").strip():
        raise HTTPException(status_code=400, detail="Latest user message is empty.")


@router.post("/chat", response_model=ChatResponse)
def edit_chat(
    body: ChatRequestBody,
    user_id: str = Depends(get_current_user_id),
):
    """
    Synchronous conversational edit. Blocks until the turn completes. Kept for
    backward-compat and scripted use; the UI prefers /chat/async + SSE so it
    can show progress and cancel. Both paths share run_chat_turn().
    """
    _validate_chat_body(body)
    try:
        payload = run_chat_turn(_body_to_turn_input(body), user_id)
        return _payload_to_chat_response(payload)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Chat edit failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# ASYNC chat: kick off a turn, stream progress over SSE, allow cancel.
#
# Execution model: the turn runs in a background thread of THIS API process.
# The heavy step is the Anthropic HTTP call, so a thread frees the event loop
# for the SSE stream. Live progress flows through the in-memory broker; the
# chat_turns row is the durable mirror (status/phase/result) for reconnects.
# ---------------------------------------------------------------------------

class StartTurnResponse(BaseModel):
    turn_id: str
    status: str


class TurnStatusResponse(BaseModel):
    id: str
    status: str
    phase: Optional[str] = None
    progress_pct: int = 0
    error: Optional[str] = None
    project_id: Optional[str] = None
    edl_version_id: Optional[str] = None
    render_id: Optional[str] = None
    # Present once status == 'done'.
    result: Optional[ChatResponse] = None


def _run_turn_in_thread(turn_id: str, user_id: str, body: ChatRequestBody) -> None:
    """Blocking worker body. Runs in a thread via run_in_executor."""
    inp = _body_to_turn_input(body)

    def emit(phase: str, pct: int, label: str) -> None:
        broker.emit(turn_id, "phase", {"phase": phase, "pct": pct, "label": label})
        # Durable mirror so a reconnecting client sees current state. Cheap:
        # a handful of phase changes per turn.
        try:
            turns_store.update_turn(
                turn_id,
                status="running" if phase != "done" else "running",
                phase=phase,
                progress_pct=pct,
            )
        except Exception:
            logger.exception("turn %s: failed to mirror phase to DB", turn_id)

    def should_cancel() -> bool:
        # In-process flag (instant) OR durable flag (set by a cancel that hit
        # a different process / after a reconnect).
        if broker.is_cancelled(turn_id):
            return True
        try:
            return turns_store.is_cancel_requested(turn_id)
        except Exception:
            return False

    def on_lineage(project_id, edl_version_id, render_id) -> None:
        try:
            turns_store.update_turn(
                turn_id,
                project_id=project_id,
                edl_version_id=edl_version_id,
                render_id=render_id,
            )
        except Exception:
            logger.exception("turn %s: failed to persist lineage", turn_id)

    try:
        turns_store.update_turn(turn_id, status="running", phase="retrieving", progress_pct=2)
        payload = run_chat_turn(
            inp, user_id, emit=emit, should_cancel=should_cancel, on_lineage=on_lineage
        )
        resp = _payload_to_chat_response(payload).model_dump()
        turns_store.update_turn(
            turn_id, status="done", phase="done", progress_pct=100, result_json=resp
        )
        broker.emit(turn_id, "done", {"result": resp})
    except TurnCancelled:
        turns_store.update_turn(turn_id, status="cancelled", phase="cancelled")
        broker.emit(turn_id, "cancelled", {"message": "Turn cancelled."})
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("turn %s failed", turn_id)
        try:
            turns_store.update_turn(turn_id, status="failed", error=msg)
        except Exception:
            logger.exception("turn %s: failed to record failure", turn_id)
        broker.emit(turn_id, "error", {"message": msg})


@router.post("/chat/async", response_model=StartTurnResponse)
async def edit_chat_async(
    body: ChatRequestBody,
    user_id: str = Depends(get_current_user_id),
):
    """Create a chat turn, start it in the background, return its id immediately.
    Stream progress from GET /chat/turn/{id}/stream."""
    _validate_chat_body(body)
    row = turns_store.create_turn(user_id, body.model_dump())
    turn_id = row["id"]
    broker.create(turn_id)

    loop = asyncio.get_event_loop()
    # Schedule the blocking work on the default thread pool. We keep a
    # reference on the broker state so the task isn't GC'd mid-flight.
    task = loop.run_in_executor(None, _run_turn_in_thread, turn_id, user_id, body)
    st = broker.get(turn_id)
    if st is not None:
        setattr(st, "_task", task)  # prevent GC; not part of the dataclass schema

    return StartTurnResponse(turn_id=turn_id, status="queued")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/chat/turn/{turn_id}/stream")
async def stream_chat_turn(
    turn_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Server-Sent Events stream of a turn's progress.

    Event types: phase | warning | done | error | cancelled.
    Low-latency path drains the in-memory broker. If this process isn't the
    one running the turn (broker miss after restart / different worker), we
    fall back to polling the durable chat_turns row.
    """
    row = turns_store.get_turn(turn_id)
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="turn not found")

    async def gen():
        st = broker.get(turn_id)

        if st is None:
            # No live broker entry: poll the DB until terminal.
            async for chunk in _poll_db_stream(turn_id):
                yield chunk
            return

        # Live path: replay current snapshot, then drain the event queue.
        yield _sse("phase", {
            "phase": st.snapshot.get("phase"),
            "pct": st.snapshot.get("pct"),
            "label": st.snapshot.get("label"),
        })
        while True:
            drained = False
            try:
                while True:
                    evt = st.events.get_nowait()
                    drained = True
                    yield _sse(evt.type, evt.payload)
                    if evt.type in ("done", "error", "cancelled"):
                        return
            except _queue.Empty:
                pass
            if st.terminal and not drained:
                # Terminal and nothing left to drain: ensure client got a
                # terminal event (it should have via the queue) then stop.
                return
            await asyncio.sleep(0.2)
            yield ": keepalive\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _poll_db_stream(turn_id: str):
    """Fallback SSE source: poll chat_turns until terminal. Coarser (no live
    sub-phase events) but correct after a process restart."""
    last_phase = None
    for _ in range(1200):  # ~5 min @ 0.25s
        row = turns_store.get_turn(turn_id)
        if not row:
            yield _sse("error", {"message": "turn disappeared"})
            return
        phase = row.get("phase")
        if phase != last_phase:
            last_phase = phase
            yield _sse("phase", {
                "phase": phase,
                "pct": row.get("progress_pct", 0),
                "label": phase or "",
            })
        status = row["status"]
        if status == "done":
            yield _sse("done", {"result": row.get("result_json")})
            return
        if status == "failed":
            yield _sse("error", {"message": row.get("error") or "Turn failed"})
            return
        if status == "cancelled":
            yield _sse("cancelled", {"message": "Turn cancelled."})
            return
        await asyncio.sleep(0.25)
    yield _sse("error", {"message": "stream timed out"})


@router.post("/chat/turn/{turn_id}/cancel", response_model=TurnStatusResponse)
def cancel_chat_turn(
    turn_id: str,
    user_id: str = Depends(get_current_user_id),
):
    row = turns_store.get_turn(turn_id)
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="turn not found")
    # Dual-channel cancel: in-process event (instant) + durable flag.
    broker.request_cancel(turn_id)
    turns_store.request_cancel(turn_id)
    row = turns_store.get_turn(turn_id) or row
    return _turn_row_to_status(row)


@router.get("/chat/turn/{turn_id}", response_model=TurnStatusResponse)
def get_chat_turn(
    turn_id: str,
    user_id: str = Depends(get_current_user_id),
):
    row = turns_store.get_turn(turn_id)
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="turn not found")
    return _turn_row_to_status(row)


def _turn_row_to_status(row: dict) -> TurnStatusResponse:
    result = None
    if row.get("status") == "done" and row.get("result_json"):
        try:
            result = ChatResponse(**row["result_json"])
        except Exception:
            logger.exception("turn %s: stored result_json failed validation", row.get("id"))
    return TurnStatusResponse(
        id=row["id"],
        status=row["status"],
        phase=row.get("phase"),
        progress_pct=row.get("progress_pct", 0),
        error=row.get("error"),
        project_id=row.get("project_id"),
        edl_version_id=row.get("edl_version_id"),
        render_id=row.get("render_id"),
        result=result,
    )


# ---------------------------------------------------------------------------
# Render preview from a precomputed timeline (used by the chat UI so we
# don't re-run Claude just to render the same timeline).
# ---------------------------------------------------------------------------

class TimelineClipIn(BaseModel):
    file_id: str
    file_name: Optional[str] = None
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: Optional[int] = None
    timeline_end_ms: Optional[int] = None
    score: Optional[float] = 0.0


class PreviewFromTimelineBody(BaseModel):
    timeline: list[TimelineClipIn]


@router.post("/preview-from-timeline")
def render_preview_from_timeline(
    body: PreviewFromTimelineBody,
    user_id: str = Depends(get_current_user_id),
):
    if not body.timeline:
        raise HTTPException(status_code=400, detail="Timeline is empty.")

    # We need r2_keys for each clip. Fetch them by file_id, scoped to the user.
    file_ids = list({c.file_id for c in body.timeline})
    import psycopg
    from psycopg.rows import dict_row
    from app.config import get_settings as _gs
    settings = _gs()
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        cur = conn.execute(
            "select id, name, r2_key, r2_proxy_key from files where id = any(%s::uuid[]) and user_id = %s",
            (file_ids, user_id),
        )
        files_meta = {str(r["id"]): r for r in cur.fetchall()}
    missing = [fid for fid in file_ids if fid not in files_meta]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown file ids: {missing}")

    # Reconstruct TimelineClip objects (with timeline timestamps).
    clips: list[TimelineClip] = []
    cursor = 0
    for c in body.timeline:
        m = files_meta[c.file_id]
        dur = max(0, c.source_out_ms - c.source_in_ms)
        ts = c.timeline_start_ms if c.timeline_start_ms is not None else cursor
        te = c.timeline_end_ms if c.timeline_end_ms is not None else cursor + dur
        clips.append(TimelineClip(
            file_id=c.file_id,
            file_name=c.file_name or m["name"],
            file_r2_key=m["r2_key"],
            file_r2_proxy_key=m.get("r2_proxy_key"),
            source_in_ms=c.source_in_ms,
            source_out_ms=c.source_out_ms,
            timeline_start_ms=ts,
            timeline_end_ms=te,
            score=c.score or 0.0,
        ))
        cursor = te

    try:
        url = render_preview(clips)
    except Exception as e:
        logger.exception("Preview render failed (from timeline)")
        raise HTTPException(status_code=500, detail=f"Preview render failed: {e}") from e
    return {"preview_url": url, "clip_count": len(clips)}


@router.post("/preview")
def render_edit_preview(
    body: EditRequestBody,
    user_id: str = Depends(get_current_user_id),
):
    """
    Re-run the parse->search->logic flow, then render the resulting timeline
    via FFmpeg `-c copy` concat. Returns a presigned MP4 URL.

    Synchronous: blocks until rendering finishes. For long timelines this can
    take a while; the frontend should show a spinner. If you want this in the
    background, we can wrap it as a procrastinate task later.
    """
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")
    query = parse_prompt(body.prompt)
    candidates = run_query(user_id, query, body.folder_id, body.candidate_limit, raw_prompt=body.prompt)
    needs_l2 = bool(query.get("needs_l2")) and body.enrich_l2
    if needs_l2 and candidates:
        try:
            l2_orch.enrich(shot_ids=[c.shot_id for c in candidates[:50]])
            candidates = run_query(user_id, query, body.folder_id, body.candidate_limit, raw_prompt=body.prompt)
        except Exception:
            logger.exception("Inline L2 enrichment failed during preview render.")
    timeline = build_timeline_full(candidates, query) if needs_l2 else build_timeline(candidates, query)
    if not timeline:
        raise HTTPException(status_code=404, detail="No matching shots to render.")
    try:
        url = render_preview(timeline)
    except Exception as e:
        logger.exception("Preview render failed")
        raise HTTPException(status_code=500, detail=f"Preview render failed: {e}") from e
    return {"preview_url": url, "clip_count": len(timeline)}


@router.post("/download")
def download_edit_xml(
    body: EditRequestBody,
    user_id: str = Depends(get_current_user_id),
):
    """Same flow as POST / but streams the .xml directly as an attachment."""
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")
    query = parse_prompt(body.prompt)
    candidates = run_query(user_id, query, body.folder_id, body.candidate_limit, raw_prompt=body.prompt)
    needs_l2 = bool(query.get("needs_l2")) and body.enrich_l2
    if needs_l2 and candidates:
        try:
            l2_orch.enrich(shot_ids=[c.shot_id for c in candidates[:50]])
            candidates = run_query(user_id, query, body.folder_id, body.candidate_limit, raw_prompt=body.prompt)
        except Exception:
            logger.exception("Inline L2 enrichment failed; continuing with L1 data only.")
    timeline = build_timeline_full(candidates, query) if needs_l2 else build_timeline(candidates, query)
    xml = build_fcp7_xml(body.sequence_name, timeline, _pathurl_for, body.fps)
    safe_name = "".join(c if c.isalnum() else "_" for c in body.sequence_name)[:60]
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.xml"'},
    )
