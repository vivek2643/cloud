"""
EDL router: manual cut-only editing surface (Phase 2).

The EDL is the source of truth. Manual edits commit a new edl_versions row
with author_kind='user' and auto-enqueue a render -- identical lineage to
Claude's chat turns, so AI and manual edits live on the same version chain
(enabling undo/branch later).

Endpoints
  GET  /api/edl/projects/{project_id}/latest          -> enriched latest EDL
  GET  /api/edl/projects/{project_id}/versions        -> version history
  GET  /api/edl/versions/{version_id}                 -> enriched specific version
  POST /api/edl/projects/{project_id}/commit          -> write user version + render
  GET  /api/edl/projects/{project_id}/search-shots    -> shots to insert from catalog

Cut-only: a clip is (shot_id, source_in_ms, source_out_ms). timeline_in/out
are always recomputed server-side (sequential concat). No transitions, no
effects -- those are future polish and slot in additively.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user_id
from app.services.chat import turns_store
from app.services.chat.broker import broker
from app.services.edl import agent as edl_agent
from app.services.edl import patch as edl_patch
from app.services.edl import renders_store
from app.services.edl import store as edl_store
from app.services.l3.query_executor import fetch_candidates_by_shot_ids, retrieve_top_k
from app.services.r2 import generate_presigned_get

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/edl", tags=["edl"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EnrichedClip(BaseModel):
    id: str
    shot_id: str
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    source_in_ms: int
    source_out_ms: int
    timeline_in_ms: int
    timeline_out_ms: int
    duration_ms: int
    # Bounds for trimming: the shot's natural extent and the file's length.
    shot_start_ms: Optional[int] = None
    shot_end_ms: Optional[int] = None
    file_duration_ms: Optional[int] = None
    thumbnail_url: Optional[str] = None
    transcript_text: Optional[str] = None
    # Presigned playable URL for the source file (proxy transcode preferred),
    # used by the timeline preview monitor to scrub/play source segments.
    source_url: Optional[str] = None


class VersionMeta(BaseModel):
    id: str
    parent_id: Optional[str] = None
    author_kind: str
    commit_msg: Optional[str] = None
    created_at: Optional[str] = None
    clip_count: int


class EnrichedEdlResponse(BaseModel):
    project_id: str
    version: VersionMeta
    fps: int
    resolution: List[int]
    clips: List[EnrichedClip]
    total_duration_ms: int


class CommitClipIn(BaseModel):
    id: Optional[str] = None
    shot_id: str
    source_in_ms: int
    source_out_ms: int
    # Carried so a v2 (A/V-split) lineage can rebuild video clips that retain
    # their source file. Optional for backward-compat with v1-only callers.
    file_id: Optional[str] = None


class CommitBody(BaseModel):
    # Cut-only clip list. The server rebuilds the timeline from these. For a v2
    # lineage the music bed / audio track is preserved from the parent version.
    clips: List[CommitClipIn] = []
    # Full EDL passthrough (v1 or v2). When provided it takes precedence over
    # `clips`: used to apply an AI agent's `proposed_edl` verbatim so a v2
    # A/V-split edit round-trips without being flattened to v1.
    edl: Optional[Dict[str, Any]] = None
    commit_msg: Optional[str] = None
    # Base the new version on this one (for optimistic-concurrency / branching).
    # When omitted, we branch off the current latest version.
    parent_id: Optional[str] = None
    fps: int = 30
    # Who authored this commit. Manual edits = 'user'; applying an AI agent
    # proposal = 'claude'. Validated against the EDL schema's allowed set.
    author_kind: str = "user"


class CommitResponse(BaseModel):
    project_id: str
    edl_version_id: str
    render_id: Optional[str] = None
    clip_count: int
    total_duration_ms: int


class SearchShot(BaseModel):
    shot_id: str
    file_id: str
    file_name: str
    start_ms: int
    end_ms: int
    duration_ms: int
    score: float
    thumbnail_url: Optional[str] = None
    transcript_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_project(project_id: str, user_id: str) -> Dict[str, Any]:
    project = edl_store.get_project(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _thumb_url(keyframe_r2_key: Optional[str]) -> Optional[str]:
    if not keyframe_r2_key:
        return None
    try:
        return generate_presigned_get(keyframe_r2_key, expires_in=86400)
    except Exception:
        logger.exception("Failed to presign keyframe %s", keyframe_r2_key)
        return None


def _source_url(m: Any) -> Optional[str]:
    """Presigned URL for a source file, preferring the lighter proxy transcode."""
    if not m:
        return None
    key = getattr(m, "file_r2_proxy_key", None) or getattr(m, "file_r2_key", None)
    if not key:
        return None
    try:
        return generate_presigned_get(key, expires_in=86400)
    except Exception:
        logger.exception("Failed to presign source %s", key)
        return None


def _v2_video_as_clips(edl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a v2 EDL's video track into the v1 clip shape for enrichment/UI."""
    out: List[Dict[str, Any]] = []
    for c in sorted(edl.get("video_track") or [], key=lambda x: x["timeline_in_ms"]):
        out.append({
            "id": c["id"],
            "shot_id": c.get("shot_id"),
            "source_in_ms": int(c["source_in_ms"]),
            "source_out_ms": int(c["source_out_ms"]),
            "timeline_in_ms": int(c["timeline_in_ms"]),
            "timeline_out_ms": int(c["timeline_out_ms"]),
        })
    return out


def _enrich(version: Dict[str, Any], user_id: str) -> EnrichedEdlResponse:
    edl = version["edl_json"] or {}
    clips = _v2_video_as_clips(edl) if edl.get("version") == 2 else (edl.get("clips") or [])
    shot_ids = list({c.get("shot_id") for c in clips if c.get("shot_id")})
    meta_by_shot: Dict[str, Any] = {}
    if shot_ids:
        for cand in fetch_candidates_by_shot_ids(user_id=user_id, shot_ids=shot_ids):
            meta_by_shot[cand.shot_id] = cand

    enriched: List[EnrichedClip] = []
    for c in clips:
        sid = c.get("shot_id")
        m = meta_by_shot.get(sid)
        dur = int(c["source_out_ms"]) - int(c["source_in_ms"])
        file_dur_ms = None
        if m and m.duration_seconds:
            file_dur_ms = int(m.duration_seconds * 1000)
        enriched.append(EnrichedClip(
            id=c["id"],
            shot_id=sid,
            file_id=(m.file_id if m else None),
            file_name=(m.file_name if m else None),
            source_in_ms=int(c["source_in_ms"]),
            source_out_ms=int(c["source_out_ms"]),
            timeline_in_ms=int(c["timeline_in_ms"]),
            timeline_out_ms=int(c["timeline_out_ms"]),
            duration_ms=dur,
            shot_start_ms=(m.start_ms if m else None),
            shot_end_ms=(m.end_ms if m else None),
            file_duration_ms=file_dur_ms,
            thumbnail_url=_thumb_url(m.keyframe_r2_key if m else None),
            transcript_text=(m.transcript_text if m else None),
            source_url=_source_url(m),
        ))

    total = enriched[-1].timeline_out_ms if enriched else 0
    return EnrichedEdlResponse(
        project_id=version["project_id"],
        version=VersionMeta(
            id=version["id"],
            parent_id=version.get("parent_id"),
            author_kind=version["author_kind"],
            commit_msg=version.get("commit_msg"),
            created_at=version.get("created_at"),
            clip_count=len(enriched),
        ),
        fps=int(edl.get("fps", 30)),
        resolution=list(edl.get("resolution", [1920, 1080])),
        clips=enriched,
        total_duration_ms=total,
    )


def _enqueue_render(render_id: str) -> bool:
    try:
        from app.services.jobs import app as proc_app, register_tasks
        register_tasks()
        with proc_app.open():
            proc_app.tasks["render_edl"].defer(render_id=render_id)
        return True
    except Exception:
        logger.exception("Could not enqueue render %s", render_id)
        try:
            renders_store.update_status(
                render_id, status="failed",
                error="Worker unavailable: could not enqueue render job.",
            )
        except Exception:
            logger.exception("Also failed to mark render %s failed", render_id)
        return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class EnsureProjectBody(BaseModel):
    source_file_ids: List[str] = []
    name: str = "Untitled"


class ProjectMeta(BaseModel):
    id: str
    name: str
    source_file_ids: List[str]


@router.post("/projects/ensure", response_model=ProjectMeta)
def ensure_project(body: EnsureProjectBody, user_id: str = Depends(get_current_user_id)):
    """
    Find-or-create the default project for a source set, mirroring the chat
    flow's project resolution. Lets the manual editor persist edits (e.g. a raw
    file dropped onto an empty timeline) before any AI turn has created one.
    """
    project = edl_store.find_or_create_default_project(
        user_id=user_id,
        source_file_ids=body.source_file_ids,
        name=body.name,
    )
    return ProjectMeta(
        id=project["id"],
        name=project.get("name") or "Untitled",
        source_file_ids=project.get("source_file_ids") or [],
    )


class ProjectSummary(BaseModel):
    id: str
    name: str
    source_file_ids: List[str]
    updated_at: Optional[str] = None
    clip_count: int
    duration_ms: int
    author_kind: str
    version_count: int
    thumbnail_url: Optional[str] = None


class RenameProjectBody(BaseModel):
    name: str


@router.get("/projects", response_model=List[ProjectSummary])
def list_projects(user_id: str = Depends(get_current_user_id)):
    """Saved edits for the Edits library: one card per project, newest first."""
    rows = edl_store.list_project_summaries(user_id)
    # Batch-resolve a thumbnail for each project's first clip.
    shot_ids = [r["first_shot_id"] for r in rows if r.get("first_shot_id")]
    thumb_by_shot: Dict[str, Optional[str]] = {}
    if shot_ids:
        try:
            for cand in fetch_candidates_by_shot_ids(user_id, shot_ids):
                thumb_by_shot[cand.shot_id] = _thumb_url(cand.keyframe_r2_key)
        except Exception:
            logger.exception("list_projects: thumbnail resolution failed")
    out: List[ProjectSummary] = []
    for r in rows:
        out.append(
            ProjectSummary(
                id=r["id"],
                name=r["name"],
                source_file_ids=r["source_file_ids"],
                updated_at=r["updated_at"],
                clip_count=r["clip_count"],
                duration_ms=r["duration_ms"],
                author_kind=r["author_kind"],
                version_count=r["version_count"],
                thumbnail_url=thumb_by_shot.get(r.get("first_shot_id") or ""),
            )
        )
    return out


@router.patch("/projects/{project_id}", response_model=ProjectMeta)
def rename_project(
    project_id: str,
    body: RenameProjectBody,
    user_id: str = Depends(get_current_user_id),
):
    name = (body.name or "").strip() or "Untitled"
    project = edl_store.rename_project(project_id, user_id, name)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return ProjectMeta(
        id=project["id"],
        name=project.get("name") or "Untitled",
        source_file_ids=project.get("source_file_ids") or [],
    )


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, user_id: str = Depends(get_current_user_id)):
    ok = edl_store.delete_project(project_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="project not found")
    return {"ok": True}


@router.get("/projects/{project_id}/latest", response_model=Optional[EnrichedEdlResponse])
def get_latest_edl(project_id: str, user_id: str = Depends(get_current_user_id)):
    _require_project(project_id, user_id)
    version = edl_store.get_latest_edl_version(project_id)
    if not version:
        return None
    return _enrich(version, user_id)


@router.get("/projects/{project_id}/versions", response_model=List[VersionMeta])
def list_versions(project_id: str, user_id: str = Depends(get_current_user_id)):
    _require_project(project_id, user_id)
    out: List[VersionMeta] = []
    for v in edl_store.list_edl_versions(project_id):
        ej = v["edl_json"] or {}
        clips = (ej.get("video_track") or []) if ej.get("version") == 2 else (ej.get("clips") or [])
        out.append(VersionMeta(
            id=v["id"],
            parent_id=v.get("parent_id"),
            author_kind=v["author_kind"],
            commit_msg=v.get("commit_msg"),
            created_at=v.get("created_at"),
            clip_count=len(clips),
        ))
    return out


@router.get("/versions/{version_id}", response_model=EnrichedEdlResponse)
def get_version(version_id: str, user_id: str = Depends(get_current_user_id)):
    version = edl_store.get_edl_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="version not found")
    _require_project(version["project_id"], user_id)
    return _enrich(version, user_id)


@router.post("/projects/{project_id}/commit", response_model=CommitResponse)
def commit_edl(
    project_id: str,
    body: CommitBody,
    user_id: str = Depends(get_current_user_id),
):
    _require_project(project_id, user_id)

    # Default parent = current latest (so the version chain stays linear unless
    # the client explicitly branches via parent_id). We also need the parent's
    # EDL json to decide whether to preserve a v2 (A/V-split) structure.
    parent_id = body.parent_id
    parent_edl: Optional[Dict[str, Any]] = None
    if parent_id is not None:
        pv = edl_store.get_edl_version(parent_id)
        parent_edl = (pv or {}).get("edl_json") if pv else None
    else:
        latest = edl_store.get_latest_edl_version(project_id)
        if latest:
            parent_id = latest["id"]
            parent_edl = latest.get("edl_json")

    # Build the EDL to persist. Three cases, in priority order:
    #   1. Full EDL passthrough (`body.edl`) -- e.g. applying an AI agent's
    #      `proposed_edl`. Re-materialized via the patch engine to recompute the
    #      timeline + audio and validate; preserves v2 verbatim.
    #   2. clips + a v2 parent -> rebuild a v2 EDL from the submitted video
    #      clips, carrying the parent's music bed / audio track forward so a
    #      manual tweak doesn't flatten the edit to v1.
    #   3. clips + a v1 (or no) parent -> classic v1 build.
    try:
        if body.edl is not None:
            src = body.edl
            work = src.get("video_track") if src.get("version") == 2 else src.get("clips")
            edl_json, _ = edl_patch.rebuild_from_clips(src, list(work or []))
        elif parent_edl and parent_edl.get("version") == 2:
            work = [c.model_dump() for c in body.clips]
            edl_json, _ = edl_patch.rebuild_from_clips(parent_edl, work)
        else:
            edl_json = edl_store.build_edl_from_user_clips(
                [c.model_dump() for c in body.clips],
                fps=body.fps,
            )
    except (ValueError, edl_patch.PatchError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Resolve the timeline track regardless of version for counts + render gating.
    track = edl_json["video_track"] if edl_json.get("version") == 2 else edl_json.get("clips", [])
    total = track[-1]["timeline_out_ms"] if track else 0

    author_kind = body.author_kind if body.author_kind in ("user", "claude", "system") else "user"
    new_version = edl_store.write_edl_version(
        project_id=project_id,
        edl_json=edl_json,
        author_kind=author_kind,
        parent_id=parent_id,
        commit_msg=body.commit_msg or ("AI edit" if author_kind == "claude" else "Manual edit"),
    )

    render_id: Optional[str] = None
    if track:
        res = list(edl_json.get("resolution") or [1920, 1080])
        preset = "preview_vertical" if len(res) == 2 and res[1] > res[0] else "preview"
        render_row = renders_store.create_render(new_version["id"], preset=preset)
        render_id = render_row["id"]
        _enqueue_render(render_id)

    return CommitResponse(
        project_id=project_id,
        edl_version_id=new_version["id"],
        render_id=render_id,
        clip_count=len(track),
        total_duration_ms=total,
    )


@router.get("/projects/{project_id}/search-shots", response_model=List[SearchShot])
def search_shots(
    project_id: str,
    q: str,
    k: int = 24,
    user_id: str = Depends(get_current_user_id),
):
    """Semantic shot search scoped to the project's source files, for inserting
    new clips into the timeline."""
    project = _require_project(project_id, user_id)
    file_ids = project.get("source_file_ids") or None
    if not q.strip():
        return []
    cands = retrieve_top_k(
        user_id=user_id,
        prompt=q.strip(),
        k=max(1, min(k, 50)),
        file_ids=file_ids,
    )
    return [
        SearchShot(
            shot_id=c.shot_id,
            file_id=c.file_id,
            file_name=c.file_name,
            start_ms=c.start_ms,
            end_ms=c.end_ms,
            duration_ms=c.end_ms - c.start_ms,
            score=round(float(c.score), 4),
            thumbnail_url=_thumb_url(c.keyframe_r2_key),
            transcript_text=c.transcript_text,
        )
        for c in cands
    ]


# ---------------------------------------------------------------------------
# Agent: Claude proposes cut-only edits to the EDL (Phase 3).
#
# Runs as an async turn (reusing the chat_turns table + broker), so the UI
# watches progress over the shared SSE endpoint
#   GET /api/edit-request/chat/turn/{turn_id}/stream
# and cancels via
#   POST /api/edit-request/chat/turn/{turn_id}/cancel
# The agent does NOT commit; it returns a PROPOSAL the user applies (which
# commits an author_kind='claude' version via /commit) or discards.
# ---------------------------------------------------------------------------

class AgentStartBody(BaseModel):
    instruction: str
    base_version_id: Optional[str] = None


class AgentStartResponse(BaseModel):
    turn_id: str
    status: str


def _enrich_proposed(clips: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
    """Attach display metadata (file name, thumbnail, transcript) to proposed
    clips so the diff overlay can render them richly."""
    shot_ids = list({c["shot_id"] for c in clips if c.get("shot_id")})
    meta = {}
    if shot_ids:
        for cand in fetch_candidates_by_shot_ids(user_id=user_id, shot_ids=shot_ids):
            meta[cand.shot_id] = cand
    out = []
    cursor = 0
    for c in clips:
        m = meta.get(c["shot_id"])
        dur = c["source_out_ms"] - c["source_in_ms"]
        out.append({
            "id": c["id"],
            "shot_id": c["shot_id"],
            "file_id": (m.file_id if m else None),
            "file_name": (m.file_name if m else None),
            "source_in_ms": c["source_in_ms"],
            "source_out_ms": c["source_out_ms"],
            "timeline_in_ms": cursor,
            "timeline_out_ms": cursor + dur,
            "duration_ms": dur,
            "thumbnail_url": _thumb_url(m.keyframe_r2_key if m else None),
            "transcript_text": (m.transcript_text if m else None),
        })
        cursor += dur
    return out


def _run_agent_in_thread(turn_id: str, user_id: str, project_id: str,
                         instruction: str, base_version_id: Optional[str]) -> None:
    def emit(phase: str, pct: int, label: str) -> None:
        broker.emit(turn_id, "phase", {"phase": phase, "pct": pct, "label": label})
        try:
            turns_store.update_turn(turn_id, status="running", phase=phase, progress_pct=pct)
        except Exception:
            logger.exception("agent turn %s: phase mirror failed", turn_id)

    def should_cancel() -> bool:
        if broker.is_cancelled(turn_id):
            return True
        try:
            return turns_store.is_cancel_requested(turn_id)
        except Exception:
            return False

    try:
        turns_store.update_turn(turn_id, status="running", phase="planning", progress_pct=2)
        result = edl_agent.run_agent(
            project_id=project_id,
            user_id=user_id,
            instruction=instruction,
            base_version_id=base_version_id,
            emit=emit,
            should_cancel=should_cancel,
        )
        # Enrich proposed clips for the diff overlay.
        result["proposed_enriched"] = _enrich_proposed(result.get("proposed_clips", []), user_id)
        turns_store.update_turn(turn_id, status="done", phase="done", progress_pct=100, result_json=result)
        broker.emit(turn_id, "done", {"result": result})
    except edl_agent.AgentCancelled:
        turns_store.update_turn(turn_id, status="cancelled", phase="cancelled")
        broker.emit(turn_id, "cancelled", {"message": "Agent cancelled."})
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("agent turn %s failed", turn_id)
        try:
            turns_store.update_turn(turn_id, status="failed", error=msg)
        except Exception:
            logger.exception("agent turn %s: failure record failed", turn_id)
        broker.emit(turn_id, "error", {"message": msg})


@router.post("/projects/{project_id}/agent", response_model=AgentStartResponse)
async def start_agent(
    project_id: str,
    body: AgentStartBody,
    user_id: str = Depends(get_current_user_id),
):
    _require_project(project_id, user_id)
    if not body.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction is required")

    row = turns_store.create_turn(user_id, {
        "kind": "edl_agent",
        "project_id": project_id,
        "instruction": body.instruction,
        "base_version_id": body.base_version_id,
    })
    turn_id = row["id"]
    turns_store.update_turn(turn_id, project_id=project_id)
    broker.create(turn_id)

    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(
        None, _run_agent_in_thread, turn_id, user_id, project_id,
        body.instruction, body.base_version_id,
    )
    st = broker.get(turn_id)
    if st is not None:
        setattr(st, "_task", task)

    return AgentStartResponse(turn_id=turn_id, status="queued")
