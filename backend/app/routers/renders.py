"""
Renders router.

POST   /api/renders         -> create a render job for a given EDL version
GET    /api/renders/:id     -> poll status; returns presigned URL when done
GET    /api/renders/by-version/:edl_version_id -> latest render for that version

The work is done by a procrastinate worker; the API just enqueues and reads
status. See app/services/render/* for the actual rendering pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user_id
from app.services.edl import renders_store, store as edl_store
from app.services.render import cuts_renderer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/renders", tags=["renders"])


class CreateRenderBody(BaseModel):
    edl_version_id: str
    preset: str = "preview"


class RenderResponse(BaseModel):
    id: str
    edl_version_id: str
    preset: str
    status: str
    progress_pct: int
    output_r2_key: Optional[str] = None
    output_url: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _enqueue(render_id: str) -> bool:
    """Defer the render task. Falls back gracefully if no worker is configured."""
    try:
        from app.services.jobs import app as proc_app, register_tasks
        register_tasks()
        with proc_app.open():
            proc_app.tasks["render_edl"].defer(render_id=render_id)
        return True
    except Exception:
        logger.exception("Could not enqueue render %s", render_id)
        return False


def _to_response(row: dict) -> RenderResponse:
    output_url = None
    if row.get("output_r2_key") and row.get("status") == "done":
        try:
            output_url = cuts_renderer.presigned_url_for(row["output_r2_key"])
        except Exception:
            logger.exception("Failed to presign render output for %s", row.get("id"))
    return RenderResponse(
        id=row["id"],
        edl_version_id=row["edl_version_id"],
        preset=row["preset"],
        status=row["status"],
        progress_pct=row["progress_pct"],
        output_r2_key=row.get("output_r2_key"),
        output_url=output_url,
        duration_ms=row.get("duration_ms"),
        error=row.get("error"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@router.post("", response_model=RenderResponse)
def create_render(
    body: CreateRenderBody,
    user_id: str = Depends(get_current_user_id),
):
    if body.preset not in cuts_renderer.PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset {body.preset!r}. Known: {list(cuts_renderer.PRESETS)}",
        )

    # Defense in depth: only allow rendering versions whose project belongs
    # to this user. (The dev mode user_id is constant; doesn't matter today,
    # but matters the moment we re-enable real auth.)
    version = edl_store.get_edl_version(body.edl_version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"edl_version {body.edl_version_id} not found")
    project = edl_store.get_project(version["project_id"], user_id)
    if not project:
        raise HTTPException(status_code=404, detail="EDL version's project not accessible")

    row = renders_store.create_render(body.edl_version_id, preset=body.preset)
    enqueued = _enqueue(row["id"])
    if not enqueued:
        # Mark as failed immediately so the UI doesn't poll forever.
        renders_store.update_status(
            row["id"],
            status="failed",
            error="Worker unavailable: could not enqueue render job.",
        )
        row = renders_store.get_render(row["id"]) or row
    return _to_response(row)


@router.get("/{render_id}", response_model=RenderResponse)
def get_render(
    render_id: str,
    user_id: str = Depends(get_current_user_id),
):
    row = renders_store.get_render(render_id)
    if not row:
        raise HTTPException(status_code=404, detail="render not found")
    # Authorize: render -> edl_version -> project -> user.
    version = edl_store.get_edl_version(row["edl_version_id"])
    if not version:
        raise HTTPException(status_code=404, detail="render's EDL version is missing")
    project = edl_store.get_project(version["project_id"], user_id)
    if not project:
        raise HTTPException(status_code=404, detail="render not accessible")
    return _to_response(row)


@router.get("/by-version/{edl_version_id}", response_model=Optional[RenderResponse])
def latest_render_for_version(
    edl_version_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return the most recent render row for the given EDL version, if any."""
    version = edl_store.get_edl_version(edl_version_id)
    if not version:
        raise HTTPException(status_code=404, detail="edl_version not found")
    project = edl_store.get_project(version["project_id"], user_id)
    if not project:
        raise HTTPException(status_code=404, detail="not accessible")
    rows = renders_store.get_renders_for_version(edl_version_id)
    return _to_response(rows[0]) if rows else None
