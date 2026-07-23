"""
Cuts project API.

  POST /api/projects                 find-or-create a project for a file set
  POST /api/projects/{id}/ingest      kick (or re-kick) the LLM ingest
  GET  /api/projects/{id}/cuts        latest ingest_run + every cut_record

See cuts_v3.plan.md section 7. Ingest itself runs on the procrastinate `l2`
queue (``app.services.l3.ingest.l3_cuts_ingest``) -- this router only
enqueues it, so a request never blocks on a real Claude call.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.services.l3 import cuts_read as read
from app.services.l3.ingest import defer_ingest
from app.services.l3.projects import find_or_create_project

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectBody(BaseModel):
    file_ids: List[str] = Field(min_length=1)


def _owned_project(project_id: str, user_id: str) -> None:
    from app.services.l3.cuts_read import _pg_conn
    with _pg_conn() as conn:
        row = conn.execute(
            "select 1 from projects where id = %s and user_id = %s", (project_id, user_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="project not found")


@router.post("")
def create_project(body: CreateProjectBody, user_id: str = Depends(get_current_user_id)):
    project_id = find_or_create_project(user_id, body.file_ids)
    return {"project_id": project_id}


@router.post("/{project_id}/ingest")
def kick_ingest(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Enqueue (or re-enqueue) the cuts ingest for this project. Returns
    immediately; poll GET /{project_id}/cuts for status."""
    _owned_project(project_id, user_id)
    try:
        defer_ingest(project_id)
    except Exception:
        logger.exception("could not enqueue ingest for project %s", project_id)
        raise HTTPException(status_code=503, detail="could not enqueue ingest")
    return {"project_id": project_id, "status": "queued"}


@router.get("/{project_id}/cuts")
def get_cuts(project_id: str, user_id: str = Depends(get_current_user_id)):
    """All cut_records + project summary + ingest status. Pure DB read --
    zero model calls."""
    result = read.load_cuts(project_id, user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="project not found")
    return result
