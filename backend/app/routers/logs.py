"""
Read-only access to the audit logs.

Edit-request logs are still file-based (backend/logs/edits/), but the L1 and
L2 analysis logs are sourced directly from Postgres. That's deliberate: video
analysis now runs on remote GPU workers whose local disks the API server can't
read, whereas Postgres is shared by every worker and the API.

Endpoints:
  GET /api/logs/edits           list recent edit-request logs
  GET /api/logs/edits/{id}      full content of a single edit-request log
  GET /api/logs/l1              list per-file L1 analyses (+ seconds taken)
  GET /api/logs/l1/{file_id}    full L1 analysis for a file
  GET /api/logs/l2              list per-file L2 analyses (+ seconds taken)
  GET /api/logs/l2/{file_id}    full analysis (L1+L2) for a file
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_current_user_id
from app.services import audit_log
from app.services.l1 import snapshot

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/edits")
def list_edits(
    limit: int = Query(50, ge=1, le=500),
    user_id: str = Depends(get_current_user_id),
):
    return {"items": audit_log.list_edit_logs(limit=limit)}


@router.get("/edits/{log_id}")
def get_edit(
    log_id: str,
    user_id: str = Depends(get_current_user_id),
):
    blob = audit_log.read_edit_log(log_id)
    if not blob:
        raise HTTPException(status_code=404, detail="Edit log not found")
    return blob


@router.get("/l1")
def list_l1(user_id: str = Depends(get_current_user_id)):
    return {"items": snapshot.list_l1_analyses(user_id)}


@router.get("/l1/{file_id}")
def get_l1(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    blob = snapshot.build_l1_snapshot(file_id)
    if not blob.get("file"):
        raise HTTPException(status_code=404, detail="L1 analysis not found for file")
    return blob


@router.get("/l2")
def list_l2(user_id: str = Depends(get_current_user_id)):
    return {"items": snapshot.list_l2_analyses(user_id)}


@router.get("/l2/{file_id}")
def get_l2(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    # The L1 snapshot already includes every L2 column (narrative, framing,
    # dinov2 presence, audio events, and the 'l2' timing row), so it doubles
    # as the L2 detail view.
    blob = snapshot.build_l1_snapshot(file_id)
    if not blob.get("file"):
        raise HTTPException(status_code=404, detail="L2 analysis not found for file")
    return blob
