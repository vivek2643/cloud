"""
Read-only access to the audit logs that the L1 pipeline and the edit
endpoint write to backend/logs/. The actual files live on the API server's
disk; this router just exposes a JSON view of them.

Endpoints:
  GET /api/logs/edits           list recent edit-request logs
  GET /api/logs/edits/{id}      full content of a single edit-request log
  GET /api/logs/l1              list per-file L1 analyses
  GET /api/logs/l1/{file_id}    full L1 analysis for a file
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_current_user_id
from app.services import audit_log

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
    return {"items": audit_log.list_l1_logs()}


@router.get("/l1/{file_id}")
def get_l1(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    blob = audit_log.read_l1_log(file_id)
    if not blob:
        raise HTTPException(status_code=404, detail="L1 log not found for file")
    return blob
