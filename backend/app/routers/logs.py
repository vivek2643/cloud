"""
Read-only access to the L1 analysis logs, sourced directly from Postgres.

That's deliberate: video analysis runs on remote GPU workers whose local disks
the API server can't read, whereas Postgres is shared by every worker and the API.

Endpoints:
  GET /api/logs/l1              list per-file L1 analyses (+ seconds taken)
  GET /api/logs/l1/{file_id}    full L1 analysis for a file
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user_id
from app.services.l1 import snapshot

router = APIRouter(prefix="/api/logs", tags=["logs"])


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
