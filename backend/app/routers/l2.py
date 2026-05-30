"""
L2 trigger endpoints.

  POST /api/files/{id}/l2          -> enqueue full-file L2 enrichment
  POST /api/l2/enrich              -> enqueue L2 on a specific shot id list

Both endpoints enqueue procrastinate jobs and return immediately. The worker
process actually runs the heavy stages (DINOv2, faces, audio events, VLM).
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["l2"])


class L2ShotEnrichBody(BaseModel):
    shot_ids: List[str]


def _enqueue(task_name: str, **kwargs) -> bool:
    """Defer a task by name; tolerate missing DB so the API doesn't 500."""
    try:
        from app.services.jobs import app as proc_app, register_tasks
        register_tasks()
        with proc_app.open():
            proc_app.tasks[task_name].defer(**kwargs)
        return True
    except Exception:
        logger.exception("Could not enqueue %s with %s", task_name, kwargs)
        return False


@router.post("/files/{file_id}/l2")
def enqueue_file_l2(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    row = sb.table("files").select("id,l2_status,l1_status").eq("id", file_id).eq("user_id", user_id).execute()
    if not row.data:
        raise HTTPException(status_code=404, detail="File not found")
    file_row = row.data[0]
    if file_row.get("l1_status") != "ready":
        raise HTTPException(status_code=400, detail="File must finish L1 indexing before L2 enrichment.")

    sb.table("files").update({"l2_status": "running"}).eq("id", file_id).execute()
    ok = _enqueue("l2_enrich_file", file_id=file_id)
    if not ok:
        sb.table("files").update({"l2_status": "failed"}).eq("id", file_id).execute()
        raise HTTPException(status_code=503, detail="Could not enqueue L2 job (worker DB unavailable).")
    return {"ok": True, "file_id": file_id, "l2_status": "running"}


@router.post("/l2/enrich")
def enqueue_shots_l2(
    body: L2ShotEnrichBody,
    user_id: str = Depends(get_current_user_id),
):
    if not body.shot_ids:
        raise HTTPException(status_code=400, detail="shot_ids must be non-empty")

    # Verify the caller owns the parent files for all requested shots.
    sb = get_supabase()
    res = (
        sb.table("shots")
        .select("id,file_id,files(user_id)")
        .in_("id", body.shot_ids)
        .execute()
    )
    found_ids = {row["id"] for row in (res.data or [])}
    missing = [sid for sid in body.shot_ids if sid not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown shot ids: {missing[:5]}")
    bad = [row for row in (res.data or []) if (row.get("files") or {}).get("user_id") != user_id]
    if bad:
        raise HTTPException(status_code=403, detail="Some shots belong to another user.")

    ok = _enqueue("l2_enrich_shots", shot_ids=body.shot_ids)
    if not ok:
        raise HTTPException(status_code=503, detail="Could not enqueue L2 job.")
    return {"ok": True, "shot_count": len(body.shot_ids)}
