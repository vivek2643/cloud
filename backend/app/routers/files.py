from __future__ import annotations
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase
from app.services.r2 import generate_presigned_get
from app.models.schemas import FileResponse, FileUpdate, FileMoveRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


def _analysis_progress(f: dict) -> tuple[float, str]:
    """Coarse, monotonic (progress, phase) for a file, from its lifecycle flags.

    Tracks L1 analysis, not the raw upload -- in the client-proxy fast path L1
    runs on the proxies while the raw is still uploading. The final 100% is
    gated on status=='ready' so the bar completes only once the file is actually
    playable (editing proxy done)."""
    if f.get("file_type") not in ("video", "audio"):
        return 1.0, "ready"
    status = f.get("status")
    l1 = f.get("l1_status") or "pending"
    if l1 == "failed":
        return 1.0, "failed"
    if l1 == "pending":
        return (0.03, "uploading") if status == "uploading" else (0.08, "queued")
    if l1 == "running":
        return 0.4, "analyzing"
    # L1 done (ready/skipped) -> waiting on the editing proxy for playback.
    return (1.0, "ready") if status == "ready" else (0.95, "finishing")


def _with_progress(f: dict) -> dict:
    p, phase = _analysis_progress(f)
    f["analysis_progress"] = p
    f["analysis_phase"] = phase
    return f


@router.get("", response_model=List[FileResponse])
def list_files(
    folder_id: Optional[str] = Query(None),
    root: bool = Query(False),
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    query = sb.table("files").select("*").eq("user_id", user_id)

    if root or folder_id is None:
        query = query.is_("folder_id", "null")
    else:
        query = query.eq("folder_id", folder_id)

    result = query.order("created_at", desc=True).execute()
    return [_with_progress(f) for f in result.data]


@router.get("/{file_id}", response_model=FileResponse)
def get_file(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    result = sb.table("files").select("*").eq("id", file_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")
    return _with_progress(result.data[0])


@router.patch("/{file_id}", response_model=FileResponse)
def rename_file(
    file_id: str,
    body: FileUpdate,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    result = (
        sb.table("files")
        .update({"name": body.name})
        .eq("id", file_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")
    return result.data[0]


@router.post("/{file_id}/move", response_model=FileResponse)
def move_file(
    file_id: str,
    body: FileMoveRequest,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()

    if body.folder_id:
        folder = sb.table("folders").select("id").eq("id", body.folder_id).eq("user_id", user_id).execute()
        if not folder.data:
            raise HTTPException(status_code=404, detail="Target folder not found")

    result = (
        sb.table("files")
        .update({"folder_id": body.folder_id})
        .eq("id", file_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")
    return result.data[0]


@router.delete("/{file_id}")
def delete_file(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    file_result = sb.table("files").select(
        "r2_key, r2_proxy_key, r2_proxy_a_key, r2_proxy_b_key, r2_thumbnail_key"
    ).eq("id", file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")

    sb.table("files").delete().eq("id", file_id).eq("user_id", user_id).execute()

    from app.services.r2 import delete_object
    f = file_result.data[0]
    for key in [
        f.get("r2_key"), f.get("r2_proxy_key"),
        f.get("r2_proxy_a_key"), f.get("r2_proxy_b_key"),
        f.get("r2_thumbnail_key"),
    ]:
        if key:
            try:
                delete_object(key)
            except Exception:
                pass

    return {"ok": True}


@router.get("/{file_id}/playback")
def get_playback_url(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return a presigned URL for proxy video playback (or raw if no proxy)."""
    sb = get_supabase()
    result = sb.table("files").select("r2_proxy_key, r2_key").eq("id", file_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")

    f = result.data[0]
    key = f.get("r2_proxy_key") or f["r2_key"]
    url = generate_presigned_get(key, expires_in=7200)
    return {"url": url}


@router.get("/{file_id}/download")
def get_download_url(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return a presigned URL for downloading the original file."""
    sb = get_supabase()
    result = sb.table("files").select("r2_key").eq("id", file_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")

    url = generate_presigned_get(result.data[0]["r2_key"], expires_in=7200)
    return {"url": url}


@router.get("/{file_id}/dialogues")
def get_dialogues(
    file_id: str,
    level: Optional[str] = Query(None, description="sentence | topic (omit for both)"),
    user_id: str = Depends(get_current_user_id),
):
    """Return the precomputed Dialogues-lens selects for a file.

    Zero recompute: this just reads the `dialogue_segments` document the L1
    `dialogue_segments` stage wrote. `ready` is false when the stage hasn't run
    yet (or the clip has no speech)."""
    sb = get_supabase()
    owns = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not owns.data:
        raise HTTPException(status_code=404, detail="File not found")

    row = (
        sb.table("dialogue_segments")
        .select("segments")
        .eq("file_id", file_id)
        .execute()
    )
    segs = (row.data[0]["segments"] if row.data else {}) or {}
    sentence = segs.get("sentence", []) or []
    topic = segs.get("topic", []) or []
    if level == "sentence":
        return {"sentence": sentence, "ready": bool(row.data)}
    if level == "topic":
        return {"topic": topic, "ready": bool(row.data)}
    return {"sentence": sentence, "topic": topic, "ready": bool(row.data)}


@router.get("/{file_id}/l1")
def get_l1_index(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return the full L1 analysis for a file as one JSON document
    (file row + transcript + audio_features + cut grids + processing_jobs + summary).

    The same payload also lives on disk at backend/logs/l1/<file_id>.json.
    """
    from app.services.l1.snapshot import build_l1_snapshot

    sb = get_supabase()
    owns = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not owns.data:
        raise HTTPException(status_code=404, detail="File not found")

    return build_l1_snapshot(file_id)
