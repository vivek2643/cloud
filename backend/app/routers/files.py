from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase
from app.services.r2 import generate_presigned_get
from app.models.schemas import FileResponse, FileUpdate, FileMoveRequest

router = APIRouter(prefix="/api/files", tags=["files"])


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
    return result.data


@router.get("/{file_id}", response_model=FileResponse)
def get_file(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    result = sb.table("files").select("*").eq("id", file_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")
    return result.data[0]


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
    file_result = sb.table("files").select("r2_key, r2_proxy_key, r2_thumbnail_key").eq("id", file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")

    sb.table("files").delete().eq("id", file_id).eq("user_id", user_id).execute()

    from app.services.r2 import delete_object
    f = file_result.data[0]
    for key in [f.get("r2_key"), f.get("r2_proxy_key"), f.get("r2_thumbnail_key")]:
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


@router.get("/{file_id}/shots")
def get_file_shots(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return the ordered shot list for a file plus its total duration in ms.

    Used by the editor when a whole raw file is dropped onto the timeline: the
    client expands the file into its constituent shots (each carrying a real
    shot_id) and coalesces the contiguous span back into a single clip, so the
    dropped file is committable to the EDL and renders as one continuous segment.
    """
    sb = get_supabase()
    f = (
        sb.table("files")
        .select("id, duration_seconds")
        .eq("id", file_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not f.data:
        raise HTTPException(status_code=404, detail="File not found")

    rows = (
        sb.table("shots")
        .select("id, shot_index, start_ms, end_ms")
        .eq("file_id", file_id)
        .order("shot_index")
        .execute()
    )
    shots = [
        {
            "shot_id": str(r["id"]),
            "shot_index": r.get("shot_index"),
            "start_ms": r.get("start_ms"),
            "end_ms": r.get("end_ms"),
        }
        for r in (rows.data or [])
    ]

    duration_ms: Optional[int] = None
    dur_s = f.data[0].get("duration_seconds")
    if dur_s is not None:
        duration_ms = int(round(float(dur_s) * 1000))
    elif shots:
        duration_ms = max((s["end_ms"] or 0) for s in shots)

    return {"file_id": file_id, "duration_ms": duration_ms, "shots": shots}


@router.get("/{file_id}/l1")
def get_l1_index(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return the full L1 analysis for a file as one JSON document
    (file row + transcript + shots + audio_features + processing_jobs + summary).

    The same payload also lives on disk at backend/logs/l1/<file_id>.json.
    """
    from app.services.l1.snapshot import build_l1_snapshot

    sb = get_supabase()
    owns = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not owns.data:
        raise HTTPException(status_code=404, detail="File not found")

    return build_l1_snapshot(file_id)
