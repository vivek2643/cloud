from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase
from app.models.schemas import FolderCreate, FolderUpdate, FolderResponse, BreadcrumbItem

router = APIRouter(prefix="/api/folders", tags=["folders"])


@router.get("", response_model=List[FolderResponse])
def list_folders(
    parent_id: Optional[str] = Query(None),
    root: bool = Query(False),
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    query = sb.table("folders").select("*").eq("user_id", user_id)

    if root or parent_id is None:
        query = query.is_("parent_id", "null")
    else:
        query = query.eq("parent_id", parent_id)

    result = query.order("name").execute()
    return result.data


@router.post("", response_model=FolderResponse)
def create_folder(
    body: FolderCreate,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()

    if body.parent_id:
        parent = sb.table("folders").select("id").eq("id", body.parent_id).eq("user_id", user_id).execute()
        if not parent.data:
            raise HTTPException(status_code=404, detail="Parent folder not found")

    result = sb.table("folders").insert({
        "user_id": user_id,
        "name": body.name,
        "parent_id": body.parent_id,
    }).execute()

    return result.data[0]


@router.patch("/{folder_id}", response_model=FolderResponse)
def rename_folder(
    folder_id: str,
    body: FolderUpdate,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    result = (
        sb.table("folders")
        .update({"name": body.name})
        .eq("id", folder_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Folder not found")
    return result.data[0]


@router.delete("/{folder_id}")
def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    result = (
        sb.table("folders")
        .delete()
        .eq("id", folder_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"ok": True}


@router.get("/{folder_id}/breadcrumb", response_model=List[BreadcrumbItem])
def get_breadcrumb(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Build breadcrumb path from root to this folder."""
    sb = get_supabase()
    crumbs: List[BreadcrumbItem] = []
    current_id: Optional[str] = folder_id

    while current_id:
        result = (
            sb.table("folders")
            .select("id, name, parent_id")
            .eq("id", current_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not result.data:
            break
        folder = result.data[0]
        crumbs.append(BreadcrumbItem(id=folder["id"], name=folder["name"]))
        current_id = folder["parent_id"]

    crumbs.reverse()
    return crumbs
