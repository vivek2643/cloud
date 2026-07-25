from __future__ import annotations
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase
from app.services.r2 import generate_presigned_get
from app.models.schemas import FolderCreate, FolderUpdate, FolderResponse, BreadcrumbItem

logger = logging.getLogger(__name__)

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


def _descendant_folder_ids(sb, folder_id: str, user_id: str) -> List[str]:
    """BFS the folders tree (user-scoped) to collect the target folder plus
    every descendant subfolder id, root first.

    ``folders.parent_id`` is ON DELETE CASCADE, so deleting the root row would
    tear down subfolder rows for free -- but ``files.folder_id`` is ON DELETE
    SET NULL, so a bare folder delete ORPHANS the files to root instead of
    removing them. We therefore enumerate every level so the caller can delete
    the files under each subfolder explicitly."""
    ids: List[str] = [folder_id]
    seen = {folder_id}
    frontier = [folder_id]
    while frontier:
        res = (
            sb.table("folders")
            .select("id")
            .in_("parent_id", frontier)
            .eq("user_id", user_id)
            .execute()
        )
        children = [r["id"] for r in (res.data or []) if r["id"] not in seen]
        seen.update(children)
        ids.extend(children)
        frontier = children
    return ids


def _purge_related_jobs(
    conn,
    file_ids: List[str],
    project_ids: List[str],
    thread_ids: List[str],
    render_ids: List[str],
    export_ids: List[str],
) -> None:
    """Best-effort delete of the Procrastinate jobs that name any of the
    deleted entities. Procrastinate 3.x stores each task's kwargs FLAT in the
    ``args`` jsonb column (verified against a live row: e.g.
    ``{"file_id": "..."}``, ``{"project_id": "..."}``), NOT nested under
    ``args->'kwargs'`` -- so we match on ``args->>'<key>'`` directly. Deleting a
    ``doing`` row does NOT stop an already-running worker; that's acceptable."""
    conn.execute(
        """
        delete from procrastinate_jobs
        where (args->>'file_id')    = any(%s::text[])
           or (args->>'project_id') = any(%s::text[])
           or (args->>'thread_id')  = any(%s::text[])
           or (args->>'render_id')  = any(%s::text[])
           or (args->>'export_id')  = any(%s::text[])
        """,
        (file_ids, project_ids, thread_ids, render_ids, export_ids),
    )


def _delete_projects_threads_and_jobs(user_id: str, file_ids: List[str]) -> None:
    """Delete the ``projects`` and ``edit_threads`` that reference any of the
    deleted files, then best-effort purge the queue jobs that name any deleted
    file / project / thread / render / export.

    Both a project (migration 006, ``source_file_ids uuid[]``) and an edit
    thread (migration 014, ``file_ids uuid[]``) key their clip set by a uuid
    array of file ids, NOT by folder -- neither has a folder_id column. So we
    resolve them via array-overlap (``&&``) with the files we're deleting: any
    project/thread that touched at least one deleted file is removed. Each then
    FK-cascades its own downstream:
      - projects -> ingest_runs -> cut_records; edl_versions; sync_groups
        (chat_turns.project_id is ON DELETE SET NULL -- logs orphan, by design)
      - edit_threads -> edit_documents; edit_turns; renders; exports;
        resolved_grades; grade_jobs
    """
    if not file_ids:
        return
    from app.services import db

    with db.connection() as conn:
        project_ids = [
            r[0]
            for r in conn.execute(
                "select id::text from projects where user_id = %s and source_file_ids && %s::uuid[]",
                (user_id, file_ids),
            ).fetchall()
        ]
        thread_ids = [
            r[0]
            for r in conn.execute(
                "select id::text from edit_threads where user_id = %s and file_ids && %s::uuid[]",
                (user_id, file_ids),
            ).fetchall()
        ]
        # Collect render/export ids up front so their queue jobs (keyed on
        # render_id/export_id, not thread_id) get purged too, before the
        # thread cascade deletes the rows.
        render_ids: List[str] = []
        export_ids: List[str] = []
        if thread_ids:
            render_ids = [
                r[0]
                for r in conn.execute(
                    "select id::text from renders where thread_id = any(%s::uuid[])",
                    (thread_ids,),
                ).fetchall()
            ]
            export_ids = [
                r[0]
                for r in conn.execute(
                    "select id::text from exports where thread_id = any(%s::uuid[])",
                    (thread_ids,),
                ).fetchall()
            ]

        if project_ids:
            conn.execute("delete from projects where id = any(%s::uuid[])", (project_ids,))
        if thread_ids:
            conn.execute("delete from edit_threads where id = any(%s::uuid[])", (thread_ids,))

        # Queue cleanup must never fail the whole delete -- the DB rows are
        # already gone; a stuck/failed purge is a housekeeping issue, not a
        # data-integrity one.
        try:
            _purge_related_jobs(conn, file_ids, project_ids, thread_ids, render_ids, export_ids)
        except Exception:
            logger.exception("folder delete: procrastinate_jobs cleanup failed (ignored)")


@router.delete("/{folder_id}")
def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a project (== a top-level folder) and EVERYTHING under it.

    Cascade order (children before parents so no FK trips):
      1. Enumerate the folder + all descendant subfolders.
      2. Delete every file row under them (cascades all L1/L3 analysis off
         ``files.id``), then remove each file's R2 objects best-effort.
      3. Delete the projects + edit_threads that referenced those files
         (each cascades its own downstream), and purge related queue jobs.
      4. Delete the folder rows.
    """
    sb = get_supabase()
    owns = (
        sb.table("folders")
        .select("id")
        .eq("id", folder_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not owns.data:
        raise HTTPException(status_code=404, detail="Folder not found")

    folder_ids = _descendant_folder_ids(sb, folder_id, user_id)

    files_res = (
        sb.table("files")
        .select("id, r2_key, r2_proxy_key, r2_proxy_a_key, r2_proxy_b_key, r2_thumbnail_key")
        .in_("folder_id", folder_ids)
        .eq("user_id", user_id)
        .execute()
    )
    file_rows = files_res.data or []
    file_ids = [f["id"] for f in file_rows]

    # 1) File rows first -- this cascades every analysis table keyed on files.id.
    if file_ids:
        sb.table("files").delete().in_("id", file_ids).eq("user_id", user_id).execute()

    # 2) Projects + edit threads that referenced those files, plus queue jobs.
    _delete_projects_threads_and_jobs(user_id, file_ids)

    # 3) R2 teardown per file (same best-effort pattern as delete_file).
    from app.services.r2 import delete_object

    for f in file_rows:
        for key in (
            f.get("r2_key"),
            f.get("r2_proxy_key"),
            f.get("r2_proxy_a_key"),
            f.get("r2_proxy_b_key"),
            f.get("r2_thumbnail_key"),
        ):
            if key:
                try:
                    delete_object(key)
                except Exception:
                    pass

    # 4) Folder rows, deepest first (parent_id also cascades, but be explicit).
    for fid in reversed(folder_ids):
        sb.table("folders").delete().eq("id", fid).eq("user_id", user_id).execute()

    return {"ok": True}


@router.get("/{folder_id}/covers")
def get_folder_covers(
    folder_id: str,
    limit: int = Query(3, ge=1, le=6),
    user_id: str = Depends(get_current_user_id),
):
    """Return presigned thumbnail URLs for the first few videos in a folder.

    Used to render the project card's stacked-clip preview. Best-effort: files
    without a generated thumbnail are skipped."""
    sb = get_supabase()
    result = (
        sb.table("files")
        .select("id, r2_thumbnail_key, created_at")
        .eq("user_id", user_id)
        .eq("folder_id", folder_id)
        .eq("file_type", "video")
        .order("created_at")
        .execute()
    )
    keys = [
        row["r2_thumbnail_key"]
        for row in (result.data or [])
        if row.get("r2_thumbnail_key")
    ][:limit]
    urls = [generate_presigned_get(k, expires_in=7200) for k in keys]
    return {"urls": urls}


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
