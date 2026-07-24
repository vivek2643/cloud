import logging
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user_id
from app.config import get_settings
from app.services.r2 import (
    abort_multipart_upload,
    complete_multipart_upload,
    create_multipart_upload,
    generate_presigned_put,
    generate_presigned_upload_parts,
    part_size_for,
)
from app.services.supabase_client import get_supabase
from app.models.schemas import (
    AnalysisProxyPresignResponse,
    FileResponse,
    MultipartAbortRequest,
    MultipartCompleteRequest,
    MultipartCreateRequest,
    MultipartCreateResponse,
    PresignRequest,
    PresignResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])


def _detect_file_type(content_type: str) -> str:
    if content_type.startswith("video/"):
        return "video"
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("audio/"):
        return "audio"
    if content_type.startswith("application/pdf") or content_type.startswith("text/"):
        return "document"
    return "other"


@router.post("/presign", response_model=PresignResponse)
def presign_upload(
    body: PresignRequest,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()

    if body.folder_id:
        folder = sb.table("folders").select("id").eq("id", body.folder_id).eq("user_id", user_id).execute()
        if not folder.data:
            raise HTTPException(status_code=404, detail="Folder not found")

    file_id = str(uuid.uuid4())
    r2_key = f"raw/{user_id}/{file_id}/{body.filename}"

    sb.table("files").insert({
        "id": file_id,
        "user_id": user_id,
        "folder_id": body.folder_id,
        "name": body.filename,
        "filename": body.filename,
        "mime_type": body.content_type,
        "file_size": body.file_size,
        "file_type": _detect_file_type(body.content_type),
        "r2_key": r2_key,
        "status": "uploading",
    }).execute()

    upload_url = generate_presigned_put(r2_key, body.content_type)

    return PresignResponse(file_id=file_id, upload_url=upload_url)


def _analysis_proxy_keys(file_id: str) -> tuple[str, str]:
    """Deterministic R2 keys for the two client analysis proxies. Deterministic
    (not client-chosen) so the /complete step can register them without trusting
    a client-sent key."""
    return (
        f"proxies/{file_id}/analysis_a.mp4",
        f"proxies/{file_id}/motion_b.mp4",
    )


def _enqueue_task(task_name: str, file_id: str, r2_key: str, user_id: str) -> bool:
    """Defer a GPU-queue ingest task via a short-lived, per-call procrastinate
    connector (see the long note on the original _enqueue_l1 for why a per-call
    connector avoids the concurrent-complete race). Best-effort.

    scale_architecture.plan.md Pillar 5: `lock` keyed on the file (not the
    task name -- l1_orchestrate/l1_editing_proxy are mutually exclusive
    alternatives for the SAME file, never both) makes jobs.py's own claimed
    invariant ("never run two L1 pipelines for the same file concurrently")
    actually true -- a retried /complete call or a duplicate multipart
    finalize can't start a second GPU pipeline run while one is still
    `doing`; it just waits behind the lock instead.

    Pillar 6: priority (not a hard cap -- upload already succeeded, the file
    needs SOME L1 run eventually) is lower the more of this user's OWN files
    are already mid-L1, so one user's bulk upload doesn't bury a quieter
    user's single file behind the whole batch on a contended gpu queue."""
    try:
        from procrastinate import App, PsycopgConnector

        from app.services import fairness

        priority = fairness.priority_for_l1(user_id)

        enqueue_app = App(connector=PsycopgConnector(
            conninfo=get_settings().database_url, min_size=1, max_size=2))
        with enqueue_app.open():
            enqueue_app.configure_task(
                task_name, queue="gpu", lock=f"l1:{file_id}", priority=priority,
            ).defer(file_id=file_id, r2_key=r2_key)
        return True
    except Exception:
        logger.exception("Could not enqueue %s for %s; file is still uploaded.", task_name, file_id)
        return False


def _enqueue_l1(file_id: str, r2_key: str, user_id: str) -> bool:
    """
    Defer the L1 orchestrator onto procrastinate. Returns True on success.

    Uses a short-lived App with its OWN connector per call instead of the shared
    global `proc_app`. FastAPI runs these sync endpoints in a threadpool, so two
    uploads finishing at once both ran `with proc_app.open(): ... ` on the same
    global app -- the first thread's context-exit closed the connector pool out
    from under the second thread's defer(), which then raised, got swallowed, and
    left that file stuck at l1_status='pending' with no job ever enqueued. A
    per-call connector is isolated, so concurrent completes can't race.

    `configure_task` defers by task name, so the API process doesn't import the
    heavy ML task modules. Still best-effort: if DATABASE_URL isn't set the
    upload itself succeeds and the file stays pending until a worker is up.
    """
    # queue="gpu": GPU fleet runs ingest; CPU render workers ignore it.
    return _enqueue_task("l1_orchestrate", file_id, r2_key, user_id)


def _finalize_upload(sb, file_record: dict) -> dict:
    """Flip an uploaded RAW file to processing/ready and kick off the right
    ingest task. Shared by the single-PUT and multipart completion paths.

    - Client fast path (analysis proxy A already registered): analysis was
      already fired on the proxies by /analysis-proxies/complete, so the raw
      only needs to become the 1080p editing proxy.
    - Fallback (no client proxies): regenerate every analysis input from the raw
      via the full L1 orchestrator, exactly as before.

    Audio uploads run the video-free music path; everything non-analyzable is
    immediately ready with no analysis."""
    analyzable = file_record["file_type"] in ("video", "audio")
    new_status = "processing" if analyzable else "ready"
    result = (
        sb.table("files").update({"status": new_status}).eq("id", file_record["id"]).execute()
    )
    if analyzable:
        if file_record.get("r2_proxy_a_key"):
            _enqueue_task("l1_editing_proxy", file_record["id"], file_record["r2_key"], file_record["user_id"])
        else:
            _enqueue_l1(file_record["id"], file_record["r2_key"], file_record["user_id"])
    return result.data[0]


# --- Multipart upload (files > 5 GiB, and any large upload) -------------------

@router.post("/multipart/create", response_model=MultipartCreateResponse)
def multipart_create(
    body: MultipartCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()

    if body.folder_id:
        folder = sb.table("folders").select("id").eq("id", body.folder_id).eq("user_id", user_id).execute()
        if not folder.data:
            raise HTTPException(status_code=404, detail="Folder not found")

    file_id = str(uuid.uuid4())
    r2_key = f"raw/{user_id}/{file_id}/{body.filename}"

    upload_id = create_multipart_upload(r2_key, body.content_type)
    psize = part_size_for(body.file_size)
    part_count = max(1, math.ceil(body.file_size / psize))
    part_urls = generate_presigned_upload_parts(r2_key, upload_id, part_count)

    sb.table("files").insert({
        "id": file_id,
        "user_id": user_id,
        "folder_id": body.folder_id,
        "name": body.filename,
        "filename": body.filename,
        "mime_type": body.content_type,
        "file_size": body.file_size,
        "file_type": _detect_file_type(body.content_type),
        "r2_key": r2_key,
        "status": "uploading",
    }).execute()

    return MultipartCreateResponse(
        file_id=file_id,
        r2_key=r2_key,
        upload_id=upload_id,
        part_size=psize,
        part_urls=part_urls,
    )


@router.post("/multipart/complete", response_model=FileResponse)
def multipart_complete(
    body: MultipartCompleteRequest,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    file_result = sb.table("files").select("*").eq("id", body.file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")
    file_record = file_result.data[0]
    if file_record["status"] != "uploading":
        raise HTTPException(status_code=400, detail="File is not in uploading state")

    try:
        complete_multipart_upload(file_record["r2_key"], body.upload_id)
    except Exception as e:
        logger.exception("Multipart complete failed for %s", body.file_id)
        raise HTTPException(status_code=400, detail=f"Could not complete upload: {e}")

    return _finalize_upload(sb, file_record)


@router.post("/multipart/abort")
def multipart_abort(
    body: MultipartAbortRequest,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    file_result = sb.table("files").select("*").eq("id", body.file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")
    file_record = file_result.data[0]

    abort_multipart_upload(file_record["r2_key"], body.upload_id)
    # Drop the placeholder row so it doesn't linger as a stuck 'uploading' file.
    sb.table("files").delete().eq("id", body.file_id).eq("user_id", user_id).execute()
    return {"ok": True}


# --- Client analysis proxies (see client_proxy.plan.md) ----------------------
#
# The desktop app decodes the local file ONCE and produces two tiny proxies that
# upload in seconds while the raw uploads in the background. Analysis fires the
# moment these land -- no waiting on the multi-GB raw.

@router.post("/{file_id}/analysis-proxies/presign", response_model=AnalysisProxyPresignResponse)
def presign_analysis_proxies(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    file_result = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")

    key_a, key_b = _analysis_proxy_keys(file_id)
    return AnalysisProxyPresignResponse(
        proxy_a_url=generate_presigned_put(key_a, "video/mp4"),
        proxy_a_key=key_a,
        proxy_b_url=generate_presigned_put(key_b, "video/mp4"),
        proxy_b_key=key_b,
    )


@router.post("/{file_id}/analysis-proxies/complete", response_model=FileResponse)
def complete_analysis_proxies(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Register the two uploaded analysis proxies and fire L1 immediately -- this
    is what decouples analysis from the raw upload. The raw's own /complete then
    only has to make the editing proxy. Idempotent: re-registering is harmless,
    and L1's stages are individually idempotent.

    NOTE: we deliberately do NOT flip status here. The raw genuinely is still
    uploading, and the raw-complete endpoints guard on status=='uploading' -- so
    the file stays 'uploading' until the raw lands, then flips to processing.
    Analysis lifecycle is tracked separately by l1_status."""
    sb = get_supabase()
    file_result = sb.table("files").select("*").eq("id", file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")
    file_record = file_result.data[0]
    if file_record["file_type"] != "video":
        raise HTTPException(status_code=400, detail="Analysis proxies are only for video files")

    key_a, key_b = _analysis_proxy_keys(file_id)
    sb.table("files").update({
        "r2_proxy_a_key": key_a,
        "r2_proxy_b_key": key_b,
    }).eq("id", file_id).execute()

    _enqueue_l1(file_id, file_record["r2_key"])
    file_record.update({"r2_proxy_a_key": key_a, "r2_proxy_b_key": key_b})
    return file_record


# NOTE: This dynamic route must be registered AFTER the static /multipart/* routes,
# otherwise "/multipart/complete" gets captured here with file_id="multipart".
@router.post("/{file_id}/complete", response_model=FileResponse)
def complete_upload(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    file_result = sb.table("files").select("*").eq("id", file_id).eq("user_id", user_id).execute()
    if not file_result.data:
        raise HTTPException(status_code=404, detail="File not found")
    file_record = file_result.data[0]
    if file_record["status"] != "uploading":
        raise HTTPException(status_code=400, detail="File is not in uploading state")
    return _finalize_upload(sb, file_record)
