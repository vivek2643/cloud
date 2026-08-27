"""
frontend_project_ux.plan.md Stage 2: anonymous upload links.

Owner-facing endpoints (all Depends(get_current_user_id)): create / list /
revoke a link for a folder the caller owns.

Public endpoints (NO auth dependency at all): everything an anonymous
visitor needs to validate a link and upload through it, reusing the SAME
core upload logic app/routers/upload.py exposes (presign_core,
multipart_create_core, etc.) rather than a second copy -- two copies of
multipart/analysis-proxy/L1-kick logic would drift, and the failure mode is
"files uploaded through a link are silently never analyzed."

Security model (see the plan's own "Traps"): user_id and folder_id for
every public call come from the resolved LINK ROW ONLY, never from a
client-supplied field. DEV_USER_ID currently makes every request resolve to
the same user, so a mistake here would look like it works locally and
become a cross-tenant write the moment real auth is restored.
"""
from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase
from app.routers.upload import (
    complete_analysis_proxies_core,
    complete_upload_core,
    multipart_abort_core,
    multipart_complete_core,
    multipart_create_core,
    presign_analysis_proxies_core,
    presign_core,
)
from app.models.schemas import (
    AnalysisProxyPresignResponse,
    CreateUploadLinkRequest,
    FileResponse,
    MultipartCreateResponse,
    PresignResponse,
    PublicMultipartAbortRequest,
    PublicMultipartCompleteRequest,
    PublicMultipartCreateRequest,
    PublicPresignRequest,
    PublicUploadLinkInfo,
    UploadLinkResponse,
)

logger = logging.getLogger(__name__)

# Mixed path prefixes (/api/folders/{folder_id}/upload-links AND
# /api/upload-links/{link_id}), so this stays prefix-less with full paths
# per route -- matching how app/main.py registers it, "next to the others."
router = APIRouter(tags=["upload-links"])
public_router = APIRouter(prefix="/api/public/upload-links", tags=["upload-links-public"])


def _parse_pg_timestamptz(s: str) -> datetime:
    """Supabase/PostgREST emits variable-precision fractional seconds
    (trailing zeros stripped, e.g. '...:50.25913+00:00') -- Python 3.9's
    datetime.fromisoformat REJECTS this outright (it requires exactly 3 or
    6 fractional digits; verified live against this exact codebase's own
    Supabase project). Pad/truncate to 6 before parsing rather than add a
    new dependency (python-dateutil parses it fine, but it's only ever a
    transitive one here -- not worth pinning for one call site)."""
    m = re.match(r"^(.*\.)(\d+)(\+.*)$", s)
    if m:
        frac = m.group(2)[:6].ljust(6, "0")
        s = f"{m.group(1)}{frac}{m.group(3)}"
    return datetime.fromisoformat(s)


# --------------------------------------------------------------------------
# Owner-facing
# --------------------------------------------------------------------------

def _check_folder_ownership(sb, folder_id: str, user_id: str) -> None:
    folder = sb.table("folders").select("id").eq("id", folder_id).eq("user_id", user_id).execute()
    if not folder.data:
        raise HTTPException(status_code=404, detail="Folder not found")


@router.post("/api/folders/{folder_id}/upload-links", response_model=UploadLinkResponse)
def create_upload_link(
    folder_id: str,
    body: CreateUploadLinkRequest,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    _check_folder_ownership(sb, folder_id, user_id)

    expires_at = None
    if body.expires_in_hours is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=body.expires_in_hours)).isoformat()

    # A bearer credential: anyone holding it can write into the project, so
    # it must be unguessable -- token_urlsafe(32) is 256 bits of entropy.
    token = secrets.token_urlsafe(32)
    result = sb.table("upload_links").insert({
        "token": token,
        "folder_id": folder_id,
        "user_id": user_id,
        "expires_at": expires_at,
        "max_files": body.max_files,
    }).execute()
    return result.data[0]


@router.get("/api/folders/{folder_id}/upload-links", response_model=List[UploadLinkResponse])
def list_upload_links(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
):
    sb = get_supabase()
    _check_folder_ownership(sb, folder_id, user_id)
    result = (
        sb.table("upload_links")
        .select("*")
        .eq("folder_id", folder_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@router.delete("/api/upload-links/{link_id}")
def revoke_upload_link(
    link_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Soft-delete only (revoked = true) -- a revoked link must still be
    able to say "this link was turned off" to a visitor holding the URL,
    which a hard-deleted row (404, indistinguishable from a typo) can't."""
    sb = get_supabase()
    result = (
        sb.table("upload_links")
        .update({"revoked": True})
        .eq("id", link_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Link not found")
    return {"ok": True}


# --------------------------------------------------------------------------
# Public -- no auth dependency on anything below this line
# --------------------------------------------------------------------------

def _resolve_link(token: str) -> dict:
    """Shared guard for every public route: unknown token -> 404; revoked,
    expired, or quota-exhausted -> 410 (the resource existed but is gone,
    which is the honest distinction from "never existed")."""
    sb = get_supabase()
    result = sb.table("upload_links").select("*").eq("token", token).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Link not found")
    link = result.data[0]
    if link["revoked"]:
        raise HTTPException(status_code=410, detail="This link has been revoked")
    if link["expires_at"] and datetime.now(timezone.utc) > _parse_pg_timestamptz(link["expires_at"]):
        raise HTTPException(status_code=410, detail="This link has expired")
    if link["max_files"] is not None and link["used_count"] >= link["max_files"]:
        raise HTTPException(status_code=410, detail="This link has reached its upload limit")
    return link


def _increment_used_count(link_id: str) -> None:
    """Called at COMPLETION only, never at presign -- an abandoned or failed
    presign must not burn the quota (2.3's explicit rule). Best-effort
    read-modify-write: supabase-py has no atomic increment, and a lost
    increment under a rare concurrent-complete race on the SAME link only
    under-counts, never over-counts -- the safe direction for a quota.
    Upload success matters more than the counter, so this never raises."""
    try:
        sb = get_supabase()
        row = sb.table("upload_links").select("used_count").eq("id", link_id).execute()
        if row.data:
            sb.table("upload_links").update(
                {"used_count": row.data[0]["used_count"] + 1}
            ).eq("id", link_id).execute()
    except Exception:
        logger.exception("upload_links: failed to increment used_count for %s", link_id)


@public_router.get("/{token}", response_model=PublicUploadLinkInfo)
def get_public_link_info(token: str):
    """Validate + return ONLY {project_name, expires_at, remaining} -- never
    user_id, folder_id, or anything about other files in the project."""
    link = _resolve_link(token)
    sb = get_supabase()
    folder = sb.table("folders").select("name").eq("id", link["folder_id"]).execute()
    project_name = folder.data[0]["name"] if folder.data else "Untitled"
    remaining = None if link["max_files"] is None else max(0, link["max_files"] - link["used_count"])
    return PublicUploadLinkInfo(
        project_name=project_name, expires_at=link["expires_at"], remaining=remaining)


@public_router.post("/{token}/presign", response_model=PresignResponse)
def public_presign(token: str, body: PublicPresignRequest):
    link = _resolve_link(token)
    return presign_core(
        link["user_id"], link["folder_id"], body.filename, body.content_type, body.file_size)


@public_router.post("/{token}/multipart/create", response_model=MultipartCreateResponse)
def public_multipart_create(token: str, body: PublicMultipartCreateRequest):
    link = _resolve_link(token)
    return multipart_create_core(
        link["user_id"], link["folder_id"], body.filename, body.content_type, body.file_size)


@public_router.post("/{token}/multipart/complete", response_model=FileResponse)
def public_multipart_complete(token: str, body: PublicMultipartCompleteRequest):
    link = _resolve_link(token)
    result = multipart_complete_core(
        link["user_id"], body.file_id, body.upload_id, folder_id=link["folder_id"])
    _increment_used_count(link["id"])
    return result


@public_router.post("/{token}/multipart/abort")
def public_multipart_abort(token: str, body: PublicMultipartAbortRequest):
    link = _resolve_link(token)
    return multipart_abort_core(
        link["user_id"], body.file_id, body.upload_id, folder_id=link["folder_id"])


@public_router.post(
    "/{token}/files/{file_id}/analysis-proxies/presign", response_model=AnalysisProxyPresignResponse)
def public_presign_analysis_proxies(token: str, file_id: str):
    link = _resolve_link(token)
    return presign_analysis_proxies_core(link["user_id"], file_id, folder_id=link["folder_id"])


@public_router.post(
    "/{token}/files/{file_id}/analysis-proxies/complete", response_model=FileResponse)
def public_complete_analysis_proxies(token: str, file_id: str):
    link = _resolve_link(token)
    return complete_analysis_proxies_core(link["user_id"], file_id, folder_id=link["folder_id"])


@public_router.post("/{token}/files/{file_id}/complete", response_model=FileResponse)
def public_complete_upload(token: str, file_id: str):
    link = _resolve_link(token)
    result = complete_upload_core(link["user_id"], file_id, folder_id=link["folder_id"])
    _increment_used_count(link["id"])
    return result
