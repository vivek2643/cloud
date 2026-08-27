from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Folders ---

class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None


class FolderUpdate(BaseModel):
    name: str


class FolderResponse(BaseModel):
    id: str
    user_id: str
    name: str
    parent_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class BreadcrumbItem(BaseModel):
    id: Optional[str]
    name: str


# --- Files ---

class FileResponse(BaseModel):
    id: str
    user_id: str
    folder_id: Optional[str]
    name: str
    filename: str
    mime_type: str
    file_size: int
    file_type: str
    r2_key: str
    r2_proxy_key: Optional[str]
    r2_thumbnail_key: Optional[str]
    duration_seconds: Optional[float]
    width: Optional[int]
    height: Optional[int]
    status: str
    l1_status: Optional[str] = None
    # Coarse, monotonic analysis progress (0..1) + a short phase label, derived
    # from status/l1_status so the UI can show a determinate bar instead of a
    # perpetual "Processing…". Defaults to done for responses that don't compute
    # it (the file list/get endpoints do).
    analysis_progress: float = 1.0
    analysis_phase: str = "ready"
    created_at: datetime
    updated_at: datetime


class FileUpdate(BaseModel):
    name: str


class FileMoveRequest(BaseModel):
    folder_id: Optional[str]


# --- Upload ---

class PresignRequest(BaseModel):
    filename: str
    content_type: str
    file_size: int
    folder_id: Optional[str] = None


class PresignResponse(BaseModel):
    file_id: str
    upload_url: str


# --- Multipart upload (large files > 5 GiB) ---

class MultipartCreateRequest(BaseModel):
    filename: str
    content_type: str
    file_size: int
    folder_id: Optional[str] = None


class MultipartCreateResponse(BaseModel):
    file_id: str
    r2_key: str
    upload_id: str
    part_size: int
    part_urls: list[str]


class MultipartCompleteRequest(BaseModel):
    file_id: str
    upload_id: str


class MultipartAbortRequest(BaseModel):
    file_id: str
    upload_id: str


# --- Client analysis proxies (see client_proxy.plan.md) ---
# The desktop app decodes the local file once and uploads two tiny proxies
# (A: 480p@1fps + audio for L2 + speech/audio; B: 160x90@10fps for motion) so
# analysis starts in seconds, decoupled from the multi-GB raw upload.

class AnalysisProxyPresignResponse(BaseModel):
    proxy_a_url: str
    proxy_a_key: str
    proxy_b_url: str
    proxy_b_key: str


# --- Anonymous upload links (frontend_project_ux.plan.md Stage 2) ---

class CreateUploadLinkRequest(BaseModel):
    # gt=0: an expiry of 0/negative hours or a 0-file cap would mint a link
    # that's already exhausted -- confusing to create even though
    # _resolve_link would reject it correctly either way.
    expires_in_hours: Optional[int] = Field(default=None, gt=0)
    max_files: Optional[int] = Field(default=None, gt=0)


class UploadLinkResponse(BaseModel):
    id: str
    token: str
    folder_id: str
    expires_at: Optional[datetime]
    max_files: Optional[int]
    used_count: int
    revoked: bool
    created_at: datetime


class PublicUploadLinkInfo(BaseModel):
    """What an anonymous visitor is allowed to know about a link -- never
    user_id, folder_id, or anything about other files in the project."""
    project_name: str
    expires_at: Optional[datetime]
    remaining: Optional[int]   # null = unlimited


class PublicPresignRequest(BaseModel):
    """Same as PresignRequest minus folder_id -- that comes from the link
    row only, never the client, on a public endpoint."""
    filename: str
    content_type: str
    file_size: int


class PublicMultipartCreateRequest(BaseModel):
    filename: str
    content_type: str
    file_size: int


class PublicMultipartCompleteRequest(BaseModel):
    file_id: str
    upload_id: str


class PublicMultipartAbortRequest(BaseModel):
    file_id: str
    upload_id: str
