from __future__ import annotations
from pydantic import BaseModel
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
    l2_status: Optional[str] = None
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
