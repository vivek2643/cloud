"""
Multicam sync endpoints (audio_sync.plan.md SS10):

  POST   /api/sync/detect                          run correlation, PREVIEW only (no write)
  POST   /api/sync/groups                           persist a declared group
  GET    /api/sync/groups?project_id=               list groups for a project
  GET    /api/sync/groups/{group_id}                one group
  PATCH  /api/sync/groups/{group_id}/authoritative   manual override (SS10)
  PATCH  /api/sync/groups/{group_id}/members/{fid}   manual offset nudge (SS10)
  DELETE /api/sync/groups/{group_id}

v1 is user-declared (SS1): detect is a preview the user reviews/nudges
before POSTing groups to commit it -- nothing is written to `sync_groups`
until that explicit second call.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.services.l3.projects import find_or_create_project
from app.services.l3.sync import store as sync_store
from app.services.l3.sync.authoritative import GroupMember, pick_authoritative
from app.services.l3.sync.detect import HIGH_CONFIDENCE_THRESHOLD, envelope, partition_by_overlap
from app.services.processing import _download_from_r2

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sync", tags=["sync"])


def _pg_conn():
    from app.services import db
    return db.connection()


def _owned_files(file_ids: List[str], user_id: str) -> List[dict]:
    """Every file row (id, r2_key, r2_proxy_key, file_type) the user owns
    among `file_ids`. Silently drops any id that isn't theirs -- never
    reveals whether a file exists under someone else's account."""
    if not file_ids:
        return []
    with _pg_conn() as conn:
        rows = conn.execute(
            "select id::text, r2_key, r2_proxy_key, file_type from files"
            " where id = any(%s::uuid[]) and user_id = %s",
            (file_ids, user_id),
        ).fetchall()
    return [{"id": r[0], "r2_key": r[1], "r2_proxy_key": r[2], "file_type": r[3]} for r in rows]


def _demux_wav(raw_path: str, out_path: str) -> None:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path],
        check=True, capture_output=True, timeout=300,
    )


def _fetch_audio_features(file_ids: List[str]) -> dict:
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, integrated_lufs, true_peak_db, silence_intervals"
            " from audio_features where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {r[0]: {"integrated_lufs": r[1], "true_peak_db": r[2], "silence_intervals": r[3]} for r in rows}


class DetectBody(BaseModel):
    file_ids: List[str] = Field(min_length=2)
    # {file_id: "video_angle" | "audio"}; defaults every file to video_angle
    # except any whose own `file_type` is "audio".
    roles: Optional[dict] = None


@router.post("/detect")
def detect_sync(body: DetectBody, user_id: str = Depends(get_current_user_id)) -> dict:
    files = _owned_files(body.file_ids, user_id)
    found_ids = {f["id"] for f in files}
    missing = [fid for fid in body.file_ids if fid not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"file(s) not found: {missing}")

    envelopes = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            key = f["r2_proxy_key"] or f["r2_key"]
            raw_path = os.path.join(tmp, f"{f['id']}.raw")
            wav_path = os.path.join(tmp, f"{f['id']}.wav")
            try:
                _download_from_r2(key, raw_path)
                _demux_wav(raw_path, wav_path)
                envelopes[f["id"]] = envelope(wav_path)
            except Exception:
                logger.exception("sync detect: envelope failed for %s", f["id"])
                envelopes[f["id"]] = None

    usable = {fid: env for fid, env in envelopes.items() if env is not None and env.size > 0}
    if len(usable) < 2:
        raise HTTPException(status_code=422, detail="Could not extract usable audio from at least 2 files")

    # All-pairs overlap partition: the selected files may resolve into MORE THAN
    # ONE same-audio group (e.g. two camera pairs split across a recording break
    # -- block-1 cams overlap each other, block-2 cams overlap each other, but
    # the two blocks don't overlap at all). Each detected group is previewed
    # separately; files that overlap nobody come back as `ungrouped_file_ids`.
    detected, ungrouped = partition_by_overlap(usable)
    roles = body.roles or {}
    audio_features = _fetch_audio_features(list(found_ids))
    role_of = {f["id"]: roles.get(f["id"]) or ("audio" if f["file_type"] == "audio" else "video_angle")
               for f in files}

    groups = []
    for grp in detected:
        auth_fid = pick_authoritative([
            GroupMember(file_id=fid, role=role_of.get(fid, "video_angle"), audio_features=audio_features.get(fid))
            for fid in grp
        ])
        groups.append({
            "members": [
                {
                    "file_id": fid, "offset_ms": pa.offset_ms, "confidence": round(pa.confidence, 3),
                    "role": role_of.get(fid, "video_angle"), "aligned_by": "auto",
                    "high_confidence": pa.confidence >= HIGH_CONFIDENCE_THRESHOLD,
                }
                for fid, pa in grp.items()
            ],
            "suggested_authoritative_file_id": auth_fid,
        })

    return {
        "groups": groups,
        "ungrouped_file_ids": ungrouped,
        "unusable_file_ids": [fid for fid in body.file_ids if fid not in usable],
    }


class MemberIn(BaseModel):
    file_id: str
    offset_ms: int
    role: str
    confidence: Optional[float] = None
    aligned_by: str = "auto"


class CreateGroupBody(BaseModel):
    members: List[MemberIn] = Field(min_length=2)
    authoritative_audio_file_id: Optional[str] = None


@router.post("/groups")
def create_group(body: CreateGroupBody, user_id: str = Depends(get_current_user_id)) -> dict:
    file_ids = [m.file_id for m in body.members]
    files = _owned_files(file_ids, user_id)
    found_ids = {f["id"] for f in files}
    missing = [fid for fid in file_ids if fid not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"file(s) not found: {missing}")

    auth_fid = body.authoritative_audio_file_id
    if auth_fid and auth_fid not in found_ids:
        raise HTTPException(status_code=422, detail="authoritative_audio_file_id must be one of the members")
    if not auth_fid:
        audio_features = _fetch_audio_features(file_ids)
        auth_fid = pick_authoritative([
            GroupMember(file_id=m.file_id, role=m.role, audio_features=audio_features.get(m.file_id))
            for m in body.members
        ])

    project_id = find_or_create_project(user_id, file_ids)
    group_id = sync_store.create_sync_group(
        project_id,
        [m.model_dump() for m in body.members],
        auth_fid,
        created_by="user",
    )
    return sync_store.get_sync_group(group_id) or {"id": group_id}


@router.get("/groups")
def list_groups(project_id: str, user_id: str = Depends(get_current_user_id)) -> list:
    with _pg_conn() as conn:
        row = conn.execute(
            "select 1 from projects where id = %s and user_id = %s", (project_id, user_id)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return sync_store.list_sync_groups_for_project(project_id)


def _owned_group(group_id: str, user_id: str) -> dict:
    group = sync_store.get_sync_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Sync group not found")
    with _pg_conn() as conn:
        row = conn.execute(
            "select 1 from projects where id = %s and user_id = %s",
            (group["project_id"], user_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Sync group not found")
    return group


@router.get("/groups/{group_id}")
def get_group(group_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    return _owned_group(group_id, user_id)


class AuthoritativeBody(BaseModel):
    file_id: str


@router.patch("/groups/{group_id}/authoritative")
def set_authoritative(
    group_id: str, body: AuthoritativeBody, user_id: str = Depends(get_current_user_id)
) -> dict:
    group = _owned_group(group_id, user_id)
    member_ids = {m["file_id"] for m in group["members"]}
    if body.file_id not in member_ids:
        raise HTTPException(status_code=422, detail="file_id must be a member of this group")
    sync_store.set_authoritative(group_id, body.file_id)
    return sync_store.get_sync_group(group_id) or {}


class OffsetBody(BaseModel):
    offset_ms: int


@router.patch("/groups/{group_id}/members/{file_id}")
def nudge_offset(
    group_id: str, file_id: str, body: OffsetBody, user_id: str = Depends(get_current_user_id)
) -> dict:
    group = _owned_group(group_id, user_id)
    member_ids = {m["file_id"] for m in group["members"]}
    if file_id not in member_ids:
        raise HTTPException(status_code=422, detail="file_id must be a member of this group")
    sync_store.set_member_offset(group_id, file_id, body.offset_ms)
    return sync_store.get_sync_group(group_id) or {}


@router.delete("/groups/{group_id}")
def delete_group(group_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    _owned_group(group_id, user_id)
    sync_store.delete_sync_group(group_id)
    return {"deleted": True}
