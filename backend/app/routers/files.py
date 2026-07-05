from __future__ import annotations
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from app.auth import get_current_user_id
from app.services.supabase_client import get_supabase
from app.services.r2 import generate_presigned_get
from app.models.schemas import FileResponse, FileUpdate, FileMoveRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


def _analysis_progress(f: dict) -> tuple[float, str]:
    """Coarse, monotonic (progress, phase) for a file, from its lifecycle flags.

    Tracks the full analysis (L1 + optional L2), not the raw upload -- in the
    client-proxy fast path L1 runs on the proxies while the raw is still
    uploading. L2 is only awaited when it was actually enqueued (l2_status set);
    audio / ineligible clips have l2_status=None and finish at L1. The final
    100% is gated on status=='ready' so the bar completes only once the file is
    actually playable (editing proxy done)."""
    if f.get("file_type") not in ("video", "audio"):
        return 1.0, "ready"
    status = f.get("status")
    l1 = f.get("l1_status") or "pending"
    l2 = f.get("l2_status")
    if l1 == "failed":
        return 1.0, "failed"
    if l1 == "pending":
        return (0.03, "uploading") if status == "uploading" else (0.08, "queued")
    if l1 == "running":
        return 0.4, "analyzing"
    # L1 done (ready/skipped) -> optional Gemini (L2) phase.
    if l2 == "queued":
        return 0.72, "perceiving"
    if l2 == "running":
        return 0.85, "perceiving"
    # Analysis complete (l2 None/ready/skipped/failed).
    return (1.0, "ready") if status == "ready" else (0.95, "finishing")


def _with_progress(f: dict) -> dict:
    p, phase = _analysis_progress(f)
    f["analysis_progress"] = p
    f["analysis_phase"] = phase
    return f


class HeroCutsFeedRequest(BaseModel):
    file_ids: List[str] = Field(default_factory=list)
    energy: float = Field(0.5, ge=0.0, le=1.0)


class CutsFeedRequest(BaseModel):
    file_ids: List[str] = Field(default_factory=list)
    energy: float = Field(0.5, ge=0.0, le=1.0)


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


@router.post("/{file_id}/reanalyze")
def reanalyze_file(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Force a fresh L2 perception run for ONE owned clip (on-demand backfill).

    Re-running L2 cascades: it re-defers thought segmentation + the hero-cuts
    precompute, so the footage map rebuilds off the fresh perception (e.g. to
    pick up a new schema tag like `valence`)."""
    sb = get_supabase()
    owned = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not owned.data:
        raise HTTPException(status_code=404, detail="File not found")
    from app.services.l2.perception import reenqueue_l2

    state = reenqueue_l2(file_id)
    return {"file_id": file_id, "state": state}


@router.post("/reanalyze-stale")
def reanalyze_stale(
    user_id: str = Depends(get_current_user_id),
):
    """Backfill: re-enqueue L2 for every owned clip whose stored perception
    predates the current schema (or was never perceived). Idempotent enough to
    call repeatedly -- a clip already at the current schema is not listed."""
    from app.services.l2.perception import reenqueue_l2, stale_perception_file_ids

    stale = stale_perception_file_ids(user_id)
    results = {fid: reenqueue_l2(fid) for fid in stale}
    queued = sum(1 for s in results.values() if s == "queued")
    return {"candidates": len(stale), "queued": queued, "results": results}


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


@router.get("/{file_id}/hero-cuts")
def get_hero_cuts(
    file_id: str,
    energy: float = Query(0.5, ge=0.0, le=1.0, description="0=broad/calm .. 1=sharp/punchy"),
    user_id: str = Depends(get_current_user_id),
):
    """Return the ranked hero-cuts feed for a file (the V1 product surface).

    Compute-on-read over already-persisted L1/L2/L3 artifacts (dialogue
    segments + clip_perception + motion grids); deterministic given `energy`.
    `ready` is false when none of the source artifacts exist yet."""
    sb = get_supabase()
    owns = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not owns.data:
        raise HTTPException(status_code=404, detail="File not found")

    from app.services.l3.hero_store import get_hero_feed

    heroes = get_hero_feed([file_id], energy=energy)
    return {"heroes": heroes, "energy": energy, "ready": bool(heroes)}


@router.post("/hero-cuts")
def get_hero_cuts_feed(
    payload: HeroCutsFeedRequest,
    user_id: str = Depends(get_current_user_id),
):
    """One combined hero-cuts feed across many clips, so repeated takes of the
    same content stack across files (best in front). Ownership is verified for
    every requested file; unknown/foreign ids are dropped."""
    file_ids = list(dict.fromkeys(payload.file_ids or []))
    if not file_ids:
        return {"heroes": [], "energy": payload.energy, "ready": False}

    sb = get_supabase()
    owned = (
        sb.table("files").select("id").eq("user_id", user_id).in_("id", file_ids).execute()
    )
    owned_ids = [r["id"] for r in (owned.data or [])]
    if not owned_ids:
        raise HTTPException(status_code=404, detail="No matching files")

    from app.services.l3.hero_store import get_hero_feed

    heroes = get_hero_feed(owned_ids, energy=payload.energy)
    return {"heroes": heroes, "energy": payload.energy, "ready": bool(heroes)}


# --------------------------------------------------------------------------
# Cuts v2 (parallel to /hero-cuts): the deterministic non-overlapping
# partition, served as one contiguous filmstrip per file. See cuts_v2.plan.md
# (Phase B4). Compute-on-read over the same L1/L3 artifacts; no VLM, no energy
# ladder -- the row is a contiguous, non-overlapping partition by construction.
# --------------------------------------------------------------------------

def _build_cuts_for(file_ids: List[str], energy: float = 0.5) -> List[dict]:
    """Partition every file into its non-overlapping, tag-bearing cuts at
    ``energy`` (as plain dicts, each with a convenience ``duration_ms``).
    Best-effort per file: a partition failure yields no cuts for that file,
    never a 500."""
    from app.services.l3.partition import build_partition

    out: List[dict] = []
    for fid in file_ids:
        try:
            cuts = build_partition(fid, energy)
        except Exception:
            logger.exception("cuts v2: partition failed for %s", fid)
            continue
        for c in cuts:
            d = c.to_dict()
            d["duration_ms"] = c.src_out_ms - c.src_in_ms
            out.append(d)
    out.sort(key=lambda d: (d["file_id"], d["src_in_ms"]))
    return out


@router.get("/{file_id}/cuts")
def get_cuts(
    file_id: str,
    energy: float = Query(0.5, ge=0.0, le=1.0, description="0=broad/loose .. 1=tight/split"),
    user_id: str = Depends(get_current_user_id),
):
    """The cuts-v2 partition for ONE file -- a non-overlapping, tag-bearing
    filmstrip in ``src_in_ms`` order. Deterministic given ``energy``; `ready`
    is false when the file has no usable L1 artifacts yet."""
    sb = get_supabase()
    owns = sb.table("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
    if not owns.data:
        raise HTTPException(status_code=404, detail="File not found")

    cuts = _build_cuts_for([file_id], energy)
    return {"cuts": cuts, "energy": energy, "ready": bool(cuts)}


@router.post("/cuts")
def get_cuts_feed(
    payload: CutsFeedRequest,
    user_id: str = Depends(get_current_user_id),
):
    """The cuts-v2 partition across many files -- one flat list the client
    groups into per-file rows. Ownership is verified; foreign/unknown ids are
    dropped."""
    file_ids = list(dict.fromkeys(payload.file_ids or []))
    if not file_ids:
        return {"cuts": [], "energy": payload.energy, "ready": False}

    sb = get_supabase()
    owned = (
        sb.table("files").select("id").eq("user_id", user_id).in_("id", file_ids).execute()
    )
    owned_ids = [r["id"] for r in (owned.data or [])]
    if not owned_ids:
        raise HTTPException(status_code=404, detail="No matching files")

    cuts = _build_cuts_for(owned_ids, payload.energy)
    return {"cuts": cuts, "energy": payload.energy, "ready": bool(cuts)}


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
