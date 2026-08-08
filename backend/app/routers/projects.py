"""
Cuts project API.

  POST /api/projects                 find-or-create a project for a file set
  POST /api/projects/{id}/ingest      kick (or re-kick) the LLM ingest
  GET  /api/projects/{id}/cuts        latest ingest_run + every cut_record
  POST /api/projects/{id}/cuts/energy re-derive video cut boundaries (vcut only)

See cuts_v3.plan.md section 7. Ingest itself runs on the procrastinate
``ingest`` queue (``app.services.l3.ingest.l3_cuts_ingest`` or, when
``settings.cuts_pipeline == "vcut"``, ``app.services.vcut.orchestrate.
vcut_ingest`` -- seam_cut_pipeline.plan.md section 11) -- this router only
enqueues it, so a request never blocks on a real model call.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.config import get_settings
from app.services.l3 import cuts_read as read
from app.services.l3.ingest import defer_ingest
from app.services.l3.projects import find_or_create_project

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectBody(BaseModel):
    file_ids: List[str] = Field(min_length=1)


class EnergyBody(BaseModel):
    energy: float = Field(ge=0.0, le=1.0)


def _owned_project(project_id: str, user_id: str) -> None:
    from app.services.l3.cuts_read import _pg_conn
    with _pg_conn() as conn:
        row = conn.execute(
            "select 1 from projects where id = %s and user_id = %s", (project_id, user_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="project not found")


@router.post("")
def create_project(body: CreateProjectBody, user_id: str = Depends(get_current_user_id)):
    project_id = find_or_create_project(user_id, body.file_ids)
    return {"project_id": project_id}


@router.post("/{project_id}/ingest")
def kick_ingest(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Enqueue (or re-enqueue) the cuts ingest for this project. Returns
    immediately; poll GET /{project_id}/cuts for status.

    seam_cut_pipeline.plan.md section 11: settings.cuts_pipeline selects
    which pipeline runs -- "v3" (default, app.services.l3.ingest) or "vcut"
    (app.services.vcut.orchestrate), so the two can run side by side per
    environment without deleting either."""
    from procrastinate.exceptions import AlreadyEnqueued

    from app.services.fairness import CapacityExceeded

    _owned_project(project_id, user_id)
    try:
        if get_settings().cuts_pipeline == "vcut":
            from app.services.vcut.orchestrate import defer_vcut_ingest
            defer_vcut_ingest(project_id, user_id)
        else:
            defer_ingest(project_id, user_id)
    except AlreadyEnqueued:
        # scale_architecture.plan.md Pillar 5: defer_ingest's queueing_lock
        # already caught this -- a run for this project is already pending.
        # A double-click/retry, not a failure: same response either way.
        pass
    except CapacityExceeded as e:
        # Pillar 6: a real, user-visible limit, not a transient/server
        # failure -- 429 so the client can distinguish "try again shortly"
        # from "something is actually broken" (503, below).
        raise HTTPException(status_code=429, detail=str(e))
    except Exception:
        logger.exception("could not enqueue ingest for project %s", project_id)
        raise HTTPException(status_code=503, detail="could not enqueue ingest")
    return {"project_id": project_id, "status": "queued"}


@router.get("/{project_id}/cuts")
def get_cuts(project_id: str, user_id: str = Depends(get_current_user_id)):
    """All cut_records + project summary + ingest status. Pure DB read --
    zero model calls."""
    result = read.load_cuts(project_id, user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="project not found")
    return result


def _latest_run_id(project_id: str) -> str | None:
    from app.services.l3.cuts_read import _pg_conn
    with _pg_conn() as conn:
        row = conn.execute(
            "select id::text from ingest_runs where project_id = %s order by created_at desc limit 1",
            (project_id,),
        ).fetchone()
    return row["id"] if row else None


def _resolve_energy_preserving(
    project_id: str,
    user_id: str,
    ingest_run_id: str,
    plan,
    seam_cache: Dict[str, dict],
    energy: float,
) -> dict:
    """resolve -> insert the kind='video' cuts at ``energy``. vcut_pass2_
    video_specifics.plan.md section 7.4 retires this function's own former
    band-aid (a hero-containment snapshot/re-map of Pass-2 specifics across
    the delete+reinsert an energy change causes): specifics now live on the
    plan's own MomentFlags and are composed onto whatever cut contains them
    by resolve_cuts itself, at ANY energy -- so a plain resolve+insert (via
    vstore.insert_video_cuts, section 7.3) already yields specifics-bearing
    rows with zero ambiguity, no remap needed. Shared by set_cuts_energy and
    the energy_levels prefetch."""
    from app.services.vcut import store as vstore
    from app.services.vcut.resolve import resolve_cuts

    resolved = resolve_cuts(plan, seam_cache, energy=energy)
    vstore.insert_video_cuts(ingest_run_id, resolved, seam_cache)
    return read.load_cuts(project_id, user_id)


@router.post("/{project_id}/cuts/energy")
def set_cuts_energy(project_id: str, body: EnergyBody, user_id: str = Depends(get_current_user_id)):
    """seam_cut_pipeline.plan.md section 9: re-derive video cut boundaries
    at a new global energy from persisted seam_cache/loose_plan alone --
    NO model call, no R2 download, no motion recompute. vcut runs only:
    the latest run's speech cuts (copied verbatim at ingest time) are
    untouched -- only kind='video' rows are replaced.

    vcut_moment_energy.plan.md section 6.3: deliberately does NOT re-enqueue
    vcut_enrich here (it used to, per vcut_pass2_rich.plan.md section 8) --
    on a live system with a worker, dragging the dial would fire a paid
    Pass-2 Gemini call on every tick, re-deriving scene_specifics for
    geometry that's about to change again on the next drag. Geometry-only;
    enrich stays whatever it was from ingest (option (a), the plan's own
    "simplest, recommended for v1" choice)."""
    _owned_project(project_id, user_id)

    from app.services.vcut import store as vstore
    from app.services.vcut.resolve import MomentPlan

    ingest_run_id = _latest_run_id(project_id)
    if not ingest_run_id:
        raise HTTPException(status_code=404, detail="no ingest run for this project")

    seam_cache, loose_plan_dict = vstore.load_seam_and_plan(ingest_run_id)
    if not seam_cache or not loose_plan_dict:
        raise HTTPException(
            status_code=409,
            detail="this project's latest run has no vcut artifacts to re-derive from",
        )

    plan = MomentPlan.from_dict(loose_plan_dict)
    return _resolve_energy_preserving(project_id, user_id, ingest_run_id, plan, seam_cache, body.energy)


# The five discrete dial stops (EnergyBar snaps to Math.round(t*4)/4). Kept as
# an ordered tuple so the endpoint below and the frontend key set stay in sync.
_ENERGY_STOPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _energy_key(energy: float) -> str:
    # "0","0.25","0.5","0.75","1" -- matches the frontend's numeric->string map
    # (a trailing ".0"/".00" would drift from setData(cachedLevels[key])).
    return str(int(energy)) if energy == int(energy) else str(energy)


@router.get("/{project_id}/cuts/energy_levels")
def get_cuts_energy_levels(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Precompute the fully-assembled cut response for ALL five dial stops in
    ONE call so the frontend can swap tightness instantly client-side, with no
    per-drag round-trip. Reuses set_cuts_energy's exact resolve -> build ->
    delete -> insert path per stop (so every level is byte-for-byte shape-
    identical to load_cuts), then re-resolves at energy 0.0 once more so the
    persisted rows match the frontend's default-on-load state. vcut runs only;
    409 when the latest run has no seam_cache/loose_plan to re-derive from."""
    _owned_project(project_id, user_id)

    from app.services.vcut import store as vstore
    from app.services.vcut.resolve import MomentPlan

    ingest_run_id = _latest_run_id(project_id)
    if not ingest_run_id:
        raise HTTPException(status_code=404, detail="no ingest run for this project")

    seam_cache, loose_plan_dict = vstore.load_seam_and_plan(ingest_run_id)
    if not seam_cache or not loose_plan_dict:
        raise HTTPException(
            status_code=409,
            detail="this project's latest run has no vcut artifacts to re-derive from",
        )

    plan = MomentPlan.from_dict(loose_plan_dict)

    def _resolve_at(energy: float) -> dict:
        return _resolve_energy_preserving(project_id, user_id, ingest_run_id, plan, seam_cache, energy)

    levels = {_energy_key(energy): _resolve_at(energy) for energy in _ENERGY_STOPS}

    # Leave the DB at energy 0.0 -- the frontend defaults a vcut dial to 0 on
    # load, so persisted rows should match what's shown before any drag.
    _resolve_at(0.0)

    return {"levels": levels}
