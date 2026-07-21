"""
Color-grade cube endpoint (color_grading.plan.md SS4/SS11): bakes (or reuses
a cached) 3D LUT for a resolved grade and serves it as `.cube` text -- the
SAME bytes `render/compositor.py` bakes for export, so the frontend's WebGL
LUT shader and the ffmpeg render never diverge (both call
`grade.lut_bake.bake_cube_text` -- see that module's docstring).

The frontend NEVER computes a `grade_hash` itself (see
`resolve-timeline.ts::resolveClipGrade`) -- it sends the raw CDL /
creative-lut-ref / working-space values and THIS endpoint is the one place
that hashes + caches, so there is exactly one hash implementation, not two
that could silently drift apart.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.auth import get_current_user_id
from app.services.l3.grade.cache import ensure_cube_file
from app.services.l3.grade.cdl import Grade, grade_hash
from app.services.l3.grade.look_engine import list_engine_looks
from app.services.l3.grade.lut_bake import parse_cube_text
from app.services.l3.grade.presets import list_presets
from app.services.processing import _download_from_r2, _upload_to_r2

logger = logging.getLogger(__name__)
router = APIRouter(tags=["grade"])

CUBE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "edso_grade_cubes")
DEFAULT_LUT_SIZE = 33
MAX_LUT_UPLOAD_BYTES = 8 * 1024 * 1024  # generous headroom over a 65^3 cube (~5MB text)


@router.get("/api/grade/presets")
def get_grade_presets(_user_id: str = Depends(get_current_user_id)) -> list:
    """Look layer gallery listing (SS7/SS12): CDL presets (mode 1) PLUS
    color_response_engine.plan.md's engine looks -- one combined list, each
    entry tagged `mode` ("preset" vs "engine") so the frontend knows which
    id field (`preset_id` vs `look_id`) to set when the user picks one. The
    frontend bakes its own live thumbnail per entry via the cube endpoint;
    this just names them. Additive: existing preset entries are unchanged."""
    return list_presets() + list_engine_looks()


@router.post("/api/grade/lut")
async def upload_grade_lut(
    request: Request, user_id: str = Depends(get_current_user_id)
) -> dict:
    """Look layer mode 3 (SS7.3): accepts raw `.cube` text in the request
    body, validates it actually parses, and stores it in R2. Returns the R2
    key to use as `EditLook.lut_ref` / `creative_lut_ref`. A plain body
    upload (not the multipart flow `upload.py` uses for raw footage) since
    `.cube` files are small text -- no reason for presigned multipart here."""
    body = await request.body()
    if len(body) > MAX_LUT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="LUT file too large")
    text = body.decode("utf-8", errors="replace")
    try:
        parse_cube_text(text)
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid .cube file")

    key = f"grades/luts/{user_id}/{uuid.uuid4().hex}.cube"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False, mode="w") as f:
            tmp_path = f.name
            f.write(text)
        _upload_to_r2(tmp_path, key, "text/plain")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return {"lut_ref": key}


def _fetch_creative_lut(ref: str) -> Optional[str]:
    """Resolve a creative-LUT reference (an R2 key) to `.cube` text -- for the
    Look layer's LUT-upload mode (SS7.3); a no-op today since nothing
    produces a `creative_lut_ref` yet."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as f:
            tmp_path = f.name
        _download_from_r2(ref, tmp_path)
        with open(tmp_path, "r") as f:
            return f.read()
    except Exception:
        logger.exception("Failed to fetch creative LUT %s", ref)
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@router.get("/api/grade/cube")
def get_grade_cube(
    cdl: str = Query(..., description="JSON-encoded {slope,offset,power,sat}"),
    working_space: str = Query("rec709"),
    creative_lut_ref: Optional[str] = Query(None),
    size: int = Query(DEFAULT_LUT_SIZE, ge=2, le=65),
    tone_contrast: float = Query(0.0, description="color_tone_contrast.plan.md filmic S-curve strength"),
    look_engine: Optional[str] = Query(
        None, description="color_response_engine.plan.md: JSON-encoded LookSpec dict",
    ),
    _user_id: str = Depends(get_current_user_id),
) -> Response:
    try:
        cdl_dict = json.loads(cdl)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="cdl must be a JSON object")
    look_engine_dict = None
    if look_engine:
        try:
            look_engine_dict = json.loads(look_engine)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=400, detail="look_engine must be a JSON object")

    grade_obj = Grade.from_dict(cdl_dict)
    h = grade_hash(
        grade_obj, creative_lut_ref=creative_lut_ref, working_space=working_space, lut_size=size,
        tone_contrast=tone_contrast, look_engine=look_engine_dict,
    )
    descriptor = {
        "cdl": grade_obj.to_dict(),
        "creative_lut_ref": creative_lut_ref,
        "working_space": working_space,
        "tone_contrast": tone_contrast,
        "look_engine": look_engine_dict,
        "grade_hash": h,
    }
    path = ensure_cube_file(
        descriptor, CUBE_CACHE_DIR, lut_size=size, fetch_creative_lut=_fetch_creative_lut,
    )
    if not path:
        raise HTTPException(status_code=500, detail="Failed to bake grade cube")

    with open(path, "r") as f:
        cube_text = f.read()
    return Response(
        content=cube_text,
        media_type="text/plain",
        headers={
            # Content-addressed by grade_hash -- identical params always bake
            # identical bytes, so this is safe to cache forever.
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": h,
        },
    )
