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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.auth import get_current_user_id
from app.services.l3.grade.cache import ensure_cube_file
from app.services.l3.grade.cdl import Grade, grade_hash
from app.services.processing import _download_from_r2

logger = logging.getLogger(__name__)
router = APIRouter(tags=["grade"])

CUBE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "edso_grade_cubes")
DEFAULT_LUT_SIZE = 33


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
    _user_id: str = Depends(get_current_user_id),
) -> Response:
    try:
        cdl_dict = json.loads(cdl)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="cdl must be a JSON object")

    grade_obj = Grade.from_dict(cdl_dict)
    h = grade_hash(
        grade_obj, creative_lut_ref=creative_lut_ref, working_space=working_space, lut_size=size,
    )
    descriptor = {
        "cdl": grade_obj.to_dict(),
        "creative_lut_ref": creative_lut_ref,
        "working_space": working_space,
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
