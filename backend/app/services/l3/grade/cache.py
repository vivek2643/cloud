"""
Local-disk, content-addressed cache for baked `.cube` files, keyed by
`grade_hash` (color_grading.plan.md SS4/SS11). Both callers -- the render
compositor and the (later) cube-serving preview endpoint -- go through
`ensure_cube_file`, which is a thin wrapper around the pure
`lut_bake.bake_cube_text`; the cache never changes what gets baked, only
whether it gets baked twice.

Known limitation (tracked for color_grading.plan.md build order #12,
"re-grade triggers + caches hardened"): this is a LOCAL disk cache, not
shared across API instances / render workers. Correct today (a render job's
own tmp dir, or a single API process's cache dir) but re-bakes per instance
in a horizontally-scaled deployment -- fine since baking is cheap (a few ms
for CDL-only; still fast composed with one creative LUT), just not free.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from app.services.l3.grade.cdl import Grade
from app.services.l3.grade.lut_bake import bake_cube_text, parse_cube_text


def ensure_cube_file(
    grade: Optional[Dict[str, Any]],
    cache_dir: str,
    *,
    lut_size: int = 33,
    fetch_creative_lut: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[str]:
    """Bake (or reuse a cached) `.cube` file for a resolved grade dict
    (`{cdl, creative_lut_ref, working_space, grade_hash}` -- see
    `grade.resolver.resolve_clip_grade`). Returns the local file path, or
    `None` when there's nothing to bake (no grade / no hash). `fetch_creative_lut`
    resolves a creative-LUT reference to `.cube` text to compose on top of the
    CDL -- unused until the Look layer's LUT-upload mode (SS7.3) needs it."""
    if not grade:
        return None
    h = grade.get("grade_hash")
    if not h:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{h}.cube")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    cdl = Grade.from_dict(grade.get("cdl"))
    creative_grid = None
    look_engine = grade.get("look_engine")
    if look_engine:
        # color_response_engine.plan.md: the engine grid and an uploaded LUT
        # both fill this one slot and are mutually exclusive (the resolver
        # already nulls creative_lut_ref when an engine look is active) --
        # prefer the engine grid so a stale lut_ref on the same look dict
        # can never silently win.
        from app.services.l3.grade.look_engine import LookSpec, build_look_grid
        creative_grid = build_look_grid(LookSpec.from_dict(look_engine), size=lut_size)
    else:
        lut_ref = grade.get("creative_lut_ref")
        if lut_ref and fetch_creative_lut is not None:
            text = fetch_creative_lut(lut_ref)
            if text:
                creative_grid = parse_cube_text(text)

    working_space = str(grade.get("working_space") or "rec709")
    tone_contrast = float(grade.get("tone_contrast") or 0.0)
    cube_text = bake_cube_text(cdl, size=lut_size, creative_lut_grid=creative_grid, title=h,
                               working_space=working_space, tone_contrast=tone_contrast)
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(cube_text)
    os.replace(tmp_path, path)
    return path
