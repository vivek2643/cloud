#!/usr/bin/env python3
"""Offline verification of the frontend look-gallery THUMBNAILS.

Reproduces exactly what `frontend/src/components/preview/look-thumbnail.ts`
renders: for each engine look, bake its thumbnail cube through the REAL
`bake_cube_text` path (identity CDL + the look's look_engine grid, working
space rec709_v1, tone_contrast 0 -- the same synthetic grade the renderer
builds), then trilinearly sample the bundled reference still through it. Tiles
the results into one labeled contact sheet so the thumbnail CONTENT can be
eyeballed without a logged-in browser session.

Note: halation/grain are spatial (not in the cube), so -- exactly like the
real thumbnails -- film looks show their COLOR only here (the gallery adds a
"grain" badge for that). No DB / no auth needed.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import cv2  # noqa: E402

from app.services.l3.grade.cdl import Grade  # noqa: E402
from app.services.l3.grade.look_engine import LOOKS, build_look_grid  # noqa: E402
from app.services.l3.grade.lut_bake import (  # noqa: E402
    _sample_lut_trilinear,
    bake_cube_text,
    parse_cube_text,
)

REF = os.path.join(BACKEND, "..", "frontend", "public", "look-thumb-ref.jpg")
OUT = os.path.join(BACKEND, "logs", "look_thumbs_sheet.png")

TILE_W, TILE_H = 300, 169  # 16:9-ish
PAD = 8
LABEL_H = 24


def thumb_cube_output(spec, ref_rgb01: np.ndarray) -> np.ndarray:
    """The EXACT thumbnail cube path: bake identity-CDL + look grid, then
    trilinearly sample the reference still through it (what the WebGL renderer
    does with the cube fetched from /api/grade/cube)."""
    grid = build_look_grid(spec)  # (grid, size) creative_lut_grid tuple
    cube_text = bake_cube_text(Grade(), working_space="rec709_v1",
                               creative_lut_grid=grid, tone_contrast=0.0)
    lut_grid, _size = parse_cube_text(cube_text)
    return np.clip(_sample_lut_trilinear(lut_grid, ref_rgb01), 0.0, 1.0)


def main() -> None:
    bgr = cv2.imread(REF, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read reference still: {REF}")
    bgr = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    ref_rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Gallery order: skip engine_identity (hidden in the picker); group family.
    looks = [lk for lk in LOOKS if lk.look_id != "engine_identity"]
    order = {"creator": 0, "film": 1, "ad": 2}
    looks.sort(key=lambda lk: (order.get(lk.family, 9), lk.label))

    cols = 4
    rows = (len(looks) + cols - 1) // cols
    sheet_w = cols * TILE_W + (cols + 1) * PAD
    sheet_h = rows * (TILE_H + LABEL_H) + (rows + 1) * PAD
    sheet = np.full((sheet_h, sheet_w, 3), 24, dtype=np.uint8)  # near-black bg

    for i, lk in enumerate(looks):
        out01 = thumb_cube_output(lk.spec, ref_rgb01)
        tile_bgr = cv2.cvtColor((out01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        r, c = divmod(i, cols)
        x = PAD + c * (TILE_W + PAD)
        y = PAD + r * (TILE_H + LABEL_H + PAD)
        sheet[y:y + TILE_H, x:x + TILE_W] = tile_bgr
        label = f"[{lk.family[:3]}] {lk.label}"
        cv2.putText(sheet, label, (x + 2, y + TILE_H + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, sheet)
    print(f"ok  wrote {OUT}  ({len(looks)} looks, {rows}x{cols})")


if __name__ == "__main__":
    main()
