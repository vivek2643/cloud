"""
Pure unit tests for app.services.vcut.reframe.solve_crops.

Run:  .venv/bin/python scripts/test_vcut_reframe.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.reframe import solve_crops  # noqa: E402

_SRC_W, _SRC_H = 1920, 1080


def _center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _close(a, b, tol=1e-3):
    return abs(a - b) < tol


def test_centered_subject_yields_centered_crops():
    box = (0.4, 0.4, 0.2, 0.2)  # center = (0.5, 0.5)
    crops = solve_crops(box, _SRC_W, _SRC_H)
    for key in ("crop_16x9", "crop_9x16", "crop_1x1"):
        cx, cy = _center(crops[key])
        assert _close(cx, 0.5) and _close(cy, 0.5), f"{key} not centered: {crops[key]}"
    print("ok  test_centered_subject_yields_centered_crops")


def test_subject_none_yields_centered_crops():
    crops = solve_crops(None, _SRC_W, _SRC_H)
    for key in ("crop_16x9", "crop_9x16", "crop_1x1"):
        cx, cy = _center(crops[key])
        assert _close(cx, 0.5) and _close(cy, 0.5), f"{key} not centered: {crops[key]}"
    print("ok  test_subject_none_yields_centered_crops")


def test_subject_near_edge_clamps_but_stays_covered():
    # Subject hugs the top-left corner -- the crop should clamp to that
    # corner (can't recenter past the frame edge) while still covering it.
    box = (0.0, 0.0, 0.05, 0.05)
    crops = solve_crops(box, _SRC_W, _SRC_H)
    for key in ("crop_16x9", "crop_9x16", "crop_1x1"):
        x, y, w, h = crops[key]
        assert x == 0.0 and y == 0.0, f"{key} did not clamp to the top-left edge: {crops[key]}"
        assert x <= box[0] + box[2] and y <= box[1] + box[3], f"{key} lost the subject: {crops[key]}"
    print("ok  test_subject_near_edge_clamps_but_stays_covered")


def test_crop_aspects_match_target():
    crops = solve_crops((0.3, 0.3, 0.1, 0.1), _SRC_W, _SRC_H)
    targets = {"crop_16x9": 16.0 / 9.0, "crop_9x16": 9.0 / 16.0, "crop_1x1": 1.0}
    for key, target in targets.items():
        _x, _y, w, h = crops[key]
        px_w, px_h = w * _SRC_W, h * _SRC_H
        assert _close(px_w / px_h, target, tol=1e-2), f"{key} aspect off: {px_w / px_h} != {target}"
    print("ok  test_crop_aspects_match_target")


def test_rotation_swaps_dims():
    # A portrait source rotated 90 degrees to upright should solve
    # identically to the already-upright (swapped) landscape dims.
    portrait = solve_crops(None, _SRC_H, _SRC_W, rotation_deg=90)
    landscape = solve_crops(None, _SRC_W, _SRC_H, rotation_deg=0)
    assert portrait == landscape, f"{portrait} != {landscape}"
    # -90 and 270 behave the same way (only |90|/|270| trigger the swap).
    assert solve_crops(None, _SRC_H, _SRC_W, rotation_deg=-90) == landscape
    assert solve_crops(None, _SRC_H, _SRC_W, rotation_deg=270) == landscape
    # 180 does NOT swap dims.
    upright_180 = solve_crops(None, _SRC_W, _SRC_H, rotation_deg=180)
    assert upright_180 == landscape
    print("ok  test_rotation_swaps_dims")


def test_degenerate_dims_return_all_none():
    crops = solve_crops(None, 0, 0)
    assert all(v is None for v in crops.values())
    crops = solve_crops((0.1, 0.1, 0.1, 0.1), -5, 100)
    assert all(v is None for v in crops.values())
    print("ok  test_degenerate_dims_return_all_none")


def main():
    test_centered_subject_yields_centered_crops()
    test_subject_none_yields_centered_crops()
    test_subject_near_edge_clamps_but_stays_covered()
    test_crop_aspects_match_target()
    test_rotation_swaps_dims()
    test_degenerate_dims_return_all_none()
    print("\nall vcut reframe tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
