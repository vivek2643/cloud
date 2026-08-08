"""
reframe_vcut_geometry.plan.md -- deterministic crop solving. Pass 2 emits
only the subject ANCHOR (subject_box); this module solves the actual
crop_16x9/9x16/1x1 rectangles. "The model never picks a crop rectangle by
hand" -- same locked framing philosophy `l3/grade/steer.py` documents for
its own (unrelated) timeline-focus mechanism. Pure geometry, no I/O, no
model call.

Note on rotation_deg: vcut always calls this with rotation_deg=0.0 in
practice (see store.py) -- the proxy vcut reads is already upright
(`l1.pipeline._encode_proxy` auto-applies rotation during transcode, and
nothing downstream persists the raw file's original rotation metadata), so
there is never a remaining correction to make. The parameter and its w/h
swap stay here so this function is still correct for a hypothetical caller
that does have a genuine rotation value.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

Box = Tuple[float, float, float, float]  # normalized x, y, w, h

_ASPECTS: Dict[str, float] = {"crop_16x9": 16.0 / 9.0, "crop_9x16": 9.0 / 16.0, "crop_1x1": 1.0}


def _center_px(box: Optional[Box], w: int, h: int) -> Tuple[float, float]:
    """The subject's center in pixel coords -- the frame's own center when
    there's no box (the documented centered-crop fallback)."""
    if box is None:
        return w / 2.0, h / 2.0
    x, y, bw, bh = box
    return (x + bw / 2.0) * w, (y + bh / 2.0) * h


def _largest_rect_of_aspect(w: int, h: int, aspect: float) -> Tuple[float, float]:
    """The largest axis-aligned (rect_w, rect_h), in pixels, of ``aspect``
    (w/h) that fits inside a w x h frame."""
    src_aspect = w / h
    if aspect >= src_aspect:
        rect_w = float(w)
        rect_h = rect_w / aspect
    else:
        rect_h = float(h)
        rect_w = rect_h * aspect
    return rect_w, rect_h


def _position_rect(
    rect_w: float, rect_h: float, cx: float, cy: float, w: int, h: int,
) -> Tuple[float, float]:
    """Top-left (x, y) in pixels, centered on (cx, cy), clamped so the
    whole rect stays inside [0, w] x [0, h]."""
    x = max(0.0, min(cx - rect_w / 2.0, w - rect_w))
    y = max(0.0, min(cy - rect_h / 2.0, h - rect_h))
    return x, y


def solve_crops(
    subject_box: Optional[Box], src_w: int, src_h: int, rotation_deg: float = 0.0,
) -> Dict[str, Optional[Box]]:
    """{crop_16x9, crop_9x16, crop_1x1}: for each target aspect, the
    largest axis-aligned rect of that aspect fitting inside the (uprighted)
    source, positioned to keep ``subject_box``'s center in view, clamped to
    the source edges. ``subject_box=None`` -> a centered crop of each
    aspect (still an improvement over an empty ``framing``: centered +
    uprighted). Works in the uprighted frame: a +/-90 ``rotation_deg``
    swaps ``src_w``/``src_h`` before solving. Degenerate/zero source dims
    return all-None crops rather than raising."""
    w, h = int(src_w), int(src_h)
    if abs(rotation_deg) % 360 in (90, 270):
        w, h = h, w
    if w <= 0 or h <= 0:
        return {key: None for key in _ASPECTS}

    cx, cy = _center_px(subject_box, w, h)
    out: Dict[str, Optional[Box]] = {}
    for key, aspect in _ASPECTS.items():
        rect_w, rect_h = _largest_rect_of_aspect(w, h, aspect)
        x, y = _position_rect(rect_w, rect_h, cx, cy, w, h)
        out[key] = (round(x / w, 4), round(y / h, 4), round(rect_w / w, 4), round(rect_h / h, 4))
    return out
