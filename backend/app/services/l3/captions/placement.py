"""
Caption placement engine (caption_style_mvp.plan.md #2): resolves ONE
normalized `[x, y, w, h]` box for the WHOLE video from the style's fixed
`position` -- lower_third / center / top / bottom_dynamic -- clamped into the
aspect's safe margins. No dynamic/speaker/per-cut `caption_zones` routing for
these styles (the MVP is a global-position choice, not per-cut placement);
`cut_rows` is still accepted so a fixed position can nudge off a subject box
it happens to land on, same "never on subject_box" guarantee as before, but
never changes WHICH of the 4 positions is used.

Pure function, no I/O.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]  # x, y, w, h, normalized 0..1

# Platform safe-area gutters: extra inset subtracted from the full-bleed
# frame before any position rect is placed. Portrait needs the most room
# (bottom UI rail + right-side action buttons); landscape/square barely
# need any.
SAFE_MARGIN = {
    "portrait": {"top": 0.08, "bottom": 0.16, "left": 0.04, "right": 0.10},
    "square": {"top": 0.05, "bottom": 0.08, "left": 0.04, "right": 0.04},
    "landscape": {"top": 0.04, "bottom": 0.06, "left": 0.06, "right": 0.06},
}

_BAND_H = 0.22  # generous enough for 2 lines at a typical caption size

# caption_style_mvp.plan.md #2: approximate vertical CENTER per position, as
# a fraction of the full (pre-safe-margin) frame height. The box is centered
# on this point, then clamped into the aspect's safe rect.
POSITION_VERTICAL_CENTER = {
    "lower_third": 0.81,      # 80-82%
    "center": 0.50,           # 50%
    "top": 0.135,             # 12-15%
    "bottom_dynamic": 0.715,  # 70-73%
}
POSITIONS = tuple(POSITION_VERTICAL_CENTER.keys())
DEFAULT_POSITION = "lower_third"


def _safe_rect(aspect: str) -> Box:
    m = SAFE_MARGIN.get(aspect, SAFE_MARGIN["landscape"])
    return (m["left"], m["top"], 1.0 - m["left"] - m["right"], 1.0 - m["top"] - m["bottom"])


def _position_rect(position: str, aspect: str) -> Box:
    sx, sy, sw, sh = _safe_rect(aspect)
    center_y = POSITION_VERTICAL_CENTER.get(position, POSITION_VERTICAL_CENTER[DEFAULT_POSITION])
    y = center_y - _BAND_H / 2.0
    return (sx, y, sw, _BAND_H)


def _area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _overlaps(a: Box, b: Box, *, threshold: float = 0.15) -> bool:
    """True when `a` and `b` overlap by more than `threshold` of `a`'s own
    area -- a light touch, not exact-pixel collision (subject_box is itself
    an LLM-estimated box, not a hard mask)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a_area = _area(a)
    return a_area > 0 and (inter / a_area) > threshold


def _subject_boxes(cut_rows: Sequence[Dict[str, Any]]) -> list:
    out = []
    for r in cut_rows:
        sb = (r.get("framing") or {}).get("subject_box")
        if sb and len(sb) == 4:
            out.append(tuple(float(v) for v in sb))
    return out


def _clamp_into_safe(box: Box, aspect: str) -> Box:
    sx, sy, sw, sh = _safe_rect(aspect)
    x, y, w, h = box
    w, h = min(w, sw), min(h, sh)
    x = max(sx, min(x, sx + sw - w))
    y = max(sy, min(y, sy + sh - h))
    return (x, y, w, h)


def resolve_placement(
    cut_rows: Sequence[Dict[str, Any]],
    *,
    position: Optional[str] = None,
    aspect: str = "landscape",
    safe_area: bool = True,
    anchor: Optional[str] = None,
) -> Dict[str, Any]:
    """ONE fixed box for the whole video (or weld-run, if a caller still
    chunks by run -- the box is identical for every run since it doesn't
    depend on `cut_rows` content, only on `position`/`aspect`).

    `position` falls back to Lower Third when unset or unrecognized (the
    plan's "fall back to Lower Third when analysis is unavailable"; here
    that reduces to "when no position was resolved at all", since this
    function no longer does its own analysis). `anchor` is accepted as a
    backward-compatible alias for `position` (the pre-MVP call-site name).

    Returns `{box: [x,y,w,h], source: "fixed", shot_size: None}` --
    `shot_size` is kept in the return shape for callers that still read it,
    always None now (shot-size-driven placement is gone; suggestion RANKING
    may still inspect shot-size separately, see suggest.py).
    """
    pos = position or anchor or DEFAULT_POSITION
    if pos not in POSITIONS:
        pos = DEFAULT_POSITION
    box = _position_rect(pos, aspect)

    subject_boxes = _subject_boxes(cut_rows) if safe_area else []
    if subject_boxes and any(_overlaps(box, sb) for sb in subject_boxes):
        for alt in POSITIONS:
            if alt == pos:
                continue
            alt_box = _position_rect(alt, aspect)
            if not any(_overlaps(alt_box, sb) for sb in subject_boxes):
                box = alt_box
                break

    if safe_area:
        box = _clamp_into_safe(box, aspect)
    return {"box": list(box), "source": "fixed", "shot_size": None}
