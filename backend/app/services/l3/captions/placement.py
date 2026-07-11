"""
Caption placement engine (captions.plan.md SS9): resolves ONE normalized
`[x, y, w, h]` box per weld-run of contiguous spine segments, from that
run's `cut_records` (`caption_zones`, `framing.subject_box`/`shot_size`) plus
the style's placement preference (a fixed anchor, or "dynamic" -- trust the
per-cut safe zones pass2b already computed).

Stability (SS9 "compute one placement per run of welded cuts; only move when
the safe zone genuinely changes"): the caller (resolver.py) already groups
spine layers into weld-runs via `cut_records.continuity` before calling this
-- one call here = one placement for the WHOLE run, which is what makes
"only recompute per run, not per cut" true by construction; there's no
separate hysteresis timer to get right.

Pure function, no I/O.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]  # x, y, w, h, normalized 0..1

# Platform safe-area gutters (SS9 "auto-avoid TikTok/Reels UI gutters"):
# extra inset subtracted from the full-bleed frame before any anchor rect is
# placed. Portrait needs the most room (bottom UI rail + right-side action
# buttons); landscape/square barely need any.
SAFE_MARGIN = {
    "portrait": {"top": 0.08, "bottom": 0.16, "left": 0.04, "right": 0.10},
    "square": {"top": 0.05, "bottom": 0.08, "left": 0.04, "right": 0.04},
    "landscape": {"top": 0.04, "bottom": 0.06, "left": 0.06, "right": 0.06},
}

_TIGHT_SHOTS = {"extreme_close_up", "close_up", "medium_close_up"}

# Fixed-anchor archetypes as a fraction of the SAFE (post-margin) area, before
# per-run subject-avoidance nudging. Height is generous enough for 2 lines at
# a typical caption font size; the resolver's own box is a placement region,
# not a tight text bound.
_ANCHOR_BAND_H = 0.22


def _safe_rect(aspect: str) -> Box:
    m = SAFE_MARGIN.get(aspect, SAFE_MARGIN["landscape"])
    return (m["left"], m["top"], 1.0 - m["left"] - m["right"], 1.0 - m["top"] - m["bottom"])


def _anchor_rect(anchor: str, aspect: str) -> Box:
    sx, sy, sw, sh = _safe_rect(aspect)
    if anchor == "top":
        return (sx, sy, sw, _ANCHOR_BAND_H)
    if anchor == "center":
        return (sx, sy + (sh - _ANCHOR_BAND_H) / 2.0, sw, _ANCHOR_BAND_H)
    # "lower_third" (and the "speaker" placeholder -- SS15 phase 2 not wired
    # yet, falls back to lower_third rather than a made-up position) and the
    # no-caption_zones fallback for "dynamic" all land here.
    return (sx, sy + sh - _ANCHOR_BAND_H, sw, _ANCHOR_BAND_H)


def _area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _overlaps(a: Box, b: Box, *, threshold: float = 0.15) -> bool:
    """True when `a` and `b` overlap by more than `threshold` of `a`'s own
    area -- a light touch, not exact-pixel collision (subject_box/caption_zone
    are themselves LLM-estimated boxes, not hard masks)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a_area = _area(a)
    return a_area > 0 and (inter / a_area) > threshold


def _dominant_shot_size(cut_rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    sizes = [((r.get("framing") or {}).get("shot_size")) for r in cut_rows]
    sizes = [s for s in sizes if s]
    if not sizes:
        return None
    # Mode, ties broken by first-seen (stable, no randomness).
    counts: Dict[str, int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    return max(sizes, key=lambda s: (counts[s], -sizes.index(s)))


def _subject_boxes(cut_rows: Sequence[Dict[str, Any]]) -> List[Box]:
    out: List[Box] = []
    for r in cut_rows:
        sb = (r.get("framing") or {}).get("subject_box")
        if sb and len(sb) == 4:
            out.append(tuple(float(v) for v in sb))  # type: ignore[arg-type]
    return out


def _candidate_zones(cut_rows: Sequence[Dict[str, Any]]) -> List[Box]:
    out: List[Box] = []
    for r in cut_rows:
        for z in r.get("caption_zones") or []:
            if len(z) == 4:
                out.append(tuple(float(v) for v in z))  # type: ignore[arg-type]
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
    anchor: str,
    aspect: str,
    safe_area: bool = True,
) -> Dict[str, Any]:
    """One box for a whole weld-run. `cut_rows` = every `cut_records` row
    (SS2 shape, i.e. `cuts_v3_read.rows_for_run`'s raw dicts) whose source
    span overlaps this run, same file, in any order.

    Returns `{box: [x,y,w,h], source: "caption_zone"|"default", shot_size}`.
    """
    subject_boxes = _subject_boxes(cut_rows) if safe_area else []
    shot_size = _dominant_shot_size(cut_rows)

    if anchor == "dynamic":
        zones = _candidate_zones(cut_rows)
        if zones:
            safe = [z for z in zones if not any(_overlaps(z, sb) for sb in subject_boxes)]
            pool = safe or zones  # best-effort: still place SOMETHING, never crash
            chosen = max(pool, key=_area)
            box = _clamp_into_safe(chosen, aspect) if safe_area else chosen
            return {"box": list(box), "source": "caption_zone", "shot_size": shot_size}
        # No perceived zones at all for this run (cut_records missing/not yet
        # ingested) -- fail open to a sensible default instead of blocking.
        default_anchor = "top" if shot_size in _TIGHT_SHOTS else "lower_third"
        box = _anchor_rect(default_anchor, aspect)
        return {"box": list(box), "source": "default", "shot_size": shot_size}

    # Fixed anchor: an editorial choice, but still nudged off the subject if
    # it happens to land there (SS9 "never on subject_box" is absolute, not
    # just for the dynamic path).
    box = _anchor_rect(anchor, aspect)
    if subject_boxes and any(_overlaps(box, sb) for sb in subject_boxes):
        for alt in ("top", "lower_third", "center"):
            if alt == anchor:
                continue
            alt_box = _anchor_rect(alt, aspect)
            if not any(_overlaps(alt_box, sb) for sb in subject_boxes):
                box = alt_box
                break
    if safe_area:
        box = _clamp_into_safe(box, aspect)
    return {"box": list(box), "source": "fixed", "shot_size": shot_size}
