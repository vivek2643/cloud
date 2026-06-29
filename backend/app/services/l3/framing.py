"""
L3 framing pass: annotate each segment/operation with a spatial FOCUS point so a
reframe (cover-crop to another delivery aspect, e.g. a 9:16 reel) keeps the
subject in frame instead of cropping the middle blindly.

It reads the L2 perception regions for the SOURCE clip behind each layer
(`speaking` spans -> events -> a person's `frame_region`) plus the L1 motion
centroid, picks the region that dominates the layer's source range, and writes
its CENTER as `transform.focus = {cx, cy}` (0..1). It also maps the clip's
`frame_orientation` to an orthogonal `transform.rotate` so sideways footage is
turned upright.

Why a separate author-time pass (not inside `layers.resolve`):
  * resolve must stay pure + DB-free so the client preview can mirror it. The
    spatial facts live in the database, so we BAKE the focus into the document
    here (orchestrator, before save). The client preview and the server render
    then both read `transform.focus` off the segment and agree -- preview ==
    render. Clips without v2 perception simply get no focus -> centered framing
    (the Phase-1 behavior), so this is purely additive.

The geometry is deterministic; the only inputs are the stored perception/motion
rows, so the result is stable across runs.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# frame_orientation enum -> clockwise degrees to APPLY to make the clip upright.
_ORIENT_TO_ROTATE = {
    "upright": 0,
    "rotate_cw90": 90,
    "rotate_ccw90": 270,
    "rotate_180": 180,
}

# --- Phase 3 motion (the one user-chosen knob: format.motion_style / _feel) ---
MOTION_STYLES = ("static", "punch_in", "push_in", "follow")
MOTION_FEELS = ("snappy", "glide")
_PUNCH_ZOOM = 1.12      # static, held tighter
_PUSH_FROM, _PUSH_TO = 1.0, 1.18   # animated push-in
_FOLLOW_ZOOM = 1.15    # zoomed enough to leave room to pan to the subject
_FOLLOW_DWELL = 0.06   # ignore focus moves smaller than this (hysteresis)
_EDGE_MS = 1000        # window at each end used to sample start/end focus


def _feel_ease(feel: Optional[str]) -> str:
    return "smooth" if feel == "glide" else "linear"


# --------------------------------------------------------------------------
# Signal access (one batched perception read; motion read per file, cached)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg

    from app.config import get_settings

    return psycopg.connect(get_settings().database_url, autocommit=True)


def _load_perceptions(file_ids: List[str]) -> Dict[str, dict]:
    """file_id -> parsed L2 perception dict (skips missing/unparseable)."""
    if not file_ids:
        return {}
    import json

    out: Dict[str, dict] = {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, perception from clip_perception where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    for fid, perception in rows:
        doc = perception if isinstance(perception, dict) else (
            json.loads(perception) if perception else None
        )
        if doc and not doc.get("_parse_error"):
            out[fid] = doc
    return out


def _load_motion_centroids(file_id: str) -> List[dict]:
    """Action points (with centroids) for a clip, or [] if none/unavailable."""
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select action_points from motion_dynamics where file_id = %s",
                (file_id,),
            ).fetchone()
        if row and row[0] and isinstance(row[0], list):
            return row[0]
    except Exception:
        logger.debug("framing: motion load failed for %s", file_id, exc_info=True)
    return []


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _region_center(region: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not isinstance(region, dict):
        return None
    try:
        cx = float(region["x"]) + float(region["w"]) / 2.0
        cy = float(region["y"]) + float(region["h"]) / 2.0
    except (KeyError, TypeError, ValueError):
        return None
    return _clamp01(cx), _clamp01(cy)


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _dominant_region_center(
    spans: List[dict], src_in: int, src_out: int
) -> Optional[Tuple[Tuple[float, float], dict]]:
    """Center of the region whose span overlaps [src_in,src_out] the most.
    Returns ((cx,cy), winning_span) or None."""
    best: Optional[Tuple[int, Tuple[float, float], dict]] = None
    for s in spans:
        c = _region_center(s.get("region"))
        if c is None:
            continue
        ov = _overlap_ms(int(s.get("start_ms", 0)), int(s.get("end_ms", 0)), src_in, src_out)
        if ov <= 0:
            continue
        if best is None or ov > best[0]:
            best = (ov, c, s)
    if best is None:
        return None
    return best[1], best[2]


def _main_person_center(persons: List[dict]) -> Optional[Tuple[float, float]]:
    """Center of the main subject's frame_region (role hints 'main' first)."""
    ranked = sorted(
        persons,
        key=lambda p: 0 if "main" in str(p.get("role") or "").lower() else 1,
    )
    for p in ranked:
        c = _region_center(p.get("frame_region"))
        if c is not None:
            return c
    return None


def _motion_centroid(action_points: List[dict], src_in: int, src_out: int) -> Optional[Tuple[float, float]]:
    """Average subject-motion centroid over the source range, if any impacts."""
    pts = [
        p["centroid"] for p in action_points
        if isinstance(p.get("centroid"), (list, tuple)) and len(p["centroid"]) == 2
        and src_in <= int(p.get("ts_ms", -1)) <= src_out
    ]
    if not pts:
        return None
    cx = sum(float(p[0]) for p in pts) / len(pts)
    cy = sum(float(p[1]) for p in pts) / len(pts)
    return _clamp01(cx), _clamp01(cy)


# --------------------------------------------------------------------------
# Per-range focus resolution
# --------------------------------------------------------------------------

def focus_for_range(
    perception: Optional[dict],
    action_points: List[dict],
    src_in: int,
    src_out: int,
) -> Optional[Dict[str, object]]:
    """Resolve the framing focus for one source range.

    Priority: who is SPEAKING on camera -> where the ACTION beat is (region) ->
    the subject-motion centroid -> the main person's resting frame_region. Each
    is a coarse, editorial signal; the first one that exists wins. Returns
    {cx, cy, source, evidence} or None when there is nothing to point at.
    """
    if src_out < src_in:
        src_in, src_out = src_out, src_in
    perc = perception or {}

    hit = _dominant_region_center(perc.get("speaking") or [], src_in, src_out)
    if hit:
        (cx, cy), span = hit
        return {"cx": round(cx, 4), "cy": round(cy, 4), "source": "speaking",
                "evidence": f"speaker {span.get('subject', '?')}"}

    hit = _dominant_region_center(perc.get("atoms") or [], src_in, src_out)
    if hit:
        (cx, cy), at = hit
        return {"cx": round(cx, 4), "cy": round(cy, 4), "source": "atom",
                "evidence": (str(at.get("label") or "action"))[:60]}

    cm = _motion_centroid(action_points, src_in, src_out)
    if cm:
        return {"cx": round(cm[0], 4), "cy": round(cm[1], 4), "source": "motion",
                "evidence": "subject-motion centroid"}

    mp = _main_person_center(perc.get("persons") or [])
    if mp:
        return {"cx": round(mp[0], 4), "cy": round(mp[1], 4), "source": "person",
                "evidence": "main subject frame_region"}

    return None


def orientation_rotate(perception: Optional[dict]) -> int:
    """Clockwise degrees to apply to make the clip upright (0 if upright/unknown)."""
    o = (perception or {}).get("frame_orientation")
    return _ORIENT_TO_ROTATE.get(str(o), 0)


# --------------------------------------------------------------------------
# Document annotation (author-time; persisted)
# --------------------------------------------------------------------------

def _xy(focus: Optional[Dict[str, object]]) -> Dict[str, float]:
    if focus is None:
        return {"cx": 0.5, "cy": 0.5}
    return {"cx": float(focus["cx"]), "cy": float(focus["cy"])}


def _build_motion(
    style: str, feel: str, dur_ms: int,
    focus_start: Optional[Dict[str, object]], focus_end: Optional[Dict[str, object]],
) -> Optional[dict]:
    """The motion (or static zoom marker) for a layer under the chosen style.
    Returns either {"motion": {...}} (animated) or {"zoom": float} (static
    punch), or None for `static`. push-in works even without perception (center)."""
    ease = _feel_ease(feel)
    dur_ms = max(1, int(dur_ms))
    if style == "punch_in":
        return {"zoom": _PUNCH_ZOOM}
    if style == "push_in":
        a = _xy(focus_start)
        return {"motion": {
            "from": {"scale": _PUSH_FROM, "cx": a["cx"], "cy": a["cy"]},
            "to": {"scale": _PUSH_TO, "cx": a["cx"], "cy": a["cy"]},
            "ease": ease, "dur_ms": dur_ms}}
    if style == "follow":
        a, b = _xy(focus_start), _xy(focus_end)
        # Hysteresis/dwell: hold still unless the subject moved meaningfully.
        if abs(a["cx"] - b["cx"]) < _FOLLOW_DWELL and abs(a["cy"] - b["cy"]) < _FOLLOW_DWELL:
            b = a
        return {"motion": {
            "from": {"scale": _FOLLOW_ZOOM, "cx": a["cx"], "cy": a["cy"]},
            "to": {"scale": _FOLLOW_ZOOM, "cx": b["cx"], "cy": b["cy"]},
            "ease": ease, "dur_ms": dur_ms}}
    return None


def _apply(layer: dict, file_id: str, src_in: int, src_out: int,
           perceptions: Dict[str, dict], motion_cache: Dict[str, List[dict]],
           style: str, feel: str) -> None:
    """Bake focus + orientation (+ motion for the chosen style) onto one
    segment/op's transform (in place)."""
    perc = perceptions.get(file_id)
    if file_id not in motion_cache:
        motion_cache[file_id] = _load_motion_centroids(file_id)
    points = motion_cache[file_id]
    rotate = orientation_rotate(perc)
    focus = focus_for_range(perc, points, src_in, src_out)

    motion_patch: Optional[dict] = None
    if style in MOTION_STYLES and style != "static":
        span = max(1, src_out - src_in)
        edge = min(_EDGE_MS, span)
        f_start = focus_for_range(perc, points, src_in, src_in + edge) or focus
        f_end = focus_for_range(perc, points, src_out - edge, src_out) or focus
        motion_patch = _build_motion(style, feel, src_out - src_in, f_start, f_end)

    # Clear any prior auto-baked motion/zoom so re-runs (incl. switching the
    # style back to static) are idempotent and don't leave stale animation.
    prev = layer.get("transform")
    if isinstance(prev, dict):
        prev.pop("motion", None)
        prev.pop("zoom", None)

    if rotate == 0 and focus is None and not motion_patch:
        return  # nothing to add; leave centered (Phase-1 behavior)
    t = layer.setdefault("transform", {})
    if not isinstance(t, dict):
        t = {}
        layer["transform"] = t
    if rotate:
        t["rotate"] = rotate
    if focus is not None:
        t["focus"] = {"cx": focus["cx"], "cy": focus["cy"]}
    if motion_patch:
        t.update(motion_patch)


def annotate_document(document: dict) -> dict:
    """Bake an auto framing focus (+ the chosen motion) onto every spine segment
    + video operation.

    Idempotent: recomputes from the stored perception/motion each call. Mutates
    and returns `document`. Safe when perception is absent (no-op per layer)."""
    timeline = document.get("timeline") or []
    operations = document.get("operations") or []
    fmt = document.get("format") or {}
    style = fmt.get("motion_style") if fmt.get("motion_style") in MOTION_STYLES else "static"
    feel = fmt.get("motion_feel") if fmt.get("motion_feel") in MOTION_FEELS else "snappy"

    file_ids = {str(seg.get("file_id")) for seg in timeline if seg.get("file_id")}
    for op in operations:
        if op.get("type") in ("place_video", "pick_angle") and op.get("source_file_id"):
            file_ids.add(str(op["source_file_id"]))
    file_ids.discard("None")
    if not file_ids:
        return document

    try:
        perceptions = _load_perceptions(sorted(file_ids))
    except Exception:
        logger.debug("framing: perception load failed", exc_info=True)
        perceptions = {}
    motion_cache: Dict[str, List[dict]] = {}

    for seg in timeline:
        fid = seg.get("file_id")
        if fid:
            _apply(seg, str(fid), int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0)),
                   perceptions, motion_cache, style, feel)

    for op in operations:
        if op.get("type") in ("place_video", "pick_angle") and op.get("source_file_id"):
            _apply(op, str(op["source_file_id"]),
                   int(op.get("src_in_ms", 0)), int(op.get("src_out_ms", 0)),
                   perceptions, motion_cache, style, feel)

    return document
