"""
L3 framing pass: annotate each segment/operation with a spatial FOCUS point so a
reframe (cover-crop to another delivery aspect, e.g. a 9:16 reel) keeps the
subject in frame instead of cropping the middle blindly.

It reads the L1 subject-motion centroid for the SOURCE clip behind each layer,
averages it over the layer's source range, and writes its CENTER as
`transform.focus = {cx, cy}` (0..1). Per-cut recomposed crops + upright rotation
now come from the Cuts v3 pass-2 image judgment (stored on `cut_records.framing`)
-- this pass only adds the time-varying motion follow / push-in that a still
crop can't express.

Why a separate author-time pass (not inside `layers.resolve`):
  * resolve must stay pure + DB-free so the client preview can mirror it. The
    motion facts live in the database, so we BAKE the focus into the document
    here (orchestrator, before save). The client preview and the server render
    then both read `transform.focus` off the segment and agree -- preview ==
    render. Clips without motion simply get no focus -> centered framing (the
    Phase-1 behavior), so this is purely additive.

The geometry is deterministic; the only input is the stored motion row, so the
result is stable across runs.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
# Signal access (motion read per file, cached)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg

    from app.config import get_settings

    return psycopg.connect(get_settings().database_url, autocommit=True)


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
    action_points: List[dict],
    src_in: int,
    src_out: int,
) -> Optional[Dict[str, object]]:
    """Resolve the framing focus for one source range from the subject-motion
    centroid. Returns {cx, cy, source, evidence} or None when there is no motion
    to point at (-> centered framing)."""
    if src_out < src_in:
        src_in, src_out = src_out, src_in
    cm = _motion_centroid(action_points, src_in, src_out)
    if cm:
        return {"cx": round(cm[0], 4), "cy": round(cm[1], 4), "source": "motion",
                "evidence": "subject-motion centroid"}
    return None


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
    punch), or None for `static`. push-in works even without motion (center)."""
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
           motion_cache: Dict[str, List[dict]], style: str, feel: str) -> None:
    """Bake focus (+ motion for the chosen style) onto one segment/op's
    transform (in place)."""
    if file_id not in motion_cache:
        motion_cache[file_id] = _load_motion_centroids(file_id)
    points = motion_cache[file_id]
    focus = focus_for_range(points, src_in, src_out)

    motion_patch: Optional[dict] = None
    if style in MOTION_STYLES and style != "static":
        span = max(1, src_out - src_in)
        edge = min(_EDGE_MS, span)
        f_start = focus_for_range(points, src_in, src_in + edge) or focus
        f_end = focus_for_range(points, src_out - edge, src_out) or focus
        motion_patch = _build_motion(style, feel, src_out - src_in, f_start, f_end)

    # Clear any prior auto-baked motion/zoom so re-runs (incl. switching the
    # style back to static) are idempotent and don't leave stale animation.
    prev = layer.get("transform")
    if isinstance(prev, dict):
        prev.pop("motion", None)
        prev.pop("zoom", None)

    if focus is None and not motion_patch:
        return  # nothing to add; leave centered (Phase-1 behavior)
    t = layer.setdefault("transform", {})
    if not isinstance(t, dict):
        t = {}
        layer["transform"] = t
    if focus is not None:
        t["focus"] = {"cx": focus["cx"], "cy": focus["cy"]}
    if motion_patch:
        t.update(motion_patch)


def annotate_document(document: dict) -> dict:
    """Bake an auto framing focus (+ the chosen motion) onto every spine segment
    + video operation.

    Idempotent: recomputes from the stored motion each call. Mutates and returns
    `document`. Safe when motion is absent (no-op per layer)."""
    timeline = document.get("timeline") or []
    operations = document.get("operations") or []
    fmt = document.get("format") or {}
    style = fmt.get("motion_style") if fmt.get("motion_style") in MOTION_STYLES else "static"
    feel = fmt.get("motion_feel") if fmt.get("motion_feel") in MOTION_FEELS else "snappy"

    motion_cache: Dict[str, List[dict]] = {}

    for seg in timeline:
        fid = seg.get("file_id")
        if fid:
            _apply(seg, str(fid), int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0)),
                   motion_cache, style, feel)

    for op in operations:
        if op.get("type") == "place_video" and op.get("source_file_id"):
            _apply(op, str(op["source_file_id"]),
                   int(op.get("src_in_ms", 0)), int(op.get("src_out_ms", 0)),
                   motion_cache, style, feel)

    return document
