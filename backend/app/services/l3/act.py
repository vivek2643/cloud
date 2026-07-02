"""
Act: the brain's deterministic EDIT VERBS -- immutable ``document -> document``
transforms.

These mirror EXACTLY what the manual timeline editor does (edit ``timeline`` +
``operations`` in place, then re-resolve layers) -- one edit model for both human
and brain, no parallel engine. A verb never renders or calls an LLM; it returns
a new document with a stale ``resolved`` dropped, and the loop re-resolves once
via ``observe.resolve_doc``.

Channel model (see the V1/V2/A1/A2 vocabulary): the main line is V1 video + A1
audio (the ``timeline`` spine); a silent video cutaway over the ongoing audio is
a V2 ``place_video`` op; audio beds are A2 ``place_audio`` ops. ``place`` writes
onto V1 or V2; the span-level verbs (``trim``/``set_audio``) act on whichever
clip owns the id.

Every verb is total: an unknown id or illegal arg is a no-op returning the doc
unchanged (the caller diagnoses), never an exception that could crash a turn.
"""
from __future__ import annotations

import copy
import uuid
from typing import List, Optional

from app.services.l3 import layers
from app.services.l3.arrange import Placement, ResolvedCut, _MapIndex


def _clone(document: dict) -> dict:
    """A working copy with fresh timeline/operations lists and no stale resolve.
    Deep-copies the mutable edit surface only; the rest is shared (read-only)."""
    doc = dict(document)
    doc["timeline"] = [dict(s) for s in (document.get("timeline") or [])]
    doc["operations"] = [dict(o) for o in (document.get("operations") or [])]
    if document.get("layout_regions"):
        doc["layout_regions"] = [dict(r) for r in document["layout_regions"]]
    doc.pop("resolved", None)
    return doc


def _new_seg_id() -> str:
    return f"u{uuid.uuid4().hex[:8]}"


def _segments_from_cut(rc: ResolvedCut) -> List[dict]:
    """A resolved cut -> one or more main-line segments (one per keep_span so a
    breath-excised jump-cut survives). The segment shape matches what the timeline
    / render read; adjacent contiguous slices are later merged by
    ``arrange._weld_segments`` in ``observe.resolve_doc``. ``rc.keep_spans`` is the
    canonical ``[(in_ms, out_ms), ...]`` (normalized in ``_MapIndex.resolve``)."""
    spans = rc.keep_spans or [(rc.src_in_ms, rc.src_out_ms)]
    out: List[dict] = []
    for a, b in spans:
        in_ms, out_ms = int(a), int(b)
        if out_ms <= in_ms:
            continue
        out.append({
            "seg_id": _new_seg_id(),
            "file_id": rc.file_id,
            "in_ms": in_ms,
            "out_ms": out_ms,
            "axis": "speech" if rc.channel == "said" else "any",
            "beat_id": None,
            "content": rc.label,
            "rationale": rc.reason or None,
            "priority": 3,
            "cut_in_cost": 0.0,
            "cut_out_cost": 0.0,
            "warnings": [],
            "mute": True if rc.mute else None,
            "ref": rc.ref or None,
            "level": rc.level,
        })
    return out


def _program_end(document: dict) -> int:
    _, total = layers.spine_spans(document.get("timeline") or [])
    return total


def place_span(document: dict, file_id: str, *, in_ms: int, out_ms: int,
               channel: str = "V1", at: Optional[int] = None,
               from_ms: Optional[int] = None, audio: Optional[str] = None,
               axis: str = "any", content: str = "", reason: str = "") -> dict:
    """Place an ARBITRARY source span ``[in_ms, out_ms]`` of ``file_id`` -- the
    continuous-editing verb. Unlike ``place`` (which can only place a pre-baked
    map ref), this addresses the clip as a continuous source, so the brain can
    lift a person's *silent* reaction, a held beat, or any window the awareness
    lanes revealed -- not just a minted cut.

    channel="V1": insert on the main line at index ``at`` (default append).
    channel="V2": lay a video cutaway over the program at ``from_ms``.
    ``axis="speech"`` marks the span as audio-load-bearing (protects it in the
    weld/coverage). Boundaries are expected to be seam-snapped by the caller.
    Empty/invalid span or missing file -> unchanged doc."""
    try:
        a, b = int(in_ms), int(out_ms)
    except (TypeError, ValueError):
        return document
    if not file_id or b <= a:
        return document

    doc = _clone(document)
    if channel.upper() == "V1":
        seg = {
            "seg_id": _new_seg_id(),
            "file_id": file_id,
            "in_ms": a,
            "out_ms": b,
            "axis": "speech" if axis == "speech" else "any",
            "beat_id": None,
            "content": content,
            "rationale": reason or None,
            "priority": 3,
            "cut_in_cost": 0.0,
            "cut_out_cost": 0.0,
            "warnings": [],
            "mute": True if audio == "mute" else None,
            "ref": None,
            "level": "span",
        }
        tl = doc["timeline"]
        idx = len(tl) if at is None else max(0, min(int(at), len(tl)))
        doc["timeline"] = tl[:idx] + [seg] + tl[idx:]
        return doc

    anchor = _program_end(doc) if from_ms is None else max(0, int(from_ms))
    span = b - a
    doc["operations"].append({
        "op_id": f"sp_{uuid.uuid4().hex[:6]}",
        "type": "place_video",
        "source_file_id": file_id,
        "src_in_ms": a,
        "src_out_ms": b,
        "from_ms": anchor,
        "to_ms": anchor + span,
        "layout": layers.DEFAULT_LAYOUT,
        "z": layers.Z_COVERAGE,
        "opacity": 1.0,
        "rationale": reason or None,
        "warnings": [],
        "mute": False if audio == "keep" else True,
    })
    return doc


# --------------------------------------------------------------------------
# Verbs
# --------------------------------------------------------------------------

def place(document: dict, index: _MapIndex, ref: str, *,
          level: str = "balanced", channel: str = "V1",
          at: Optional[int] = None, from_ms: Optional[int] = None,
          audio: Optional[str] = None, reason: str = "") -> dict:
    """Add a cut from a map ``ref``.

    channel="V1": insert on the main line at index ``at`` (default append).
    channel="V2": lay a SILENT (unless audio="keep") video cutaway over the
    program at ``from_ms`` (default the current program end -> effectively a tail
    cutaway; pass from_ms to place it precisely).
    Unknown/illegal ref -> unchanged doc.
    """
    p = Placement(ref=ref, level=level, track=(0 if channel.upper() == "V1" else 1),
                  from_ms=from_ms, reason=reason, audio=audio)
    rc = index.resolve(p)
    if rc is None:
        return document
    doc = _clone(document)

    if channel.upper() == "V1":
        segs = _segments_from_cut(rc)
        if not segs:
            return document
        tl = doc["timeline"]
        idx = len(tl) if at is None else max(0, min(int(at), len(tl)))
        doc["timeline"] = tl[:idx] + segs + tl[idx:]
        return doc

    # V2 (and up): a program-anchored video cutaway operation.
    anchor = _program_end(doc) if from_ms is None else max(0, int(from_ms))
    span = rc.src_out_ms - rc.src_in_ms
    doc["operations"].append({
        "op_id": f"ov_{uuid.uuid4().hex[:6]}",
        "type": "place_video",
        "source_file_id": rc.file_id,
        "src_in_ms": int(rc.src_in_ms),
        "src_out_ms": int(rc.src_in_ms + span),
        "from_ms": anchor,
        "to_ms": anchor + span,
        "layout": layers.DEFAULT_LAYOUT,
        "z": layers.Z_COVERAGE,
        "opacity": 1.0,
        "rationale": reason or None,
        "warnings": [],
        "mute": False if audio == "keep" else True,
    })
    return doc


def remove(document: dict, target_id: str) -> dict:
    """Drop a main-line segment (by seg_id), an operation (by op_id), or a layout
    region (by region_id). Removing an op that a layout region references also
    drops that now-dangling region (so a split/PiP tears down cleanly)."""
    doc = _clone(document)
    regions = doc.get("layout_regions") or []
    before = len(doc["timeline"]) + len(doc["operations"]) + len(regions)
    doc["timeline"] = [s for s in doc["timeline"] if s.get("seg_id") != target_id]
    doc["operations"] = [o for o in doc["operations"] if o.get("op_id") != target_id]
    regions = [
        r for r in regions
        if r.get("region_id") != target_id
        and not any((sel or {}).get("layer") == target_id for sel in (r.get("cells") or {}).values())
    ]
    if regions:
        doc["layout_regions"] = regions
    else:
        doc.pop("layout_regions", None)
    after = len(doc["timeline"]) + len(doc["operations"]) + len(regions)
    if after == before:
        return document  # nothing matched -> unchanged
    return doc


def move(document: dict, seg_id: str, to_index: int) -> dict:
    """Reorder a main-line segment to ``to_index`` (0-based). Op reposition is a
    separate concern (``set_from``); this is the main-line sequence."""
    doc = _clone(document)
    tl = doc["timeline"]
    src = next((i for i, s in enumerate(tl) if s.get("seg_id") == seg_id), None)
    if src is None:
        return document
    seg = tl.pop(src)
    dst = max(0, min(int(to_index), len(tl)))
    tl.insert(dst, seg)
    return doc


def trim(document: dict, target_id: str, *,
         in_ms: Optional[int] = None, out_ms: Optional[int] = None,
         delta_in_ms: Optional[int] = None, delta_out_ms: Optional[int] = None) -> dict:
    """Adjust the SOURCE span of a main-line segment (or a place_video op).

    Absolute (``in_ms``/``out_ms``) or relative (``delta_in_ms``/``delta_out_ms``,
    e.g. delta_in_ms=+200 nudges the in-point 200ms later). The result is clamped
    so out stays > in; a no-op span is rejected (unchanged doc)."""
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        cur_in, cur_out = int(seg["in_ms"]), int(seg["out_ms"])
        new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
        new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
        new_in = max(0, new_in)
        if new_out <= new_in:
            return document
        seg["in_ms"], seg["out_ms"] = new_in, new_out
        return doc
    op = next((o for o in doc["operations"]
               if o.get("op_id") == target_id and o.get("type") == "place_video"), None)
    if op is None:
        return document
    cur_in, cur_out = int(op["src_in_ms"]), int(op["src_out_ms"])
    new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    new_in = max(0, new_in)
    if new_out <= new_in:
        return document
    op["src_in_ms"], op["src_out_ms"] = new_in, new_out
    op["to_ms"] = int(op.get("from_ms", 0)) + (new_out - new_in)
    return doc


def split_edit(document: dict, seam_seg_id: str, *, audio_offset_ms: int) -> dict:
    """J/L cut: decouple the AUDIO boundary from the VIDEO boundary at the seam
    just BEFORE main-line segment ``seam_seg_id``.

    offset > 0 (L-cut): the previous cut's audio lingers over the incoming
    picture. offset < 0 (J-cut): the incoming cut's audio leads under the
    previous picture. offset 0 clears any existing split at that seam.

    One split per seam: re-issuing replaces the previous offset. The first
    segment has no seam before it -> unchanged doc. The offset itself is applied
    at resolve time (``layers._apply_split_edits``), so it survives welds that
    keep the seam and simply no-ops if the seam disappears."""
    try:
        offset = int(audio_offset_ms)
    except (TypeError, ValueError):
        return document
    timeline = document.get("timeline") or []
    idx = next((i for i, s in enumerate(timeline)
                if s.get("seg_id") == seam_seg_id), None)
    if idx is None or idx == 0:
        return document

    doc = _clone(document)
    ops = [o for o in doc["operations"]
           if not (o.get("type") == "split_edit"
                   and o.get("seam_seg_id") == seam_seg_id)]
    if offset != 0:
        ops.append({
            "op_id": f"se_{uuid.uuid4().hex[:6]}",
            "type": "split_edit",
            "seam_seg_id": seam_seg_id,
            "audio_offset_ms": offset,
        })
    doc["operations"] = ops
    return doc


def set_audio(document: dict, target_id: str, *, mute: bool) -> dict:
    """Mute/unmute the SOURCE audio of a main-line segment (A1) or a V2 cutaway,
    keeping its picture. mute=True silences; mute=False plays the source sound."""
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        seg["mute"] = True if mute else None
        return doc
    op = next((o for o in doc["operations"]
               if o.get("op_id") == target_id and o.get("type") == "place_video"), None)
    if op is None:
        return document
    op["mute"] = bool(mute)
    return doc


# Default cell assignment per template: (spine cell, added-ref cell).
_SPLIT_CELLS = {
    "split_h": ("left", "right"),
    "split_v": ("top", "bottom"),
    "pip": ("base", "inset"),
}


def split_screen(document: dict, index: _MapIndex, ref: str, *,
                 template: str = "split_h", from_ms: int, to_ms: int,
                 level: str = "balanced", audio: Optional[str] = None,
                 reason: str = "") -> dict:
    """Show the ongoing main line (V1) AND a second source `ref` side-by-side (or
    PiP) over the program window [from_ms, to_ms].

    Adds a V2 place_video op for `ref` and a LAYOUT REGION that assigns the spine
    to one cell and the op to the other per the template (split_h/split_v/pip).
    The added cell is silent by default (audio="keep" plays its sound). This is a
    user-owned look -- the brain should ``ask_user`` before calling it. Unknown
    template / bad window -> unchanged doc."""
    cells = _SPLIT_CELLS.get(template)
    if cells is None:
        return document
    rc = index.resolve(Placement(ref=ref, level=level))
    if rc is None:
        return document
    try:
        f, t = int(from_ms), int(to_ms)
    except (TypeError, ValueError):
        return document
    if t <= f:
        return document
    span = min(t - f, rc.src_out_ms - rc.src_in_ms)
    if span < 200:
        return document

    doc = _clone(document)
    doc.setdefault("layout_regions", list(document.get("layout_regions") or []))
    doc["layout_regions"] = [dict(r) for r in doc["layout_regions"]]
    spine_cell, ref_cell = cells
    op_id = f"sp_{uuid.uuid4().hex[:6]}"
    doc["operations"].append({
        "op_id": op_id,
        "type": "place_video",
        "source_file_id": rc.file_id,
        "src_in_ms": int(rc.src_in_ms),
        "src_out_ms": int(rc.src_in_ms + span),
        "from_ms": f,
        "to_ms": f + span,
        "layout": layers.DEFAULT_LAYOUT,
        "z": layers.Z_COVERAGE,
        "opacity": 1.0,
        "rationale": reason or None,
        "warnings": [],
        "mute": False if audio == "keep" else True,
    })
    doc["layout_regions"].append({
        "region_id": f"lr_{uuid.uuid4().hex[:6]}",
        "from_ms": f,
        "to_ms": f + span,
        "template": template,
        "cells": {spine_cell: {"layer": "spine"}, ref_cell: {"layer": op_id}},
    })
    return doc


def tighten(document: dict, index: _MapIndex, *,
            seg_id: Optional[str] = None, level: str = "tight") -> dict:
    """Re-take main-line cut(s) at a different energy ``level`` (pacing).

    With ``seg_id`` -> just that cut; without -> every main-line cut that has a
    map ref and a variant at ``level``. A cut whose ref lacks that level is left
    as-is (partial tighten is fine). Re-resolves each cut's span from the map at
    the new level, preserving order + provenance."""
    doc = _clone(document)
    changed = False
    new_tl: List[dict] = []
    for seg in doc["timeline"]:
        ref = seg.get("ref")
        want = seg_id is None or seg.get("seg_id") == seg_id
        if want and ref and index.level_ok(ref, level):
            rc = index.resolve(Placement(ref=ref, level=level))
            if rc is not None:
                segs = _segments_from_cut(rc)
                if segs:
                    # keep the first slice's id so selections/refs stay stable
                    segs[0]["seg_id"] = seg.get("seg_id") or segs[0]["seg_id"]
                    new_tl.extend(segs)
                    changed = True
                    continue
        new_tl.append(seg)
    if not changed:
        return document
    doc["timeline"] = new_tl
    return doc
