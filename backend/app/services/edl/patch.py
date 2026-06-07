"""
Addressable EDL patch engine.

Controllability pillar: a chat turn should *adjust* the existing cut, not
regenerate it. This module applies a list of addressable operations (keyed by
stable clip id) to an EDL and returns a new, validated EDL plus a structured
diff -- deterministically, server-owning the timeline positions.

Works on BOTH EDL versions:
  v1 (cut-only):  one `clips` list, video+audio coupled per clip.
  v2 (A/V split): a `video_track` is the editable spine; the `audio_track` is
                  rebuilt to match -- a music bed is re-spanned to the new total
                  length, otherwise audio follows the video (coupled).

Operations (each a dict with an "op" key):
  {"op": "trim",   "clip_id", "source_in_ms"?, "source_out_ms"?}
  {"op": "move",   "clip_id", "to_index"}
  {"op": "delete", "clip_id"}
  {"op": "insert", "file_id"?, "shot_id"?, "source_in_ms", "source_out_ms", "at_index"?}
  {"op": "set_gain","clip_id", "gain_db"}      # audio bed / clip gain (v2)
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.services.edl.store import validate_edl

MIN_CLIP_MS = 200


class PatchError(ValueError):
    pass


def _editable_track(edl: Dict[str, Any]) -> List[Dict[str, Any]]:
    if edl.get("version") == 2:
        return list(edl.get("video_track") or [])
    return list(edl.get("clips") or [])


def _find(clips: List[Dict[str, Any]], clip_id: str) -> int:
    for i, c in enumerate(clips):
        if str(c.get("id")) == str(clip_id):
            return i
    raise PatchError(f"clip_id {clip_id!r} not found")


def _apply_one(clips: List[Dict[str, Any]], op: Dict[str, Any]) -> None:
    kind = str(op.get("op") or "").lower()

    if kind == "trim":
        c = clips[_find(clips, op.get("clip_id"))]
        new_in = int(op["source_in_ms"]) if op.get("source_in_ms") is not None else int(c["source_in_ms"])
        new_out = int(op["source_out_ms"]) if op.get("source_out_ms") is not None else int(c["source_out_ms"])
        new_in = max(0, new_in)
        if new_out - new_in < MIN_CLIP_MS:
            raise PatchError(f"trim would make clip < {MIN_CLIP_MS}ms")
        c["source_in_ms"], c["source_out_ms"] = new_in, new_out

    elif kind == "move":
        i = _find(clips, op.get("clip_id"))
        to = max(0, min(int(op.get("to_index", i)), len(clips) - 1))
        c = clips.pop(i)
        clips.insert(to, c)

    elif kind == "delete":
        clips.pop(_find(clips, op.get("clip_id")))

    elif kind == "insert":
        in_ms = max(0, int(op["source_in_ms"]))
        out_ms = int(op["source_out_ms"])
        if out_ms - in_ms < MIN_CLIP_MS:
            raise PatchError(f"insert clip < {MIN_CLIP_MS}ms")
        if not (op.get("file_id") or op.get("shot_id")):
            raise PatchError("insert needs a file_id or shot_id")
        clip = {
            "id": str(op.get("id") or uuid.uuid4()),
            "file_id": str(op["file_id"]) if op.get("file_id") else None,
            "shot_id": str(op["shot_id"]) if op.get("shot_id") else None,
            "source_in_ms": in_ms,
            "source_out_ms": out_ms,
        }
        at = op.get("at_index")
        if at is None:
            clips.append(clip)
        else:
            clips.insert(max(0, min(int(at), len(clips))), clip)

    elif kind == "set_gain":
        c = clips[_find(clips, op.get("clip_id"))]
        c["gain_db"] = float(op["gain_db"])

    else:
        raise PatchError(f"unknown op {kind!r}")


def _seq_timeline(clips: List[Dict[str, Any]]) -> int:
    """Recompute timeline positions as a sequential concat. Returns total ms."""
    cursor = 0
    for c in clips:
        dur = int(c["source_out_ms"]) - int(c["source_in_ms"])
        c["timeline_in_ms"] = cursor
        c["timeline_out_ms"] = cursor + dur
        cursor += dur
    return cursor


def _rebuild_v1(edl: Dict[str, Any], clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    for c in clips:
        if not c.get("shot_id"):
            raise PatchError("v1 clip requires a shot_id")
        out.append({
            "id": str(c.get("id") or uuid.uuid4()),
            "shot_id": str(c["shot_id"]),
            "source_in_ms": int(c["source_in_ms"]),
            "source_out_ms": int(c["source_out_ms"]),
            "timeline_in_ms": 0,
            "timeline_out_ms": 0,
        })
    _seq_timeline(out)
    return {
        "version": 1,
        "fps": edl.get("fps", 30),
        "resolution": list(edl.get("resolution") or [1920, 1080]),
        "clips": out,
    }


def _rebuild_v2(edl: Dict[str, Any], video_clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    video: List[Dict[str, Any]] = []
    for c in video_clips:
        clip = {
            "id": str(c.get("id") or uuid.uuid4()),
            "file_id": str(c["file_id"]) if c.get("file_id") else None,
            "shot_id": str(c["shot_id"]) if c.get("shot_id") else None,
            "source_in_ms": int(c["source_in_ms"]),
            "source_out_ms": int(c["source_out_ms"]),
            "timeline_in_ms": 0,
            "timeline_out_ms": 0,
        }
        for k in ("role_in_edit", "section", "why", "gain_db", "transition"):
            if c.get(k) is not None:
                clip[k] = c[k]
        video.append(clip)
    total = _seq_timeline(video)

    # Rebuild audio to match. A music bed (single span) is re-stretched to the
    # new total; otherwise audio is coupled (mirrors each video clip).
    orig_audio = list(edl.get("audio_track") or [])
    beds = [a for a in orig_audio if a.get("role_in_edit") == "music_bed"]
    audio: List[Dict[str, Any]] = []
    if beds and total >= MIN_CLIP_MS:
        bed = beds[0]
        src_in = int(bed.get("source_in_ms") or 0)
        audio.append({
            "id": str(uuid.uuid4()),
            "file_id": str(bed["file_id"]) if bed.get("file_id") else None,
            "shot_id": None,
            "source_in_ms": src_in,
            "source_out_ms": src_in + total,
            "timeline_in_ms": 0,
            "timeline_out_ms": total,
            "role_in_edit": "music_bed",
            "gain_db": bed.get("gain_db"),
        })
    else:
        for vc in video:
            audio.append({
                "id": str(uuid.uuid4()),
                "file_id": vc.get("file_id"),
                "shot_id": vc.get("shot_id"),
                "source_in_ms": vc["source_in_ms"],
                "source_out_ms": vc["source_out_ms"],
                "timeline_in_ms": vc["timeline_in_ms"],
                "timeline_out_ms": vc["timeline_out_ms"],
            })

    return {
        "version": 2,
        "fps": edl.get("fps", 30),
        "resolution": list(edl.get("resolution") or [1920, 1080]),
        "video_track": video,
        "audio_track": audio,
        "sections": list(edl.get("sections") or []),
    }


def _diff(base: List[Dict[str, Any]], work: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_by_id = {str(c["id"]): c for c in base}
    work_by_id = {str(c["id"]): c for c in work}
    added = [c for c in work if str(c["id"]) not in base_by_id]
    removed = [c for c in base if str(c["id"]) not in work_by_id]
    trimmed = []
    for c in work:
        b = base_by_id.get(str(c["id"]))
        if b and (int(b["source_in_ms"]) != int(c["source_in_ms"])
                  or int(b["source_out_ms"]) != int(c["source_out_ms"])):
            trimmed.append({
                "clip_id": str(c["id"]),
                "from": {"source_in_ms": int(b["source_in_ms"]), "source_out_ms": int(b["source_out_ms"])},
                "to": {"source_in_ms": int(c["source_in_ms"]), "source_out_ms": int(c["source_out_ms"])},
            })
    common_base = [str(c["id"]) for c in base if str(c["id"]) in work_by_id]
    common_work = [str(c["id"]) for c in work if str(c["id"]) in base_by_id]
    moved = []
    if common_base != common_work:
        pos_b = {cid: i for i, cid in enumerate(common_base)}
        pos_w = {cid: i for i, cid in enumerate(common_work)}
        for cid in common_work:
            if pos_b.get(cid) != pos_w.get(cid):
                moved.append({"clip_id": cid, "from_index": pos_b.get(cid), "to_index": pos_w.get(cid)})
    return {
        "added": added, "removed": removed, "trimmed": trimmed, "moved": moved,
        "changed": bool(added or removed or trimmed or moved),
    }


def rebuild_from_clips(
    base_edl: Dict[str, Any],
    work_clips: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rebuild a validated EDL (same version as `base_edl`) from an already-mutated
    minimal clip list, recomputing timeline positions and audio. Returns
    (new_edl, diff vs the base track). Used by the chat agent, which mutates its
    own working copy via tools and then materializes the result here."""
    base = [dict(c) for c in _editable_track(base_edl)]
    if base_edl.get("version") == 2:
        new_edl = _rebuild_v2(base_edl, work_clips)
        new_track = new_edl["video_track"]
    else:
        new_edl = _rebuild_v1(base_edl, work_clips)
        new_track = new_edl["clips"]
    try:
        validate_edl(new_edl)
    except ValueError as e:
        raise PatchError(f"patched EDL failed validation: {e}") from e
    return new_edl, _diff(base, new_track)


def apply_ops(edl: Dict[str, Any], ops: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply addressable ops to `edl`, returning (new_edl, diff). The new EDL is
    validated; raises PatchError on an invalid op or result."""
    base = [dict(c) for c in _editable_track(edl)]
    work = [dict(c) for c in base]
    for op in ops or []:
        _apply_one(work, op)
    if edl.get("version") == 2:
        new_edl = _rebuild_v2(edl, work)
        new_track = new_edl["video_track"]
    else:
        new_edl = _rebuild_v1(edl, work)
        new_track = new_edl["clips"]
    try:
        validate_edl(new_edl)
    except ValueError as e:
        raise PatchError(f"patched EDL failed validation: {e}") from e
    return new_edl, _diff(base, new_track)
