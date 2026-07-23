"""
Authoritative audio bed routing (audio_sync.plan.md SS8): for every spine
segment whose source cut belongs to a synced group, resolve where its
DIALOGUE AUDIO should actually come from -- the group's authoritative
source, offset-mapped -- while its picture stays whatever angle the edit
shows. `layers.resolve()` applies the result; this module does the (impure)
DB lookups (`cut_records.sync_group_id` -> `sync_groups`/`sync_group_members`),
mirroring `grade.measure.fetch_color_stats`'s "fetch once, resolve pure"
split.

A segment with no synced cut (the overwhelming common case -- no sync groups
declared at all, or this particular span isn't one) is simply absent from
the returned dict; `layers.resolve` then falls back to today's coupled
audio, byte-identical to before this feature existed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.l3 import cuts_read
from app.services.l3.sync import store as sync_store


def _covering_cut(rows: List[Dict[str, Any]], in_ms: int, out_ms: int) -> Optional[Dict[str, Any]]:
    """The cut_record row with the largest overlap against a spine segment's
    (possibly trimmed) `[in_ms, out_ms)` span. A segment normally sits fully
    inside the cut that produced it, but trimming can shrink it -- overlap,
    not exact-match, is the robust test."""
    best: Optional[Dict[str, Any]] = None
    best_overlap = 0
    for r in rows:
        lo, hi = max(in_ms, int(r["src_in_ms"])), min(out_ms, int(r["src_out_ms"]))
        overlap = hi - lo
        if overlap > best_overlap:
            best_overlap, best = overlap, r
    return best


def resolve_audio_routes(timeline: List[dict]) -> Dict[str, Dict[str, Any]]:
    """`{seg_id: {"source_file_id", "src_in_ms", "src_out_ms"}}` for every
    spine segment whose cut carries a `sync_group_id` -- the re-routed
    authoritative audio span for that segment's exact (possibly trimmed)
    program window."""
    file_ids = list({str(s["file_id"]) for s in timeline if s.get("file_id")})
    if not file_ids:
        return {}
    run_id = cuts_read.latest_run_for_files(file_ids)
    if run_id is None:
        return {}
    rows_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for row in cuts_read.rows_for_run(run_id, file_ids):
        rows_by_file.setdefault(row["file_id"], []).append(row)

    routes: Dict[str, Dict[str, Any]] = {}
    group_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for seg in timeline:
        fid, in_ms, out_ms = str(seg.get("file_id") or ""), int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
        cut = _covering_cut(rows_by_file.get(fid, []), in_ms, out_ms)
        group_id = cut.get("sync_group_id") if cut else None
        if not group_id:
            continue
        if group_id not in group_cache:
            group_cache[group_id] = sync_store.get_sync_group(group_id)
        group = group_cache[group_id]
        if not group or not group.get("authoritative_audio_file_id"):
            continue
        auth_fid = group["authoritative_audio_file_id"]
        members = {m["file_id"]: m for m in group["members"]}
        if fid not in members or auth_fid not in members:
            continue
        # group_ms = angle_ms + angle_offset = auth_ms + auth_offset
        # => auth_ms = angle_ms + angle_offset - auth_offset (see sync/detect.py).
        delta_ms = int(members[fid]["offset_ms"]) - int(members[auth_fid]["offset_ms"])
        routes[seg["seg_id"]] = {
            "source_file_id": auth_fid,
            "src_in_ms": in_ms + delta_ms,
            "src_out_ms": out_ms + delta_ms,
        }
    return routes
