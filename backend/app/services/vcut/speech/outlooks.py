"""
speech_cuts_pipeline.plan.md section 7 -- Stage 2: audio-sync grouping +
hero audio + collapse to take instances. Reads sync_groups/sync_group_members
directly (a small, non-fiddly SELECT + one-line offset formula) rather than
importing app.services.l3.sync.store/audio_route -- section 18's own lean
("read tables directly, reimplement the small offset mapping; only import
if it gets fiddly"), consistent with vcut's established isolation pattern
elsewhere (spans.py mirrors transcripts' shape the same way).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SyncMember:
    file_id: str
    offset_ms: int
    role: str                    # "video_angle" | "audio"
    confidence: Optional[float]


@dataclass
class SyncGroup:
    group_id: str
    authoritative_file_id: str
    members: Dict[str, SyncMember] = field(default_factory=dict)  # file_id -> SyncMember


def _pg_conn():
    from app.services import db
    return db.connection_dict_row()


def load_sync_groups(file_ids: List[str]) -> Dict[str, SyncGroup]:
    """{group_id: SyncGroup} covering any of ``file_ids`` -- mirrors
    l3.sync.store.sync_groups_for_files' own SELECT shape (not imported,
    see module docstring)."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select sg.id::text as group_id, sg.authoritative_audio_file_id::text as auth_id,
                   sgm.file_id::text as file_id, sgm.offset_ms, sgm.role, sgm.confidence
              from sync_group_members sgm
              join sync_groups sg on sg.id = sgm.group_id
             where sgm.group_id in (
                 select group_id from sync_group_members where file_id = any(%s::uuid[])
             )
            """,
            (file_ids,),
        ).fetchall()
    groups: Dict[str, SyncGroup] = {}
    for r in rows:
        if not r["auth_id"]:
            continue  # a group with no authoritative audio chosen yet contributes nothing
        g = groups.setdefault(
            r["group_id"], SyncGroup(group_id=r["group_id"], authoritative_file_id=r["auth_id"]))
        g.members[r["file_id"]] = SyncMember(
            file_id=r["file_id"], offset_ms=int(r["offset_ms"] or 0),
            role=r["role"], confidence=r["confidence"])
    # A group whose authoritative file isn't even among its own members is
    # malformed data -- drop it rather than route audio to a file with no
    # known offset.
    return {gid: g for gid, g in groups.items() if g.authoritative_file_id in g.members}


def angle_ms(auth_ms: int, auth_offset_ms: int, angle_offset_ms: int) -> int:
    """audio_route.py's own formula: group_ms = angle_ms + angle_offset_ms
    = auth_ms + auth_offset_ms -> angle_ms = auth_ms + auth_offset_ms -
    angle_offset_ms."""
    return auth_ms + auth_offset_ms - angle_offset_ms


def group_for_file(groups: Dict[str, SyncGroup], file_id: str) -> Optional[SyncGroup]:
    for g in groups.values():
        if file_id in g.members:
            return g
    return None


def collapse_to_take_instances(file_ids: List[str], groups: Dict[str, SyncGroup]) -> List[str]:
    """One take-instance file_id per performance (section 7): a sync group
    contributes only its authoritative (hero) file; a file with no sync
    group is its own take instance. Order-preserving over ``file_ids`` for
    the ungrouped files, then any authoritative files not already present."""
    grouped_ids = {fid for g in groups.values() for fid in g.members}
    instances = [fid for fid in file_ids if fid not in grouped_ids]
    seen = set(instances)
    for g in groups.values():
        if g.authoritative_file_id not in seen:
            instances.append(g.authoritative_file_id)
            seen.add(g.authoritative_file_id)
    return instances


def video_angle_files(group: SyncGroup) -> List[str]:
    """Every member with a picture (role='video_angle') -- these are the
    files an outlook fan-out (store.py section 12) emits a cut for. A
    pure-audio member (role='audio', e.g. a boom mic with no camera)
    contributes routing only, never its own picture cut."""
    return [fid for fid, m in group.members.items() if m.role == "video_angle"]
