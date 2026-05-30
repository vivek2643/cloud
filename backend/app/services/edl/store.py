"""
EDL persistence: projects + edl_versions.

The EDL is the source of truth for the timeline. Every actor (Claude, the
user, the renderer) reads and writes through this module.

EDL JSON shape (Phase 1, cut-only):
{
  "version": 1,
  "fps": 30,
  "resolution": [1920, 1080],
  "clips": [
    {
      "id":              "<uuid or string>",
      "shot_id":         "<uuid>",
      "source_in_ms":    1200,
      "source_out_ms":   4800,
      "timeline_in_ms":  0,
      "timeline_out_ms": 3600
    }
  ]
}

Future polish phases extend this with optional fields (transitions, captions,
multi-track, per-clip volume/speed/effects). Anything not strictly needed
for hard cuts stays out of v1.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings

logger = logging.getLogger(__name__)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def find_or_create_default_project(
    user_id: str,
    source_file_ids: List[str],
    name: str = "Untitled",
) -> Dict[str, Any]:
    """
    Find a project for this user that has the same source set, or create one.

    "Default" project = the most recently created project for this user
    whose source_file_ids exactly matches the input set (order-independent).

    Phase 2 will replace this with explicit project picker UI; for now this
    keeps the chat flow working without a separate "create project" step.
    """
    sorted_ids = sorted(set(str(fid) for fid in source_file_ids if fid))

    with _pg() as conn:
        # Look for an existing project owned by this user with the same set.
        # We use the array equality on sorted ids; the gin index helps with
        # the `<@`/`@>` containment, then we filter exact-match in Python.
        cur = conn.execute(
            """
            select id, user_id, name, source_file_ids, created_at, updated_at
            from public.projects
            where user_id = %s
              and source_file_ids @> %s::uuid[]
              and source_file_ids <@ %s::uuid[]
            order by updated_at desc
            limit 1
            """,
            (user_id, sorted_ids, sorted_ids),
        )
        row = cur.fetchone()
        if row:
            # Touch updated_at so the most-recently-used project sorts first.
            conn.execute(
                "update public.projects set updated_at = now() where id = %s",
                (row["id"],),
            )
            return _row_to_project(row)

        # Create.
        cur = conn.execute(
            """
            insert into public.projects (user_id, name, source_file_ids)
            values (%s, %s, %s::uuid[])
            returning id, user_id, name, source_file_ids, created_at, updated_at
            """,
            (user_id, name, sorted_ids),
        )
        row = cur.fetchone()
        return _row_to_project(row)


def get_project(project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, user_id, name, source_file_ids, created_at, updated_at
            from public.projects
            where id = %s and user_id = %s
            """,
            (project_id, user_id),
        )
        row = cur.fetchone()
        return _row_to_project(row) if row else None


def list_projects(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, user_id, name, source_file_ids, created_at, updated_at
            from public.projects
            where user_id = %s
            order by updated_at desc
            limit %s
            """,
            (user_id, limit),
        )
        return [_row_to_project(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# EDL versions
# ---------------------------------------------------------------------------

def get_latest_edl_version(project_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, project_id, parent_id, edl_json, author_kind, commit_msg, created_at
            from public.edl_versions
            where project_id = %s
            order by created_at desc
            limit 1
            """,
            (project_id,),
        )
        row = cur.fetchone()
        return _row_to_edl_version(row) if row else None


def get_edl_version(version_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, project_id, parent_id, edl_json, author_kind, commit_msg, created_at
            from public.edl_versions
            where id = %s
            """,
            (version_id,),
        )
        row = cur.fetchone()
        return _row_to_edl_version(row) if row else None


def list_edl_versions(project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select id, project_id, parent_id, edl_json, author_kind, commit_msg, created_at
            from public.edl_versions
            where project_id = %s
            order by created_at desc
            limit %s
            """,
            (project_id, limit),
        )
        return [_row_to_edl_version(r) for r in cur.fetchall()]


def write_edl_version(
    project_id: str,
    edl_json: Dict[str, Any],
    author_kind: str,
    parent_id: Optional[str] = None,
    commit_msg: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a new immutable EDL version. Returns the inserted row."""
    if author_kind not in ("user", "claude", "system"):
        raise ValueError(f"Invalid author_kind: {author_kind!r}")
    validate_edl(edl_json)

    with _pg() as conn:
        cur = conn.execute(
            """
            insert into public.edl_versions
                (project_id, parent_id, edl_json, author_kind, commit_msg)
            values (%s, %s, %s::jsonb, %s, %s)
            returning id, project_id, parent_id, edl_json, author_kind, commit_msg, created_at
            """,
            (project_id, parent_id, json.dumps(edl_json), author_kind, commit_msg),
        )
        row = cur.fetchone()
        # Touch the project so updated_at reflects activity.
        conn.execute(
            "update public.projects set updated_at = now() where id = %s",
            (project_id,),
        )
        return _row_to_edl_version(row)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_edl(edl: Dict[str, Any]) -> None:
    """
    Lightweight schema check. Raises ValueError on malformed EDL.

    We intentionally only enforce the cut-only invariants here. Future
    polish phases extend with optional fields whose absence is fine.
    """
    if not isinstance(edl, dict):
        raise ValueError("EDL must be a JSON object")
    if edl.get("version") != 1:
        raise ValueError(f"Unsupported EDL version: {edl.get('version')!r}")
    fps = edl.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"EDL.fps must be a positive number, got {fps!r}")
    res = edl.get("resolution")
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(x, int) and x > 0 for x in res)):
        raise ValueError(f"EDL.resolution must be [w, h] of positive ints, got {res!r}")
    clips = edl.get("clips")
    if not isinstance(clips, list):
        raise ValueError("EDL.clips must be a list")
    seen_ids = set()
    for i, c in enumerate(clips):
        if not isinstance(c, dict):
            raise ValueError(f"clips[{i}] must be an object")
        for k in ("id", "shot_id", "source_in_ms", "source_out_ms",
                  "timeline_in_ms", "timeline_out_ms"):
            if k not in c:
                raise ValueError(f"clips[{i}] missing field {k!r}")
        if not isinstance(c["source_in_ms"], int) or not isinstance(c["source_out_ms"], int):
            raise ValueError(f"clips[{i}] source_in/out_ms must be ints")
        if c["source_out_ms"] - c["source_in_ms"] < 1:
            raise ValueError(f"clips[{i}] has zero or negative duration")
        if c["id"] in seen_ids:
            raise ValueError(f"duplicate clip id {c['id']!r} at index {i}")
        seen_ids.add(c["id"])


# ---------------------------------------------------------------------------
# Helpers to translate between Claude's TimelineClip output and EDL JSON
# ---------------------------------------------------------------------------

def empty_edl(fps: int = 30, resolution: tuple[int, int] = (1920, 1080)) -> Dict[str, Any]:
    return {
        "version": 1,
        "fps": fps,
        "resolution": [resolution[0], resolution[1]],
        "clips": [],
    }


def build_edl_from_user_clips(
    clips: List[Dict[str, Any]],
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
) -> Dict[str, Any]:
    """
    Build a validated EDL from a minimal, user-authored clip list.

    Each input clip needs at least: shot_id, source_in_ms, source_out_ms.
    `id` is preserved if present (so the UI can keep stable React keys across
    a save) or freshly generated otherwise. timeline_in/out are ALWAYS
    recomputed here from clip durations -- cut-only timelines are a simple
    sequential concatenation, so the server owns timeline positions and the
    client can't desync them.
    """
    out_clips: List[Dict[str, Any]] = []
    cursor_ms = 0
    for i, c in enumerate(clips):
        try:
            in_ms = int(c["source_in_ms"])
            out_ms = int(c["source_out_ms"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"clips[{i}] missing/invalid source_in_ms/source_out_ms")
        if out_ms - in_ms < 1:
            raise ValueError(f"clips[{i}] has zero or negative duration")
        shot_id = c.get("shot_id")
        if not shot_id:
            raise ValueError(f"clips[{i}] missing shot_id")
        dur = out_ms - in_ms
        out_clips.append({
            "id": str(c.get("id") or uuid.uuid4()),
            "shot_id": str(shot_id),
            "source_in_ms": in_ms,
            "source_out_ms": out_ms,
            "timeline_in_ms": cursor_ms,
            "timeline_out_ms": cursor_ms + dur,
        })
        cursor_ms += dur
    edl = {
        "version": 1,
        "fps": fps,
        "resolution": [resolution[0], resolution[1]],
        "clips": out_clips,
    }
    validate_edl(edl)
    return edl


def edl_from_timeline_clips(
    timeline_clips: List[Any],
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
) -> Dict[str, Any]:
    """
    Convert a list of TimelineClip objects (the existing Phase-0 representation)
    into a Phase-1 EDL JSON dict.

    The timeline clips already carry shot_id (when produced by claude_editor's
    smart path); we use it as the editorial pointer. Clip ids are freshly
    generated UUIDs so they can be referenced by later edits.
    """
    clips: List[Dict[str, Any]] = []
    cursor_ms = 0
    for tc in timeline_clips:
        dur = max(0, tc.source_out_ms - tc.source_in_ms)
        clip = {
            "id": str(uuid.uuid4()),
            "shot_id": getattr(tc, "shot_id", None) or "",
            "source_in_ms": int(tc.source_in_ms),
            "source_out_ms": int(tc.source_out_ms),
            "timeline_in_ms": cursor_ms,
            "timeline_out_ms": cursor_ms + dur,
        }
        clips.append(clip)
        cursor_ms += dur
    return {
        "version": 1,
        "fps": fps,
        "resolution": [resolution[0], resolution[1]],
        "clips": clips,
    }


# ---------------------------------------------------------------------------
# Row converters (psycopg returns dict_row, so basically a passthrough plus
# normalization of uuids/datetimes to JSON-friendly strings).
# ---------------------------------------------------------------------------

def _row_to_project(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "name": row["name"],
        "source_file_ids": [str(x) for x in (row.get("source_file_ids") or [])],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


def _row_to_edl_version(row: Dict[str, Any]) -> Dict[str, Any]:
    edl_json = row["edl_json"]
    if isinstance(edl_json, str):
        # In rare cases jsonb comes back as text (older psycopg); normalize.
        edl_json = json.loads(edl_json)
    return {
        "id": str(row["id"]),
        "project_id": str(row["project_id"]),
        "parent_id": str(row["parent_id"]) if row.get("parent_id") else None,
        "edl_json": edl_json,
        "author_kind": row["author_kind"],
        "commit_msg": row.get("commit_msg"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }
