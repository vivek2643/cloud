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


def list_project_summaries(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Projects that have at least one committed version, joined with their latest
    version, for the "Edits" library. Each row carries enough to render a card
    (clip count, duration, author, first shot id for a thumbnail) without
    fetching the full enriched EDL.
    """
    with _pg() as conn:
        cur = conn.execute(
            """
            select p.id, p.name, p.source_file_ids, p.updated_at,
                   v.edl_json, v.author_kind, v.created_at as version_created_at,
                   (select count(*) from public.edl_versions ev
                      where ev.project_id = p.id) as version_count
            from public.projects p
            join lateral (
                select edl_json, author_kind, created_at
                from public.edl_versions
                where project_id = p.id
                order by created_at desc
                limit 1
            ) v on true
            where p.user_id = %s
            order by p.updated_at desc
            limit %s
            """,
            (user_id, limit),
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            edl = r["edl_json"] or {}
            clips = edl.get("clips") or []
            duration_ms = 0
            first_shot_id: Optional[str] = None
            for c in clips:
                duration_ms = max(duration_ms, int(c.get("timeline_out_ms") or 0))
                if first_shot_id is None and c.get("shot_id"):
                    first_shot_id = str(c["shot_id"])
            out.append(
                {
                    "id": str(r["id"]),
                    "name": r["name"] or "Untitled",
                    "source_file_ids": [str(x) for x in (r["source_file_ids"] or [])],
                    "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
                    "clip_count": len(clips),
                    "duration_ms": duration_ms,
                    "author_kind": r["author_kind"],
                    "version_count": int(r["version_count"]),
                    "first_shot_id": first_shot_id,
                }
            )
        return out


def rename_project(project_id: str, user_id: str, name: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            update public.projects
               set name = %s, updated_at = now()
             where id = %s and user_id = %s
            returning id, user_id, name, source_file_ids, created_at, updated_at
            """,
            (name, project_id, user_id),
        )
        row = cur.fetchone()
        return _row_to_project(row) if row else None


def delete_project(project_id: str, user_id: str) -> bool:
    """Delete a project; edl_versions and renders cascade via FK."""
    with _pg() as conn:
        cur = conn.execute(
            "delete from public.projects where id = %s and user_id = %s returning id",
            (project_id, user_id),
        )
        return cur.fetchone() is not None


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

    Supports two versions:
      v1 (cut-only): a single ``clips`` list, video+audio coupled per clip.
      v2 (A/V split): independent ``video_track`` and ``audio_track`` lists so
          we can lay a music bed under b-roll, do J/L cuts, or cut away from a
          speaker while keeping their audio.

    Both are accepted forever -- old edits and the existing chat flow keep
    working. New work emits v2.
    """
    if not isinstance(edl, dict):
        raise ValueError("EDL must be a JSON object")
    version = edl.get("version")
    if version == 1:
        _validate_common(edl)
        clips = edl.get("clips")
        if not isinstance(clips, list):
            raise ValueError("EDL.clips must be a list")
        _validate_clip_list(clips, label="clips", require_shot_id=True)
        return
    if version == 2:
        _validate_common(edl)
        vt = edl.get("video_track")
        at = edl.get("audio_track")
        if not isinstance(vt, list) or not isinstance(at, list):
            raise ValueError("EDL v2 requires list 'video_track' and 'audio_track'")
        if not vt and not at:
            raise ValueError("EDL v2 has no clips on either track")
        _validate_clip_list(vt, label="video_track", require_shot_id=False)
        _validate_clip_list(at, label="audio_track", require_shot_id=False)
        return
    raise ValueError(f"Unsupported EDL version: {version!r}")


def _validate_common(edl: Dict[str, Any]) -> None:
    fps = edl.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"EDL.fps must be a positive number, got {fps!r}")
    res = edl.get("resolution")
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(x, int) and x > 0 for x in res)):
        raise ValueError(f"EDL.resolution must be [w, h] of positive ints, got {res!r}")


def _validate_clip_list(clips: List[Dict[str, Any]], label: str, require_shot_id: bool) -> None:
    seen_ids: set = set()
    for i, c in enumerate(clips):
        if not isinstance(c, dict):
            raise ValueError(f"{label}[{i}] must be an object")
        required = ["id", "source_in_ms", "source_out_ms", "timeline_in_ms", "timeline_out_ms"]
        if require_shot_id:
            required.append("shot_id")
        for k in required:
            if k not in c:
                raise ValueError(f"{label}[{i}] missing field {k!r}")
        # v2 clips must carry a file reference so the renderer can resolve media.
        if not require_shot_id and not (c.get("file_id") or c.get("shot_id")):
            raise ValueError(f"{label}[{i}] needs a file_id or shot_id")
        if not isinstance(c["source_in_ms"], int) or not isinstance(c["source_out_ms"], int):
            raise ValueError(f"{label}[{i}] source_in/out_ms must be ints")
        if c["source_out_ms"] - c["source_in_ms"] < 1:
            raise ValueError(f"{label}[{i}] has zero or negative duration")
        if c["id"] in seen_ids:
            raise ValueError(f"duplicate clip id {c['id']!r} at {label}[{i}]")
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


def empty_av_edl(fps: int = 30, resolution: tuple[int, int] = (1920, 1080)) -> Dict[str, Any]:
    return {
        "version": 2,
        "fps": fps,
        "resolution": [resolution[0], resolution[1]],
        "video_track": [],
        "audio_track": [],
        "sections": [],
    }


def _normalize_track_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce a track's clips into the validated v2 shape (fill id, ints)."""
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(clips):
        in_ms = int(c["source_in_ms"])
        out_ms = int(c["source_out_ms"])
        if out_ms - in_ms < 1:
            raise ValueError(f"track clip[{i}] has zero or negative duration")
        clip: Dict[str, Any] = {
            "id": str(c.get("id") or uuid.uuid4()),
            "file_id": str(c["file_id"]) if c.get("file_id") else None,
            "shot_id": str(c["shot_id"]) if c.get("shot_id") else None,
            "source_in_ms": in_ms,
            "source_out_ms": out_ms,
            "timeline_in_ms": int(c["timeline_in_ms"]),
            "timeline_out_ms": int(c["timeline_out_ms"]),
        }
        # Optional editorial metadata kept for UI/audit (never required).
        for k in ("role_in_edit", "section", "why", "gain_db", "transition"):
            if c.get(k) is not None:
                clip[k] = c[k]
        out.append(clip)
    return out


def build_av_edl(
    video_track: List[Dict[str, Any]],
    audio_track: List[Dict[str, Any]],
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
    sections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a validated v2 (A/V split) EDL from independent video + audio clip
    lists produced by the composer/recipes. Each clip dict needs at least:
    file_id, source_in_ms, source_out_ms, timeline_in_ms, timeline_out_ms.
    ``sections`` is optional metadata describing the per-section styles.
    """
    edl = {
        "version": 2,
        "fps": fps,
        "resolution": [resolution[0], resolution[1]],
        "video_track": _normalize_track_clips(video_track),
        "audio_track": _normalize_track_clips(audio_track),
        "sections": list(sections or []),
    }
    validate_edl(edl)
    return edl


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
