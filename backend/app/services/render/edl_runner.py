"""
EDL render runner: glue between an EDL row and the cuts_renderer.

Responsibilities:
  1. Hydrate an EDL (load shot_id -> file_id mapping, file_lookup table).
  2. Stamp file_id onto each clip in a working copy of the EDL.
  3. Drive the cuts_renderer with a progress callback that updates the
     `renders` row.
  4. On success: record output_r2_key + duration_ms + status='done'.
  5. On failure: record error + status='failed'.

This module is intentionally separate from the procrastinate task entry
point so it can be exercised synchronously from a test or REPL.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services.edl import store as edl_store
from app.services.edl import renders_store
from app.services.render import cuts_renderer

logger = logging.getLogger(__name__)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _iter_edl_clips(edl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All clip dicts across v1 (clips) and v2 (video_track + audio_track)."""
    if edl.get("version") == 2:
        return list(edl.get("video_track") or []) + list(edl.get("audio_track") or [])
    return list(edl.get("clips") or [])


def _resolve_file_lookup_for_edl(edl: Dict[str, Any]) -> tuple[Dict[str, cuts_renderer.FileEntry], Dict[str, str]]:
    """
    Given an EDL (v1 or v2), return (file_lookup_by_file_id, shot_to_file).

    v2 clips usually carry file_id directly; v1 clips carry shot_id. We resolve
    shot_id -> file_id via the shots table and fetch r2 keys for every file_id
    referenced (directly or via a shot).
    """
    clips = _iter_edl_clips(edl)
    shot_ids = list({str(c["shot_id"]) for c in clips if c.get("shot_id")})
    direct_file_ids = list({str(c["file_id"]) for c in clips if c.get("file_id")})
    if not shot_ids and not direct_file_ids:
        return {}, {}

    shot_to_file: Dict[str, str] = {}
    file_lookup: Dict[str, cuts_renderer.FileEntry] = {}

    with _pg() as conn:
        if shot_ids:
            rows = conn.execute(
                """
                select s.id as shot_id, f.id as file_id, f.r2_key, f.r2_proxy_key
                from shots s
                join files f on f.id = s.file_id
                where s.id = any(%s::uuid[])
                """,
                (shot_ids,),
            ).fetchall()
            for r in rows:
                sid = str(r["shot_id"])
                fid = str(r["file_id"])
                shot_to_file[sid] = fid
                file_lookup.setdefault(fid, cuts_renderer.FileEntry(
                    file_id=fid, r2_key=r["r2_key"], r2_proxy_key=r.get("r2_proxy_key"),
                ))

        missing_files = [fid for fid in direct_file_ids if fid not in file_lookup]
        if missing_files:
            rows = conn.execute(
                "select id as file_id, r2_key, r2_proxy_key from files where id = any(%s::uuid[])",
                (missing_files,),
            ).fetchall()
            for r in rows:
                fid = str(r["file_id"])
                file_lookup.setdefault(fid, cuts_renderer.FileEntry(
                    file_id=fid, r2_key=r["r2_key"], r2_proxy_key=r.get("r2_proxy_key"),
                ))

    missing = [sid for sid in shot_ids if sid not in shot_to_file]
    if missing:
        raise RuntimeError(
            f"EDL references shot_ids that are not in the catalog: {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )
    return file_lookup, shot_to_file


def _stamp_file_ids(edl: Dict[str, Any], shot_to_file: Dict[str, str]) -> Dict[str, Any]:
    """Return a shallow copy of the EDL with file_id stamped on each clip
    (from shot_to_file when only a shot_id is present). Handles v1 + v2."""
    def stamp(c: Dict[str, Any]) -> Dict[str, Any]:
        if c.get("file_id"):
            return dict(c)
        return {**c, "file_id": shot_to_file.get(c.get("shot_id"))}

    if edl.get("version") == 2:
        return {
            **edl,
            "video_track": [stamp(c) for c in (edl.get("video_track") or [])],
            "audio_track": [stamp(c) for c in (edl.get("audio_track") or [])],
        }
    return {**edl, "clips": [stamp(c) for c in (edl.get("clips") or [])]}


def run_render(render_id: str) -> None:
    """
    Synchronous render. Used by the procrastinate task and by tests.
    """
    render_row = renders_store.get_render(render_id)
    if not render_row:
        raise RuntimeError(f"renders.{render_id} not found")
    if render_row["status"] not in ("queued", "running"):
        # Idempotent: if someone already finished it, don't redo.
        logger.info("render %s already in terminal state %s; skipping.",
                    render_id, render_row["status"])
        return

    version = edl_store.get_edl_version(render_row["edl_version_id"])
    if not version:
        renders_store.update_status(
            render_id, status="failed",
            error=f"edl_version {render_row['edl_version_id']} not found",
        )
        return

    edl = version["edl_json"]
    if not (edl and isinstance(edl, dict) and _iter_edl_clips(edl)):
        renders_store.update_status(
            render_id, status="failed",
            error="EDL has no clips to render",
        )
        return

    renders_store.update_status(render_id, status="running", progress_pct=1)

    try:
        file_lookup, shot_to_file = _resolve_file_lookup_for_edl(edl)
        rendered_edl = _stamp_file_ids(edl, shot_to_file)

        def progress(pct: int, _label: str) -> None:
            try:
                renders_store.update_status(render_id, progress_pct=pct)
            except Exception:
                logger.exception("progress update failed for render %s", render_id)

        out_key, duration_ms = cuts_renderer.render_edl(
            edl=rendered_edl,
            file_lookup=file_lookup,
            preset=render_row["preset"] or "preview",
            progress_cb=progress,
        )
        renders_store.update_status(
            render_id,
            status="done",
            progress_pct=100,
            output_r2_key=out_key,
            duration_ms=duration_ms,
        )
        logger.info("render %s done -> %s (%dms)", render_id, out_key, duration_ms)
    except Exception as e:
        logger.exception("render %s failed", render_id)
        renders_store.update_status(
            render_id,
            status="failed",
            error=f"{type(e).__name__}: {e}",
        )
        raise
