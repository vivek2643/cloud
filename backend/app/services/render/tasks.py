"""Procrastinate task that renders one edit-document version to MP4.

Runs on its own `render` queue so the (concurrency=1) L3 agent loop on the `l3`
queue isn't blocked behind a long ffmpeg pass. Start a worker with
WORKER_QUEUES including `render`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.jobs import app
from app.services.l3 import layers
from app.services.render import compositor, store

logger = logging.getLogger(__name__)


def _pg():
    from app.services import db
    return db.connection()


def file_ids_in(resolved: Dict[str, Any]) -> List[str]:
    ids = set()
    for v in resolved.get("video_layers") or []:
        ids.add(v["source_file_id"])
    for a in resolved.get("audio_layers") or []:
        ids.add(a["source_file_id"])
    return list(ids)


def _durations(file_ids: List[str]) -> Dict[str, int]:
    if not file_ids:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            "select id::text, coalesce(duration_seconds, 0) from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {r[0]: int(float(r[1]) * 1000) for r in rows}


def _resolve_audio_routes(timeline: List[dict]) -> Dict[str, dict]:
    try:
        from app.services.l3.sync.audio_route import resolve_audio_routes
        return resolve_audio_routes(timeline)
    except Exception:
        logger.exception("resolve_document: sync audio-route lookup failed (continuing without re-routing)")
        return {}


def _resolve_captions(document: Dict[str, Any], resolved: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from app.services.l3.captions.resolver import resolve_captions_for_document
        return resolve_captions_for_document(document, resolved, aspect=str(resolved.get("aspect") or "landscape"))
    except Exception:
        logger.exception("resolve_document: captions resolve failed (continuing without captions)")
        return []


def _grade_lookup_for(thread_id: Optional[str], document: Dict[str, Any]) -> Dict[str, dict]:
    """color_grading_upgrade.plan.md Step 1.0: pre-fetch every gradeable
    shot's freshest persisted grade so `layers.resolve` can just read (never
    compute). Empty (and therefore a
    plain identity fallback inside `layers.resolve`) when there's no
    thread_id to key off of, or the lookup itself fails -- never blocks a
    render/resolve on the grade store being reachable."""
    if not thread_id:
        return {}
    try:
        from app.services.l3.grade.job import fetch_latest_grades, ordered_shots
        shot_keys = [s.key for s in ordered_shots(document)]
        return fetch_latest_grades(thread_id, shot_keys)
    except Exception:
        logger.exception("resolve_document: grade lookup failed for thread %s (continuing)", thread_id)
        return {}


def resolve_document(document: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
    """The resolved layer set for a document. Prefer the snapshot the agent
    persisted; otherwise recompute deterministically from spine + operations.
    `thread_id` (Step 1.0) is needed for the recompute path's grade lookup --
    the persisted-snapshot fast path already baked the right grades in at
    save time."""
    res = document.get("resolved")
    # Grades are produced ASYNCHRONOUSLY by run_grade_job and read from
    # resolved_grades at resolve time -- so a persisted snapshot's baked
    # grades go stale the instant a grade job re-runs (e.g. a look change).
    # Never serve the snapshot's grades: recompute from timeline+operations
    # (cheap -- grade resolution is a DB lookup, NO frame measurement, which
    # only happens inside the job) so a completed grade job actually surfaces
    # in preview/render.
    can_recompute = bool(document.get("timeline") or document.get("operations"))
    use_snapshot = isinstance(res, dict) and (res.get("video_layers") or res.get("audio_layers")) and not can_recompute
    if use_snapshot:
        # Snapshots predating the format field carry no aspect; backfill it from
        # the document so the render uses the declared delivery shape.
        res.setdefault("aspect", layers.aspect_of(document))
        # Snapshots predating the captions feature (or a captions-off save)
        # carry no captions key -- backfill so an old render still reflects
        # the CURRENT captions selection (captions.plan.md SS4).
        if "captions" not in res:
            res["captions"] = _resolve_captions(document, res)
        return res
    timeline = document.get("timeline") or []
    audio_routes = _resolve_audio_routes(timeline)
    fids = list({s["file_id"] for s in timeline})
    # A synced group's authoritative source may be a file that's never itself
    # a spine angle (e.g. a dedicated external mic) -- durations must cover
    # it too so `_apply_split_edits`'s clamp has real footage room to work
    # with (audio_sync.plan.md SS8).
    route_fids = {r["source_file_id"] for r in audio_routes.values()}
    grade_lookup = _grade_lookup_for(thread_id, document)
    resolved = layers.resolve(
        document, _durations(list(set(fids) | route_fids)), audio_routes=audio_routes,
        grade_lookup=grade_lookup,
    ).to_dict()
    resolved["captions"] = _resolve_captions(document, resolved)
    return resolved


def file_lookup(file_ids: List[str]) -> Dict[str, compositor.FileEntry]:
    if not file_ids:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            "select id::text, r2_key, r2_proxy_key, file_type, width, height "
            "from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {
        r[0]: compositor.FileEntry(
            file_id=r[0], r2_key=r[1], r2_proxy_key=r[2], has_video=(r[3] == "video"),
            width=r[4], height=r[5],
        )
        for r in rows
    }


def load_document_version(thread_id: str, version: int) -> Optional[Dict[str, Any]]:
    import json
    with _pg() as conn:
        row = conn.execute(
            "select document from edit_documents where thread_id = %s and version = %s",
            (thread_id, version),
        ).fetchone()
    if not row:
        return None
    return row[0] if isinstance(row[0], dict) else json.loads(row[0])


@app.task(name="render_edit", queue="render", retry=False)
def render_edit(render_id: str) -> None:
    row = store.get_render(render_id)
    if row is None:
        logger.warning("render %s gone; skipping", render_id)
        return
    store.update_status(render_id, status="running", progress_pct=1)
    try:
        document = load_document_version(row["thread_id"], row["document_version"])
        if document is None:
            raise RuntimeError(f"document v{row['document_version']} not found")
        resolved = resolve_document(document, thread_id=row["thread_id"])
        file_ids = file_ids_in(resolved)
        lookup = file_lookup(file_ids)
        missing = [f for f in file_ids if f not in lookup]
        if missing:
            raise RuntimeError(f"source file(s) missing for render: {missing}")

        def progress(pct: int, _label: str) -> None:
            store.update_status(render_id, progress_pct=max(1, min(99, pct)))

        out_key, duration_ms = compositor.render_resolved(
            resolved, lookup, preset=row["preset"], progress_cb=progress
        )
        store.update_status(
            render_id, status="done", progress_pct=100,
            output_r2_key=out_key, duration_ms=duration_ms,
        )
        logger.info("render %s done: %s (%dms)", render_id, out_key, duration_ms)
    except Exception as e:  # noqa: BLE001
        logger.exception("render %s failed", render_id)
        store.update_status(render_id, status="failed", error=str(e)[:1000])
