"""Procrastinate task that renders one edit-document version to MP4.

Runs on its own `render` queue so the (concurrency=1) L3 agent loop on the `l3`
queue isn't blocked behind a long ffmpeg pass. Start a worker with
WORKER_QUEUES including `render`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import psycopg

from app.config import get_settings
from app.services.jobs import app
from app.services.l3 import layers
from app.services.render import compositor, store

logger = logging.getLogger(__name__)


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _file_ids_in(resolved: Dict[str, Any]) -> List[str]:
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


def resolve_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """The resolved layer set for a document. Prefer the snapshot the agent
    persisted; otherwise recompute deterministically from spine + operations."""
    res = document.get("resolved")
    if isinstance(res, dict) and (res.get("video_layers") or res.get("audio_layers")):
        return res
    fids = list({s["file_id"] for s in (document.get("timeline") or [])})
    return layers.resolve(document, _durations(fids)).to_dict()


def _file_lookup(file_ids: List[str]) -> Dict[str, compositor.FileEntry]:
    if not file_ids:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            "select id::text, r2_key, r2_proxy_key, file_type from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {
        r[0]: compositor.FileEntry(
            file_id=r[0], r2_key=r[1], r2_proxy_key=r[2], has_video=(r[3] == "video")
        )
        for r in rows
    }


def _load_document_version(thread_id: str, version: int) -> Optional[Dict[str, Any]]:
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
        document = _load_document_version(row["thread_id"], row["document_version"])
        if document is None:
            raise RuntimeError(f"document v{row['document_version']} not found")
        resolved = resolve_document(document)
        file_ids = _file_ids_in(resolved)
        lookup = _file_lookup(file_ids)
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
