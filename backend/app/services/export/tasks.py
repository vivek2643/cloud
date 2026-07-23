"""Procrastinate task that builds one export (export_options.plan.md Phase 5).

Resolves the edit document ONCE, then dispatches to whichever deliverable
was requested: 'mp4' delegates to the existing render stack at the chosen
quality preset, 'srt' formats the sidecar subtitles, 'rough_cut' bundles the
self-relinking FCPXML+SRT ZIP. Runs on its own `export` queue -- see
run_workers.sh Phase 0 wiring.
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Dict, List

import psycopg

from app.config import get_settings
from app.services.export import bundle, srt, store
from app.services.jobs import app
from app.services.processing import _upload_to_r2
from app.services.render import compositor
from app.services.render.tasks import (
    file_ids_in,
    file_lookup as render_file_lookup,
    load_document_version,
    resolve_document,
)

logger = logging.getLogger(__name__)


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _bundle_file_lookup(file_ids: List[str]) -> Dict[str, bundle.BundleFileEntry]:
    if not file_ids:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            "select id::text, filename, r2_key, file_size, duration_seconds, width, height "
            "from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {
        r[0]: bundle.BundleFileEntry(
            file_id=r[0], filename=r[1], r2_key=r[2],
            file_size_bytes=int(r[3] or 0), duration_ms=int(float(r[4] or 0) * 1000),
            width=r[5], height=r[6],
        )
        for r in rows
    }


def _project_name_for_thread(thread_id: str) -> str:
    from app.services.l3 import store as l3_store
    thread = l3_store.get_thread(thread_id)
    title = (thread or {}).get("title")
    return title if title else "Untitled"


def _upload_text(text: str, ext: str, content_type: str) -> str:
    with tempfile.TemporaryDirectory(prefix="edso_export_text_") as tmp:
        local_path = os.path.join(tmp, f"out.{ext}")
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        out_key = f"{bundle.EXPORT_PREFIX}/{uuid.uuid4().hex}.{ext}"
        _upload_to_r2(local_path, out_key, content_type)
    return out_key


@app.task(name="build_export", queue="export", retry=False)
def build_export(export_id: str) -> None:
    row = store.get_export(export_id)
    if row is None:
        logger.warning("export %s gone; skipping", export_id)
        return
    store.update_status(export_id, status="running")
    try:
        thread_id = row["thread_id"]
        document = load_document_version(thread_id, row["document_version"])
        if document is None:
            raise RuntimeError(f"document v{row['document_version']} not found")
        resolved = resolve_document(document, thread_id=thread_id)
        file_ids = file_ids_in(resolved)
        kind = row["kind"]

        if kind == "mp4":
            lookup = render_file_lookup(file_ids)
            missing = [f for f in file_ids if f not in lookup]
            if missing:
                raise RuntimeError(f"source file(s) missing for export: {missing}")
            out_key, _duration_ms = compositor.render_resolved(resolved, lookup, preset=row["quality"])
        elif kind == "srt":
            text = srt.build_srt(resolved.get("captions") or [])
            out_key = _upload_text(text, "srt", "text/plain; charset=utf-8")
        elif kind == "rough_cut":
            b_lookup = _bundle_file_lookup(file_ids)
            missing = [f for f in file_ids if f not in b_lookup]
            if missing:
                raise RuntimeError(f"source file(s) missing for export: {missing}")
            out_key = bundle.build_rough_cut_bundle(
                resolved, b_lookup, project_name=_project_name_for_thread(thread_id),
                include_media=bool(row["include_media"]),
                resolved_captions=resolved.get("captions") or [],
            )
        else:
            raise RuntimeError(f"unknown export kind {kind!r}")

        store.update_status(export_id, status="done", output_r2_key=out_key)
        logger.info("export %s done: %s", export_id, out_key)
    except Exception as e:  # noqa: BLE001
        logger.exception("export %s failed", export_id)
        store.update_status(export_id, status="failed", error=str(e)[:1000])
