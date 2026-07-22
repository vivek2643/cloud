#!/usr/bin/env python3
"""color_qa_harness.plan.md A.1: corpus enumeration -- one gradeable thread
per project, written to backend/scripts/_out/qa/corpus.json. Read-only
(copies `_grade_all_projects.py`'s project/thread enumeration verbatim:
`projects.name` is "Untitled" for every real row, so the readable label
comes from the project's folder names, same as that script). `--regrade`
additionally runs `run_grade_job` per thread first, so grades are current
for the deployed `INPUT_HASH_SCHEMA_VERSION` before anything downstream
reads them; the default just reads whatever's freshest already persisted.

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_corpus.py [--regrade]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l3 import store as edit_store  # noqa: E402
from app.services.l3.grade import job as grade_job  # noqa: E402

OUT_DIR = os.path.join(HERE, "_out", "qa")
OUT_PATH = os.path.join(OUT_DIR, "corpus.json")


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def build_corpus(regrade: bool) -> List[Dict[str, Any]]:
    with _pg() as c:
        projects = c.execute("select id::text, source_file_ids from projects").fetchall()
        folders = c.execute("select id::text, name from folders").fetchall()
        frows = c.execute("select id::text, folder_id::text from files").fetchall()
    f2folder = {r[0]: r[1] for r in frows}
    folder_name = {r[0]: r[1] for r in folders}

    def label_for(fids: List[str]) -> str:
        names = {folder_name.get(f2folder.get(f)) for f in fids}
        names.discard(None)
        return ", ".join(sorted(n for n in names if n)) or "?"

    manifest: List[Dict[str, Any]] = []
    for pid, sfids in projects:
        fids = [str(x) for x in (sfids or [])]
        if not fids:
            continue
        label = label_for(fids)
        with _pg() as c:
            tids = [r[0] for r in c.execute(
                "select id::text from edit_threads where file_ids && %s::uuid[] order by updated_at desc",
                (fids,),
            ).fetchall()]

        chosen = None
        for tid in tids:
            try:
                doc, _ = edit_store.latest_document(tid)
            except Exception:
                doc = None
            if doc and (doc.get("timeline") or doc.get("operations")):
                chosen = (tid, doc)
                break
        if not chosen:
            print(f"[skip] {label} ({pid[:8]}): no gradeable thread")
            continue
        tid, doc = chosen

        if regrade:
            fn = getattr(grade_job.run_grade_job, "func", grade_job.run_grade_job)
            try:
                fn(tid)
                # Re-read: the job may have welded/annotated the document.
                doc, _ = edit_store.latest_document(tid)
            except Exception as e:
                print(f"[warn] {label} ({pid[:8]}) thread={tid[:8]}: run_grade_job failed: {e}")

        shots = grade_job.ordered_shots(doc)
        if not shots:
            print(f"[skip] {label} ({pid[:8]}) thread={tid[:8]}: no gradeable shots")
            continue
        manifest.append({
            "project_id": pid,
            "project_label": label,
            "thread_id": tid,
            "shot_count": len(shots),
            "file_ids": sorted({s.file_id for s in shots}),
        })
        print(f"[ok  ] {label} ({pid[:8]}) thread={tid[:8]} shots={len(shots)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regrade", action="store_true",
        help="run_grade_job(tid) for every thread before reading (grades current for this deploy)",
    )
    args = parser.parse_args()

    manifest = build_corpus(regrade=args.regrade)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    total_shots = sum(p["shot_count"] for p in manifest)
    print(f"\nwrote {OUT_PATH}: {len(manifest)} projects, {total_shots} shots")


if __name__ == "__main__":
    main()
