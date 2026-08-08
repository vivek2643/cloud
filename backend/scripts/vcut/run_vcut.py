"""
seam_cut_pipeline.plan.md section 13.4 -- dev/smoke CLI for the vcut
pipeline: runs the whole foreground path (spans -> sample -> Pass 1 ->
resolve -> write cut_records) for one project, synchronously (no
procrastinate worker needed). THIS SPENDS REAL MONEY (one Gemini call per
project) -- a separate, explicit decision each time it's run, same
disclosure as app.services.vcut.orchestrate.run_vcut_ingest.

Run:
  .venv/bin/python scripts/vcut/run_vcut.py --project <project_id>
  .venv/bin/python scripts/vcut/run_vcut.py --folder <folder_id>   # find-or-create a project from every video in a folder
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_vcut")

_LOCAL_USER = "00000000-0000-0000-0000-000000000001"


def _project_for_folder(folder_id: str) -> str:
    from app.services import db
    from app.services.l3.projects import find_or_create_project

    with db.connection_dict_row() as conn:
        rows = conn.execute(
            "select id::text from files where folder_id = %s and file_type = 'video' order by filename",
            (folder_id,),
        ).fetchall()
    file_ids = [r["id"] for r in rows]
    if not file_ids:
        raise SystemExit(f"folder {folder_id} has no video files")
    project_id = find_or_create_project(_LOCAL_USER, file_ids)
    logger.info("folder %s -> project %s (%d file(s))", folder_id, project_id, len(file_ids))
    return project_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=None, help="run vcut for an existing project id")
    ap.add_argument("--folder", default=None, help="find-or-create a project from a folder's video files")
    args = ap.parse_args()
    if not args.project and not args.folder:
        raise SystemExit("pass --project <id> or --folder <id>")

    from app.services.vcut.orchestrate import run_vcut_ingest

    project_id = args.project or _project_for_folder(args.folder)
    logger.info("running vcut ingest for project %s ...", project_id)
    ingest_run_id = run_vcut_ingest(project_id)
    logger.info("done: ingest_run_id=%s -- inspect via GET /api/projects/%s/cuts "
               "or the cutviz debug overlay", ingest_run_id, project_id)


if __name__ == "__main__":
    main()
