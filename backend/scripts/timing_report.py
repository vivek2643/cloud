#!/usr/bin/env python3
"""scale_architecture.plan.md Pillar 7: per-stage timing breakdown for an L3
ingest run/project, or an L1 file. Purely a formatter over what run_ingest /
l1_orchestrate already persisted (ingest_runs.timings_ms, processing_jobs)
-- nothing here re-derives or re-measures anything.

Usage:
    .venv/bin/python scripts/timing_report.py --run <ingest_run_id>
    .venv/bin/python scripts/timing_report.py --project <project_id> [--limit N]
    .venv/bin/python scripts/timing_report.py --file <file_id>
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services import db  # noqa: E402

_STAGE_ORDER = ("pass1", "extract", "pass2", "pass2_max_batch", "identity", "post", "total")


def _fmt_run(row: Dict[str, Any]) -> str:
    timings = row.get("timings_ms") or {}
    lines = [
        f"run {row['id']}  project={row['project_id']}  status={row['status']}  "
        f"created={row['created_at']}",
    ]
    if not timings:
        lines.append("  (no timings recorded -- pre-Pillar-7 run, or failed before any stage completed)")
        return "\n".join(lines)
    for stage in _STAGE_ORDER:
        if stage in timings:
            lines.append(f"  {stage:<16} {timings[stage] / 1000:8.1f}s")
    extra = {k: v for k, v in timings.items() if k not in _STAGE_ORDER}
    for k, v in sorted(extra.items()):
        lines.append(f"  {k:<16} {v / 1000:8.1f}s")
    return "\n".join(lines)


def report_run(ingest_run_id: str) -> None:
    with db.connection_dict_row() as conn:
        row = conn.execute(
            "select id::text, project_id::text, status, timings_ms, created_at "
            "from ingest_runs where id = %s",
            (ingest_run_id,),
        ).fetchone()
    if row is None:
        print(f"no ingest_run {ingest_run_id}")
        return
    print(_fmt_run(row))


def report_project(project_id: str, limit: int = 10) -> None:
    with db.connection_dict_row() as conn:
        rows = conn.execute(
            "select id::text, project_id::text, status, timings_ms, created_at "
            "from ingest_runs where project_id = %s order by created_at desc limit %s",
            (project_id, limit),
        ).fetchall()
    if not rows:
        print(f"no ingest_runs for project {project_id}")
        return
    for row in rows:
        print(_fmt_run(row))
        print()


def report_file(file_id: str) -> None:
    """L1's per-stage timing was already gated by processing_jobs
    (started_at/finished_at) before this pillar existed -- report it the
    same way rather than inventing a parallel mechanism."""
    with db.connection_dict_row() as conn:
        rows = conn.execute(
            "select stage, status, started_at, finished_at, attempts, error "
            "from processing_jobs where file_id = %s order by started_at",
            (file_id,),
        ).fetchall()
    if not rows:
        print(f"no processing_jobs for file {file_id}")
        return
    print(f"file {file_id}")
    total_s = 0.0
    for r in rows:
        started, finished = r["started_at"], r["finished_at"]
        secs = (finished - started).total_seconds() if started and finished else None
        total_s += secs or 0.0
        secs_str = f"{secs:8.1f}s" if secs is not None else "     n/a"
        note = f"  ({r['error'][:60]})" if r.get("error") else ""
        print(f"  {r['stage']:<18} {r['status']:<10} {secs_str} attempts={r['attempts']}{note}")
    print(f"  {'total':<18} {'':<10} {total_s:8.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="ingest_run id (L3)")
    p.add_argument("--project", help="project id (L3, most recent runs)")
    p.add_argument("--file", help="file id (L1, per-stage processing_jobs)")
    p.add_argument("--limit", type=int, default=10, help="max runs to show for --project")
    args = p.parse_args()

    if sum(bool(x) for x in (args.run, args.project, args.file)) != 1:
        p.error("pass exactly one of --run, --project, --file")

    if args.run:
        report_run(args.run)
    elif args.project:
        report_project(args.project, args.limit)
    else:
        report_file(args.file)


if __name__ == "__main__":
    main()
