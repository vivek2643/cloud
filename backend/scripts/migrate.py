#!/usr/bin/env python3
"""
Migrations CLI (migration_runner.plan.md) -- the applier.

    python backend/scripts/migrate.py apply       # apply pending migrations
    python backend/scripts/migrate.py check        # read-only: pending/drifted?
    python backend/scripts/migrate.py reconcile <filename>   # re-checksum only

`apply` is the ONLY command that writes schema. It takes a Postgres advisory
lock first so exactly one process applies at a time even if invoked
concurrently -- see app/services/db_migrations.py::advisory_lock(). It is
meant to be called from exactly one place in the deploy path (today:
run_workers.sh, before any worker fork) -- never from every worker/app
process at startup.

Run:  .venv/bin/python scripts/migrate.py <command> [args]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.config import get_settings  # noqa: E402
from app.services.db_migrations import (  # noqa: E402
    advisory_lock,
    apply_pending,
    pending,
    reconcile as _reconcile,
)


def _connect():
    import psycopg
    return psycopg.connect(get_settings().database_url, autocommit=False)


def cmd_apply() -> int:
    with _connect() as conn:
        with advisory_lock(conn):
            applied = apply_pending(conn)
    if applied:
        print(f"Applied {len(applied)} migration(s):")
        for name in applied:
            print(" -", name)
    else:
        print("Nothing pending.")
    return 0


def cmd_check() -> int:
    with _connect() as conn:
        not_applied, drifted = pending(conn)
    if not not_applied and not drifted:
        print("Clean -- nothing pending, nothing drifted.")
        return 0
    if not_applied:
        print(f"PENDING ({len(not_applied)}):")
        for f in not_applied:
            print(" -", f.name)
    if drifted:
        print(f"DRIFTED -- checksum changed since applied ({len(drifted)}):")
        for name in drifted:
            print(" -", name)
    return 1


def cmd_reconcile(filename: str) -> int:
    with _connect() as conn:
        _reconcile(conn, filename)
    print(f"Reconciled {filename} -- checksum updated, nothing re-run.")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("apply", "check", "reconcile"):
        print(__doc__)
        return 2
    command = sys.argv[1]
    if command == "apply":
        return cmd_apply()
    if command == "check":
        return cmd_check()
    if command == "reconcile":
        if len(sys.argv) != 3:
            print("usage: migrate.py reconcile <filename>")
            return 2
        return cmd_reconcile(sys.argv[2])
    return 2


if __name__ == "__main__":
    sys.exit(main())
