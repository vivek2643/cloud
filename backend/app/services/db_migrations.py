"""
Migrations-tracking + runner logic (migration_runner.plan.md). Shared by the
CLI (scripts/migrate.py -- the applier) and the startup guard
(worker.py, app/main.py -- read-only via assert_up_to_date()).

Two different jobs, two different safety properties: apply_pending() is the
thing that ever writes schema, and it must only ever run as one gated
process at a time (see advisory_lock()). assert_up_to_date() is read-only
and safe to call from every process unconditionally.

What assert_up_to_date() proves, and does NOT prove: a green result means
"every file in migrations/ has either never run, or ran once through this
system and hasn't been edited since." It does NOT mean "the live database
currently reflects what these files say" -- that's a different claim. The
motivating incident (018) was applied by hand, outside this system, and
landed only partially (2 of 7 target column drops); its checksum would
still match today since the file itself was never edited. Going forward,
apply_pending()'s per-file transaction (for transactional files) closes that
gap for anything applied *through* this system -- but a no-transaction file
partially failing mid-flight, or anyone hand-running DDL outside migrate.py
entirely, is invisible to this ledger. See migration_runner.plan.md's "What
the guard does -- and does not -- catch" for the full writeup.
"""
from __future__ import annotations

import hashlib
import logging
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

# A file whose first line is exactly this marker is applied in autocommit
# mode (one statement, no wrapping transaction) instead of the default
# single-transaction-per-file path. Needed because some legitimate Postgres
# statements -- CREATE INDEX CONCURRENTLY, ALTER TYPE ... ADD VALUE pre-12,
# VACUUM -- cannot run inside a transaction block at all. Convention (not
# mechanically enforced by a parser): a no-transaction file should contain
# exactly one statement -- if it bundles more than one genuinely
# non-transactional statement, Postgres's own multi-statement grouping in
# the simple-query protocol still applies to a single execute() call
# regardless of autocommit, so it fails loudly with the same "cannot run
# inside a transaction block" error, which is itself the signal to split it.
NO_TRANSACTION_MARKER = "-- migrate:no-transaction"

# Fixed, arbitrary advisory-lock key so exactly one process at a time runs
# apply_pending() against a given database -- derived from a fixed string
# (not random) so every process computes the same key by construction.
# Procrastinate (this repo's job queue) does not use Postgres advisory locks
# itself (checked in its installed source at implementation time), so there
# is no collision risk with it.
_ADVISORY_LOCK_KEY = zlib.crc32(b"edso:schema_migrations")


class SchemaDriftError(RuntimeError):
    """Raised by assert_up_to_date() when live schema doesn't match migrations/."""


def list_migration_files() -> List[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _normalize(text: str) -> str:
    # Strip trailing whitespace per line + force a single trailing newline,
    # so an innocuous re-save (CRLF flip, a linter touching whitespace)
    # doesn't change the checksum of an already-applied file and trip the
    # guard fleet-wide over nothing but formatting.
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines) + "\n"


def checksum(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _is_no_transaction(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    return first_line == NO_TRANSACTION_MARKER


def fetch_applied(conn) -> dict:
    """filename -> checksum, from schema_migrations. {} if the table doesn't
    exist yet -- that's the pre-bootstrap state, never an error."""
    try:
        rows = conn.execute(
            "select filename, checksum from public.schema_migrations"
        ).fetchall()
    except Exception:
        conn.rollback()
        return {}
    return {filename: cksum for filename, cksum in rows}


def pending(conn) -> Tuple[List[Path], List[str]]:
    """(files not yet applied, filenames whose live checksum no longer
    matches the tracked one) -- the second list is the drift signal."""
    applied = fetch_applied(conn)
    files = list_migration_files()
    not_applied = [f for f in files if f.name not in applied]
    drifted = [
        f.name for f in files
        if f.name in applied and applied[f.name] != checksum(f)
    ]
    return not_applied, drifted


def _ensure_tracking_table(conn) -> None:
    conn.execute(
        "create table if not exists public.schema_migrations ("
        "  filename text primary key,"
        "  checksum text not null,"
        "  applied_at timestamptz not null default now()"
        ")"
    )
    conn.commit()


def apply_pending(conn) -> List[str]:
    """
    Apply every pending migration file, in filename order.

    Transactional files (the default) run + get recorded as one atomic
    transaction: either both happen or neither does. A no-transaction file
    (see NO_TRANSACTION_MARKER) runs in autocommit mode instead, which means
    apply+record for THAT file is not atomic -- write no-transaction
    migrations idempotently (e.g. `create index concurrently if not
    exists ...`) so a retry after a crash between the two is safe.

    Stops at the first failure and raises with the filename named -- never
    skips ahead to a later file.
    """
    _ensure_tracking_table(conn)
    to_apply, _drifted = pending(conn)
    applied_names: List[str] = []
    for path in to_apply:
        logger.info("Applying migration %s", path.name)
        sql_text = path.read_text(encoding="utf-8")
        cksum = checksum(path)
        if _is_no_transaction(path):
            conn.autocommit = True
            try:
                conn.execute(sql_text)
                conn.execute(
                    "insert into public.schema_migrations (filename, checksum) "
                    "values (%s, %s)",
                    (path.name, cksum),
                )
            except Exception:
                logger.exception(
                    "Migration %s (no-transaction) failed -- apply+record for "
                    "this file is NOT atomic; verify DB state by hand before "
                    "retrying.",
                    path.name,
                )
                raise
            finally:
                conn.autocommit = False
        else:
            try:
                conn.execute(sql_text)
                conn.execute(
                    "insert into public.schema_migrations (filename, checksum) "
                    "values (%s, %s)",
                    (path.name, cksum),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(
                    "Migration %s failed; rolled back, stopping before any "
                    "later file.",
                    path.name,
                )
                raise
        applied_names.append(path.name)
    return applied_names


def reconcile(conn, filename: str) -> None:
    """
    Update the tracked checksum for an already-applied file to match its
    current on-disk contents, WITHOUT re-running it. The deliberate, explicit
    escape hatch for "this old file's text changed for a legitimate reason
    (e.g. a comment fix) and I've confirmed the DB itself needs no change" --
    never used silently, never used automatically.
    """
    path = MIGRATIONS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No such migration file: {filename}")
    cur = conn.execute(
        "update public.schema_migrations set checksum = %s where filename = %s",
        (checksum(path), filename),
    )
    if cur.rowcount == 0:
        conn.rollback()
        raise LookupError(
            f"{filename} has never been applied -- reconcile only updates the "
            "checksum of an already-tracked file, it does not apply new ones."
        )
    conn.commit()


@contextmanager
def advisory_lock(conn) -> Iterator[None]:
    """
    Session-level Postgres advisory lock (not pg_advisory_XACT_lock -- this
    must survive apply_pending()'s per-file commits, so it has to be
    session-scoped, released explicitly in `finally`, not tied to any one
    transaction). A concurrent second caller blocks here until the first
    releases, then finds nothing pending and returns immediately.
    """
    conn.execute("select pg_advisory_lock(%s)", (_ADVISORY_LOCK_KEY,))
    conn.commit()
    try:
        yield
    finally:
        conn.execute("select pg_advisory_unlock(%s)", (_ADVISORY_LOCK_KEY,))
        conn.commit()


def assert_up_to_date(conn) -> None:
    """
    Read-only startup check. Raises SchemaDriftError, naming every
    pending/drifted filename explicitly, if the live schema doesn't match
    migrations/. See module docstring for exactly what this does and does
    not prove.

    Bypass: MIGRATION_GUARD=off (Settings.migration_guard) is a sanctioned,
    loud, per-process escape hatch for local dev -- never set in production.
    """
    from app.config import get_settings

    if get_settings().migration_guard == "off":
        logger.warning(
            "MIGRATION_GUARD=off -- schema drift checks DISABLED for this "
            "process. Do not use in production."
        )
        return

    not_applied, drifted = pending(conn)
    if not not_applied and not drifted:
        return

    parts = []
    if not_applied:
        parts.append("pending: " + ", ".join(f.name for f in not_applied))
    if drifted:
        parts.append("drifted (checksum changed since applied): " + ", ".join(drifted))
    raise SchemaDriftError(
        "Schema out of date with backend/migrations/ -- "
        + "; ".join(parts)
        + ". Run `python backend/scripts/migrate.py apply`, or set "
        "MIGRATION_GUARD=off for local dev."
    )
