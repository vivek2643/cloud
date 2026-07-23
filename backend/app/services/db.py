"""
Process-global Postgres connection pool over Supabase's TRANSACTION pooler
(port 6543) -- scale_architecture.plan.md Pillar 1.

Every business-query module's local `_pg()`/`_pg_conn()` helper borrows from
this ONE pool instead of opening a brand-new socket per call (`psycopg.
connect(...)`), so N concurrent jobs each running many queries becomes "N
jobs share a bounded backend pool," not "N x (queries per job) sockets."
That is the precondition every other concurrency knob in this plan assumes.

Transaction-mode pooling has NO session state: server-side prepared
statements and LISTEN/NOTIFY break under it. `prepare_threshold=None`
disables psycopg3's default statement preparation -- mandatory here, not
an optimization. `autocommit=True` matches what every existing `_pg()`/
`_pg_conn()` helper already did (all ~24 call sites), so migrating a module
onto this pool changes zero query semantics.

Session-state consumers MUST NOT use this pool -- they stay on
`settings.database_url` (session pooler / direct, port 5432) via their own
separate connections:
  - Procrastinate's connector (`jobs.py`) -- needs LISTEN/NOTIFY.
  - The migration runner (`scripts/migrate.py`, `db_migrations.py`'s
    `advisory_lock`) -- `pg_advisory_lock`/`pg_advisory_unlock` are
    SESSION-scoped and require the same physical backend connection to stay
    pinned across multiple statements; a transaction-mode pooler can hand
    consecutive transactions to different backend PIDs, silently breaking
    that pin.
  - `app/main.py` / `backend/worker.py`'s own startup schema-drift check
    (`_check_schema`), for the same reason as the migration runner.

Lazy pool creation (opened on first `connection()` call, never at import
time) so a test or script that never touches the DB never needs
DATABASE_URL/DATABASE_POOL_URL set.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

_pool: "ConnectionPool | None" = None
_pool_lock = threading.Lock()


def _build_pool() -> ConnectionPool:
    settings = get_settings()
    pool_url = settings.database_pool_url or settings.database_url
    if not pool_url:
        raise RuntimeError(
            "db.py: neither DATABASE_POOL_URL nor DATABASE_URL is set -- "
            "cannot open the business-query connection pool."
        )
    pool = ConnectionPool(
        conninfo=pool_url,
        min_size=1,
        max_size=max(1, settings.db_pool_max_size),
        kwargs={"autocommit": True, "prepare_threshold": None},
        open=False,
    )
    pool.open()
    return pool


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # re-check inside the lock (double-checked init)
                _pool = _build_pool()
    return _pool


def connection() -> "contextlib.AbstractContextManager[psycopg.Connection]":
    """`with db.connection() as conn:` -- borrows a connection from the
    shared pool and returns it to the pool (never closes the socket) on
    exit. Plain tuple rows, matching every existing `_pg()`/`_pg_conn()`
    helper's default (no `row_factory` set)."""
    return _get_pool().connection()


@contextlib.contextmanager
def connection_dict_row() -> Iterator[psycopg.Connection]:
    """Same as `connection()`, but rows come back as dicts (psycopg's
    `dict_row` factory) -- for the handful of modules that already relied
    on that (`l3/cuts_read.py`, `l3/sync/store.py`, `l1/snapshot.py`)."""
    with connection() as conn:
        conn.row_factory = dict_row
        yield conn


def close() -> None:
    """Close the pool (all idle connections). For test teardown / clean
    process shutdown; never required for correctness (the pool's own
    `max_lifetime`/`max_idle` already recycle connections)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
