#!/usr/bin/env python3
"""Tests for app/services/db_migrations.py (migration_runner.plan.md).

No real DB -- a small FakeConn test double stands in for a psycopg
connection (records every execute() call, keeps an in-memory
filename->checksum store), mirroring the rest of this suite's "no DB"
convention (see test_grade.py's docstring). The one exception is the
advisory-lock concurrency test, which uses a real threading.Lock to prove
serialization the same way a real Postgres session-level advisory lock
would.

Run:  .venv/bin/python scripts/test_db_migrations.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services import db_migrations  # noqa: E402


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class FakeConn:
    """filename -> checksum store + a call log. `table_exists=False` mimics
    the pre-bootstrap state (fetch_applied must return {} without erroring)."""

    def __init__(self, existing=None, table_exists=True, fail_on=None):
        self.rows = dict(existing or {})
        self.table_exists = table_exists
        self.fail_on = fail_on
        self.executed = []
        self.autocommit_log = []
        self.commits = 0
        self.rollbacks = 0
        self.autocommit = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self.autocommit_log.append(self.autocommit)
        norm = " ".join(sql.split()).lower()
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("boom: " + self.fail_on)
        if norm.startswith("create table"):
            self.table_exists = True
            return FakeCursor()
        if norm.startswith("select filename, checksum"):
            if not self.table_exists:
                raise RuntimeError('relation "schema_migrations" does not exist')
            return FakeCursor(rows=list(self.rows.items()))
        if norm.startswith("insert into public.schema_migrations"):
            filename, cksum = params
            self.rows[filename] = cksum
            return FakeCursor()
        if norm.startswith("update public.schema_migrations"):
            cksum, filename = params
            if filename in self.rows:
                self.rows[filename] = cksum
                return FakeCursor(rowcount=1)
            return FakeCursor(rowcount=0)
        if "pg_advisory_lock" in norm or "pg_advisory_unlock" in norm:
            return FakeCursor()
        # Anything else is a migration file's own arbitrary DDL/content --
        # just record that it ran.
        return FakeCursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _write(tmpdir: str, name: str, content: str) -> Path:
    path = Path(tmpdir) / name
    path.write_text(content)
    return path


def _in_tmp_migrations_dir(files: dict):
    """Context manager: point db_migrations.MIGRATIONS_DIR at a fresh temp
    dir populated with `files` (name -> content), clean up after."""
    tmpdir = tempfile.mkdtemp()
    for name, content in files.items():
        _write(tmpdir, name, content)
    patcher = mock.patch.object(db_migrations, "MIGRATIONS_DIR", Path(tmpdir))
    return patcher, tmpdir


# --------------------------------------------------------------------------
# checksum()
# --------------------------------------------------------------------------

def test_checksum_ignores_trailing_whitespace_and_crlf():
    tmpdir = tempfile.mkdtemp()
    try:
        a = _write(tmpdir, "a.sql", "select 1;\nselect 2;\n")
        b = _write(tmpdir, "b.sql", "select 1;   \r\nselect 2;\r\n")
        assert db_migrations.checksum(a) == db_migrations.checksum(b)

        c = _write(tmpdir, "c.sql", "select 1;\nselect 3;\n")
        assert db_migrations.checksum(a) != db_migrations.checksum(c)
    finally:
        shutil.rmtree(tmpdir)


# --------------------------------------------------------------------------
# pending()
# --------------------------------------------------------------------------

def test_pending_separates_never_applied_from_drifted():
    patcher, tmpdir = _in_tmp_migrations_dir({
        "001_a.sql": "select 1;\n",
        "002_b.sql": "select 2;\n",
        "003_c.sql": "select 3;\n",
    })
    with patcher:
        a_path = db_migrations.MIGRATIONS_DIR / "001_a.sql"
        conn = FakeConn(existing={
            "001_a.sql": db_migrations.checksum(a_path),  # matches, up to date
            "002_b.sql": "stale-checksum-from-before-an-edit",  # drifted
            # 003_c.sql never recorded -> pending
        })
        not_applied, drifted = db_migrations.pending(conn)
        assert [f.name for f in not_applied] == ["003_c.sql"]
        assert drifted == ["002_b.sql"]
    shutil.rmtree(tmpdir)


def test_pending_returns_all_files_when_table_does_not_exist_yet():
    patcher, tmpdir = _in_tmp_migrations_dir({"001_a.sql": "select 1;\n"})
    with patcher:
        conn = FakeConn(existing={}, table_exists=False)
        not_applied, drifted = db_migrations.pending(conn)
        assert [f.name for f in not_applied] == ["001_a.sql"]
        assert drifted == []
        assert conn.rollbacks == 1  # fetch_applied recovers from the failed select
    shutil.rmtree(tmpdir)


# --------------------------------------------------------------------------
# apply_pending()
# --------------------------------------------------------------------------

def test_apply_pending_stops_at_first_failure_and_never_runs_later_files():
    patcher, tmpdir = _in_tmp_migrations_dir({
        "001_a.sql": "select 1;\n",
        "002_b.sql": "BOOM_MARKER select 2;\n",
        "003_c.sql": "select 3;\n",
    })
    with patcher:
        conn = FakeConn(existing={}, table_exists=False, fail_on="BOOM_MARKER")
        try:
            db_migrations.apply_pending(conn)
            raise AssertionError("expected apply_pending to raise")
        except RuntimeError as e:
            assert "BOOM_MARKER" in str(e)
        assert "001_a.sql" in conn.rows
        assert "002_b.sql" not in conn.rows
        assert "003_c.sql" not in conn.rows
        assert conn.rollbacks == 1
    shutil.rmtree(tmpdir)


def test_apply_pending_no_transaction_file_uses_autocommit():
    patcher, tmpdir = _in_tmp_migrations_dir({
        "001_a.sql": "select 1;\n",
        "002_idx.sql": "-- migrate:no-transaction\ncreate index concurrently if not exists x on y(z);\n",
    })
    with patcher:
        conn = FakeConn(existing={}, table_exists=False)
        applied = db_migrations.apply_pending(conn)
        assert applied == ["001_a.sql", "002_idx.sql"]
        # autocommit was True for exactly the 2 execute() calls belonging to
        # the no-transaction file (its DDL + its own tracking insert), and
        # False again immediately after.
        assert conn.autocommit is False
        assert conn.autocommit_log.count(True) == 2
    shutil.rmtree(tmpdir)


# --------------------------------------------------------------------------
# assert_up_to_date()
# --------------------------------------------------------------------------

class _FakeSettings:
    def __init__(self, migration_guard="on"):
        self.migration_guard = migration_guard


def test_assert_up_to_date_raises_naming_every_pending_and_drifted_filename():
    patcher, tmpdir = _in_tmp_migrations_dir({
        "001_a.sql": "select 1;\n",
        "002_b.sql": "select 2;\n",
        "003_c.sql": "select 3;\n",
    })
    with patcher:
        a_path = db_migrations.MIGRATIONS_DIR / "001_a.sql"
        conn = FakeConn(existing={
            "001_a.sql": db_migrations.checksum(a_path),
            "002_b.sql": "stale",
        })
        with mock.patch("app.config.get_settings", return_value=_FakeSettings("on")):
            try:
                db_migrations.assert_up_to_date(conn)
                raise AssertionError("expected SchemaDriftError")
            except db_migrations.SchemaDriftError as e:
                msg = str(e)
                assert "003_c.sql" in msg
                assert "002_b.sql" in msg
    shutil.rmtree(tmpdir)


def test_assert_up_to_date_bypassed_when_migration_guard_off():
    patcher, tmpdir = _in_tmp_migrations_dir({"001_a.sql": "select 1;\n"})
    with patcher:
        conn = FakeConn(existing={}, table_exists=False)
        with mock.patch("app.config.get_settings", return_value=_FakeSettings("off")):
            db_migrations.assert_up_to_date(conn)  # must not raise
        assert conn.executed == []  # never even queried
    shutil.rmtree(tmpdir)


# --------------------------------------------------------------------------
# reconcile()
# --------------------------------------------------------------------------

def test_reconcile_updates_checksum_without_rerunning():
    patcher, tmpdir = _in_tmp_migrations_dir({"001_a.sql": "select 1;\n"})
    with patcher:
        conn = FakeConn(existing={"001_a.sql": "old-and-wrong"})
        db_migrations.reconcile(conn, "001_a.sql")
        a_path = db_migrations.MIGRATIONS_DIR / "001_a.sql"
        assert conn.rows["001_a.sql"] == db_migrations.checksum(a_path)
        assert conn.commits == 1
        # no DDL/select was ever sent for the file's own content
        assert all("select 1" not in (sql or "") for sql, _ in conn.executed)
    shutil.rmtree(tmpdir)


def test_reconcile_raises_for_never_applied_file():
    patcher, tmpdir = _in_tmp_migrations_dir({"001_a.sql": "select 1;\n"})
    with patcher:
        conn = FakeConn(existing={})
        try:
            db_migrations.reconcile(conn, "001_a.sql")
            raise AssertionError("expected LookupError")
        except LookupError:
            pass
        assert conn.rollbacks == 1
    shutil.rmtree(tmpdir)


def test_reconcile_raises_for_missing_file():
    patcher, tmpdir = _in_tmp_migrations_dir({})
    with patcher:
        conn = FakeConn(existing={"999_ghost.sql": "x"})
        try:
            db_migrations.reconcile(conn, "999_ghost.sql")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
    shutil.rmtree(tmpdir)


# --------------------------------------------------------------------------
# advisory_lock() -- real threading.Lock, proving serialization the same
# way a real Postgres session-level advisory lock would.
# --------------------------------------------------------------------------

class _RacingConn:
    """Shares `rows`/a real lock across two 'connections' (one per thread).
    Every execute() sleeps briefly to widen the race window; `inside`
    tracks concurrent occupancy of the locked section so the test can prove
    the two threads never overlapped."""

    def __init__(self, rows: dict, lock: threading.Lock, occupancy: dict):
        self.rows = rows
        self._lock = lock
        self._occupancy = occupancy
        self.autocommit = False

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).lower()
        if "pg_advisory_lock" in norm and "unlock" not in norm:
            self._lock.acquire()
            return FakeCursor()
        if "pg_advisory_unlock" in norm:
            self._lock.release()
            return FakeCursor()

        self._occupancy["count"] += 1
        if self._occupancy["count"] > 1:
            self._occupancy["overlap"] = True
        time.sleep(0.005)
        try:
            if norm.startswith("create table"):
                return FakeCursor()
            if norm.startswith("select filename, checksum"):
                return FakeCursor(rows=list(self.rows.items()))
            if norm.startswith("insert into public.schema_migrations"):
                filename, cksum = params
                self.rows[filename] = cksum
                return FakeCursor()
            # Migration file's own arbitrary content.
            return FakeCursor()
        finally:
            self._occupancy["count"] -= 1

    def commit(self):
        pass

    def rollback(self):
        pass


def test_advisory_lock_serializes_two_concurrent_appliers():
    patcher, tmpdir = _in_tmp_migrations_dir({"001_a.sql": "select 1;\n"})
    with patcher:
        shared_rows: dict = {}
        lock = threading.Lock()
        occupancy = {"count": 0, "overlap": False}
        results = []

        def worker():
            conn = _RacingConn(shared_rows, lock, occupancy)
            with db_migrations.advisory_lock(conn):
                results.append(db_migrations.apply_pending(conn))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not occupancy["overlap"], "both threads were inside the locked section at once"
        applied_by_either = [r for r in results if r]
        assert applied_by_either == [["001_a.sql"]], (
            "expected exactly one caller to apply the file and the other to "
            f"find nothing pending, got {results}"
        )
        a_path = db_migrations.MIGRATIONS_DIR / "001_a.sql"
        assert shared_rows == {"001_a.sql": db_migrations.checksum(a_path)}
    shutil.rmtree(tmpdir)


def main() -> None:
    test_checksum_ignores_trailing_whitespace_and_crlf()
    test_pending_separates_never_applied_from_drifted()
    test_pending_returns_all_files_when_table_does_not_exist_yet()
    test_apply_pending_stops_at_first_failure_and_never_runs_later_files()
    test_apply_pending_no_transaction_file_uses_autocommit()
    test_assert_up_to_date_raises_naming_every_pending_and_drifted_filename()
    test_assert_up_to_date_bypassed_when_migration_guard_off()
    test_reconcile_updates_checksum_without_rerunning()
    test_reconcile_raises_for_never_applied_file()
    test_reconcile_raises_for_missing_file()
    test_advisory_lock_serializes_two_concurrent_appliers()
    print("\nall db_migrations tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
