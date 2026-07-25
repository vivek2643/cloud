"""Regression test: db.connection_dict_row() must restore row_factory on exit
so it can't poison a pooled connection for the next db.connection() borrower.

This reproduces the l1_active_speaker `KeyError: 0` seen on a warm RunPod worker:
build_l1_snapshot borrowed a dict-row connection, and the next task's
db.connection() borrow of the SAME pooled socket then handed back dict rows.
"""
import contextlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg.rows import dict_row, tuple_row  # noqa: E402

import app.services.db as db  # noqa: E402


class _FakeConn:
    def __init__(self):
        self.row_factory = tuple_row


class _FakePool:
    """Mimics one pooled connection handed back and reused."""

    def __init__(self, conn):
        self._conn = conn

    @contextlib.contextmanager
    def connection(self):
        # A real pool does NOT reset row_factory itself -- that's the whole point.
        yield self._conn


def test_dict_row_restores_and_does_not_poison_pool():
    conn = _FakeConn()
    db._pool = _FakePool(conn)  # inject a fake pool (bypass real DB)
    try:
        with db.connection_dict_row() as c:
            assert c.row_factory is dict_row, "dict_row not applied inside the ctx"
        assert conn.row_factory is tuple_row, (
            f"row_factory NOT restored on exit: {conn.row_factory}")

        # The next plain borrow of the same pooled socket must see tuple rows.
        with db.connection() as c2:
            assert c2.row_factory is tuple_row, (
                f"pool poisoned -- next connection() got {c2.row_factory}")
    finally:
        db._pool = None
    print("ok: connection_dict_row restores row_factory; pool not poisoned")


if __name__ == "__main__":
    test_dict_row_restores_and_does_not_poison_pool()
    print("1/1 passed")
