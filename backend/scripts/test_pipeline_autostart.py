"""
Unit tests for L1's cuts-autostart hook (``pipeline._maybe_autostart_cuts``).

The helper decides -- purely from the files table -- whether a folder's whole
L1 upload batch has finished and, if so, kicks the L3 cuts ingest ONCE. Here
the DB (`pipeline._pg_conn`) and the two l3 seams it calls
(`find_or_create_project`, `defer_ingest`) are fully faked, so nothing real is
queried, enqueued, or spent. The only real imports are the exception types the
helper is contracted to swallow (procrastinate's ``AlreadyEnqueued`` and
fairness' ``CapacityExceeded``).

Run:  .venv/bin/python scripts/test_pipeline_autostart.py
"""
from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from procrastinate.exceptions import AlreadyEnqueued  # noqa: E402

from app.services.fairness import CapacityExceeded  # noqa: E402
from app.services.l1 import pipeline  # noqa: E402


# --- fakes ----------------------------------------------------------------

class _FakeCursor:
    def __init__(self, one=None, many=None):
        self._one = one
        self._many = many

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._many or [])


class _FakeConn:
    """Answers only the three queries the helper issues, matched by a
    fragment of the SQL. Any other query is a test bug, not a silent pass."""

    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if "from ingest_runs" in s:
            return _FakeCursor(one=self.cfg.get("ingest_run_row"))
        if "file_type = 'video'" in s:
            return _FakeCursor(many=self.cfg.get("video_rows"))
        if "from files where id = %s" in s:
            return _FakeCursor(one=self.cfg.get("user_row"))
        raise AssertionError(f"unexpected SQL: {s}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Env:
    """Installs fake DB + fake l3 seams for one test, records the calls made,
    and restores everything on exit."""

    def __init__(self, cfg, find_or_create=None, defer=None):
        self.cfg = cfg
        self.find_calls = []
        self.defer_calls = []
        self._find_or_create = find_or_create
        self._defer = defer
        self._orig_pg = None
        self._orig_modules = {}

    def _find(self, user_id, file_ids):
        self.find_calls.append((user_id, list(file_ids)))
        if self._find_or_create is not None:
            return self._find_or_create(user_id, file_ids)
        return "proj-X"

    def _defer_ingest(self, project_id, user_id):
        self.defer_calls.append((project_id, user_id))
        if self._defer is not None:
            return self._defer(project_id, user_id)
        return None

    def __enter__(self):
        self._orig_pg = pipeline._pg_conn
        pipeline._pg_conn = lambda: _FakeConn(self.cfg)

        proj_mod = types.ModuleType("app.services.l3.projects")
        proj_mod.find_or_create_project = self._find
        ingest_mod = types.ModuleType("app.services.l3.ingest")
        ingest_mod.defer_ingest = self._defer_ingest
        for name, mod in (("app.services.l3.projects", proj_mod),
                          ("app.services.l3.ingest", ingest_mod)):
            self._orig_modules[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *a):
        pipeline._pg_conn = self._orig_pg
        for name, mod in self._orig_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        return False


# --- tests ----------------------------------------------------------------

def test_no_kick_while_sibling_still_running():
    cfg = {
        "user_row": ("user-1", "folder-1"),
        "video_rows": [("v1", "ready"), ("v2", "running")],
        "ingest_run_row": None,
    }
    with _Env(cfg) as env:
        pipeline._maybe_autostart_cuts("v1")
    assert env.find_calls == [], env.find_calls
    assert env.defer_calls == [], env.defer_calls
    print("ok  test_no_kick_while_sibling_still_running")


def test_kicks_once_when_batch_terminal_with_a_ready_video():
    cfg = {
        "user_row": ("user-1", "folder-1"),
        # mixed terminal states; only 'ready' ones are candidates
        "video_rows": [("v1", "ready"), ("v2", "skipped"), ("v3", "failed")],
        "ingest_run_row": None,
    }
    with _Env(cfg) as env:
        pipeline._maybe_autostart_cuts("v1")
    assert env.find_calls == [("user-1", ["v1"])], env.find_calls
    assert env.defer_calls == [("proj-X", "user-1")], env.defer_calls
    print("ok  test_kicks_once_when_batch_terminal_with_a_ready_video")


def test_root_folder_null_id_groups_together():
    """folder_id NULL (drive root) must still resolve a scope + kick."""
    cfg = {
        "user_row": ("user-1", None),
        "video_rows": [("v1", "ready"), ("v2", "ready")],
        "ingest_run_row": None,
    }
    with _Env(cfg) as env:
        pipeline._maybe_autostart_cuts("v1")
    assert env.find_calls == [("user-1", ["v1", "v2"])], env.find_calls
    assert env.defer_calls == [("proj-X", "user-1")], env.defer_calls
    print("ok  test_root_folder_null_id_groups_together")


def test_no_kick_when_no_ready_videos():
    cfg = {
        "user_row": ("user-1", "folder-1"),
        "video_rows": [("v1", "skipped"), ("v2", "failed")],
        "ingest_run_row": None,
    }
    with _Env(cfg) as env:
        pipeline._maybe_autostart_cuts("v1")
    assert env.find_calls == [], env.find_calls
    assert env.defer_calls == [], env.defer_calls
    print("ok  test_no_kick_when_no_ready_videos")


def test_no_rekick_when_project_already_has_an_ingest_run():
    cfg = {
        "user_row": ("user-1", "folder-1"),
        "video_rows": [("v1", "ready"), ("v2", "ready")],
        "ingest_run_row": (1,),  # one-shot guard: already ran
    }
    with _Env(cfg) as env:
        pipeline._maybe_autostart_cuts("v1")
    assert env.find_calls == [("user-1", ["v1", "v2"])], env.find_calls
    assert env.defer_calls == [], env.defer_calls
    print("ok  test_no_rekick_when_project_already_has_an_ingest_run")


def test_swallows_already_enqueued_and_capacity_exceeded():
    for exc in (AlreadyEnqueued("already queued"),
                CapacityExceeded("user-1", in_flight=5, max_inflight=5)):
        cfg = {
            "user_row": ("user-1", "folder-1"),
            "video_rows": [("v1", "ready")],
            "ingest_run_row": None,
        }

        def _boom(project_id, user_id, _exc=exc):
            raise _exc

        with _Env(cfg, defer=_boom) as env:
            pipeline._maybe_autostart_cuts("v1")  # must not raise
        assert env.defer_calls == [("proj-X", "user-1")], env.defer_calls
    print("ok  test_swallows_already_enqueued_and_capacity_exceeded")


def test_never_raises_even_if_a_dependency_throws():
    cfg = {
        "user_row": ("user-1", "folder-1"),
        "video_rows": [("v1", "ready")],
        "ingest_run_row": None,
    }

    def _explode(user_id, file_ids):
        raise RuntimeError("db is on fire")

    # find_or_create_project blowing up must be swallowed.
    with _Env(cfg, find_or_create=_explode):
        pipeline._maybe_autostart_cuts("v1")

    # even a totally broken _pg_conn must be swallowed.
    orig = pipeline._pg_conn
    try:
        def _broken():
            raise RuntimeError("no pool")
        pipeline._pg_conn = _broken
        pipeline._maybe_autostart_cuts("v1")
    finally:
        pipeline._pg_conn = orig
    print("ok  test_never_raises_even_if_a_dependency_throws")


def main():
    test_no_kick_while_sibling_still_running()
    test_kicks_once_when_batch_terminal_with_a_ready_video()
    test_root_folder_null_id_groups_together()
    test_no_kick_when_no_ready_videos()
    test_no_rekick_when_project_already_has_an_ingest_run()
    test_swallows_already_enqueued_and_capacity_exceeded()
    test_never_raises_even_if_a_dependency_throws()
    print("\nall pipeline autostart tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
