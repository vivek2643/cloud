"""
Smoke tests for the full-cascade folder (project) delete
(``app.routers.folders.delete_folder``) -- a real FastAPI TestClient against
the actual app, with the DB (supabase + raw SQL) and R2 fully faked. No real
Postgres, no real R2.

Covers: 404 on an unowned folder; the recursive descendant collection; and the
end-to-end cascade orchestration (file rows deleted -> projects/threads/jobs
purged -> R2 objects removed -> folder rows deleted, deepest first).

Run:  .venv/bin/python scripts/test_folders_router.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from fastapi.testclient import TestClient  # noqa: E402

from app.auth import get_current_user_id  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.routers import folders  # noqa: E402


class _Patcher:
    def __init__(self):
        self._orig = {}

    def set(self, obj, name, value):
        self._orig[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def restore(self):
        for (obj, name), value in self._orig.items():
            setattr(obj, name, value)


def _as_user(user_id: str):
    fastapi_app.dependency_overrides[get_current_user_id] = lambda: user_id


def _clear_overrides():
    fastapi_app.dependency_overrides.clear()


# --- Minimal fake supabase client -------------------------------------------
# Records every delete and answers the exact select chains delete_folder issues:
#   folders.select("id").eq("id", X).eq("user_id", U)            -> ownership
#   folders.select("id").in_("parent_id", [...]).eq("user_id")   -> children
#   files.select(...).in_("folder_id", [...]).eq("user_id")      -> file rows
#   files.delete().in_("id", [...]).eq("user_id")                -> record
#   folders.delete().eq("id", X).eq("user_id")                   -> record


class _FakeQuery:
    def __init__(self, sb, table, op):
        self.sb = sb
        self.table = table
        self.op = op  # "select" | "delete"
        self.eqs = {}
        self.ins = {}

    def select(self, *_a, **_k):
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, col, val):
        self.eqs[col] = val
        return self

    def in_(self, col, vals):
        self.ins[col] = list(vals)
        return self

    def is_(self, col, _val):
        return self

    def order(self, *_a, **_k):
        return self

    def execute(self):
        if self.op == "delete":
            self.sb.deletes.append(
                {"table": self.table, "eqs": dict(self.eqs), "ins": dict(self.ins)}
            )
            return _Result([{"id": "x"}])  # non-empty = "something matched"
        # selects
        if self.table == "folders" and "parent_id" in self.ins:
            rows = []
            for parent in self.ins["parent_id"]:
                rows.extend({"id": c} for c in self.sb.children.get(parent, []))
            return _Result(rows)
        if self.table == "folders" and "id" in self.eqs:
            fid = self.eqs["id"]
            return _Result([{"id": fid}] if fid in self.sb.owned else [])
        if self.table == "files" and "folder_id" in self.ins:
            rows = [f for f in self.sb.files if f["folder_id"] in set(self.ins["folder_id"])]
            return _Result(rows)
        return _Result([])


class _Result:
    def __init__(self, data):
        self.data = data


class _FakeSupabase:
    def __init__(self, owned, children, files):
        self.owned = set(owned)
        self.children = children  # parent_id -> [child_id, ...]
        self.files = files        # [{id, folder_id, r2_key, ...}, ...]
        self.deletes = []

    def table(self, name):
        return _FakeQuery(self, name, "select")


def test_delete_folder_404s_when_not_owned():
    sb = _FakeSupabase(owned=set(), children={}, files=[])
    p = _Patcher()
    p.set(folders, "get_supabase", lambda: sb)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.delete("/api/folders/missing")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    assert sb.deletes == []
    print("ok  test_delete_folder_404s_when_not_owned")


def test_descendant_folder_ids_walks_the_whole_subtree():
    # root -> a, b ; a -> a1 ; a1 -> a1x
    sb = _FakeSupabase(
        owned={"root"},
        children={"root": ["a", "b"], "a": ["a1"], "a1": ["a1x"], "b": []},
        files=[],
    )
    ids = folders._descendant_folder_ids(sb, "root", "user-1")
    assert ids[0] == "root"
    assert set(ids) == {"root", "a", "b", "a1", "a1x"}
    # root must come before its descendants (BFS), so reversed() deletes leaves first.
    assert ids.index("root") < ids.index("a") < ids.index("a1") < ids.index("a1x")
    print("ok  test_descendant_folder_ids_walks_the_whole_subtree")


def test_delete_folder_full_cascade():
    # Subtree: root -> sub ; files f1 (root) and f2 (sub).
    files = [
        {
            "id": "f1", "folder_id": "root",
            "r2_key": "raw/f1", "r2_proxy_key": "prox/f1",
            "r2_proxy_a_key": None, "r2_proxy_b_key": None,
            "r2_thumbnail_key": "thumb/f1",
        },
        {
            "id": "f2", "folder_id": "sub",
            "r2_key": "raw/f2", "r2_proxy_key": None,
            "r2_proxy_a_key": "a/f2", "r2_proxy_b_key": "b/f2",
            "r2_thumbnail_key": None,
        },
    ]
    sb = _FakeSupabase(owned={"root"}, children={"root": ["sub"], "sub": []}, files=files)

    seen_projthread = {}
    deleted_keys = []

    def fake_projthread(user_id, file_ids):
        seen_projthread["args"] = (user_id, list(file_ids))

    p = _Patcher()
    p.set(folders, "get_supabase", lambda: sb)
    p.set(folders, "_delete_projects_threads_and_jobs", fake_projthread)
    # delete_object is imported lazily inside the endpoint from app.services.r2.
    from app.services import r2 as r2mod
    p.set(r2mod, "delete_object", lambda key: deleted_keys.append(key))

    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.delete("/api/folders/root")
    finally:
        p.restore()
        _clear_overrides()

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}

    # File rows deleted first, scoped to the collected file ids + user.
    file_deletes = [d for d in sb.deletes if d["table"] == "files"]
    assert len(file_deletes) == 1
    assert set(file_deletes[0]["ins"]["id"]) == {"f1", "f2"}
    assert file_deletes[0]["eqs"]["user_id"] == "user-1"

    # Projects/threads/jobs purge received exactly the deleted file ids.
    assert seen_projthread["args"][0] == "user-1"
    assert set(seen_projthread["args"][1]) == {"f1", "f2"}

    # Every non-null R2 key removed.
    assert set(deleted_keys) == {"raw/f1", "prox/f1", "thumb/f1", "raw/f2", "a/f2", "b/f2"}

    # Both folder rows deleted, deepest (sub) before root.
    folder_deletes = [d for d in sb.deletes if d["table"] == "folders"]
    order = [d["eqs"]["id"] for d in folder_deletes]
    assert order == ["sub", "root"], order
    for d in folder_deletes:
        assert d["eqs"]["user_id"] == "user-1"

    print("ok  test_delete_folder_full_cascade")


def main():
    test_delete_folder_404s_when_not_owned()
    test_descendant_folder_ids_walks_the_whole_subtree()
    test_delete_folder_full_cascade()
    print("\nall folders-router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
