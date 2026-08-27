"""
Tests for frontend_project_ux.plan.md Stage 2.2's upload.py refactor: every
handler's body pulled into a plain `*_core(user_id, folder_id, ...)`
function so app/routers/upload_links.py's public routes can call the exact
same logic instead of forking it. No real Postgres, no real R2, no real
procrastinate/DB connector.

Covers: folder-ownership checks, the optional folder_id scoping
`_get_owned_file` adds for the public flow, the status-guard on
multipart/plain completion, `_finalize_upload`'s branch between the client
fast-path and the full-L1 fallback, and a regression test for the
`_enqueue_l1` missing-user_id bug found and fixed while extracting
`complete_analysis_proxies_core`.

Run:  .venv/bin/python scripts/test_upload_router.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from fastapi import HTTPException  # noqa: E402

from app.routers import upload  # noqa: E402


class _Patcher:
    def __init__(self):
        self._orig = {}

    def set(self, obj, name, value):
        self._orig[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def restore(self):
        for (obj, name), value in self._orig.items():
            setattr(obj, name, value)


# --- Generic fake Supabase: select/insert/update/delete over in-memory tables.
# upload.py touches "folders" (ownership check) and "files" (everything else),
# each via .select/.insert/.update/.delete + .eq() chains only -- no .in_/.order
# needed here, unlike upload_links.py's list endpoint.

class _Result:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, sb, table):
        self.sb = sb
        self.table_name = table
        self.op = "select"
        self.payload = None
        self.eqs = {}

    def select(self, *_a, **_k):
        self.op = "select"
        return self

    def insert(self, payload):
        self.op = "insert"
        self.payload = dict(payload)
        return self

    def update(self, payload):
        self.op = "update"
        self.payload = dict(payload)
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, col, val):
        self.eqs[col] = val
        return self

    def execute(self):
        rows = self.sb.tables.setdefault(self.table_name, [])
        if self.op == "insert":
            row = dict(self.payload)
            rows.append(row)
            self.sb.inserts.append({"table": self.table_name, "row": dict(row)})
            return _Result([row])

        matches = [r for r in rows if all(r.get(k) == v for k, v in self.eqs.items())]

        if self.op == "update":
            for r in matches:
                r.update(self.payload)
            self.sb.updates.append(
                {"table": self.table_name, "eqs": dict(self.eqs), "payload": dict(self.payload)}
            )
            return _Result(matches)

        if self.op == "delete":
            self.sb.tables[self.table_name] = [r for r in rows if r not in matches]
            self.sb.deletes.append({"table": self.table_name, "eqs": dict(self.eqs)})
            return _Result(matches)

        return _Result(matches)


class _FakeSupabase:
    def __init__(self):
        self.tables = {}
        self.inserts = []
        self.updates = []
        self.deletes = []

    def seed(self, table, rows):
        self.tables[table] = [dict(r) for r in rows]

    def table(self, name):
        return _FakeQuery(self, name)


def _patch_r2(p: _Patcher):
    p.set(upload, "generate_presigned_put", lambda key, content_type: f"https://signed/{key}")
    p.set(upload, "create_multipart_upload", lambda key, content_type: "mpu-1")
    p.set(upload, "part_size_for", lambda file_size: 8 * 1024 * 1024)
    p.set(upload, "generate_presigned_upload_parts", lambda key, upload_id, count: [f"part-{i}" for i in range(count)])
    p.set(upload, "complete_multipart_upload", lambda key, upload_id: None)
    p.set(upload, "abort_multipart_upload", lambda key, upload_id: None)


def test_detect_file_type():
    assert upload._detect_file_type("video/mp4") == "video"
    assert upload._detect_file_type("image/png") == "image"
    assert upload._detect_file_type("audio/wav") == "audio"
    assert upload._detect_file_type("application/pdf") == "document"
    assert upload._detect_file_type("text/plain") == "document"
    assert upload._detect_file_type("application/zip") == "other"
    print("ok  test_detect_file_type")


def test_presign_core_404s_on_unowned_folder():
    sb = _FakeSupabase()
    sb.seed("folders", [{"id": "f1", "user_id": "someone-else"}])
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    _patch_r2(p)
    try:
        try:
            upload.presign_core("user-1", "f1", "clip.mp4", "video/mp4", 1000)
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 404
    finally:
        p.restore()
    assert sb.inserts == []
    print("ok  test_presign_core_404s_on_unowned_folder")


def test_presign_core_inserts_and_signs():
    sb = _FakeSupabase()
    sb.seed("folders", [{"id": "f1", "user_id": "user-1"}])
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    _patch_r2(p)
    try:
        resp = upload.presign_core("user-1", "f1", "clip.mp4", "video/mp4", 1000)
    finally:
        p.restore()
    assert len(sb.inserts) == 1
    row = sb.inserts[0]["row"]
    assert row["user_id"] == "user-1"
    assert row["folder_id"] == "f1"
    assert row["status"] == "uploading"
    assert row["r2_key"] == f"raw/user-1/{resp.file_id}/clip.mp4"
    assert resp.upload_url == f"https://signed/{row['r2_key']}"
    print("ok  test_presign_core_inserts_and_signs")


def test_presign_core_allows_no_folder():
    # folder_id=None (root-level upload): _check_folder_ownership is a no-op.
    sb = _FakeSupabase()
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    _patch_r2(p)
    try:
        resp = upload.presign_core("user-1", None, "clip.mp4", "video/mp4", 1000)
    finally:
        p.restore()
    assert resp.file_id
    assert sb.inserts[0]["row"]["folder_id"] is None
    print("ok  test_presign_core_allows_no_folder")


def test_get_owned_file_scopes_to_folder_id():
    # frontend_project_ux.plan.md Stage 2.3's safety rule: a public route's
    # file_id must belong to THIS link's folder, not just to the same user --
    # otherwise a link for project A becomes a write primitive against any
    # file id in project B (same owner, different folder).
    sb = _FakeSupabase()
    sb.seed("files", [{"id": "file-1", "user_id": "user-1", "folder_id": "folder-A", "status": "uploading"}])

    # Authenticated-style call (no folder_id constraint): succeeds.
    row = upload._get_owned_file(sb, "file-1", "user-1")
    assert row["id"] == "file-1"

    # Public-style call scoped to the CORRECT folder: succeeds.
    row = upload._get_owned_file(sb, "file-1", "user-1", folder_id="folder-A")
    assert row["id"] == "file-1"

    # Public-style call scoped to a DIFFERENT folder owned by the same user: 404s.
    try:
        upload._get_owned_file(sb, "file-1", "user-1", folder_id="folder-B")
        assert False, "expected HTTPException"
    except HTTPException as e:
        assert e.status_code == 404
    print("ok  test_get_owned_file_scopes_to_folder_id")


def test_multipart_complete_core_rejects_wrong_status():
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "folder_id": "folder-A",
        "status": "ready", "r2_key": "raw/user-1/file-1/clip.mp4",
    }])
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    _patch_r2(p)
    try:
        try:
            upload.multipart_complete_core("user-1", "file-1", "mpu-1")
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 400
    finally:
        p.restore()
    print("ok  test_multipart_complete_core_rejects_wrong_status")


def test_multipart_abort_core_deletes_placeholder_row():
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "folder_id": "folder-A",
        "status": "uploading", "r2_key": "raw/user-1/file-1/clip.mp4",
    }])
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    _patch_r2(p)
    try:
        result = upload.multipart_abort_core("user-1", "file-1", "mpu-1")
    finally:
        p.restore()
    assert result == {"ok": True}
    assert sb.tables["files"] == []
    print("ok  test_multipart_abort_core_deletes_placeholder_row")


def test_complete_analysis_proxies_core_enqueues_l1_with_user_id():
    # Regression test: complete_analysis_proxies previously called
    # _enqueue_l1(file_id, r2_key) with only 2 of its 3 required args, which
    # would raise TypeError on every real invocation. Fixed while extracting
    # this function for the Stage 2.2 refactor -- lock this in.
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "folder_id": "folder-A",
        "status": "uploading", "file_type": "video",
        "r2_key": "raw/user-1/file-1/clip.mp4",
    }])
    calls = []
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    p.set(upload, "_enqueue_l1", lambda *args: calls.append(args) or True)
    try:
        result = upload.complete_analysis_proxies_core("user-1", "file-1")
    finally:
        p.restore()
    assert len(calls) == 1
    assert calls[0] == ("file-1", "raw/user-1/file-1/clip.mp4", "user-1")
    assert result["r2_proxy_a_key"] == "proxies/file-1/analysis_a.mp4"
    assert result["r2_proxy_b_key"] == "proxies/file-1/motion_b.mp4"
    # Status must NOT be flipped here -- the raw is still uploading.
    assert sb.tables["files"][0]["status"] == "uploading"
    print("ok  test_complete_analysis_proxies_core_enqueues_l1_with_user_id")


def test_complete_analysis_proxies_core_rejects_non_video():
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "folder_id": "folder-A",
        "status": "uploading", "file_type": "audio",
        "r2_key": "raw/user-1/file-1/song.wav",
    }])
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    try:
        try:
            upload.complete_analysis_proxies_core("user-1", "file-1")
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 400
    finally:
        p.restore()
    print("ok  test_complete_analysis_proxies_core_rejects_non_video")


def test_finalize_upload_fast_path_uses_editing_proxy_task():
    # Client analysis-proxy fast path already ran (r2_proxy_a_key present):
    # the raw only needs the 1080p editing proxy, not a full L1 re-run.
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "file_type": "video",
        "r2_key": "raw/user-1/file-1/clip.mp4", "r2_proxy_a_key": "proxies/file-1/analysis_a.mp4",
    }])
    calls = []
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    p.set(upload, "_enqueue_task", lambda task_name, *rest: calls.append((task_name, *rest)) or True)
    p.set(upload, "_enqueue_l1", lambda *a: (_ for _ in ()).throw(AssertionError("should not run full L1")))
    try:
        result = upload._finalize_upload(sb, sb.tables["files"][0])
    finally:
        p.restore()
    assert result["status"] == "processing"
    assert calls == [("l1_editing_proxy", "file-1", "raw/user-1/file-1/clip.mp4", "user-1")]
    print("ok  test_finalize_upload_fast_path_uses_editing_proxy_task")


def test_finalize_upload_fallback_runs_full_l1():
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "file_type": "video",
        "r2_key": "raw/user-1/file-1/clip.mp4", "r2_proxy_a_key": None,
    }])
    calls = []
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    p.set(upload, "_enqueue_l1", lambda *args: calls.append(args) or True)
    try:
        result = upload._finalize_upload(sb, sb.tables["files"][0])
    finally:
        p.restore()
    assert result["status"] == "processing"
    assert calls == [("file-1", "raw/user-1/file-1/clip.mp4", "user-1")]
    print("ok  test_finalize_upload_fallback_runs_full_l1")


def test_finalize_upload_non_analyzable_skips_ingest():
    sb = _FakeSupabase()
    sb.seed("files", [{
        "id": "file-1", "user_id": "user-1", "file_type": "document",
        "r2_key": "raw/user-1/file-1/doc.pdf", "r2_proxy_a_key": None,
    }])
    calls = []
    p = _Patcher()
    p.set(upload, "get_supabase", lambda: sb)
    p.set(upload, "_enqueue_task", lambda *a: calls.append(a) or True)
    p.set(upload, "_enqueue_l1", lambda *a: calls.append(a) or True)
    try:
        result = upload._finalize_upload(sb, sb.tables["files"][0])
    finally:
        p.restore()
    assert result["status"] == "ready"
    assert calls == []
    print("ok  test_finalize_upload_non_analyzable_skips_ingest")


def main():
    test_detect_file_type()
    test_presign_core_404s_on_unowned_folder()
    test_presign_core_inserts_and_signs()
    test_presign_core_allows_no_folder()
    test_get_owned_file_scopes_to_folder_id()
    test_multipart_complete_core_rejects_wrong_status()
    test_multipart_abort_core_deletes_placeholder_row()
    test_complete_analysis_proxies_core_enqueues_l1_with_user_id()
    test_complete_analysis_proxies_core_rejects_non_video()
    test_finalize_upload_fast_path_uses_editing_proxy_task()
    test_finalize_upload_fallback_runs_full_l1()
    test_finalize_upload_non_analyzable_skips_ingest()
    print("\nall upload-router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
