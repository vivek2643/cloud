"""
Tests for frontend_project_ux.plan.md Stage 2.3's anonymous upload-link
router (app/routers/upload_links.py). No real Postgres, no real R2, no real
procrastinate/DB connector.

Covers: the Postgres-timestamp fractional-seconds parsing workaround; the
shared `_resolve_link` guard's four branches (unknown token, revoked,
expired, quota-exhausted); the owner-facing create/list/revoke endpoints via
a real FastAPI TestClient (ownership checks, soft-delete); that the public
info endpoint leaks nothing beyond {project_name, expires_at, remaining};
that `used_count` increments only at completion, never at presign; and that
the public completion routes pass the link's own folder_id through to the
upload-core functions (the cross-tenant-write guard from upload.py's
_get_owned_file).

Run:  .venv/bin/python scripts/test_upload_links_router.py
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import get_current_user_id  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.models.schemas import (  # noqa: E402
    PublicMultipartCompleteRequest,
    PublicPresignRequest,
)
from app.routers import upload_links  # noqa: E402


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


# --- Generic fake Supabase (same shape as test_upload_router.py's) ----------

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
        self.order_col = None
        self.order_desc = False

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

    def order(self, col, desc=False):
        self.order_col = col
        self.order_desc = desc
        return self

    def execute(self):
        rows = self.sb.tables.setdefault(self.table_name, [])
        if self.op == "insert":
            row = dict(self.payload)
            row.setdefault("id", str(uuid.uuid4()))
            if self.table_name == "upload_links":
                row.setdefault("used_count", 0)
                row.setdefault("revoked", False)
                row.setdefault("created_at", "2026-01-01T00:00:00+00:00")
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

        if self.order_col:
            matches = sorted(matches, key=lambda r: r.get(self.order_col), reverse=self.order_desc)
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


# --- _parse_pg_timestamptz ---------------------------------------------------

def test_parse_pg_timestamptz_handles_variable_precision():
    # The exact format observed live against this codebase's own Supabase
    # project -- Python 3.9's datetime.fromisoformat rejects this outright
    # (it demands exactly 3 or 6 fractional digits).
    dt = upload_links._parse_pg_timestamptz("2026-06-08T20:31:50.25913+00:00")
    assert dt.year == 2026 and dt.month == 6 and dt.day == 8
    assert dt.microsecond == 259130

    # No fractional part at all.
    dt = upload_links._parse_pg_timestamptz("2026-06-08T20:31:50+00:00")
    assert dt.microsecond == 0

    # Exactly 6 digits (already valid isoformat) passes through unchanged.
    dt = upload_links._parse_pg_timestamptz("2026-06-08T20:31:50.123456+00:00")
    assert dt.microsecond == 123456

    # 1 digit, padded.
    dt = upload_links._parse_pg_timestamptz("2026-06-08T20:31:50.5+00:00")
    assert dt.microsecond == 500000

    # 9 digits (nanosecond-style), truncated to microseconds.
    dt = upload_links._parse_pg_timestamptz("2026-06-08T20:31:50.123456789+00:00")
    assert dt.microsecond == 123456
    print("ok  test_parse_pg_timestamptz_handles_variable_precision")


# --- _resolve_link ------------------------------------------------------------

def test_resolve_link_404s_on_unknown_token():
    sb = _FakeSupabase()
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        try:
            upload_links._resolve_link("no-such-token")
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 404
    finally:
        p.restore()
    print("ok  test_resolve_link_404s_on_unknown_token")


def test_resolve_link_410s_when_revoked():
    sb = _FakeSupabase()
    sb.seed("upload_links", [{
        "id": "l1", "token": "tok-revoked", "folder_id": "f1", "user_id": "u1",
        "expires_at": None, "max_files": None, "used_count": 0, "revoked": True,
    }])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        try:
            upload_links._resolve_link("tok-revoked")
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 410
    finally:
        p.restore()
    print("ok  test_resolve_link_410s_when_revoked")


def test_resolve_link_410s_when_expired():
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    sb = _FakeSupabase()
    sb.seed("upload_links", [{
        "id": "l1", "token": "tok-expired", "folder_id": "f1", "user_id": "u1",
        "expires_at": past, "max_files": None, "used_count": 0, "revoked": False,
    }])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        try:
            upload_links._resolve_link("tok-expired")
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 410
    finally:
        p.restore()
    print("ok  test_resolve_link_410s_when_expired")


def test_resolve_link_410s_when_quota_exhausted():
    sb = _FakeSupabase()
    sb.seed("upload_links", [{
        "id": "l1", "token": "tok-full", "folder_id": "f1", "user_id": "u1",
        "expires_at": None, "max_files": 2, "used_count": 2, "revoked": False,
    }])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        try:
            upload_links._resolve_link("tok-full")
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 410
    finally:
        p.restore()
    print("ok  test_resolve_link_410s_when_quota_exhausted")


def test_resolve_link_passes_when_valid():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    sb = _FakeSupabase()
    sb.seed("upload_links", [{
        "id": "l1", "token": "tok-ok", "folder_id": "f1", "user_id": "u1",
        "expires_at": future, "max_files": 5, "used_count": 2, "revoked": False,
    }])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        link = upload_links._resolve_link("tok-ok")
    finally:
        p.restore()
    assert link["id"] == "l1"
    print("ok  test_resolve_link_passes_when_valid")


# --- Owner-facing router (TestClient) ----------------------------------------

def test_create_upload_link_404s_on_unowned_folder():
    sb = _FakeSupabase()
    sb.seed("folders", [{"id": "f1", "user_id": "someone-else"}])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/folders/f1/upload-links", json={})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    assert sb.inserts == []
    print("ok  test_create_upload_link_404s_on_unowned_folder")


def test_create_upload_link_generates_unguessable_token():
    sb = _FakeSupabase()
    sb.seed("folders", [{"id": "f1", "user_id": "user-1"}])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post(
            "/api/folders/f1/upload-links", json={"expires_in_hours": 24, "max_files": 10}
        )
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["token"]) > 32  # token_urlsafe(32) -- 256 bits, base64url
    assert body["folder_id"] == "f1"
    assert body["max_files"] == 10
    assert body["used_count"] == 0
    assert body["revoked"] is False
    row = sb.inserts[0]["row"]
    assert row["user_id"] == "user-1"  # owner, not the (nonexistent here) uploader
    print("ok  test_create_upload_link_generates_unguessable_token")


def test_list_upload_links_scoped_to_owner_and_folder():
    sb = _FakeSupabase()
    sb.seed("folders", [{"id": "f1", "user_id": "user-1"}])
    sb.seed("upload_links", [
        {
            "id": "l1", "token": "tok-1", "folder_id": "f1", "user_id": "user-1",
            "expires_at": None, "max_files": None, "used_count": 0, "revoked": False,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "l2", "token": "tok-2", "folder_id": "f1", "user_id": "someone-else",
            "expires_at": None, "max_files": None, "used_count": 0, "revoked": False,
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    ])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/folders/f1/upload-links")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    ids = [row["id"] for row in resp.json()]
    assert ids == ["l1"]  # only this owner's link, not the other user's
    print("ok  test_list_upload_links_scoped_to_owner_and_folder")


def test_revoke_upload_link_soft_deletes_and_checks_ownership():
    sb = _FakeSupabase()
    sb.seed("upload_links", [{
        "id": "l1", "token": "tok-1", "folder_id": "f1", "user_id": "user-1",
        "expires_at": None, "max_files": None, "used_count": 0, "revoked": False,
        "created_at": "2026-01-01T00:00:00+00:00",
    }])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    _as_user("someone-else")
    try:
        client = TestClient(fastapi_app)
        resp_wrong_owner = client.delete("/api/upload-links/l1")
    finally:
        p.restore()
        _clear_overrides()
    assert resp_wrong_owner.status_code == 404, resp_wrong_owner.text
    assert sb.tables["upload_links"][0]["revoked"] is False  # untouched

    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.delete("/api/upload-links/l1")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    assert sb.tables["upload_links"][0]["revoked"] is True  # soft delete, row still present
    print("ok  test_revoke_upload_link_soft_deletes_and_checks_ownership")


# --- Public info endpoint leaks nothing beyond the documented fields --------

def test_public_link_info_hides_owner_and_folder():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    sb = _FakeSupabase()
    sb.seed("folders", [{"id": "f1", "user_id": "user-1", "name": "My Project"}])
    sb.seed("upload_links", [{
        "id": "l1", "token": "tok-ok", "folder_id": "f1", "user_id": "user-1",
        "expires_at": future, "max_files": 5, "used_count": 2, "revoked": False,
    }])
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/public/upload-links/tok-ok")
    finally:
        p.restore()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"project_name", "expires_at", "remaining"}
    assert body["project_name"] == "My Project"
    assert body["remaining"] == 3  # max_files(5) - used_count(2)
    print("ok  test_public_link_info_hides_owner_and_folder")


def test_public_link_info_404s_on_unknown_token():
    sb = _FakeSupabase()
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/public/upload-links/does-not-exist")
    finally:
        p.restore()
    assert resp.status_code == 404, resp.text
    print("ok  test_public_link_info_404s_on_unknown_token")


# --- used_count increments at completion only, never at presign ------------

def _seed_valid_link(sb, **overrides):
    row = {
        "id": "l1", "token": "tok-ok", "folder_id": "f1", "user_id": "user-1",
        "expires_at": None, "max_files": 5, "used_count": 0, "revoked": False,
    }
    row.update(overrides)
    sb.seed("upload_links", [row])
    return row


def test_presign_does_not_increment_used_count():
    sb = _FakeSupabase()
    _seed_valid_link(sb)
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    p.set(upload_links, "presign_core", lambda *a, **k: {"file_id": "file-1", "upload_url": "https://signed"})
    try:
        upload_links.public_presign(
            "tok-ok", PublicPresignRequest(filename="clip.mp4", content_type="video/mp4", file_size=1000)
        )
    finally:
        p.restore()
    assert sb.tables["upload_links"][0]["used_count"] == 0
    print("ok  test_presign_does_not_increment_used_count")


def test_complete_increments_used_count_and_scopes_folder():
    sb = _FakeSupabase()
    _seed_valid_link(sb, used_count=1)
    seen = {}

    def fake_multipart_complete_core(user_id, file_id, upload_id, *, folder_id=None):
        seen["args"] = (user_id, file_id, upload_id, folder_id)
        return {"id": file_id, "status": "processing"}

    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    p.set(upload_links, "multipart_complete_core", fake_multipart_complete_core)
    try:
        upload_links.public_multipart_complete(
            "tok-ok", PublicMultipartCompleteRequest(file_id="file-1", upload_id="mpu-1")
        )
    finally:
        p.restore()

    # user_id and folder_id came from the LINK ROW, never the client body.
    assert seen["args"] == ("user-1", "file-1", "mpu-1", "f1")
    assert sb.tables["upload_links"][0]["used_count"] == 2
    print("ok  test_complete_increments_used_count_and_scopes_folder")


def test_analysis_proxy_routes_do_not_increment_used_count():
    # Only the two file-creating completions (multipart/complete and
    # /complete) burn quota -- the analysis-proxy pair never should, since
    # they don't represent a new uploaded file.
    sb = _FakeSupabase()
    _seed_valid_link(sb, used_count=1)
    p = _Patcher()
    p.set(upload_links, "get_supabase", lambda: sb)
    p.set(upload_links, "presign_analysis_proxies_core", lambda *a, **k: {"proxy_a_url": "x", "proxy_a_key": "a", "proxy_b_url": "y", "proxy_b_key": "b"})
    p.set(upload_links, "complete_analysis_proxies_core", lambda *a, **k: {"id": "file-1"})
    try:
        upload_links.public_presign_analysis_proxies("tok-ok", "file-1")
        upload_links.public_complete_analysis_proxies("tok-ok", "file-1")
    finally:
        p.restore()
    assert sb.tables["upload_links"][0]["used_count"] == 1  # unchanged
    print("ok  test_analysis_proxy_routes_do_not_increment_used_count")


def main():
    test_parse_pg_timestamptz_handles_variable_precision()
    test_resolve_link_404s_on_unknown_token()
    test_resolve_link_410s_when_revoked()
    test_resolve_link_410s_when_expired()
    test_resolve_link_410s_when_quota_exhausted()
    test_resolve_link_passes_when_valid()
    test_create_upload_link_404s_on_unowned_folder()
    test_create_upload_link_generates_unguessable_token()
    test_list_upload_links_scoped_to_owner_and_folder()
    test_revoke_upload_link_soft_deletes_and_checks_ownership()
    test_public_link_info_hides_owner_and_folder()
    test_public_link_info_404s_on_unknown_token()
    test_presign_does_not_increment_used_count()
    test_complete_increments_used_count_and_scopes_folder()
    test_analysis_proxy_routes_do_not_increment_used_count()
    print("\nall upload-links-router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
