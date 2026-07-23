"""
Smoke tests for the exports API (``app.routers.exports``) -- a real FastAPI
TestClient against the actual app, with every DB-touching service call
monkeypatched. No real Postgres, no real export job, no real R2.

Run:  .venv/bin/python scripts/test_exports_router.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import get_current_user_id  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.routers import exports  # noqa: E402
from app.services.render import compositor as render_compositor  # noqa: E402
from app.services.render import tasks as render_tasks  # noqa: E402


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


_THREAD = {"id": "thread-1", "user_id": "user-1"}
_DOCUMENT = {"timeline": [{"seg_id": "s0", "file_id": "f1"}], "operations": []}


def _patch_dedup_noop(p: "_Patcher") -> None:
    """Pillar 5: create_export now resolves the document + hashes it to
    check for an existing done export before enqueuing. Patch that path to a
    no-op (never finds a match) so tests unrelated to dedup itself don't
    need a real resolve/hash."""
    p.set(render_tasks, "resolve_document", lambda document, thread_id=None: {})
    p.set(render_compositor, "resolved_hash", lambda resolved, preset: "hash-noop")
    p.set(exports.export_store, "find_done", lambda tid, v, kind, quality, rhash: None)


def test_create_export_enqueues_and_returns_queued_row():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.l3_store, "latest_document", lambda thread_id: (_DOCUMENT, 3))
    _patch_dedup_noop(p)
    p.set(exports.export_store, "create_export", lambda tid, v, kind, quality, media, rhash: {
        "id": "exp-1", "thread_id": tid, "document_version": v, "kind": kind,
        "quality": quality, "include_media": media, "resolved_hash": rhash, "status": "queued",
        "output_r2_key": None, "error": None, "created_at": None, "updated_at": None,
    })
    calls = []
    p.set(exports, "_enqueue", lambda export_id: calls.append(export_id) or True)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post(
            "/api/edit/threads/thread-1/export",
            json={"kind": "mp4", "quality": "1080", "include_media": False},
        )
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert body["kind"] == "mp4" and body["quality"] == "1080"
    assert calls == ["exp-1"]
    print("ok  test_create_export_enqueues_and_returns_queued_row")


def test_create_export_short_circuits_to_existing_done_export():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.l3_store, "latest_document", lambda thread_id: (_DOCUMENT, 3))
    p.set(render_tasks, "resolve_document", lambda document, thread_id=None: {})
    p.set(render_compositor, "resolved_hash", lambda resolved, preset: "hash-match")
    done_row = {
        "id": "exp-done", "thread_id": "thread-1", "document_version": 3, "kind": "mp4",
        "quality": "1080", "include_media": False, "resolved_hash": "hash-match", "status": "done",
        "output_r2_key": "exports/existing.mp4", "error": None, "created_at": None, "updated_at": None,
    }
    p.set(exports.export_store, "find_done", lambda tid, v, kind, quality, rhash: done_row)
    create_calls = []
    p.set(exports.export_store, "create_export", lambda *a, **kw: create_calls.append((a, kw)) or {})
    enqueue_calls = []
    p.set(exports, "_enqueue", lambda export_id: enqueue_calls.append(export_id) or True)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post(
            "/api/edit/threads/thread-1/export",
            json={"kind": "mp4", "quality": "1080", "include_media": False},
        )
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == "exp-done"
    assert create_calls == [], "should not create a new row when an identical done export exists"
    assert enqueue_calls == [], "should not enqueue when short-circuiting to an existing export"
    print("ok  test_create_export_short_circuits_to_existing_done_export")


def test_create_export_rejects_unknown_kind():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/edit/threads/thread-1/export", json={"kind": "avi"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 400, resp.text
    print("ok  test_create_export_rejects_unknown_kind")


def test_create_export_rejects_unknown_quality():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/edit/threads/thread-1/export", json={"kind": "mp4", "quality": "8k"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 400, resp.text
    print("ok  test_create_export_rejects_unknown_quality")


def test_create_export_404s_on_unowned_thread():
    def not_found(thread_id):
        return None

    p = _Patcher()
    p.set(exports.l3_store, "get_thread", not_found)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/edit/threads/missing/export", json={"kind": "srt"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_create_export_404s_on_unowned_thread")


def test_create_export_409s_with_no_document():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.l3_store, "latest_document", lambda thread_id: (None, 0))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/edit/threads/thread-1/export", json={"kind": "srt"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 409, resp.text
    print("ok  test_create_export_409s_with_no_document")


def test_create_export_409s_when_document_has_no_timeline():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.l3_store, "latest_document", lambda thread_id: ({"timeline": [], "operations": []}, 1))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/edit/threads/thread-1/export", json={"kind": "srt"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 409, resp.text
    print("ok  test_create_export_409s_when_document_has_no_timeline")


def test_create_export_marks_failed_when_enqueue_fails():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.l3_store, "latest_document", lambda thread_id: (_DOCUMENT, 3))
    _patch_dedup_noop(p)
    p.set(exports.export_store, "create_export", lambda tid, v, kind, quality, media, rhash: {
        "id": "exp-2", "thread_id": tid, "document_version": v, "kind": kind,
        "quality": quality, "include_media": media, "resolved_hash": rhash, "status": "queued",
        "output_r2_key": None, "error": None, "created_at": None, "updated_at": None,
    })
    updates = []
    p.set(exports.export_store, "update_status", lambda export_id, **kw: updates.append((export_id, kw)))
    p.set(exports.export_store, "get_export", lambda export_id: {
        "id": export_id, "thread_id": "thread-1", "document_version": 3, "kind": "mp4",
        "quality": "1080", "include_media": False, "status": "failed",
        "output_r2_key": None, "error": "Worker unavailable.", "created_at": None, "updated_at": None,
    })
    p.set(exports, "_enqueue", lambda export_id: False)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/edit/threads/thread-1/export", json={"kind": "mp4"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "failed"
    assert updates and updates[0][1].get("status") == "failed"
    print("ok  test_create_export_marks_failed_when_enqueue_fails")


def test_get_export_returns_presigned_url_when_done():
    p = _Patcher()
    p.set(exports.export_store, "get_export", lambda export_id: {
        "id": export_id, "thread_id": "thread-1", "document_version": 1, "kind": "srt",
        "quality": "1080", "include_media": False, "status": "done",
        "output_r2_key": "exports/abc.srt", "error": None, "created_at": None, "updated_at": None,
    })
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.bundle, "presigned_url_for", lambda key, expires_in=86400: f"https://example.invalid/{key}")
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/exports/exp-3")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "done"
    assert body["output_url"] == "https://example.invalid/exports/abc.srt"
    print("ok  test_get_export_returns_presigned_url_when_done")


def test_get_export_404s_when_not_found():
    p = _Patcher()
    p.set(exports.export_store, "get_export", lambda export_id: None)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/exports/missing")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_get_export_404s_when_not_found")


def test_get_export_404s_on_unowned_thread():
    p = _Patcher()
    p.set(exports.export_store, "get_export", lambda export_id: {
        "id": export_id, "thread_id": "someone-elses-thread", "document_version": 1, "kind": "srt",
        "quality": "1080", "include_media": False, "status": "done",
        "output_r2_key": None, "error": None, "created_at": None, "updated_at": None,
    })

    def not_owned(thread_id, user_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    p.set(exports, "_owned_thread", not_owned)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/exports/exp-4")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_get_export_404s_on_unowned_thread")


def test_list_exports_returns_all_for_thread():
    p = _Patcher()
    p.set(exports.l3_store, "get_thread", lambda thread_id: dict(_THREAD, id=thread_id))
    p.set(exports.export_store, "list_for_thread", lambda thread_id: [
        {"id": "exp-a", "thread_id": thread_id, "document_version": 1, "kind": "srt",
         "quality": "1080", "include_media": False, "status": "done",
         "output_r2_key": None, "error": None, "created_at": None, "updated_at": None},
        {"id": "exp-b", "thread_id": thread_id, "document_version": 2, "kind": "mp4",
         "quality": "2160", "include_media": False, "status": "queued",
         "output_r2_key": None, "error": None, "created_at": None, "updated_at": None},
    ])
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/edit/threads/thread-1/exports")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    ids = [e["id"] for e in resp.json()["exports"]]
    assert ids == ["exp-a", "exp-b"], ids
    print("ok  test_list_exports_returns_all_for_thread")


def main():
    test_create_export_enqueues_and_returns_queued_row()
    test_create_export_short_circuits_to_existing_done_export()
    test_create_export_rejects_unknown_kind()
    test_create_export_rejects_unknown_quality()
    test_create_export_404s_on_unowned_thread()
    test_create_export_409s_with_no_document()
    test_create_export_409s_when_document_has_no_timeline()
    test_create_export_marks_failed_when_enqueue_fails()
    test_get_export_returns_presigned_url_when_done()
    test_get_export_404s_when_not_found()
    test_get_export_404s_on_unowned_thread()
    test_list_exports_returns_all_for_thread()
    print("\nall exports-router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
