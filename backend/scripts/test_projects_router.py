"""
Smoke tests for the cuts-v3 project API (``app.routers.projects``) -- a real
FastAPI TestClient against the actual app, with every DB-touching service
call monkeypatched. No real Postgres, no real ingest, no real R2.

Run:  .venv/bin/python scripts/test_projects_router.py
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
from app.routers import projects  # noqa: E402


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


def test_create_project_returns_project_id():
    p = _Patcher()
    p.set(projects, "find_or_create_project", lambda user_id, file_ids: "proj-123")
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects", json={"file_ids": ["f1", "f2"]})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"project_id": "proj-123"}
    print("ok  test_create_project_returns_project_id")


def test_create_project_rejects_empty_file_ids():
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects", json={"file_ids": []})
    finally:
        _clear_overrides()
    assert resp.status_code == 422, resp.text
    print("ok  test_create_project_rejects_empty_file_ids")


def test_kick_ingest_enqueues_and_returns_queued():
    calls = []
    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(projects, "defer_ingest", lambda project_id: calls.append(project_id))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/proj-123/ingest")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"project_id": "proj-123", "status": "queued"}
    assert calls == ["proj-123"]
    print("ok  test_kick_ingest_enqueues_and_returns_queued")


def test_kick_ingest_404s_on_unowned_project():
    def not_found(project_id, user_id):
        raise HTTPException(status_code=404, detail="project not found")

    p = _Patcher()
    p.set(projects, "_owned_project", not_found)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/missing/ingest")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_kick_ingest_404s_on_unowned_project")


def test_kick_ingest_503s_when_enqueue_fails():
    def boom(project_id):
        raise RuntimeError("queue unavailable")

    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(projects, "defer_ingest", boom)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/proj-123/ingest")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 503, resp.text
    print("ok  test_kick_ingest_503s_when_enqueue_fails")


def test_get_cuts_returns_result():
    fake_result = {"project_id": "proj-123", "name": "x", "ingest_run": None, "cuts": []}
    p = _Patcher()
    p.set(projects.read, "load_cuts", lambda project_id, user_id: fake_result)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/projects/proj-123/cuts")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json() == fake_result
    print("ok  test_get_cuts_returns_result")


def test_get_cuts_404s_when_not_found():
    p = _Patcher()
    p.set(projects.read, "load_cuts", lambda project_id, user_id: None)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/projects/missing/cuts")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_get_cuts_404s_when_not_found")


def main():
    test_create_project_returns_project_id()
    test_create_project_rejects_empty_file_ids()
    test_kick_ingest_enqueues_and_returns_queued()
    test_kick_ingest_404s_on_unowned_project()
    test_kick_ingest_503s_when_enqueue_fails()
    test_get_cuts_returns_result()
    test_get_cuts_404s_when_not_found()
    print("\nall projects-router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
