"""
Smoke tests for the cuts-v3 project API (``app.routers.projects``) -- a real
FastAPI TestClient against the actual app, with every DB-touching service
call monkeypatched. No real Postgres, no real ingest, no real R2.

Run:  .venv/bin/python scripts/test_projects_router.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import get_current_user_id  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.routers import projects  # noqa: E402
from app.services.vcut import orchestrate  # noqa: E402
from app.services.vcut.resolve import ResolvedCut  # noqa: E402


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
    # cuts_pipeline defaults to "vcut" (config.py), so kick_ingest dispatches
    # to app.services.vcut.orchestrate.defer_vcut_ingest -- mock that, not the
    # old v3 defer_ingest, so no real fairness/DB UUID-cast runs.
    calls = []
    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(orchestrate, "defer_vcut_ingest", lambda project_id, user_id: calls.append((project_id, user_id)))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/proj-123/ingest")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"project_id": "proj-123", "status": "queued"}
    assert calls == [("proj-123", "user-1")]
    print("ok  test_kick_ingest_enqueues_and_returns_queued")


def test_kick_ingest_treats_already_enqueued_as_a_noop():
    """scale_architecture.plan.md Pillar 5: defer_ingest's queueing_lock
    raises AlreadyEnqueued when a run for this project is already pending --
    a double-click, not a failure."""
    from procrastinate.exceptions import AlreadyEnqueued

    def already(project_id, user_id):
        raise AlreadyEnqueued("already queued")

    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(orchestrate, "defer_vcut_ingest", already)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/proj-123/ingest")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"project_id": "proj-123", "status": "queued"}
    print("ok  test_kick_ingest_treats_already_enqueued_as_a_noop")


def test_kick_ingest_429s_when_over_capacity():
    """scale_architecture.plan.md Pillar 6: defer_ingest raises
    CapacityExceeded when this user already has max_inflight_ingest_runs_
    per_user runs going -- a real, user-visible limit (429), not a 503."""
    from app.services.fairness import CapacityExceeded

    def over_capacity(project_id, user_id):
        raise CapacityExceeded(user_id, in_flight=5, max_inflight=5)

    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(orchestrate, "defer_vcut_ingest", over_capacity)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/proj-123/ingest")
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 429, resp.text
    print("ok  test_kick_ingest_429s_when_over_capacity")


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
    def boom(project_id, user_id):
        raise RuntimeError("queue unavailable")

    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(orchestrate, "defer_vcut_ingest", boom)
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


# --------------------------------------------------------------------------
# cuts/energy, cuts/energy_levels -- vcut_pass2_video_specifics.plan.md
# section 7.4: the hero-containment snapshot/re-map band-aid is retired.
# resolve_cuts's own composed specifics (section 7) make a plain resolve ->
# insert_video_cuts already yield specifics-bearing rows, at any energy.
# --------------------------------------------------------------------------

def test_snapshot_video_specifics_band_aid_is_fully_retired():
    assert not hasattr(projects, "_snapshot_video_specifics")
    print("ok  test_snapshot_video_specifics_band_aid_is_fully_retired")


def test_set_cuts_energy_resolves_and_reinserts_without_snapshot_remap():
    loose_plan_dict = {"f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x"}]}}
    seam_cache = {"f1": {"hop_ms": 100, "S": [1.0]}}
    fake_resolved = [ResolvedCut(file_id="f1", in_ms=0, out_ms=1000, peak_ms=500,
                                 tag="both", summary="x", specifics={"subject": "a dog"})]
    fake_result = {"project_id": "proj-123", "cuts": [{"id": "c1"}]}

    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(projects, "_latest_run_id", lambda project_id: "run-1")
    p.set(projects.read, "load_cuts", lambda project_id, user_id: fake_result)
    _as_user("user-1")
    try:
        with mock.patch("app.services.vcut.store.load_seam_and_plan",
                        return_value=(seam_cache, loose_plan_dict)) as seam_mock, \
             mock.patch("app.services.vcut.resolve.resolve_cuts", return_value=fake_resolved) as resolve_mock, \
             mock.patch("app.services.vcut.store.insert_video_cuts", return_value=["c1"]) as insert_mock:
            client = TestClient(fastapi_app)
            resp = client.post("/api/projects/proj-123/cuts/energy", json={"energy": 0.5})
    finally:
        p.restore()
        _clear_overrides()

    assert resp.status_code == 200, resp.text
    assert resp.json() == fake_result
    seam_mock.assert_called_once_with("run-1")
    assert resolve_mock.call_args.kwargs["energy"] == 0.5
    insert_mock.assert_called_once_with("run-1", fake_resolved, seam_cache)
    print("ok  test_set_cuts_energy_resolves_and_reinserts_without_snapshot_remap")


def test_set_cuts_energy_404s_with_no_ingest_run():
    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(projects, "_latest_run_id", lambda project_id: None)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/projects/proj-123/cuts/energy", json={"energy": 0.5})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_set_cuts_energy_404s_with_no_ingest_run")


def test_set_cuts_energy_409s_with_no_vcut_artifacts():
    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(projects, "_latest_run_id", lambda project_id: "run-1")
    _as_user("user-1")
    try:
        with mock.patch("app.services.vcut.store.load_seam_and_plan", return_value=({}, {})):
            client = TestClient(fastapi_app)
            resp = client.post("/api/projects/proj-123/cuts/energy", json={"energy": 0.5})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 409, resp.text
    print("ok  test_set_cuts_energy_409s_with_no_vcut_artifacts")


def test_energy_levels_resolves_all_five_stops_plus_final_energy_zero():
    loose_plan_dict = {"f1": {"flags": [{"t_ms": 100, "shape": "both", "summary": "x"}]}}
    seam_cache = {"f1": {"hop_ms": 100, "S": [1.0]}}
    fake_resolved = [ResolvedCut(file_id="f1", in_ms=0, out_ms=1000, peak_ms=500, tag="both", summary="x")]
    fake_result = {"project_id": "proj-123", "cuts": []}

    p = _Patcher()
    p.set(projects, "_owned_project", lambda project_id, user_id: None)
    p.set(projects, "_latest_run_id", lambda project_id: "run-1")
    p.set(projects.read, "load_cuts", lambda project_id, user_id: fake_result)
    _as_user("user-1")
    try:
        with mock.patch("app.services.vcut.store.load_seam_and_plan",
                        return_value=(seam_cache, loose_plan_dict)), \
             mock.patch("app.services.vcut.resolve.resolve_cuts", return_value=fake_resolved) as resolve_mock, \
             mock.patch("app.services.vcut.store.insert_video_cuts", return_value=["c1"]):
            client = TestClient(fastapi_app)
            resp = client.get("/api/projects/proj-123/cuts/energy_levels")
    finally:
        p.restore()
        _clear_overrides()

    assert resp.status_code == 200, resp.text
    assert set(resp.json()["levels"].keys()) == {"0", "0.25", "0.5", "0.75", "1"}
    # 5 dial stops + 1 final "leave the DB at energy 0.0" re-resolve.
    energies = [c.kwargs["energy"] for c in resolve_mock.call_args_list]
    assert energies == [0.0, 0.25, 0.5, 0.75, 1.0, 0.0], energies
    print("ok  test_energy_levels_resolves_all_five_stops_plus_final_energy_zero")


def main():
    test_create_project_returns_project_id()
    test_create_project_rejects_empty_file_ids()
    test_kick_ingest_enqueues_and_returns_queued()
    test_kick_ingest_treats_already_enqueued_as_a_noop()
    test_kick_ingest_429s_when_over_capacity()
    test_kick_ingest_404s_on_unowned_project()
    test_kick_ingest_503s_when_enqueue_fails()
    test_get_cuts_returns_result()
    test_get_cuts_404s_when_not_found()
    test_snapshot_video_specifics_band_aid_is_fully_retired()
    test_set_cuts_energy_resolves_and_reinserts_without_snapshot_remap()
    test_set_cuts_energy_404s_with_no_ingest_run()
    test_set_cuts_energy_409s_with_no_vcut_artifacts()
    test_energy_levels_resolves_all_five_stops_plus_final_energy_zero()
    print("\nall projects-router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
