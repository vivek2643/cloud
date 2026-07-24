"""
Tests for the RunPod serverless bridge + the GPU_EXECUTION task guards
(deployment.plan.md Phase 2). NO real network, NO real DB/GPU: httpx and
get_settings are faked so what's under test is the FORWARDING logic and the
guard branch, not RunPod itself.

Run:  .venv/bin/python scripts/test_runpod_bridge.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services import runpod_bridge as rb  # noqa: E402
from app.services.l1 import pipeline  # noqa: E402


class FakeSettings:
    def __init__(self, gpu_execution="runpod", endpoint="ep1", key="k1", timeout=900):
        self.gpu_execution = gpu_execution
        self.runpod_endpoint_id = endpoint
        self.runpod_api_key = key
        self.runpod_timeout_seconds = timeout


class _Resp:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class _FakeClient:
    """One client handles the single submit POST then the sequence of status
    GETs within run_remote's single `with httpx.Client()` block."""
    def __init__(self, submit_resp, status_resps):
        self._submit = submit_resp
        self._statuses = list(status_resps)
        self.posts = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.posts.append((url, json))
        return self._submit

    def get(self, url, headers=None):
        return self._statuses.pop(0)


class _FakeHttpx:
    def __init__(self, client):
        self._client = client

    def Client(self, *a, **k):
        return self._client


def _patch(monkeypatched, obj, attr, value):
    monkeypatched.append((obj, attr, getattr(obj, attr)))
    setattr(obj, attr, value)


def _restore(monkeypatched):
    for obj, attr, old in reversed(monkeypatched):
        setattr(obj, attr, old)


# --- run_remote -----------------------------------------------------------

def test_run_remote_completes_ok():
    mp = []
    try:
        _patch(mp, rb, "get_settings", lambda: FakeSettings())
        client = _FakeClient(
            submit_resp=_Resp({"id": "job-1"}),
            status_resps=[_Resp({"status": "IN_PROGRESS"}),
                          _Resp({"status": "COMPLETED", "output": {"ok": True}})],
        )
        _patch(mp, rb, "httpx", _FakeHttpx(client))
        _patch(mp, rb.time, "sleep", lambda *_: None)
        rb.run_remote("l1_orchestrate", file_id="f1", r2_key="k1")
        assert client.posts and client.posts[0][1]["input"]["task"] == "l1_orchestrate"
        assert client.posts[0][1]["input"]["kwargs"] == {"file_id": "f1", "r2_key": "k1"}
    finally:
        _restore(mp)


def test_run_remote_raises_on_failed():
    mp = []
    try:
        _patch(mp, rb, "get_settings", lambda: FakeSettings())
        client = _FakeClient(_Resp({"id": "job-2"}), [_Resp({"status": "FAILED"})])
        _patch(mp, rb, "httpx", _FakeHttpx(client))
        _patch(mp, rb.time, "sleep", lambda *_: None)
        raised = False
        try:
            rb.run_remote("l1_active_speaker", file_id="f1")
        except RuntimeError:
            raised = True
        assert raised, "run_remote must raise on a FAILED job"
    finally:
        _restore(mp)


def test_run_remote_raises_on_ok_false():
    mp = []
    try:
        _patch(mp, rb, "get_settings", lambda: FakeSettings())
        client = _FakeClient(
            _Resp({"id": "job-3"}),
            [_Resp({"status": "COMPLETED", "output": {"ok": False}})],
        )
        _patch(mp, rb, "httpx", _FakeHttpx(client))
        _patch(mp, rb.time, "sleep", lambda *_: None)
        raised = False
        try:
            rb.run_remote("l1_orchestrate", file_id="f1", r2_key="k1")
        except RuntimeError:
            raised = True
        assert raised, "run_remote must raise when handler output ok is false"
    finally:
        _restore(mp)


def test_warm_noop_when_local():
    mp = []
    try:
        # gpu_execution=local -> warm() must not touch httpx at all.
        _patch(mp, rb, "get_settings", lambda: FakeSettings(gpu_execution="local"))
        def _boom(*a, **k):
            raise AssertionError("warm() must not call httpx when local")
        _patch(mp, rb, "httpx", type("X", (), {"Client": staticmethod(_boom)}))
        rb.warm()  # no raise, no httpx use
    finally:
        _restore(mp)


# --- pipeline guards ------------------------------------------------------

def _func(task_obj):
    return getattr(task_obj, "func", task_obj)


def test_guard_forwards_when_runpod():
    mp = []
    try:
        _patch(mp, pipeline, "get_settings", lambda: FakeSettings(gpu_execution="runpod"))
        calls = []
        _patch(mp, rb, "run_remote", lambda task, **kw: calls.append((task, kw)))
        # If the guard leaks through, this would run real compute -- make it loud.
        _patch(mp, pipeline, "_l1_orchestrate",
               lambda *a, **k: (_ for _ in ()).throw(AssertionError("compute ran under runpod")))
        _func(pipeline.l1_orchestrate)(file_id="f1", r2_key="k1")
        assert calls == [("l1_orchestrate", {"file_id": "f1", "r2_key": "k1"})]
    finally:
        _restore(mp)


def test_guard_runs_local_when_local():
    mp = []
    try:
        _patch(mp, pipeline, "get_settings", lambda: FakeSettings(gpu_execution="local"))
        _patch(mp, pipeline, "_user_id_for_file", lambda _fid: None)
        ran = []
        _patch(mp, pipeline, "_l1_orchestrate", lambda fid, key: ran.append((fid, key)))
        # run_remote must NOT be called on the local path.
        _patch(mp, rb, "run_remote",
               lambda *a, **k: (_ for _ in ()).throw(AssertionError("forwarded under local")))
        _func(pipeline.l1_orchestrate)(file_id="f1", r2_key="k1")
        assert ran == [("f1", "k1")]
    finally:
        _restore(mp)


TESTS = [
    test_run_remote_completes_ok,
    test_run_remote_raises_on_failed,
    test_run_remote_raises_on_ok_false,
    test_warm_noop_when_local,
    test_guard_forwards_when_runpod,
    test_guard_runs_local_when_local,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
