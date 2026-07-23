"""
Smoke tests for the captions API (``app.routers.captions``) -- a real
FastAPI TestClient against the actual app, with every DB-touching service
call monkeypatched. No real Postgres, no real R2. Mirrors
test_projects_router.py's pattern exactly.

Run:  .venv/bin/python scripts/test_captions_router.py
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
from app.routers import captions as captions_router  # noqa: E402
from app.services.l3 import store  # noqa: E402
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


_RESOLVED = {
    "aspect": "landscape",
    "video_layers": [{"kind": "spine", "source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 5000,
                       "prog_start_ms": 0, "prog_end_ms": 5000}],
}
_CUT_ROW = {
    "file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000, "caption_zones": [[0.1, 0.7, 0.8, 0.2]],
    "junk": False, "hero_key": "hero/f1.jpg", "hero_ts_ms": 500, "on_camera": True,
    "total_quality": 0.8, "framing": {"subject_box": [0.3, 0.3, 0.2, 0.3], "shot_size": "medium"},
    "pace": {"energy_grade": "active"}, "speaker_person": "p1",
}
_TRANSCRIPT = {"f1": {"segments": [{"words": [
    {"text": "hello", "start_ms": 0, "end_ms": 300, "is_filler": False},
    {"text": "world", "start_ms": 350, "end_ms": 700, "is_filler": False},
]}]}}


def _patch_common(p: _Patcher, *, doc=None):
    p.set(store, "get_thread", lambda thread_id: {"id": thread_id, "user_id": "user-1"})
    p.set(store, "latest_document", lambda thread_id: (doc if doc is not None else {"look": {}}, 1))
    p.set(render_tasks, "resolve_document", lambda document, thread_id=None: _RESOLVED)
    p.set(captions_router, "presigned_url_for", lambda key: f"https://example.test/{key}")


def test_get_catalog_returns_fonts_colours_and_one_standard():
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/captions/catalog")
    finally:
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["fonts"]) == 4, body
    assert len(body["colours"]) == 4, body
    assert len(body["standards"]) == 1, body
    print("ok  test_get_catalog_returns_fonts_colours_and_one_standard")


def test_get_suggestions_returns_standard_plus_four():
    p = _Patcher()
    _patch_common(p)
    p.set(captions_router, "fetch_cut_records", lambda file_ids: {"f1": [_CUT_ROW]})
    p.set(captions_router, "fetch_audio_features", lambda file_ids: {"f1": {"is_musical": False}})
    p.set(captions_router, "fetch_color_stats_for_captions", lambda file_ids: {"f1": {"rgb_mean": [0.4, 0.4, 0.4]}})
    p.set(captions_router, "fetch_transcripts", lambda file_ids: _TRANSCRIPT)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/captions/suggestions", params={"thread_id": "t1"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["standard"]["style_id"] == "std_standard", body
    assert len(body["suggestions"]) == 4, body
    assert body["representative_frame"]["url"] == "https://example.test/hero/f1.jpg", body
    assert len(body["sample_words"]) > 0, body
    print("ok  test_get_suggestions_returns_standard_plus_four")


def test_get_suggestions_fails_open_when_one_signal_fetch_raises():
    """Missing colour/audio/cut analysis must not fail the endpoint."""
    p = _Patcher()
    _patch_common(p)
    p.set(captions_router, "fetch_cut_records", lambda file_ids: {"f1": [_CUT_ROW]})

    def _boom(file_ids):
        raise RuntimeError("audio_features table unavailable")
    p.set(captions_router, "fetch_audio_features", _boom)
    p.set(captions_router, "fetch_color_stats_for_captions", lambda file_ids: {})
    p.set(captions_router, "fetch_transcripts", lambda file_ids: _TRANSCRIPT)
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/captions/suggestions", params={"thread_id": "t1"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["suggestions"]) == 4, body
    print("ok  test_get_suggestions_fails_open_when_one_signal_fetch_raises")


def test_get_suggestions_no_spine_still_returns_standard_plus_four():
    p = _Patcher()
    _patch_common(p)
    p.set(render_tasks, "resolve_document", lambda document, thread_id=None: {"aspect": "landscape", "video_layers": []})
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/captions/suggestions", params={"thread_id": "t1"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["standard"]["style_id"] == "std_standard"
    assert len(body["suggestions"]) == 4, body
    assert body["representative_frame"] is None
    assert body["sample_words"] == []
    print("ok  test_get_suggestions_no_spine_still_returns_standard_plus_four")


def test_get_suggestions_404_when_thread_not_owned():
    p = _Patcher()
    p.set(store, "get_thread", lambda thread_id: {"id": thread_id, "user_id": "someone-else"})
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/captions/suggestions", params={"thread_id": "t1"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_get_suggestions_404_when_thread_not_owned")


def test_get_suggestions_404_when_no_document():
    p = _Patcher()
    p.set(store, "get_thread", lambda thread_id: {"id": thread_id, "user_id": "user-1"})
    p.set(store, "latest_document", lambda thread_id: (None, None))
    _as_user("user-1")
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/captions/suggestions", params={"thread_id": "t1"})
    finally:
        p.restore()
        _clear_overrides()
    assert resp.status_code == 404, resp.text
    print("ok  test_get_suggestions_404_when_no_document")


def main():
    test_get_catalog_returns_fonts_colours_and_one_standard()
    test_get_suggestions_returns_standard_plus_four()
    test_get_suggestions_fails_open_when_one_signal_fetch_raises()
    test_get_suggestions_no_spine_still_returns_standard_plus_four()
    test_get_suggestions_404_when_thread_not_owned()
    test_get_suggestions_404_when_no_document()
    print("\nall captions router tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
