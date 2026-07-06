"""
Tests for the cuts-v3 orchestrator (``app.services.l3.ingest``) -- NO real
API calls, NO real DB, NO real R2/ffmpeg. Every seam (pass1, image_plan,
frames, pass2, post, the DB store, hero-frame upload) is monkeypatched with
a fake so what's actually under test is the ORCHESTRATION: stage sequencing,
usage accumulation across shards, and failure handling.

Run:  .venv/bin/python scripts/test_ingest.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import ingest  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402
from app.services.llm import client as ic  # noqa: E402
from app.services.llm.client import IngestFailure  # noqa: E402


class FakeStore:
    def __init__(self):
        self.status_history = []
        self.pass1_recorded = None
        self.pass2_usage = []
        self.deleted_for = None
        self.inserted = None
        self.hero_keys = {}

    def create_ingest_run(self, project_id, pass1_model, pass2_model):
        return "run-1"

    def set_status(self, ingest_run_id, status, error=None):
        self.status_history.append((status, error))

    def record_pass1_result(self, ingest_run_id, pass1_output, usage, project_summary):
        self.pass1_recorded = (pass1_output, usage, project_summary)

    def accumulate_pass2_usage(self, ingest_run_id, usage):
        self.pass2_usage.append(usage)

    def delete_cut_records_for_run(self, ingest_run_id):
        self.deleted_for = ingest_run_id

    def insert_cut_records(self, ingest_run_id, records):
        self.inserted = records
        return [f"rec-{i}" for i in range(len(records))]

    def set_hero_key(self, cut_record_id, hero_key):
        self.hero_keys[cut_record_id] = hero_key


class _Patcher:
    """Tiny manual monkeypatch helper -- set attrs, restore all on exit."""

    def __init__(self):
        self._orig = {}

    def set(self, obj, name, value):
        self._orig[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def restore(self):
        for (obj, name), value in self._orig.items():
            setattr(obj, name, value)


def _lattice(file_id, duration_ms=2000):
    return Lattice(file_id=file_id, duration_ms=duration_ms, words=[], turns=[], hints=[], atoms=[])


_GOOD_PASS1 = {
    "speech_cuts": [], "take_candidates": [], "video_tentative_groups": [],
    "junk_suspects": [], "project_summary": "a summary", "clip_summaries": [],
}


def test_run_ingest_happy_path_sequences_stages_in_order():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]
    p = _Patcher()
    p.set(ingest, "store", fake_store)
    p.set(ingest.pass1, "load_project_file_rows", lambda pid: file_rows)
    p.set(ingest, "_proxy_keys_for_files", lambda file_ids: {"f1": "proxies/f1/proxy.mp4"})
    p.set(ingest, "build_l1_snapshot",
          lambda fid: {"motion_dynamics": {}, "scene_cuts": {}, "audio_features": {}})
    p.set(ingest.pass1, "run_pass1",
          lambda file_rows: ic.Completion(data=_GOOD_PASS1, usage={"input_tokens": 10}, attempts=1))
    p.set(ingest.ip, "build_image_plan", lambda *a, **k: [])
    p.set(ingest.fr, "extract_for_planned_frames", lambda *a, **k: {})
    p.set(ingest.pass2, "build_shards", lambda *a, **k: [])
    p.set(ingest.post, "assemble_cut_records", lambda *a, **k: [])
    p.set(ingest, "_extract_and_upload_heroes", lambda *a, **k: None)
    try:
        run_id = ingest.run_ingest("proj-1")
    finally:
        p.restore()

    assert run_id == "run-1"
    assert [s for s, _ in fake_store.status_history] == ["pass1", "images", "pass2", "post", "ready"], \
        fake_store.status_history
    assert fake_store.pass1_recorded[2] == "a summary"
    assert fake_store.deleted_for == "run-1"
    assert fake_store.inserted == []
    print("ok  test_run_ingest_happy_path_sequences_stages_in_order")


def test_run_ingest_calls_pass2_per_shard_and_collects_all_cuts():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1")), ("f2", "b.mp4", 1500, _lattice("f2"))]
    cut1 = {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
           "word_span": [0, 1], "label": "x", "summary": "y"}
    cut2 = {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f2",
           "word_span": [0, 1], "label": "a", "summary": "b"}
    shard_calls = []

    def fake_run_pass2_shard(shard_file_rows, pass1_output, shard_frames, images_b64):
        files = [r[0] for r in shard_file_rows]
        shard_calls.append(files)
        cut = cut1 if files == ["f1"] else cut2
        return ic.Completion(data={"cuts": [cut]}, usage={"input_tokens": 5}, attempts=1)

    recorded = {}

    def fake_assemble(pass2_output, lattices, motion_by_file, silences_by_file):
        recorded["cuts"] = list(pass2_output.cuts)
        return []

    p = _Patcher()
    p.set(ingest, "store", fake_store)
    p.set(ingest.pass1, "load_project_file_rows", lambda pid: file_rows)
    p.set(ingest, "_proxy_keys_for_files", lambda file_ids: {})
    p.set(ingest, "build_l1_snapshot",
          lambda fid: {"motion_dynamics": {}, "scene_cuts": {}, "audio_features": {}})
    p.set(ingest.pass1, "run_pass1",
          lambda file_rows: ic.Completion(data=_GOOD_PASS1, usage={}, attempts=1))
    p.set(ingest.ip, "build_image_plan", lambda *a, **k: [])
    p.set(ingest.fr, "extract_for_planned_frames", lambda *a, **k: {})
    p.set(ingest.pass2, "build_shards", lambda *a, **k: [["f1"], ["f2"]])
    p.set(ingest.pass2, "run_pass2_shard", fake_run_pass2_shard)
    p.set(ingest.post, "assemble_cut_records", fake_assemble)
    p.set(ingest, "_extract_and_upload_heroes", lambda *a, **k: None)
    try:
        ingest.run_ingest("proj-1")
    finally:
        p.restore()

    assert sorted(shard_calls) == [["f1"], ["f2"]], shard_calls
    assert len(recorded["cuts"]) == 2
    assert len(fake_store.pass2_usage) == 2
    print("ok  test_run_ingest_calls_pass2_per_shard_and_collects_all_cuts")


def test_run_ingest_marks_failed_and_reraises_on_pass1_failure():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]

    def boom(file_rows):
        raise IngestFailure("pass1", "schema violation twice")

    p = _Patcher()
    p.set(ingest, "store", fake_store)
    p.set(ingest.pass1, "load_project_file_rows", lambda pid: file_rows)
    p.set(ingest, "_proxy_keys_for_files", lambda file_ids: {})
    p.set(ingest, "build_l1_snapshot",
          lambda fid: {"motion_dynamics": {}, "scene_cuts": {}, "audio_features": {}})
    p.set(ingest.pass1, "run_pass1", boom)
    try:
        try:
            ingest.run_ingest("proj-1")
            assert False, "expected IngestFailure"
        except IngestFailure:
            pass
    finally:
        p.restore()

    assert fake_store.status_history[-1][0] == "failed", fake_store.status_history
    assert "schema violation" in fake_store.status_history[-1][1]
    print("ok  test_run_ingest_marks_failed_and_reraises_on_pass1_failure")


def test_run_ingest_raises_when_no_ingest_ready_files():
    fake_store = FakeStore()
    p = _Patcher()
    p.set(ingest, "store", fake_store)
    p.set(ingest.pass1, "load_project_file_rows", lambda pid: [])
    try:
        try:
            ingest.run_ingest("proj-empty")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "no ingest-ready files" in str(e)
    finally:
        p.restore()
    assert fake_store.status_history[-1][0] == "failed"
    print("ok  test_run_ingest_raises_when_no_ingest_ready_files")


def test_run_ingest_marks_failed_on_post_invariant_violation():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]

    def boom_post(*a, **k):
        raise ValueError("f1: coverage gap [0-2000] before the first cut")

    p = _Patcher()
    p.set(ingest, "store", fake_store)
    p.set(ingest.pass1, "load_project_file_rows", lambda pid: file_rows)
    p.set(ingest, "_proxy_keys_for_files", lambda file_ids: {})
    p.set(ingest, "build_l1_snapshot",
          lambda fid: {"motion_dynamics": {}, "scene_cuts": {}, "audio_features": {}})
    p.set(ingest.pass1, "run_pass1",
          lambda file_rows: ic.Completion(data=_GOOD_PASS1, usage={}, attempts=1))
    p.set(ingest.ip, "build_image_plan", lambda *a, **k: [])
    p.set(ingest.fr, "extract_for_planned_frames", lambda *a, **k: {})
    p.set(ingest.pass2, "build_shards", lambda *a, **k: [])
    p.set(ingest.post, "assemble_cut_records", boom_post)
    try:
        try:
            ingest.run_ingest("proj-1")
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        p.restore()

    assert fake_store.status_history[-1][0] == "failed"
    assert "coverage gap" in fake_store.status_history[-1][1]
    print("ok  test_run_ingest_marks_failed_on_post_invariant_violation")


def test_run_many_runs_every_project_and_collects_results_by_id():
    calls = []

    def fake_run_ingest(project_id):
        calls.append(project_id)
        return f"run-for-{project_id}"

    p = _Patcher()
    p.set(ingest, "run_ingest", fake_run_ingest)
    try:
        results = ingest.run_many(["proj-a", "proj-b", "proj-c"])
    finally:
        p.restore()

    assert sorted(calls) == ["proj-a", "proj-b", "proj-c"], calls
    assert results == {
        "proj-a": "run-for-proj-a", "proj-b": "run-for-proj-b", "proj-c": "run-for-proj-c",
    }, results
    print("ok  test_run_many_runs_every_project_and_collects_results_by_id")


def test_run_many_isolates_one_projects_failure_from_the_rest():
    def fake_run_ingest(project_id):
        if project_id == "proj-bad":
            raise ValueError("boom")
        return f"run-for-{project_id}"

    p = _Patcher()
    p.set(ingest, "run_ingest", fake_run_ingest)
    try:
        results = ingest.run_many(["proj-good", "proj-bad"])
    finally:
        p.restore()

    assert results["proj-good"] == "run-for-proj-good", results
    assert isinstance(results["proj-bad"], ValueError), results
    print("ok  test_run_many_isolates_one_projects_failure_from_the_rest")


def main():
    test_run_ingest_happy_path_sequences_stages_in_order()
    test_run_ingest_calls_pass2_per_shard_and_collects_all_cuts()
    test_run_ingest_marks_failed_and_reraises_on_pass1_failure()
    test_run_ingest_raises_when_no_ingest_ready_files()
    test_run_ingest_marks_failed_on_post_invariant_violation()
    test_run_many_runs_every_project_and_collects_results_by_id()
    test_run_many_isolates_one_projects_failure_from_the_rest()
    print("\nall ingest orchestration tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
