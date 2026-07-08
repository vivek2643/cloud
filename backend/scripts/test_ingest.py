"""
Tests for the cuts-v3 orchestrator (``app.services.l3.ingest``) -- NO real
API calls, NO real DB, NO real R2/ffmpeg. Every seam (pass1, image_plan,
frames, pass2a, pass2b, post, the DB store, hero-frame upload) is
monkeypatched with a fake so what's actually under test is the
ORCHESTRATION: stage sequencing, usage accumulation across shards/batches,
concurrency, and failure handling.

Run:  .venv/bin/python scripts/test_ingest.py
"""
from __future__ import annotations

import os
import sys
import time

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


def _basic_patches(p, fake_store, file_rows):
    """Common patches every test needs: store, pass1 loading/running,
    image_plan/frame extraction stubbed empty. build_identity_shards and
    build_visual_batches are left REAL (pure, deterministic) -- with empty
    planned_frames they naturally return [], so tests that don't care about
    shard/batch composition don't need to mock them at all."""
    p.set(ingest, "store", fake_store)
    p.set(ingest.pass1, "load_project_file_rows", lambda pid: file_rows)
    p.set(ingest, "_proxy_keys_for_files", lambda file_ids: {})
    p.set(ingest, "build_l1_snapshot",
          lambda fid: {"motion_dynamics": {}, "scene_cuts": {}, "audio_features": {}})
    p.set(ingest.pass1, "run_pass1",
          lambda file_rows: ic.Completion(data=_GOOD_PASS1, usage={}, attempts=1))
    p.set(ingest.ip, "build_image_plan", lambda *a, **k: [])
    p.set(ingest.fr, "extract_for_planned_frames", lambda *a, **k: {})


def test_run_ingest_happy_path_sequences_stages_in_order():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]
    p = _Patcher()
    _basic_patches(p, fake_store, file_rows)
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


def test_run_ingest_calls_pass2a_per_shard_and_pass2b_per_batch():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1")), ("f2", "b.mp4", 1500, _lattice("f2"))]
    identity_cut1 = {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
                     "word_span": [0, 1], "label": "x", "summary": "y"}
    identity_cut2 = {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f2",
                     "word_span": [0, 1], "label": "a", "summary": "b"}
    shard_calls = []

    def fake_run_identity_shard(shard_file_rows, pass1_output, shard_frames, images_b64):
        files = [r[0] for r in shard_file_rows]
        shard_calls.append(files)
        cut = identity_cut1 if files == ["f1"] else identity_cut2
        return ic.Completion(data={"cuts": [cut]}, usage={"input_tokens": 5}, attempts=1)

    batch_calls = []

    def fake_run_visual_batch(identity_output, batch_indices, planned_frames, images_b64):
        batch_calls.append(list(batch_indices))
        judgments = [{"cut_index": i} for i in batch_indices]
        return ic.Completion(data={"judgments": judgments}, usage={"input_tokens": 2}, attempts=1)

    recorded = {}

    def fake_assemble(pass2_output, lattices, motion_by_file, silences_by_file,
                      junk_suspects=None, audio_by_file=None):
        recorded["cuts"] = list(pass2_output.cuts)
        return []

    p = _Patcher()
    _basic_patches(p, fake_store, file_rows)
    p.set(ingest.pass2a, "build_identity_shards", lambda *a, **k: [["f1"], ["f2"]])
    p.set(ingest.pass2a, "run_identity_shard", fake_run_identity_shard)
    p.set(ingest.pass2b, "run_visual_batch", fake_run_visual_batch)
    p.set(ingest.post, "assemble_cut_records", fake_assemble)
    p.set(ingest, "_extract_and_upload_heroes", lambda *a, **k: None)
    try:
        ingest.run_ingest("proj-1")
    finally:
        p.restore()

    assert sorted(shard_calls) == [["f1"], ["f2"]], shard_calls
    assert batch_calls == [[0, 1]], batch_calls   # 2 confirmed cuts, 1 default-sized batch
    assert len(recorded["cuts"]) == 2
    assert len(fake_store.pass2_usage) == 3   # 2 identity shards + 1 visual batch
    print("ok  test_run_ingest_calls_pass2a_per_shard_and_pass2b_per_batch")


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


def test_run_ingest_runs_pass2a_shards_concurrently_not_sequentially():
    """Identity shards only share a read-only cached prefix -- they should
    run at the same time, not wait on each other. Four shards that each
    take ~0.3s should finish in well under 4x0.3s=1.2s if actually
    concurrent."""
    fake_store = FakeStore()
    file_rows = [(f"f{i}", f"{i}.mp4", 2000, _lattice(f"f{i}")) for i in range(4)]

    def fake_run_identity_shard(shard_file_rows, pass1_output, shard_frames, images_b64):
        time.sleep(0.3)
        fid = shard_file_rows[0][0]
        cut = {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": fid,
              "word_span": [0, 1], "label": "x", "summary": "y"}
        return ic.Completion(data={"cuts": [cut]}, usage={"input_tokens": 1}, attempts=1)

    def fake_run_visual_batch(identity_output, batch_indices, planned_frames, images_b64):
        judgments = [{"cut_index": i} for i in batch_indices]
        return ic.Completion(data={"judgments": judgments}, usage={"input_tokens": 1}, attempts=1)

    p = _Patcher()
    _basic_patches(p, fake_store, file_rows)
    p.set(ingest.pass2a, "build_identity_shards", lambda *a, **k: [[f"f{i}"] for i in range(4)])
    p.set(ingest.pass2a, "run_identity_shard", fake_run_identity_shard)
    p.set(ingest.pass2b, "run_visual_batch", fake_run_visual_batch)
    p.set(ingest.post, "assemble_cut_records", lambda *a, **k: [])
    p.set(ingest, "_extract_and_upload_heroes", lambda *a, **k: None)
    try:
        start = time.monotonic()
        ingest.run_ingest("proj-1")
        elapsed = time.monotonic() - start
    finally:
        p.restore()

    assert elapsed < 0.9, f"took {elapsed:.2f}s -- identity shards do not appear to run concurrently"
    print("ok  test_run_ingest_runs_pass2a_shards_concurrently_not_sequentially")


def test_run_ingest_runs_pass2b_batches_concurrently_not_sequentially():
    """Same guarantee, for visual batches: no co-location constraint at all
    (see pass2b.py), so they should run with at least as much parallelism
    as identity shards."""
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]
    # 8 confirmed cuts, batched at 2/batch (patched below) -> 4 batches.
    identity_cuts = [
        {"source_ref": f"speech_cut[{i}]", "kind": "speech", "file_id": "f1",
         "word_span": [i, i + 1], "label": "x", "summary": "y"}
        for i in range(8)
    ]

    def fake_run_identity_shard(shard_file_rows, pass1_output, shard_frames, images_b64):
        return ic.Completion(data={"cuts": identity_cuts}, usage={"input_tokens": 1}, attempts=1)

    def fake_run_visual_batch(identity_output, batch_indices, planned_frames, images_b64):
        time.sleep(0.3)
        judgments = [{"cut_index": i} for i in batch_indices]
        return ic.Completion(data={"judgments": judgments}, usage={"input_tokens": 1}, attempts=1)

    p = _Patcher()
    _basic_patches(p, fake_store, file_rows)
    p.set(ingest.pass2a, "build_identity_shards", lambda *a, **k: [["f1"]])
    p.set(ingest.pass2a, "run_identity_shard", fake_run_identity_shard)
    p.set(ingest.pass2b, "run_visual_batch", fake_run_visual_batch)
    p.set(ingest, "MAX_CUTS_PER_VISUAL_BATCH", 2)
    p.set(ingest.post, "assemble_cut_records", lambda *a, **k: [])
    p.set(ingest, "_extract_and_upload_heroes", lambda *a, **k: None)
    try:
        start = time.monotonic()
        ingest.run_ingest("proj-1")
        elapsed = time.monotonic() - start
    finally:
        p.restore()

    assert elapsed < 0.9, f"took {elapsed:.2f}s -- visual batches do not appear to run concurrently"
    print("ok  test_run_ingest_runs_pass2b_batches_concurrently_not_sequentially")


def test_run_ingest_marks_failed_on_post_invariant_violation():
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]

    def boom_post(*a, **k):
        raise ValueError("f1: overlap between [0-1100] and [1000-2000]")

    p = _Patcher()
    _basic_patches(p, fake_store, file_rows)
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
    assert "overlap" in fake_store.status_history[-1][1]
    print("ok  test_run_ingest_marks_failed_on_post_invariant_violation")


def test_run_ingest_marks_failed_when_pass2b_misses_a_judgment():
    """merge_identity_and_visual's own invariant (every confirmed cut needs
    a visual judgment) should propagate as a normal ingest failure, same as
    any other post-pass2 error."""
    fake_store = FakeStore()
    file_rows = [("f1", "a.mp4", 2000, _lattice("f1"))]
    identity_cuts = [
        {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
         "word_span": [0, 1], "label": "x", "summary": "y"},
        {"source_ref": "speech_cut[1]", "kind": "speech", "file_id": "f1",
         "word_span": [2, 3], "label": "a", "summary": "b"},
    ]

    def fake_run_identity_shard(shard_file_rows, pass1_output, shard_frames, images_b64):
        return ic.Completion(data={"cuts": identity_cuts}, usage={}, attempts=1)

    def fake_run_visual_batch(identity_output, batch_indices, planned_frames, images_b64):
        # only ever answers for cut_index 0, never 1
        return ic.Completion(data={"judgments": [{"cut_index": 0}]}, usage={}, attempts=1)

    p = _Patcher()
    _basic_patches(p, fake_store, file_rows)
    p.set(ingest.pass2a, "build_identity_shards", lambda *a, **k: [["f1"]])
    p.set(ingest.pass2a, "run_identity_shard", fake_run_identity_shard)
    p.set(ingest.pass2b, "run_visual_batch", fake_run_visual_batch)
    try:
        try:
            ingest.run_ingest("proj-1")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "1" in str(e), e
    finally:
        p.restore()

    assert fake_store.status_history[-1][0] == "failed"
    print("ok  test_run_ingest_marks_failed_when_pass2b_misses_a_judgment")


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
    test_run_ingest_calls_pass2a_per_shard_and_pass2b_per_batch()
    test_run_ingest_runs_pass2a_shards_concurrently_not_sequentially()
    test_run_ingest_runs_pass2b_batches_concurrently_not_sequentially()
    test_run_ingest_marks_failed_and_reraises_on_pass1_failure()
    test_run_ingest_raises_when_no_ingest_ready_files()
    test_run_ingest_marks_failed_on_post_invariant_violation()
    test_run_ingest_marks_failed_when_pass2b_misses_a_judgment()
    test_run_many_runs_every_project_and_collects_results_by_id()
    test_run_many_isolates_one_projects_failure_from_the_rest()
    print("\nall ingest orchestration tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
