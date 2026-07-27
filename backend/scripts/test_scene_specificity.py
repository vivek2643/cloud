#!/usr/bin/env python3
"""Tests for app.services.l3.scene_specificity (cut_structure_and_scene_
specificity.plan.md Part 3, Pass B) -- mocks the LLM, frame extraction, and
the proxy-key DB read; no real network/DB/R2.

Run:  .venv/bin/python scripts/test_scene_specificity.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import scene_specificity as ss  # noqa: E402
from app.services.llm.client import Completion  # noqa: E402


def _row(cid, file_id="f1", hero_ts_ms=1000, label="a machine", summary="a machine runs"):
    return {"id": cid, "file_id": file_id, "hero_ts_ms": hero_ts_ms, "label": label, "summary": summary}


def _taxonomy(clusters, needs_pass_b, taxonomy=None):
    return {
        "domain": "CNC machine shop", "confidence": "med", "evidence": ["x"],
        "taxonomy": taxonomy or [], "clusters": clusters, "needs_pass_b": needs_pass_b,
    }


class _Common:
    """Shared no-op mocks for the pieces every test needs stubbed: proxy
    key lookup (DB), frame extraction (R2 + ffmpeg), and cache create/
    delete (Gemini API) -- so only the LLM completion behavior varies test
    to test."""

    def __enter__(self):
        self._stack = [
            mock.patch.object(ss, "_proxy_keys_for_files", lambda file_ids: {fid: f"proxies/{fid}" for fid in file_ids}),
            mock.patch.object(ss, "extract_for_planned_frames", lambda *a, **k: {}),
            mock.patch.object(ss, "create_pass2_cache", self._create_cache),
            mock.patch.object(ss, "delete_pass2_cache", self._delete_cache),
        ]
        self.create_calls = []
        self.delete_calls = []
        for p in self._stack:
            p.start()
        return self

    def _create_cache(self, *a, **k):
        self.create_calls.append((a, k))
        return "cache-handle-1"

    def _delete_cache(self, name):
        self.delete_calls.append(name)

    def __exit__(self, *exc):
        for p in reversed(self._stack):
            p.stop()
        return False


def test_run_pass_b_writes_specific_and_label_for_triaged_cuts():
    rows = [_row("c0"), _row("c1")]
    taxonomy = _taxonomy(
        clusters=[{"cut_refs": ["c0", "c1"], "questions": ["what part and operation?"]}],
        needs_pass_b=["c0", "c1"],
        taxonomy=[{"id": "milling", "definition": "cutting with a rotating tool"}],
    )
    fake_answer = {"cuts": [
        {"cut_id": "c0", "specific": "milling a steel bracket", "label": "milling"},
        {"cut_id": "c1", "specific": "inspecting the finished part", "label": "other"},
    ]}

    with _Common() as common, \
         mock.patch.object(ss.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(ss, "complete_gemini",
                           lambda *a, **k: Completion(data=fake_answer, usage={}, attempts=1)):
        result = ss.run_pass_b("run-1", taxonomy)

    assert result == {
        "c0": {"specific": "milling a steel bracket", "label": "milling"},
        "c1": {"specific": "inspecting the finished part", "label": "other"},
    }, result
    assert len(common.create_calls) == 1, common.create_calls
    assert common.delete_calls == ["cache-handle-1"], common.delete_calls
    print("ok  test_run_pass_b_writes_specific_and_label_for_triaged_cuts")


def test_run_pass_b_excludes_cluster_members_not_in_needs_pass_b():
    """A cluster can cover more cuts than actually need Pass B -- only the
    ones ALSO in needs_pass_b get sent to the model."""
    rows = [_row("c0"), _row("c1")]
    taxonomy = _taxonomy(
        clusters=[{"cut_refs": ["c0", "c1"], "questions": ["q"]}],
        needs_pass_b=["c0"],   # c1 already specific enough -- triaged out
    )
    sent_cut_ids = []

    def fake_complete(system, blocks, schema, **kw):
        for b in blocks:
            if b.get("type") == "text" and "cut_id=" in b.get("text", ""):
                sent_cut_ids.append(b["text"].split("cut_id=")[1].split(" ")[0])
        return Completion(data={"cuts": [{"cut_id": "c0", "specific": "x", "label": ""}]}, usage={}, attempts=1)

    with _Common(), \
         mock.patch.object(ss.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(ss, "complete_gemini", fake_complete):
        result = ss.run_pass_b("run-1", taxonomy)

    assert sent_cut_ids == ["c0"], sent_cut_ids
    assert "c1" not in result, result
    print("ok  test_run_pass_b_excludes_cluster_members_not_in_needs_pass_b")


def test_run_pass_b_one_cluster_failure_is_non_fatal_others_still_run():
    rows = [_row("c0"), _row("c1", file_id="f2")]
    taxonomy = _taxonomy(
        clusters=[
            {"cut_refs": ["c0"], "questions": ["q1"]},
            {"cut_refs": ["c1"], "questions": ["q2"]},
        ],
        needs_pass_b=["c0", "c1"],
    )

    def fake_complete(system, blocks, schema, **kw):
        text = blocks[1]["text"] if len(blocks) > 1 else ""
        if "c0" in text:
            raise RuntimeError("simulated Gemini failure")
        return Completion(data={"cuts": [{"cut_id": "c1", "specific": "ok", "label": ""}]}, usage={}, attempts=1)

    with _Common() as common, \
         mock.patch.object(ss.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(ss, "complete_gemini", fake_complete):
        result = ss.run_pass_b("run-1", taxonomy)

    assert "c0" not in result, result
    assert result.get("c1") == {"specific": "ok", "label": ""}, result
    # The cache is still torn down even though one cluster raised.
    assert common.delete_calls == ["cache-handle-1"], common.delete_calls
    print("ok  test_run_pass_b_one_cluster_failure_is_non_fatal_others_still_run")


def test_run_pass_b_empty_needs_pass_b_short_circuits_with_no_calls():
    taxonomy = _taxonomy(clusters=[{"cut_refs": ["c0"], "questions": ["q"]}], needs_pass_b=[])
    calls = []
    with mock.patch.object(ss.cuts_read, "rows_for_run", lambda run_id: calls.append(1)):
        result = ss.run_pass_b("run-1", taxonomy)
    assert result == {}
    assert calls == [], "must never even read cut_records when nothing needs Pass B"
    print("ok  test_run_pass_b_empty_needs_pass_b_short_circuits_with_no_calls")


def test_run_pass_b_ignores_a_cut_id_the_model_invents():
    rows = [_row("c0")]
    taxonomy = _taxonomy(clusters=[{"cut_refs": ["c0"], "questions": ["q"]}], needs_pass_b=["c0"])
    fake_answer = {"cuts": [
        {"cut_id": "c0", "specific": "real answer", "label": ""},
        {"cut_id": "not-a-real-cut", "specific": "hallucinated", "label": ""},
    ]}
    with _Common(), \
         mock.patch.object(ss.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(ss, "complete_gemini",
                           lambda *a, **k: Completion(data=fake_answer, usage={}, attempts=1)):
        result = ss.run_pass_b("run-1", taxonomy)
    assert list(result.keys()) == ["c0"], result
    print("ok  test_run_pass_b_ignores_a_cut_id_the_model_invents")


def main():
    test_run_pass_b_writes_specific_and_label_for_triaged_cuts()
    test_run_pass_b_excludes_cluster_members_not_in_needs_pass_b()
    test_run_pass_b_one_cluster_failure_is_non_fatal_others_still_run()
    test_run_pass_b_empty_needs_pass_b_short_circuits_with_no_calls()
    test_run_pass_b_ignores_a_cut_id_the_model_invents()
    print("\nall scene_specificity tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
