#!/usr/bin/env python3
"""Tests for app.services.l3.scene_taxonomy (cut_structure_and_scene_
specificity.plan.md Part 3, middle text layer) -- mocks the LLM + the
transcript DB read, no real network/DB.

Run:  .venv/bin/python scripts/test_scene_taxonomy.py
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import scene_taxonomy as st  # noqa: E402
from app.services.llm.client import Completion  # noqa: E402


def _row(cid, label, summary, channel="shown", file_id="f1"):
    return {"id": cid, "file_id": file_id, "label": label, "summary": summary,
            "channel": channel, "characteristics": [], "screen_text": ""}


@contextmanager
def _no_transcripts():
    """Stub pass1._pg_conn so _fetch_transcripts never touches a real DB --
    every scene_taxonomy test is a pure/mocked-LLM test."""
    class _FakeCur:
        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return []

    class _FakeConn:
        def __enter__(self):
            return _FakeCur()

        def __exit__(self, *a):
            return False

    with mock.patch.object(st.pass1, "_pg_conn", lambda: _FakeConn()):
        yield


# --------------------------------------------------------------------------
# _group_cuts: pure dedupe, no LLM involved
# --------------------------------------------------------------------------

def test_group_cuts_dedupes_near_identical_label_and_summary():
    rows = [_row(f"c{i}", "machine on a line", "a machine runs") for i in range(5)]
    rows += [_row("c5", "different thing", "something else")]
    groups = st._group_cuts(rows)
    assert len(groups) == 2, groups
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 5], sizes
    print("ok  test_group_cuts_dedupes_near_identical_label_and_summary")


def test_group_cuts_is_case_and_whitespace_insensitive():
    rows = [_row("c0", "Machine On A Line", "a  machine   runs"),
           _row("c1", "machine on a line", "a machine runs")]
    groups = st._group_cuts(rows)
    assert len(groups) == 1, groups
    print("ok  test_group_cuts_is_case_and_whitespace_insensitive")


def test_group_cuts_caps_at_max_groups():
    rows = [_row(f"c{i}", f"unique label {i}", f"summary {i}") for i in range(200)]
    groups = st._group_cuts(rows)
    assert len(groups) == st._MAX_GROUPS, len(groups)
    print("ok  test_group_cuts_caps_at_max_groups")


# --------------------------------------------------------------------------
# build_scene_taxonomy: mocked LLM
# --------------------------------------------------------------------------

def test_build_scene_taxonomy_expands_group_refs_to_real_cut_ids():
    rows = [_row(f"c{i}", "machine on a line", "a machine runs") for i in range(3)]
    rows += [_row("c9", "wide shot of the floor", "an establishing shot")]

    def fake_rows_for_run(run_id):
        return rows

    fake_response = {
        "domain": "CNC machine shop", "confidence": "med",
        "evidence": ["repeated machine/tooling references"],
        "taxonomy": [], "clusters": [
            {"group_refs": ["g0"], "questions": ["what part and what operation?"]},
        ],
        "needs_pass_b_groups": ["g0"],
    }

    with mock.patch.object(st.cuts_read, "rows_for_run", fake_rows_for_run), \
         mock.patch.object(st, "complete_gemini",
                           lambda *a, **k: Completion(data=fake_response, usage={}, attempts=1)), \
         _no_transcripts():
        result = st.build_scene_taxonomy("run-1")

    assert result["domain"] == "CNC machine shop", result
    assert sorted(result["needs_pass_b"]) == ["c0", "c1", "c2"], result["needs_pass_b"]
    assert len(result["clusters"]) == 1, result["clusters"]
    assert sorted(result["clusters"][0]["cut_refs"]) == ["c0", "c1", "c2"], result["clusters"]
    print("ok  test_build_scene_taxonomy_expands_group_refs_to_real_cut_ids")


def test_build_scene_taxonomy_keeps_taxonomy_empty_when_model_gives_none():
    """Closed-set taxonomy is a SPECIAL CASE, not the default -- when the
    model returns none, the result stays empty rather than inventing one."""
    rows = [_row("c0", "a person talks", "someone explains something")]

    fake_response = {
        "domain": "unknown/mixed", "confidence": "low", "evidence": [],
        "taxonomy": [], "clusters": [], "needs_pass_b_groups": [],
    }

    with mock.patch.object(st.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(st, "complete_gemini",
                           lambda *a, **k: Completion(data=fake_response, usage={}, attempts=1)), \
         _no_transcripts():
        result = st.build_scene_taxonomy("run-1")

    assert result["domain"] == "unknown/mixed", result
    assert result["taxonomy"] == [], result
    assert result["needs_pass_b"] == [], result
    assert result["clusters"] == [], result
    print("ok  test_build_scene_taxonomy_keeps_taxonomy_empty_when_model_gives_none")


def test_build_scene_taxonomy_uses_a_real_closed_set_taxonomy_when_given():
    rows = [_row("c0", "couple circles the fire", "a ritual moment")]
    fake_response = {
        "domain": "Indian wedding (Gujarati)", "confidence": "high",
        "evidence": ["repeated ritual staging", "fire references in transcript"],
        "taxonomy": [{"id": "pheras", "definition": "circling the sacred fire"},
                     {"id": "other", "definition": ""}, {"id": "unsure", "definition": ""}],
        "clusters": [{"group_refs": ["g0"], "questions": ["which ritual step is this?"]}],
        "needs_pass_b_groups": ["g0"],
    }
    with mock.patch.object(st.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(st, "complete_gemini",
                           lambda *a, **k: Completion(data=fake_response, usage={}, attempts=1)), \
         _no_transcripts():
        result = st.build_scene_taxonomy("run-1")

    assert [t["id"] for t in result["taxonomy"]] == ["pheras", "other", "unsure"], result["taxonomy"]
    assert result["needs_pass_b"] == ["c0"], result
    print("ok  test_build_scene_taxonomy_uses_a_real_closed_set_taxonomy_when_given")


def test_build_scene_taxonomy_empty_run_short_circuits_with_no_llm_call():
    calls = []
    with mock.patch.object(st.cuts_read, "rows_for_run", lambda run_id: []), \
         mock.patch.object(st, "complete_gemini", lambda *a, **k: calls.append(1)):
        result = st.build_scene_taxonomy("run-1")
    assert calls == [], "must never call the LLM for a run with zero cuts"
    assert result["domain"] == "unknown/mixed", result
    print("ok  test_build_scene_taxonomy_empty_run_short_circuits_with_no_llm_call")


def test_build_scene_taxonomy_drops_clusters_with_no_questions_or_refs():
    rows = [_row("c0", "a", "b")]
    fake_response = {
        "domain": "unknown/mixed", "confidence": "low", "evidence": [],
        "taxonomy": [], "clusters": [
            {"group_refs": [], "questions": ["orphaned question"]},
            {"group_refs": ["g0"], "questions": []},
        ],
        "needs_pass_b_groups": [],
    }
    with mock.patch.object(st.cuts_read, "rows_for_run", lambda run_id: rows), \
         mock.patch.object(st, "complete_gemini",
                           lambda *a, **k: Completion(data=fake_response, usage={}, attempts=1)), \
         _no_transcripts():
        result = st.build_scene_taxonomy("run-1")
    assert result["clusters"] == [], result["clusters"]
    print("ok  test_build_scene_taxonomy_drops_clusters_with_no_questions_or_refs")


def main():
    test_group_cuts_dedupes_near_identical_label_and_summary()
    test_group_cuts_is_case_and_whitespace_insensitive()
    test_group_cuts_caps_at_max_groups()
    test_build_scene_taxonomy_expands_group_refs_to_real_cut_ids()
    test_build_scene_taxonomy_keeps_taxonomy_empty_when_model_gives_none()
    test_build_scene_taxonomy_uses_a_real_closed_set_taxonomy_when_given()
    test_build_scene_taxonomy_empty_run_short_circuits_with_no_llm_call()
    test_build_scene_taxonomy_drops_clusters_with_no_questions_or_refs()
    print("\nall scene_taxonomy tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
