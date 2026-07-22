#!/usr/bin/env python3
"""Tests for color_phase1.plan.md Part 1c's qa/content_source.py -- a pure
lookup, no DB / ffmpeg / R2.

Run:  .venv/bin/python scripts/test_qa_content_source.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from qa.content_source import content_type_for  # noqa: E402


def test_demo_trail_project_id_is_synthetic():
    assert content_type_for("8621c012-58c1-4bd5-8897-b8e8e4f24dca", "demo trail") == "synthetic"
    assert content_type_for("5cd8f004-13c7-43f8-a1ed-d2f7e646fae7", "demo trail") == "synthetic"
    print("ok  content_type_for: both demo-trail project_ids read as synthetic")


def test_demo_trail_2_is_not_synthetic():
    # A differently-labeled, unrelated project -- must NOT be swept in by a
    # loose "demo trail" prefix/substring match on project_id alone.
    assert content_type_for("642e9587-e3a4-43ad-8342-c58ae58c04ca", "demo trail 2") != "synthetic"
    print("ok  content_type_for: 'demo trail 2' (a different project) is not swept in")


def test_unknown_project_defaults_photographic():
    assert content_type_for("00000000-0000-0000-0000-000000000000", "siri reel") == "photographic"
    print("ok  content_type_for: an unlisted project defaults to photographic")


def test_label_fallback_is_exact_not_substring():
    # A re-created project (new id) with the exact same label should still
    # be caught by the label fallback, case-insensitively and trim-tolerant
    # -- but NOT as a loose substring match, which would also sweep in an
    # unrelated project like "demo trail 2" or "demo trail (v2 export)".
    assert content_type_for("some-new-uuid-not-in-the-map", "  Demo Trail  ") == "synthetic"
    assert content_type_for("some-new-uuid-not-in-the-map", "demo trail (v2 export)") == "photographic"
    print("ok  content_type_for: label fallback is exact (case/whitespace-insensitive), not substring")


def test_empty_label_does_not_crash():
    assert content_type_for("some-uuid", "") == "photographic"
    assert content_type_for("some-uuid", None) == "photographic"
    print("ok  content_type_for: empty/None label is handled without raising")


def main():
    test_demo_trail_project_id_is_synthetic()
    test_demo_trail_2_is_not_synthetic()
    test_unknown_project_defaults_photographic()
    test_label_fallback_is_exact_not_substring()
    test_empty_label_does_not_crash()
    print("\nall qa content_source tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
