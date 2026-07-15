"""
Tests for the target-length parser (app.services.l3.converse._extract_target_s)
-- edso_done_gate.plan.md's "anchor" step: a stated length in the user's own
words lands in brief.target_duration_s, never invented. No DB, no network.

Run:  .venv/bin/python scripts/test_converse_target.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import converse  # noqa: E402


def test_extract_target_variants():
    f = converse._extract_target_s
    assert f("make it 60s") == 60.0
    assert f("about 90 seconds punchy") == 90.0
    assert f("2 min recap") == 120.0
    assert f("keep it to a minute") == 60.0
    assert f("30-45s teaser") == 45.0          # range -> upper bound
    assert f("just make an edit") is None      # no number -> never invented
    print("ok  converse: target length parsed from the user's words")


def test_latest_user_text_reads_bare_string_and_block_content():
    f = converse._latest_user_text
    assert f([{"role": "user", "content": "cut a 5s teaser"}]) == "cut a 5s teaser"
    assert f([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "make it 90 seconds"}]},
    ]) == "make it 90 seconds"
    assert f([]) == ""
    print("ok  converse: latest user text reads both message shapes")


def main():
    test_extract_target_variants()
    test_latest_user_text_reads_bare_string_and_block_content()
    print("\nall converse-target tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
