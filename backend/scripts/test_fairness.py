"""
Tests for app.services.fairness (scale_architecture.plan.md Pillar 6) --
priority-by-busyness and the L3 ingest per-user in-flight cap.

Run:  .venv/bin/python scripts/test_fairness.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.config import get_settings  # noqa: E402
from app.services import fairness  # noqa: E402


def test_priority_for_is_zero_with_nothing_in_flight():
    assert fairness.priority_for(0) == 0


def test_priority_for_is_negative_and_monotonically_non_increasing():
    prev = fairness.priority_for(0)
    for n in (1, 2, 5, 10):
        cur = fairness.priority_for(n)
        assert cur <= prev, f"priority_for({n})={cur} should be <= priority_for(prev)={prev}"
        assert cur <= 0
        prev = cur


def test_priority_for_clamps_at_the_penalty_ceiling():
    at_ceiling = fairness.priority_for(fairness._MAX_PRIORITY_PENALTY)
    way_over = fairness.priority_for(fairness._MAX_PRIORITY_PENALTY * 10)
    assert at_ceiling == way_over, "priority should not keep dropping past the clamp"
    assert at_ceiling == -fairness._MAX_PRIORITY_PENALTY


def test_priority_for_negative_input_treated_as_zero():
    assert fairness.priority_for(-5) == fairness.priority_for(0)


def test_capacity_exceeded_message_names_the_user_and_limits():
    exc = fairness.CapacityExceeded("user-1", in_flight=5, max_inflight=5)
    assert "user-1" in str(exc)
    assert "5" in str(exc)


def test_check_ingest_capacity_raises_at_the_configured_cap(monkeypatch=None):
    settings = get_settings()
    orig_max = settings.max_inflight_ingest_runs_per_user
    orig_count_fn = fairness.count_inflight_ingest_runs
    settings.max_inflight_ingest_runs_per_user = 2
    try:
        fairness.count_inflight_ingest_runs = lambda user_id: 2
        try:
            fairness.check_ingest_capacity("user-1")
        except fairness.CapacityExceeded as e:
            assert e.in_flight == 2 and e.max_inflight == 2
        else:
            raise AssertionError("expected CapacityExceeded at the cap")

        fairness.count_inflight_ingest_runs = lambda user_id: 1
        priority = fairness.check_ingest_capacity("user-1")
        assert priority == fairness.priority_for(1)
    finally:
        settings.max_inflight_ingest_runs_per_user = orig_max
        fairness.count_inflight_ingest_runs = orig_count_fn


def test_count_inflight_functions_run_against_the_real_db_read_only():
    """Not a correctness assertion about specific rows -- just proves the
    SQL is valid and the join/columns exist, against the real schema."""
    n1 = fairness.count_inflight_ingest_runs("00000000-0000-0000-0000-000000000000")
    n2 = fairness.count_inflight_l1("00000000-0000-0000-0000-000000000000")
    assert isinstance(n1, int) and n1 >= 0
    assert isinstance(n2, int) and n2 >= 0


def main():
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("ok ", t.__name__)
        except Exception as e:
            failed += 1
            print("FAIL:", t.__name__, "-", e)
    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    print("\nall fairness tests passed")


if __name__ == "__main__":
    main()
