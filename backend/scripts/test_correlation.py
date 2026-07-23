"""
Tests for app.services.correlation (scale_architecture.plan.md Pillar 7) --
the contextvar-based scope() + logging.Filter that stamps
user_id/project_id/file_id/ingest_run_id onto every log line.

Run:  .venv/bin/python scripts/test_correlation.py
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services import correlation  # noqa: E402


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hi", args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_filter_stamps_dash_when_unset():
    f = correlation.CorrelationFilter()
    record = _record()
    assert f.filter(record) is True
    for field in correlation.FIELDS:
        assert getattr(record, field) == "-", field


def test_scope_stamps_fields_and_restores_on_exit():
    f = correlation.CorrelationFilter()
    with correlation.scope(user_id="u1", file_id="f1"):
        record = _record()
        f.filter(record)
        assert record.user_id == "u1"
        assert record.file_id == "f1"
        assert record.project_id == "-"
    record = _record()
    f.filter(record)
    assert record.user_id == "-"


def test_nested_scope_merges_without_dropping_outer_fields():
    f = correlation.CorrelationFilter()
    with correlation.scope(project_id="p1", user_id="u1"):
        with correlation.scope(ingest_run_id="r1"):
            record = _record()
            f.filter(record)
            assert record.project_id == "p1"
            assert record.user_id == "u1"
            assert record.ingest_run_id == "r1"
        # inner scope's field is gone once it exits; outer fields remain.
        record = _record()
        f.filter(record)
        assert record.project_id == "p1"
        assert record.ingest_run_id == "-"


def test_scope_restores_prior_value_on_exception():
    with correlation.scope(user_id="outer"):
        try:
            with correlation.scope(user_id="inner"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert correlation._ctx.get()["user_id"] == "outer"


def test_none_valued_field_is_dropped_not_stringified():
    with correlation.scope(user_id=None, project_id="p1"):
        assert "user_id" not in correlation._ctx.get()
        assert correlation._ctx.get()["project_id"] == "p1"


def test_run_with_scope_propagates_into_worker_thread():
    seen = {}

    def worker():
        seen["user_id"] = correlation._ctx.get().get("user_id")

    with correlation.scope(user_id="u-thread"):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = correlation.run_with_scope(ex, worker)
            fut.result()
    assert seen["user_id"] == "u-thread"


def test_bare_submit_does_not_see_the_scope():
    seen = {}

    def worker():
        seen["user_id"] = correlation._ctx.get().get("user_id")

    with correlation.scope(user_id="u-thread"):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(worker)
            fut.result()
    assert seen["user_id"] is None


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
    print("\nall correlation tests passed")


if __name__ == "__main__":
    main()
