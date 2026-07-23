"""
Tests for app.services.limits (scale_architecture.plan.md Pillar 3) -- the
process-global BoundedSemaphores every ffmpeg/ffprobe subprocess and every
R2 GET/PUT acquires before running.

Run:  .venv/bin/python scripts/test_limits.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.config import get_settings  # noqa: E402
from app.services import limits  # noqa: E402


def _reset():
    """Each test wants a fresh semaphore sized off the CURRENT setting --
    the module lazily builds one on first use and caches it, so force a
    rebuild rather than reusing whatever an earlier test sized."""
    limits._ffmpeg_sem = None
    limits._r2_sem = None


def test_ffmpeg_slot_bounds_concurrency_to_the_configured_limit():
    _reset()
    settings = get_settings()
    orig = settings.ffmpeg_concurrency
    settings.ffmpeg_concurrency = 2
    try:
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def worker():
            nonlocal in_flight, peak
            with limits.ffmpeg_slot():
                with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                time.sleep(0.05)
                with lock:
                    in_flight -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak == 2, f"expected peak concurrency 2, got {peak}"
    finally:
        settings.ffmpeg_concurrency = orig
        _reset()


def test_r2_slot_bounds_concurrency_to_the_configured_limit():
    _reset()
    settings = get_settings()
    orig = settings.r2_concurrency
    settings.r2_concurrency = 3
    try:
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def worker():
            nonlocal in_flight, peak
            with limits.r2_slot():
                with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                time.sleep(0.05)
                with lock:
                    in_flight -= 1

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak == 3, f"expected peak concurrency 3, got {peak}"
    finally:
        settings.r2_concurrency = orig
        _reset()


def test_slot_releases_on_exception():
    _reset()
    settings = get_settings()
    orig = settings.ffmpeg_concurrency
    settings.ffmpeg_concurrency = 1
    try:
        try:
            with limits.ffmpeg_slot():
                raise ValueError("boom")
        except ValueError:
            pass
        # If the first acquire leaked, this would deadlock -- run under a
        # watchdog thread instead of blocking the suite forever on failure.
        acquired = []

        def worker():
            with limits.ffmpeg_slot():
                acquired.append(True)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2)
        assert acquired == [True], "slot did not release after an exception"
    finally:
        settings.ffmpeg_concurrency = orig
        _reset()


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
    print("\nall limits tests passed")


if __name__ == "__main__":
    main()
