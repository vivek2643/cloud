"""
Process-global bounded resource pools (scale_architecture.plan.md Pillar 3).

Every ffmpeg/ffprobe subprocess and every R2 GET/PUT acquires one of these
BoundedSemaphores before running. Per-run caps (MAX_PARALLEL_PASS2_BATCHES,
L1's 3 parallel tracks, frames.py's MAX_PARALLEL_FILES) already bound how
many tasks a SINGLE run/file can have in flight; they say nothing about how
many DIFFERENT runs/files are in flight process-wide at once. These two
semaphores are the actual global backstop -- config-driven so raising the
ceiling later is a knob (Settings.ffmpeg_concurrency / .r2_concurrency),
never a rewrite.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from app.config import get_settings

_ffmpeg_sem: Optional[threading.BoundedSemaphore] = None
_r2_sem: Optional[threading.BoundedSemaphore] = None
_init_lock = threading.Lock()


def _get_ffmpeg_sem() -> threading.BoundedSemaphore:
    global _ffmpeg_sem
    if _ffmpeg_sem is None:
        with _init_lock:
            if _ffmpeg_sem is None:
                _ffmpeg_sem = threading.BoundedSemaphore(get_settings().ffmpeg_concurrency)
    return _ffmpeg_sem


def _get_r2_sem() -> threading.BoundedSemaphore:
    global _r2_sem
    if _r2_sem is None:
        with _init_lock:
            if _r2_sem is None:
                _r2_sem = threading.BoundedSemaphore(get_settings().r2_concurrency)
    return _r2_sem


@contextmanager
def ffmpeg_slot() -> Iterator[None]:
    """Acquire before spawning any ffmpeg/ffprobe subprocess; blocks if
    FFMPEG_CONCURRENCY are already running."""
    sem = _get_ffmpeg_sem()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


@contextmanager
def r2_slot() -> Iterator[None]:
    """Acquire before an R2 GET/PUT; blocks if R2_CONCURRENCY are already
    in flight."""
    sem = _get_r2_sem()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
