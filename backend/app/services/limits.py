"""
Process-global bounded resource pools (scale_architecture.plan.md Pillars 3
+ 4).

Every ffmpeg/ffprobe subprocess, every R2 GET/PUT, and every in-flight LLM
completion call acquires one of these BoundedSemaphores before running.
Per-run caps (MAX_PARALLEL_PASS2_BATCHES, L1's 3 parallel tracks,
frames.py's MAX_PARALLEL_FILES) already bound how many tasks a SINGLE
run/file can have in flight; they say nothing about how many DIFFERENT
runs/files are in flight process-wide at once. These semaphores are the
actual global backstop -- config-driven so raising a ceiling later is a
knob (Settings.ffmpeg_concurrency / .r2_concurrency /
.ingest_llm_max_inflight_{anthropic,gemini}), never a rewrite.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from app.config import get_settings

_ffmpeg_sem: Optional[threading.BoundedSemaphore] = None
_r2_sem: Optional[threading.BoundedSemaphore] = None
_llm_sems: Dict[str, threading.BoundedSemaphore] = {}
_init_lock = threading.Lock()

_LLM_PROVIDER_SETTING = {
    "anthropic": "ingest_llm_max_inflight_anthropic",
    "gemini": "ingest_llm_max_inflight_gemini",
}


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


def _get_llm_sem(provider: str) -> threading.BoundedSemaphore:
    sem = _llm_sems.get(provider)
    if sem is None:
        with _init_lock:
            sem = _llm_sems.get(provider)
            if sem is None:
                attr = _LLM_PROVIDER_SETTING.get(provider, "ingest_llm_max_inflight_anthropic")
                sem = threading.BoundedSemaphore(getattr(get_settings(), attr))
                _llm_sems[provider] = sem
    return sem


@contextmanager
def llm_slot(provider: str) -> Iterator[None]:
    """Acquire before an in-flight LLM completion call (llm/client.complete
    and the Gemini path it delegates to); blocks if that provider's
    configured max-inflight are already running. Proactive, unlike
    client.py's existing retry -- this bounds concurrency BEFORE a call goes
    out, so raising MAX_PARALLEL_PASS2_BATCHES doesn't just turn into a
    bigger retry storm against the provider's own rate limit."""
    sem = _get_llm_sem(provider)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
