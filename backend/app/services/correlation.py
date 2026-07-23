"""
Correlation IDs on every log line (scale_architecture.plan.md Pillar 7).

A ``logging.Filter`` reads from a contextvar so callers never have to thread
user_id/project_id/file_id/ingest_run_id through every ``logger.info(...)``
call by hand -- enter ``scope(...)`` once near the top of a task and every
log line inside it (this thread, or a pool worker submitted via
``run_with_scope``) picks the fields up automatically. Unset fields render
as ``"-"`` so one log format string works for every logger, worker or API.
"""
from __future__ import annotations

import contextvars
import logging
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, TypeVar

FIELDS = ("user_id", "project_id", "file_id", "ingest_run_id")

_ctx: "contextvars.ContextVar[Dict[str, str]]" = contextvars.ContextVar("correlation_ctx", default={})

T = TypeVar("T")


class CorrelationFilter(logging.Filter):
    """Stamps the current scope's fields onto every ``LogRecord`` that
    reaches the handler this filter is attached to. Must be added to a
    HANDLER (not a logger) to see records propagated up from child loggers
    -- see worker.py's ``logging.basicConfig`` wiring."""

    def filter(self, record: logging.LogRecord) -> bool:
        current = _ctx.get()
        for field in FIELDS:
            setattr(record, field, current.get(field, "-"))
        return True


@contextmanager
def scope(**fields: Any) -> Iterator[None]:
    """Merge `fields` onto the current scope for this block, restoring the
    prior scope on exit. Nests: an inner scope() only overrides the keys it
    passes, keeping whatever an outer scope already set (e.g. l3_cuts_ingest
    enters with project_id/user_id, then widens with ingest_run_id once the
    run row exists)."""
    current = _ctx.get()
    merged = {**current, **{k: str(v) for k, v in fields.items() if v is not None}}
    token = _ctx.set(merged)
    try:
        yield
    finally:
        _ctx.reset(token)


def run_with_scope(executor: Executor, fn: Callable[..., T], *args: Any, **kwargs: Any) -> "Future[T]":
    """Submit to `executor` carrying the calling thread's full contextvars
    snapshot -- plain ``Executor.submit`` does NOT propagate contextvars into
    worker threads. Same pattern as ``ingest_gemini.submit_with_cache_context``,
    generalized past just the Gemini cache handle."""
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)
