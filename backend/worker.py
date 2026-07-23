"""
Procrastinate worker entry point.

Run:
    cd backend
    python worker.py

Concurrency is fixed at 1 because Whisper + SigLIP saturate one CPU.
Scale = more worker processes, not more concurrent jobs per worker.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Allow `python backend/worker.py` from project root as well as `python worker.py`
# from inside backend/ — Python won't find the `app` package otherwise.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app.services.correlation import CorrelationFilter  # noqa: E402
from app.services.jobs import app, register_tasks  # noqa: E402

# scale_architecture.plan.md Pillar 7: every log line gets
# user/project/file/run correlation fields ("-" when unset). The filter must
# be on the HANDLER, not a logger, to see records propagated up from every
# module's own `logging.getLogger(__name__)`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s "
           "[user=%(user_id)s project=%(project_id)s file=%(file_id)s run=%(ingest_run_id)s] "
           "%(message)s",
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(CorrelationFilter())
logger = logging.getLogger("worker")


def _warmup() -> None:
    """Force model weights to download/load now so the first job isn't slow."""
    try:
        logger.info("Warming up Whisper...")
        from app.services.l1.transcript import _WhisperEngine
        _WhisperEngine.get()
    except Exception:
        logger.exception("Whisper warmup failed; will lazy-load on first job.")


def _check_schema() -> None:
    """Fail loud (uncaught SchemaDriftError) if the live schema has drifted
    from backend/migrations/ -- see app/services/db_migrations.py and
    migration_runner.plan.md for exactly what this does and does not catch.
    Bypass via MIGRATION_GUARD=off (local dev only, never in production)."""
    import psycopg
    from app.config import get_settings
    from app.services.db_migrations import assert_up_to_date

    conn = psycopg.connect(get_settings().database_url, autocommit=True)
    try:
        assert_up_to_date(conn)
    finally:
        conn.close()


async def main() -> None:
    _check_schema()
    register_tasks()

    # Concurrency defaults to 1 (Whisper + SigLIP saturate one CPU). On a GPU
    # box you can raise it via WORKER_CONCURRENCY, but a single GPU usually
    # still wants 1 to avoid VRAM contention. WORKER_QUEUES (comma-separated)
    # restricts which queues this worker pulls; empty = all queues.
    concurrency = int(os.getenv("WORKER_CONCURRENCY", "1"))
    queues_env = os.getenv("WORKER_QUEUES", "").strip()
    queues = [q.strip() for q in queues_env.split(",") if q.strip()] or None

    # Only the ingest workers run Whisper; L2 (Gemini) and L3 (Claude) workers
    # are network-bound, so skip the model warmup/load for them.
    if queues is None or "gpu" in queues:
        _warmup()

    logger.info(
        "Worker ready; concurrency=%d queues=%s; entering main loop.",
        concurrency,
        queues or "ALL",
    )
    async with app.open_async():
        await app.run_worker_async(
            concurrency=concurrency,
            queues=queues,
            install_signal_handlers=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
