"""
procrastinate app: Postgres-backed durable job queue.

Why procrastinate over FastAPI BackgroundTasks:
  - Survives API process restarts.
  - Retries with exponential backoff.
  - Single-writer locking per task name so we never run two L1 pipelines
    for the same file concurrently.

Setup (one-time):
  - Add DATABASE_URL to .env (direct Supabase Postgres URL).
  - Run: `procrastinate -a app.services.jobs.app schema --apply`
  - Start worker:  `python backend/worker.py`
"""
from __future__ import annotations

from procrastinate import App, PsycopgConnector

from app.config import get_settings


def _make_connector() -> PsycopgConnector:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add the direct Supabase Postgres URL to .env "
            "(Project Settings -> Database -> Connection string -> URI) before "
            "starting the worker or enqueuing jobs."
        )
    return PsycopgConnector(conninfo=settings.database_url)


# Single global app instance. Tasks are registered in modules imported below.
app: App = App(connector=_make_connector())


# Importing the pipeline module registers the L1 orchestrator task.
# Done lazily so unit tests / API processes that don't need the worker
# don't pay the model-import cost.
def register_tasks() -> None:
    # Local imports avoid circular dependency at module load. Each import
    # registers @app.task decorators as a side-effect.
    from app.services.l1 import pipeline  # noqa: F401
    from app.services.l2 import orchestrator  # noqa: F401
    from app.services.render import tasks as render_tasks  # noqa: F401
