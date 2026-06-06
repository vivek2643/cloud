"""
Procrastinate task entry points for rendering.

Importing this module registers @app.task handlers as a side-effect.
register_tasks() in services/jobs.py imports it for that purpose.
"""
from __future__ import annotations

import logging

from app.services.jobs import app
from app.services.render.edl_runner import run_render

logger = logging.getLogger(__name__)


@app.task(name="render_edl", queue="cpu", retry={"max_attempts": 2, "wait": "exponential"})
def render_edl_task(render_id: str) -> None:
    """Top-level render job. The runner does all the heavy lifting."""
    logger.info("render_edl task picked up: render_id=%s", render_id)
    run_render(render_id)
