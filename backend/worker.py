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

from app.services.jobs import app, register_tasks  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("worker")


def _warmup() -> None:
    """Force model weights to download/load now so the first job isn't slow."""
    try:
        logger.info("Warming up Whisper...")
        from app.services.l1.transcript import _WhisperEngine
        _WhisperEngine.get()
    except Exception:
        logger.exception("Whisper warmup failed; will lazy-load on first job.")

    try:
        logger.info("Warming up SigLIP 2...")
        from app.services.l1.embeddings import _SigLIPEngine
        _SigLIPEngine.get()
    except Exception:
        logger.exception("SigLIP warmup failed; will lazy-load on first job.")

    # Self-hosted Qwen2.5-VL for L2 narratives — only on a GPU box (the model is
    # too heavy for CPU). Downloads ~7 GB to HF_HOME the first time, then caches.
    try:
        from app.services.l2.qwen_vl import _QwenVLEngine
        if _QwenVLEngine.available():
            logger.info("Warming up Qwen2.5-VL...")
            _QwenVLEngine.get()
    except Exception:
        logger.exception("Qwen2.5-VL warmup failed; will lazy-load on first L2 job.")


async def main() -> None:
    register_tasks()
    _warmup()

    # Concurrency defaults to 1 (Whisper + SigLIP saturate one CPU). On a GPU
    # box you can raise it via WORKER_CONCURRENCY, but a single GPU usually
    # still wants 1 to avoid VRAM contention. WORKER_QUEUES (comma-separated)
    # restricts which queues this worker pulls; empty = all queues.
    concurrency = int(os.getenv("WORKER_CONCURRENCY", "1"))
    queues_env = os.getenv("WORKER_QUEUES", "").strip()
    queues = [q.strip() for q in queues_env.split(",") if q.strip()] or None

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
