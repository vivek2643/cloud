"""
RunPod Serverless entrypoint for the GPU L1 tier (deployment.plan.md Phase 1).

This container runs with GPU_EXECUTION=local, so the real l1_* task functions
execute their model compute HERE (Whisper / pyannote / insightface / optical
flow). The Render `edso-gpu-dispatcher` pulls `gpu`-queue jobs and forwards them
to this handler via app/services/runpod_bridge.py; the task functions write their
results straight to Supabase + R2, exactly as in single-box execution. Any
follow-up work a task enqueues (e.g. l1_editing_proxy -> l1_active_speaker) lands
back on the `gpu` queue and is forwarded by the dispatcher again -- it never
forwards from here (GPU_EXECUTION=local guarantees the guard falls through to
real compute), so there is no infinite bounce.

Payload contract (input):
    {"task": "warmup"}                      -> load model weights, return fast
    {"task": "l1_orchestrate",   "kwargs": {"file_id": ..., "r2_key": ...}}
    {"task": "l1_editing_proxy", "kwargs": {"file_id": ..., "r2_key": ...}}
    {"task": "l1_active_speaker","kwargs": {"file_id": ...}}
"""
from __future__ import annotations

import logging

import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runpod_handler")


def _warmup() -> dict:
    """Force Whisper weights to load now so the first real job isn't slow.
    Used by the upload pre-warm ping (deployment.plan.md Phase 3)."""
    from app.services.l1.transcript import _WhisperEngine

    _WhisperEngine.get()
    return {"ok": True, "warmed": True}


def _task_func(task: str):
    """Resolve a task name to the raw Python function behind the Procrastinate
    Task object. `getattr(..., "func", ...)` returns the wrapped function when
    the attribute exists (Procrastinate Task) and the object itself otherwise,
    so we run the real body -- including its GPU_EXECUTION guard, which falls
    through to local compute in this container."""
    from app.services.l1 import pipeline

    tasks = {
        "l1_orchestrate": pipeline.l1_orchestrate,
        "l1_editing_proxy": pipeline.l1_editing_proxy,
        "l1_active_speaker": pipeline.l1_active_speaker,
    }
    task_obj = tasks.get(task)
    if task_obj is None:
        raise ValueError(f"unknown task {task!r}")
    return getattr(task_obj, "func", task_obj)


def handler(event: dict) -> dict:
    inp = (event or {}).get("input") or {}
    task = inp.get("task")
    kwargs = inp.get("kwargs") or {}

    if task == "warmup":
        return _warmup()

    fn = _task_func(task)
    # register_tasks() + an open app so the task's own follow-up `.defer()`
    # calls (e.g. l1_editing_proxy -> l1_active_speaker) reach Postgres,
    # mirroring how worker.py runs a task.
    from app.services.jobs import app, register_tasks

    register_tasks()
    logger.info("RunPod handler: running %s kwargs=%s", task, list(kwargs))
    with app.open():
        fn(**kwargs)
    return {"ok": True, "task": task}


runpod.serverless.start({"handler": handler})
