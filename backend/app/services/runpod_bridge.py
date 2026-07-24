"""
RunPod Serverless bridge (deployment.plan.md Phase 2).

Only exercised when settings.gpu_execution == "runpod": the Render
`edso-gpu-dispatcher` pulls `gpu`-queue jobs, and each l1_* task body forwards
here instead of running the model compute locally (see the GPU_EXECUTION guards
in app/services/l1/pipeline.py).

`run_remote()` submits the job to RunPod (`/run`) and polls `/status` until it
finishes, then raises on any non-success so the caller's Procrastinate retry
kicks in exactly as a local failure would. We poll rather than use `/runsync`
because an L1 pass on a long clip can run for minutes -- past RunPod's synchronous
response window.

`warm()` fires a best-effort async warmup ping (the upload pre-warm, Phase 3) and
never raises.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_BASE = "https://api.runpod.ai/v2"
_TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}
_POLL_INTERVAL_S = 2.0


def _endpoint(path: str) -> str:
    endpoint_id = get_settings().runpod_endpoint_id
    if not endpoint_id:
        raise RuntimeError("RUNPOD_ENDPOINT_ID is not set but GPU_EXECUTION=runpod")
    return f"{_BASE}/{endpoint_id}/{path}"


def _headers() -> dict:
    api_key = get_settings().runpod_api_key
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is not set but GPU_EXECUTION=runpod")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def run_remote(task: str, **kwargs: Any) -> None:
    """Forward one gpu task to RunPod and block until it finishes. Raises on any
    non-success so Procrastinate retries the job."""
    settings = get_settings()
    payload = {"input": {"task": task, "kwargs": kwargs}}
    logger.info("runpod: dispatching %s kwargs=%s", task, list(kwargs))

    with httpx.Client(timeout=60) as client:
        resp = client.post(_endpoint("run"), json=payload, headers=_headers())
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError(f"runpod {task}: submit returned no job id: {resp.text}")

        deadline = time.monotonic() + settings.runpod_timeout_seconds
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"runpod {task} timed out after {settings.runpod_timeout_seconds}s "
                    f"(job {job_id})"
                )
            st = client.get(_endpoint(f"status/{job_id}"), headers=_headers())
            st.raise_for_status()
            body = st.json()
            status = body.get("status")
            if status == "COMPLETED":
                out = body.get("output") or {}
                if not out.get("ok"):
                    raise RuntimeError(f"runpod {task} handler reported failure: {out}")
                logger.info("runpod: %s completed (job %s)", task, job_id)
                return
            if status in _TERMINAL_FAIL:
                raise RuntimeError(f"runpod {task} {status} (job {job_id}): {body}")
            time.sleep(_POLL_INTERVAL_S)


def _warm_call() -> None:
    try:
        payload = {"input": {"task": "warmup"}}
        with httpx.Client(timeout=15) as client:
            client.post(_endpoint("run"), json=payload, headers=_headers())
        logger.info("runpod: warm ping sent")
    except Exception:  # noqa: BLE001 - pre-warm is best-effort, never fatal
        logger.warning("runpod warm ping failed (non-fatal)", exc_info=True)


def warm() -> None:
    """Fire a best-effort async warmup ping so a RunPod worker + model weights
    are hot by the time real L1 jobs arrive (upload pre-warm, Phase 3). Returns
    immediately: the HTTP call runs on a daemon thread so it can never block or
    fault the request that triggered it. No-ops when GPU_EXECUTION != runpod."""
    settings = get_settings()
    if (
        settings.gpu_execution != "runpod"
        or not settings.runpod_endpoint_id
        or not settings.runpod_api_key
    ):
        return
    threading.Thread(target=_warm_call, name="runpod-warm", daemon=True).start()
