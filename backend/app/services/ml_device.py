"""
Centralized ML device selection.

The pipeline runs identically on a laptop (CPU) or a GPU worker. Every model
engine asks here for its device instead of hardcoding "cpu", so deploying the
worker to a CUDA box is a config change, not a code change.

- `torch_device()`   -> "cuda" when a GPU is present, else "cpu".
- `whisper_device()` -> (device, compute_type) tuned for faster-whisper /
                        CTranslate2 ("cuda"/"float16" on GPU, "cpu"/"int8" off).

Set FORCE_CPU=1 to pin CPU even on a GPU host (useful for debugging).
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

_cached_device: str | None = None


def torch_device() -> str:
    """Return 'cuda' if a usable GPU is available, else 'cpu'. Cached per process."""
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    if os.getenv("FORCE_CPU") == "1":
        _cached_device = "cpu"
        logger.info("ML device selected: cpu (FORCE_CPU=1)")
        return _cached_device

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        logger.exception("torch.cuda probe failed; falling back to CPU")

    _cached_device = device
    if device == "cuda":
        try:
            import torch

            logger.info("ML device selected: cuda (%s)", torch.cuda.get_device_name(0))
        except Exception:
            logger.info("ML device selected: cuda")
    else:
        logger.info("ML device selected: cpu")
    return _cached_device


def whisper_device() -> Tuple[str, str]:
    """(device, compute_type) for faster-whisper. fp16 on GPU, int8 on CPU."""
    if torch_device() == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"
