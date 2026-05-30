"""
L2 Stage A: DINOv2 structural embedding + heuristic framing / camera dynamics.

Model: facebook/dinov2-base via HF transformers, 86M params, 768-d CLS token.

Why DINOv2 alongside SigLIP 2:
  - SigLIP is text-aligned -> great for semantic search.
  - DINOv2 ignores language -> better at composition similarity
    (Close-Up vs Wide, indoor vs outdoor) which is what jump-cut elimination
    needs in Phase 3b.

Both embeddings live on shots.dinov2_embedding (halfvec 768) populated
on-demand. Framing scale and camera dynamics are derived heuristics, not
ML predictions — good enough for v1.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/dinov2-base"
EMBED_DIM = 768
BATCH_SIZE = 4


class _DinoEngine:
    _model = None
    _processor = None
    _torch = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from transformers import AutoImageProcessor, AutoModel
            import torch

            logger.info("Loading DINOv2 base (CPU)...")
            cls._processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            cls._model = AutoModel.from_pretrained(MODEL_ID).eval()
            cls._torch = torch
            logger.info("DINOv2 loaded.")
        return cls._model, cls._processor, cls._torch


def embed_images(image_paths: Sequence[str]) -> np.ndarray:
    """Return (N, 768) L2-normalized CLS-token DINOv2 embeddings."""
    if not image_paths:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    from PIL import Image

    model, processor, torch = _DinoEngine.get()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i : i + BATCH_SIZE]
            images = []
            for p in batch:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    logger.warning("Could not open %s; using black placeholder", p)
                    images.append(Image.new("RGB", (224, 224)))
            inputs = processor(images=images, return_tensors="pt")
            outputs = model(**inputs)
            # CLS-token pooled output = [batch, hidden]
            cls = outputs.last_hidden_state[:, 0, :]
            cls = cls / cls.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out.append(cls.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)


# --- Heuristic framing & camera-dynamics ----------------------------------
# Built from primitives we already have at L1: motion_magnitude, brightness,
# focus_score. Phase 2 Stage B can refine framing_scale further once face
# bounding boxes are available.

def infer_framing_from_focus(focus_score: float | None, motion: float | None) -> str:
    """
    Very rough rule pass when no face bboxes are available yet.
      - high focus + low motion -> "MS" (medium shot)
      - high focus + high motion -> "MCU" (medium close-up, common in vlogs)
      - low focus -> "WS" (wide shot or out-of-focus)
    Phase 2 Stage B will override this when faces are detected.
    """
    if focus_score is None:
        return "MS"
    if focus_score < 50:
        return "WS"
    if focus_score >= 200 and (motion or 0) > 0.5:
        return "MCU"
    return "MS"


def infer_camera_dynamics(motion: float | None) -> str:
    """
    Lump everything into 3 buckets based on optical flow magnitude:
      < 0.4   -> Static
      < 1.5   -> Pan (or slow handheld)
      else    -> Handheld
    """
    m = motion or 0.0
    if m < 0.4:
        return "Static"
    if m < 1.5:
        return "Pan"
    return "Handheld"
