"""
L1 Stage 4: SigLIP image embeddings for every shot keyframe.

Currently using SigLIP 1 (google/siglip-base-patch16-224) because SigLIP 2's
Gemma-tokenizer needs transformers>=4.50 (which requires Python 3.10+). The
embedding dim is still 768 so no schema change when we upgrade later -- just
swap MODEL_ID back to "google/siglip2-base-patch16-256".

Output: 768-d cosine-normalized vectors -> halfvec(768) in pgvector.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip-base-patch16-224"
EMBED_DIM = 768
BATCH_SIZE = 8


class _SigLIPEngine:
    """Lazy-loaded singleton so model weights load once per worker process."""
    _model = None
    _processor = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from transformers import AutoModel, AutoProcessor
            import torch

            logger.info("Loading %s (CPU)...", MODEL_ID)
            cls._processor = AutoProcessor.from_pretrained(MODEL_ID)
            cls._model = AutoModel.from_pretrained(MODEL_ID).eval()
            cls._torch = torch
            logger.info("SigLIP loaded.")
        return cls._model, cls._processor, cls._torch


def embed_images(image_paths: Sequence[str]) -> np.ndarray:
    """Return a (N, 768) float32 array of L2-normalized image embeddings."""
    if not image_paths:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    from PIL import Image

    model, processor, torch = _SigLIPEngine.get()
    out: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    logger.warning("Could not open keyframe %s; using black placeholder", p)
                    images.append(Image.new("RGB", (256, 256)))
            inputs = processor(images=images, return_tensors="pt")
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out.append(features.cpu().numpy().astype(np.float32))

    return np.concatenate(out, axis=0) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)


def embed_text(query: str) -> np.ndarray:
    """Encode a text query for retrieval against shot_embeddings."""
    model, processor, torch = _SigLIPEngine.get()
    with torch.no_grad():
        inputs = processor(
            text=[query],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return features[0].cpu().numpy().astype(np.float32)
