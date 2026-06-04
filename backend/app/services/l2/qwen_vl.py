"""
Self-hosted Qwen2.5-VL engine for L2 Stage D narratives.

Runs the VLM locally on the worker's GPU instead of calling a paid Claude /
hosted-Qwen API per shot. Weights download once to HF_HOME (=/models on the
pod, a persistent volume) and stay resident in the long-lived worker process.

Design notes:
- Lazy singleton: the ~7 GB (3B) model loads on first use, then is reused for
  every shot of every L2 job.
- GPU-only: `available()` returns False on CPU boxes (the local API/dev never
  runs L2), so those fall back to Claude.
- A module-level lock serialises `generate()` — a single GPU model is not safe
  to drive from the 5 narrative threads concurrently, and serial local
  inference is still far faster than 5-way parallel network calls.
- Any load/inference failure degrades gracefully (returns None / marks the
  engine unavailable) so the caller can fall back to Claude.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import List, Optional

from app.config import get_settings
from app.services.ml_device import torch_device

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of the model's free-form completion."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        # ```json\n{...}\n```  ->  {...}
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class _QwenVLEngine:
    _model = None
    _processor = None
    _lock = threading.Lock()
    _load_failed = False

    @classmethod
    def available(cls) -> bool:
        """True only when a CUDA GPU is present and the model hasn't failed to load."""
        return torch_device() == "cuda" and not cls._load_failed

    @classmethod
    def get(cls):
        if cls._model is not None:
            return cls._model, cls._processor
        with cls._lock:
            if cls._model is not None:
                return cls._model, cls._processor
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            settings = get_settings()
            model_id = settings.qwen_vl_model
            logger.info("Loading Qwen2.5-VL '%s' onto GPU ...", model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("cuda")
            model.eval()
            processor = AutoProcessor.from_pretrained(model_id)
            cls._model, cls._processor = model, processor
            logger.info("Qwen2.5-VL ready.")
            return cls._model, cls._processor

    @classmethod
    def infer(
        cls,
        image_paths: List[str],
        prompt_text: str,
        max_new_tokens: int,
    ) -> Optional[dict]:
        """Run one shot's narrative analysis. Returns a parsed JSON dict or None."""
        try:
            import torch
            from qwen_vl_utils import process_vision_info

            model, processor = cls.get()

            content = [{"type": "image", "image": p} for p in image_paths]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Serialise GPU generate() across the narrative thread pool.
            with cls._lock:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
            trimmed = generated[:, inputs.input_ids.shape[1]:]
            decoded = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return _parse_json(decoded)
        except Exception:
            logger.exception("Qwen2.5-VL inference failed")
            # If the very first load blew up (e.g. OOM), stop trying so the
            # whole L2 run falls back to Claude instead of thrashing the GPU.
            if cls._model is None:
                cls._load_failed = True
            return None
