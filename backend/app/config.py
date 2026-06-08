from __future__ import annotations
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_key: str

    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str = "aerodrive"

    # Direct Postgres connection string for procrastinate + pgvector queries
    # (the supabase-py REST client can't run raw SQL or HNSW search).
    # Example: postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres
    database_url: str = ""

    cors_origins: List[str] = ["http://localhost:3000"]

    # Dev mode: when set (non-empty), the backend bypasses JWT validation and
    # treats every request as this user. Set to "" to re-enable real auth.
    dev_user_id: str = "00000000-0000-0000-0000-000000000001"

    # L1 guardrail: anything longer than this just gets S1 (proxy + thumb).
    max_l1_duration_seconds: int = 3600

    # L1 Stage 6: CPU speaker diarization (who-says-what). Labels each word with
    # a per-file speaker id ("S0", "S1", ...). "mfcc" is the dependency-free
    # default (librosa + sklearn, no GPU, no model download); a stronger
    # backend (ecapa / pyannote) can be slotted in later behind this flag.
    enable_diarization: bool = True
    diarization_backend: str = "mfcc"
    diarization_max_speakers: int = 8

    # Phase 3a: L3 query parsing via Anthropic Claude
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # Provider-agnostic LLM backbone. "anthropic" (default) or "gemini". All L3
    # model calls route through app.services.llm.get_llm() keyed on this value.
    llm_provider: str = "anthropic"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # Layer C: how many keyframe images to attach to the multimodal editor call
    # (0 disables vision -> text-only editor). ~1.3k tokens/image on Sonnet.
    editor_vision_max_images: int = 16
    editor_vision_per_shot_max: int = 1

    # Agentic perception (director view_frames loop). The director is a "blind
    # editor" that pulls keyframes on demand; these bound its appetite so cost
    # and latency stay sane.
    editor_perception_max_rounds: int = 12
    editor_perception_max_images: int = 300
    view_frames_per_shot_max: int = 8
    # Keep images from only the most recent N view_frames turns in context; older
    # frames are pruned to their captions to bound multimodal token growth.
    editor_perception_keep_image_turns: int = 3
    # Prompt caching on the stable system/catalog prefix (Anthropic only).
    llm_prompt_caching: bool = True

    # Phase 2 Stage D: hosted Qwen2.5-VL endpoint (Replicate / Anyscale style).
    # Leave empty to use the self-hosted GPU model below instead.
    qwen_vl_endpoint_url: str = ""
    qwen_vl_api_key: str = ""

    # Self-hosted Qwen2.5-VL on the worker GPU. When enabled, narratives run
    # locally on CUDA (falling back to Claude on CPU-only boxes or load failure).
    # 3B fits a 16 GB GPU alongside the other models; bump to 7B on a bigger GPU.
    qwen_vl_local: bool = True
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_vl_max_tokens: int = 256

    # L2 deep enrichment (Qwen VLM narrative, faces, dinov2). Off by default:
    # edit-time managed multimodal vision replaces per-shot pre-captioning, and
    # this was the heavy, OOM-prone GPU stage. Set true to re-enable the local
    # L2 pipeline.
    enable_l2_vlm: bool = False

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
