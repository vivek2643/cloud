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

    # Anthropic credentials (used by the L3 edit orchestrator).
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # Provider-agnostic LLM backbone. "anthropic" (default) or "gemini". All L3
    # model calls route through app.services.llm.get_llm() keyed on this value.
    llm_provider: str = "anthropic"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # --- L2: VLM perception layer (Gemini) -------------------------------
    # A single Gemini pass over the whole clip that produces the rich, single-
    # take "footage log" (clip-level look/setting, person identities, an event
    # timeline, and semantic cut-cost events). Runs after L1 completes.
    enable_l2_perception: bool = True
    # Cost/latency scale ~linearly with duration, so we gate the deep pass by
    # length. Tunable; temporarily raised to 60 min (we'll split long videos
    # later instead of just gating them).
    l2_max_duration_seconds: int = 3600
    # Reuses gemini_api_key. Flash is fast/cheap and plenty for perception;
    # bump to a pro model here if quality needs it.
    l2_gemini_model: str = "gemini-2.5-flash"
    # Frame sampling rate Gemini uses to read the video. 1 fps is Gemini's
    # default and is enough for single-take footage logging; raise for fast
    # action at a roughly linear token cost.
    l2_video_fps: float = 1.0
    # Per-frame token budget: "low" (~64 tok/frame), "default", or "high".
    l2_media_resolution: str = "default"
    # Rich logs for a few minutes of footage can be large; give the model room.
    l2_max_output_tokens: int = 32768
    # Files API upload poll: how long to wait for Gemini to finish ingesting the
    # uploaded video before giving up.
    l2_file_active_timeout_seconds: int = 300

    # --- L3: edit orchestrator (Claude Opus agentic tool-loop) ------------
    # The creative brain: picks material/order/rough timing over the L1+L2 text
    # analysis and drives the deterministic cut-engine tools; never places
    # exact frames itself (the cost grids do).
    l3_model: str = "claude-opus-4-8"
    # Guardrails so the inner reason->tool loop is always bounded.
    l3_max_iterations: int = 40
    l3_max_output_tokens: int = 16384
    # Extended-thinking budget per call (0 disables thinking).
    l3_thinking_budget_tokens: int = 8192
    # Prompt caching on the stable system/tools/catalog prefix (Anthropic only);
    # this is what makes a many-iteration Opus loop affordable.
    llm_prompt_caching: bool = True

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
