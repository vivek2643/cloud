from __future__ import annotations
from typing import List
from pydantic import AliasChoices, Field
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

    # L1 Stage 6: speaker diarization (who-says-what). Labels each word with a
    # per-file speaker id ("S0", "S1", ...). Backends, strongest first:
    #   "pyannote" (default) -- pyannote.audio 3.1 (VAD + neural segmentation +
    #       overlap-aware resegmentation). GPU when present; needs HF_TOKEN and a
    #       one-time license acceptance for the gated models. Falls back to the
    #       embedding path below if unavailable (no token / not installed).
    #   "neural" -- Resemblyzer GE2E d-vectors + agglomerative clustering
    #       (ships its weights, CPU-only, no token). The fallback authority.
    #   "mfcc"   -- classical MFCC+pitch, fully dependency-free.
    enable_diarization: bool = True
    diarization_backend: str = "pyannote"
    diarization_max_speakers: int = 8
    # Hugging Face access token (env HF_TOKEN). Required only for the pyannote
    # backend's gated models; empty on CPU/local dev triggers the neural fallback.
    huggingface_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_TOKEN", "huggingface_token"),
    )

    # Anthropic credentials (used by the L3 edit orchestrator).
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # Provider-agnostic LLM backbone. "anthropic" (default), "gemini", or
    # "openai". All L3 model calls route through app.services.llm.get_llm()
    # keyed on this value (or an explicit per-feature override).
    llm_provider: str = "anthropic"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # OpenAI credentials. Used by feature-level calls that opt in via get_llm(
    # provider="openai", ...) -- e.g. the recommendations filtration pass.
    openai_api_key: str = ""
    # Small/cheap GPT-5-class model is plenty for text classification; override
    # via the OPENAI_MODEL env var.
    openai_model: str = "gpt-5-mini"

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
    # gemini-2.5-flash supports up to 65536 output tokens; long/dense clips
    # (>10 min) overflow 32768 and return truncated, unparseable JSON.
    l2_max_output_tokens: int = 65536
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
    # Extended-thinking toggle: > 0 enables adaptive thinking on the model
    # (modern Opus ignores the exact number; depth is steered by effort below).
    l3_thinking_budget_tokens: int = 8192
    # Reasoning depth for adaptive thinking: low|medium|high|xhigh|max.
    # "high" is the default; "xhigh" is recommended for heavy agentic loops.
    l3_effort: str = "high"
    # Prompt caching on the stable system/tools/catalog prefix (Anthropic only);
    # this is what makes a many-iteration Opus loop affordable.
    llm_prompt_caching: bool = True

    # --- Recommendations: LLM filtration of the hero-cuts feed ------------
    # A single, energy-independent text call judges each dialogue sentence
    # keep/drop; the feed then flags cuts via the "contains a keeper" rule.
    # Provider/model are overridable so this feature can move independently of
    # the L3 backbone (we pilot OpenAI here, leaving L3 on Anthropic).
    enable_recommendations: bool = True
    recommend_provider: str = "openai"
    # Empty -> the provider's default model (e.g. openai_model).
    recommend_model: str = ""
    recommend_max_output_tokens: int = 4096
    # Reasoning depth for the filtration call. Sentence keep/drop is classification,
    # not deep reasoning, so "low" keeps a GPT-5 model's latency sane (the default
    # effort can take ~30s+). minimal|low|medium|high.
    recommend_effort: str = "low"

    # --- L3 v2: prompt-driven auto-editor (OpenAI) -----------------------
    # A simple, deterministic 3-call pipeline (Director -> Editor -> Coverage)
    # that turns a one-line brief into a full Edit Document: guess the energy,
    # build the hero-cuts feed at that energy, select + order the cuts, then
    # lay light coverage. Provider/model overridable so it can pilot on OpenAI
    # while the agentic L3 stays on Anthropic.
    enable_autoedit: bool = True
    autoedit_provider: str = "openai"
    # The strongest available OpenAI model: this is creative selection +
    # ordering, not cheap classification. Override via OPENAI / env if needed.
    autoedit_model: str = "gpt-5"
    autoedit_max_output_tokens: int = 16384
    # Deep reasoning for the editorial calls (taste, story, ordering).
    autoedit_effort: str = "high"

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
