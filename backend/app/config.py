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

    # Session/direct Postgres connection string -- Procrastinate (LISTEN/
    # NOTIFY) and the migration runner's advisory lock (session-scoped,
    # needs one pinned backend connection) MUST stay on this route, never
    # the transaction pooler below. Supabase's Supavisor session pooler
    # (port 5432) or a direct DB connection both work here.
    database_url: str = ""

    # scale_architecture.plan.md Pillar 1: the TRANSACTION pooler URL
    # (Supavisor, port 6543) business-query connections borrow from via
    # app/services/db.py's process-global pool. Falls back to `database_url`
    # when unset (dev / a deployment with no separate pooler configured) --
    # every business query already ran fine over a single URL before this
    # pillar, so an unset pool URL is a no-op, not a startup failure.
    database_pool_url: str = ""
    # Per-WORKER-PROCESS cap on the business-query pool (app/services/db.py).
    # Deliberately NOT named DB_POOL_MAX -- that env var already exists
    # (jobs.py) for Procrastinate's own, separate connector pool, which
    # stays on the session/direct route; reusing the name would silently
    # couple two unrelated pools. Budget: total (this x worker process
    # count) should stay under ~40-80% of the compute tier's direct-
    # connection cap when running WITHOUT a pooler in front, or well under
    # the pooler's own configured client-slot budget when one is (see the
    # plan's Supabase Pro connection-facts table).
    db_pool_max_size: int = 8

    cors_origins: List[str] = ["http://localhost:3000"]

    # Dev mode: when set (non-empty), the backend bypasses JWT validation and
    # treats every request as this user. Set to "" to re-enable real auth.
    dev_user_id: str = "00000000-0000-0000-0000-000000000001"

    # L1 guardrail: anything longer than this just gets S1 (proxy + thumb).
    max_l1_duration_seconds: int = 3600

    # L1 Stage 6: speaker diarization (who-says-what). Labels each word with a
    # per-file speaker id ("S0", "S1", ...) via pyannote.audio 3.1 (VAD + neural
    # segmentation + overlap-aware resegmentation), on GPU when present. Needs
    # HF_TOKEN and a one-time license acceptance for the gated models. Soft
    # signal: if pyannote is unavailable, speakers are simply left unset.
    enable_diarization: bool = True
    diarization_max_speakers: int = 8
    # Hugging Face access token (env HF_TOKEN). Required for the pyannote
    # diarization models; empty on CPU/local dev leaves speakers unset.
    huggingface_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_TOKEN", "huggingface_token"),
    )

    # Provider-agnostic LLM backbone. "openai" (default) or "gemini". All model
    # calls route through app.services.llm.get_llm() keyed on this value (or an
    # explicit per-feature override).
    llm_provider: str = "openai"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # OpenAI credentials. Used by feature-level calls that opt in via get_llm(
    # provider="openai", ...) -- e.g. the L3 auto-editor.
    openai_api_key: str = ""
    # Small/cheap GPT-5-class model is plenty for text classification; override
    # via the OPENAI_MODEL env var.
    openai_model: str = "gpt-5-mini"

    # Anthropic (Claude) credentials + default model. Used when a feature selects
    # provider="anthropic" (e.g. the L3 editing brain). cache_system maps to
    # Claude prompt caching (cache_control) for cheap multi-pass / multi-turn.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"

    # --- L3 editing brain (converse + tools) model backbone --------------
    # The agentic chat editor (converse.respond -> tools.run_edit_loop) runs on
    # the strongest available model -- creative selection + ordering, not cheap
    # classification. Provider-neutral: flip to "openai"/"gemini" + a model id
    # to swap backbones.
    autoedit_provider: str = "anthropic"
    autoedit_model: str = "claude-opus-4-8"
    autoedit_max_output_tokens: int = 16384

    # --- Cuts v3: LLM-grouped ingest (app.services.llm.client) -----------
    # Two structured Sonnet-class calls per project ingest (text-only pass 1,
    # then vision pass 2) decide MEANING (grouping/takes/junk/framing/etc.);
    # boundaries stay code-derived (word/atom edges), never an LLM millisecond.
    # Model ids are per-stage so either pass can be swapped independently via
    # env var -- prompts are model-agnostic (see cuts_v3.plan.md, "Model layer").
    ingest_pass1_model: str = "claude-sonnet-5"
    ingest_pass2_model: str = "gemini-3.1-flash-lite"
    # Prompt-cache TTL headroom: pass-2 shards must run back-to-back within
    # this window to keep reading the pass-1 prefix at the cheap cache rate.
    ingest_cache_ttl_seconds: int = 300

    # gemini_pass2.plan.md: Pass 2 backend. "anthropic" keeps ic.complete(
    # "pass2", ...) on the Claude tool-forced path; "gemini" routes it to
    # app.services.llm.ingest_gemini.complete_gemini instead.
    # ingest_pass2_model is the model id for WHICHEVER provider is selected.
    # Pass 1 always stays Anthropic -- this flag has no effect on it.
    # perception_upgrade.plan.md Part A: flipped to "gemini" -- A/B verified
    # across podcast + drone/b-roll reel + montage reel (6/6 ingests OK,
    # coverage at Sonnet parity, 28-47x cheaper, no IngestFailure). Flip back
    # to "anthropic" to roll back; single env-var change, no code involved.
    ingest_pass2_provider: str = "gemini"
    # Gemini thinking effort for Pass 2: "low"/"medium"/"high" (mapped to a
    # fixed thinking_budget token count) or a numeric string used as the budget
    # directly. MUST stay "low": on gemini-3.x flash-lite thinking_budget is a
    # soft target the model can overshoot without bound, and it counts INSIDE
    # max_output_tokens. At "medium"/"high" the hardest b-roll batches spiral --
    # thinking eats the entire output budget (observed think_tok=30k+,
    # finish=MAX_TOKENS) and zero JSON is emitted, failing the whole run. At
    # "low" the budget stays small enough that even hard batches finish and
    # emit. B-roll field coverage is nudged via the prompt, not more thinking.
    # Only affects the gemini pass-2 path.
    ingest_pass2_thinking: str = "low"

    # migration_runner.plan.md: the startup guard's sanctioned local-dev
    # bypass. "on" (default) means every process refuses to boot on schema
    # drift; "off" disables that check for THIS process only, loudly (a
    # warning is logged). Never set "off" in production -- it exists so a
    # dev with a deliberately divergent local DB has a named escape hatch
    # instead of quietly commenting out the check itself.
    migration_guard: str = "on"

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
