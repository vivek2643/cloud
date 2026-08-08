from __future__ import annotations
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

    # --- LOCAL-DEV DATA ISOLATION (opt-in, default == production) ---------
    # A separate Postgres SCHEMA to point ALL app tables + the Procrastinate
    # queue tables at, so local development never reads/writes production's
    # `public` tables or shares its job queue -- while production keeps using
    # `public`. UNSET (empty) is the PRODUCTION default: no search_path is
    # set on any connection, the Supabase (PostgREST) client stays on its
    # default schema, and the migration ledger stays `public.schema_migrations`
    # -- i.e. byte-for-byte today's behavior. Set to e.g. "dev" ONLY in a
    # local .env to flip local dev onto an isolated schema. The Supabase
    # `auth` schema stays shared regardless (same users locally + in prod).
    # NOTE: routing the Supabase/PostgREST client to a non-`public` schema
    # ALSO requires exposing that schema in Supabase (Project Settings -> API
    # -> Exposed schemas); psycopg/Procrastinate only need the schema to exist.
    db_schema: str = ""

    # Object-key PREFIX prepended to every R2 key at the storage boundary
    # (app/services/r2.py, app/services/processing.py) so local dev media
    # lives under its own keyspace (e.g. "dev/raw/...") and can never read,
    # overwrite, or (cascade-)delete a production object. UNSET (empty) is the
    # PRODUCTION default: keys are used verbatim, exactly as today. Stored DB
    # keys stay "logical" (unprefixed); the prefix is applied transparently on
    # every get/put/delete/multipart op, so a delete can only ever target
    # `<prefix>/<key>` and never a bare production key.
    r2_key_prefix: str = ""

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

    # scale_architecture.plan.md Pillar 3: process-global bounded pools
    # (app/services/limits.py). Every ffmpeg/ffprobe subprocess spawn and
    # every R2 GET/PUT acquires one of these before running -- per-run caps
    # (e.g. MAX_PARALLEL_PASS2_BATCHES, L1's 3 parallel tracks) only bound a
    # SINGLE run/file; these are the process-wide backstop across however
    # many runs/files are concurrently in flight. Raise as CPU/bandwidth
    # headroom allows -- pure config knobs, no code change to scale.
    ffmpeg_concurrency: int = 4
    r2_concurrency: int = 16

    # scale_architecture.plan.md Pillar 6: hard per-user cap on concurrent L3
    # ingest runs (a real, costed API call the user directly triggers, worth
    # rate-limiting outright -- unlike L1, which is upload-triggered and only
    # gets a priority penalty, not a hard cap; see app/services/fairness.py).
    # Sized for the plan's target burst (5 users x 10 videos): one user
    # shouldn't be able to occupy the whole ingest queue by kicking off ingest
    # on their entire library at once.
    max_inflight_ingest_runs_per_user: int = 10

    # Comma-separated allowed browser origins for CORS (the deployed frontend,
    # e.g. "https://app.vercel.app,https://www.myapp.com"). Kept a plain str --
    # NOT List[str] -- because pydantic-settings JSON-decodes list-typed env
    # vars, so a normal comma-separated CORS_ORIGINS value would raise at
    # startup. Split into a list at the consumer (app/main.py). The dev-origin
    # regex there still covers localhost/LAN, so local dev is unaffected.
    cors_origins: str = "http://localhost:3000"

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

    # cut_structure_and_scene_specificity.plan.md Part 3: the two-pass scene-
    # specificity enrichment (runs AFTER cuts are shown, background, never
    # blocks ingest). Middle text layer -- one TEXT-only, no-re-ask call on a
    # CAPABLE model (quality matters here: it infers the project's domain and
    # writes the targeted questions Pass B answers) that turns Pass A's
    # generic summaries into sharp, footage-derived questions.
    ingest_scene_text_model: str = "gemini-3.1-pro-preview"
    # Pass B -- targeted vision, one call per taxonomy cluster, small output.
    # flash/flash-lite is enough: it's answering a SPECIFIC, already-written
    # question against a couple of frames, not inferring the domain itself.
    ingest_pass_b_model: str = "gemini-3.1-flash-lite"
    ingest_pass_b_thinking: str = "low"
    # Short-TTL safety net for the Part 3 CachedContent (system + gist +
    # taxonomy) -- deleted on completion regardless; this only bounds the
    # cost of a crash between creation and that delete. "Never hold a
    # provider cache idle" (locked decision) -- keep this short.
    ingest_scene_cache_ttl_seconds: int = 1800

    # scale_architecture.plan.md Pillar 4: PROACTIVE limiter on in-flight LLM
    # calls, per provider -- separate from llm/client.py's existing REACTIVE
    # retry (backoff on a 429/5xx after it happens). Without this, bumping
    # MAX_PARALLEL_PASS2_BATCHES just turns a slow run into a retry storm: a
    # single run_ingest already fires up to MAX_PARALLEL_PASS2_BATCHES calls
    # at once, and run_many() can run up to 4 projects concurrently, each
    # with its own batch pool -- so today's real worst case is already
    # 4x MAX_PARALLEL_PASS2_BATCHES in flight process-wide. Pass 1 and an
    # anthropic-provider pass 2 share the anthropic slot; a gemini-provider
    # pass 2 uses the gemini slot (see llm/client.py's complete()).
    ingest_llm_max_inflight_anthropic: int = 8
    ingest_llm_max_inflight_gemini: int = 8

    # --- seam_cut_pipeline.plan.md: vcut, the seam-driven VLM cut pipeline -
    # A separate cuts pipeline from the L3 ingest above (app.services.vcut/,
    # cuts_pipeline flag below selects which one project ingest dispatches
    # to). Cost first (plan principle 6): flash-lite for both passes by
    # default, one knob per stage so pass 2 can be upgraded independently
    # later without touching pass 1.
    vcut_pass1_model: str = "gemini-3.1-flash-lite"
    vcut_pass2_model: str = "gemini-3.1-flash-lite"
    # vcut_pass2_rich.plan.md section 3.1: TTL for the shared Gemini
    # CachedContent both passes ride -- comfortably covers Pass 1 plus the
    # gap until the background vcut_enrich task runs.
    vcut_cache_ttl_s: int = 1200
    # speech_cuts_pipeline.plan.md section 15: the speech channel's ONE
    # text-only call (speech/segment_llm.py) -- a pro-class model, not
    # flash-lite, per the plan's own "not sonnet, use a pro model" decision
    # (beat segmentation + retake clustering needs stronger judgment than
    # the video channel's frame-labeling calls). The speech frame pass
    # (speech/frames.py) reuses vcut_pass2_model (flash-lite) -- no
    # separate knob for that one.
    vcut_speech_model: str = "gemini-3.1-pro-preview"
    # pass1_video_input.plan.md section 8: Pass 1's input medium. "frames"
    # (default, shipped) = today's sampled-JPEG path. "video" = feed each
    # file's non-speech video (speech cut out) instead -- flip per-env to
    # A/B and roll back instantly, no code change either way.
    vcut_pass1_input_mode: str = "video"
    vcut_video_fps: float = 2.0
    vcut_video_media_resolution: str = "low"
    # vcut_pass2_video_specifics.plan.md section 4.5/9: the question
    # planner's ONE text-only call per run -- cheap even on a slightly
    # stronger model, so it gets its own override knob; defaults to the SAME
    # model as vcut_pass1_model (flash-lite), not a sentinel, matching every
    # other vcut model setting's own concrete-default convention.
    vcut_qplan_model: str = "gemini-3.1-flash-lite"
    # Which cuts pipeline `projects.kick_ingest` enqueues: "v3" (default,
    # today's app.services.l3.ingest) or "vcut" (app.services.vcut.
    # orchestrate). Env-driven so the two can run side by side per
    # environment/A-B without deleting either (plan section 11).
    cuts_pipeline: str = "vcut"

    # migration_runner.plan.md: the startup guard's sanctioned local-dev
    # bypass. "on" (default) means every process refuses to boot on schema
    # drift; "off" disables that check for THIS process only, loudly (a
    # warning is logged). Never set "off" in production -- it exists so a
    # dev with a deliberately divergent local DB has a named escape hatch
    # instead of quietly commenting out the check itself.
    migration_guard: str = "on"

    # --- deployment.plan.md: RunPod serverless GPU execution -------------
    # Where the `gpu`-queue L1 tasks (l1_orchestrate / l1_editing_proxy /
    # l1_active_speaker) physically run their model compute. "local" (default)
    # = run it in THIS process, byte-identical to single-box/dev. "runpod" =
    # the task body forwards to a RunPod Serverless endpoint via
    # app/services/runpod_bridge.py. The Render `edso-gpu-dispatcher` sets
    # "runpod"; the RunPod handler container itself runs with "local" so it
    # does the real compute (and its follow-up enqueues bounce back through
    # the dispatcher, not into an infinite forward loop). Nothing about the
    # compute changes -- same functions, same inputs, same outputs; only the
    # machine differs.
    gpu_execution: str = "local"
    # RunPod Serverless endpoint the dispatcher forwards to (only read when
    # gpu_execution == "runpod"). Empty everywhere else.
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""
    # Upper bound (seconds) the dispatcher polls one GPU job before giving up
    # so Procrastinate can retry -- sized for a long clip's full L1 pass.
    runpod_timeout_seconds: int = 900

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    @property
    def effective_db_schema(self) -> str:
        """The schema unqualified DB access resolves to: DB_SCHEMA when set,
        else "public" (production)."""
        return self.db_schema.strip() or "public"

    @property
    def pg_options(self) -> str:
        """libpq `options` connection parameter that pins the session
        search_path to DB_SCHEMA (with `public` kept as a fallback so
        extensions -- uuid-ossp/vector/pg_trgm -- and their types/functions
        still resolve). Returns "" when DB_SCHEMA is unset, so callers pass NO
        options and behave byte-for-byte like production. `public` stays in the
        path only as a fallback; every app + Procrastinate table is created in
        DB_SCHEMA by the schema-aware migration runner, so unqualified names
        resolve there first."""
        schema = self.db_schema.strip()
        if not schema:
            return ""
        return f"-c search_path={schema},public"

    def pg_connect_kwargs(self) -> dict:
        """Extra psycopg connect kwargs to isolate this process onto DB_SCHEMA.
        Empty dict when DB_SCHEMA is unset (production: pass nothing)."""
        opts = self.pg_options
        return {"options": opts} if opts else {}

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
