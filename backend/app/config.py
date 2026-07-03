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

    # --- L3: prompt-driven auto-editor (OpenAI) --------------------------
    # A simple, deterministic 3-call pipeline (Director -> Editor -> Coverage)
    # that turns a one-line brief into a full Edit Document: guess the energy,
    # build the hero-cuts feed at that energy, select + order the cuts, then
    # lay light coverage. Provider/model overridable.
    enable_autoedit: bool = True
    # The editing brain (converse + arranger) runs on the strongest available
    # model -- this is creative selection + ordering, not cheap classification.
    # Provider-neutral: flip to "openai"/"gemini" + a model id to swap backbones.
    autoedit_provider: str = "anthropic"
    autoedit_model: str = "claude-opus-4-8"
    autoedit_max_output_tokens: int = 16384
    # Deep reasoning for the editorial calls (taste, story, ordering).
    autoedit_effort: str = "high"
    # A pure clip ASSEMBLER: pick the right clips and order them, nothing else.
    # Leave False to skip the coverage pass (no V2 cutaway ops); flip True to let
    # it lay B-roll / reaction cutaways as V2 over the V1 spine.
    autoedit_coverage: bool = False
    # Conversational arranger (edit-thread brain) version.
    #   "v1" -- the original MOMENT-framed loop: clips are a bag of pre-scored
    #           speech moments; place/split_screen only. Kept for fallback.
    #   "v2" -- the CONTINUOUS-SOURCE loop, WORKFLOW-framed: same senses/verbs as
    #           v3 but the prompt prescribes "lay the spoken spine, then lift
    #           reactions via place_span". Kept for A/B against v3.
    #   "v3" -- the BLIND-EDITOR loop (default): identical awareness + tools, but
    #           NO workflow prescription. The brain is told who it is, given a
    #           plan-first discipline (look -> picture the finished piece -> plan
    #           -> pin -> execute -> check), and reads the senses/verbs as neutral
    #           capabilities -- the material and the craft are its to decide.
    autoedit_arranger_version: str = "v3"

    # --- L3 thought segmentation (the speech primitive) ------------------
    # A post-L2 pass that splits a clip's speaker-tagged transcript into generic
    # THOUGHTS -- one speaker's self-contained idea -- each with a zoom hierarchy
    # (punchline -> core sentence -> thought -> + setup). This replaces the ragged
    # ASR sentence/topic units as the speech source for the energy bands. Cached
    # once per file; falls back to L1 dialogue_segments when the LLM is
    # unavailable. Provider-neutral; empty -> reuse the autoedit (Opus) backbone.
    enable_thought_segments: bool = True
    thoughts_provider: str = ""
    thoughts_model: str = ""
    thoughts_max_output_tokens: int = 16384
    # Payload guard: clips with more words than this skip the LLM pass and use
    # the deterministic L1 fallback (paging the pass is a later concern).
    thoughts_max_words: int = 8000

    # --- L3 arranger: the cut-picking reasoning brain --------------------
    # Resident mode (default) holds the whole footage map in one context and
    # runs a draft -> self-critique cycle. When the map text exceeds this many
    # chars it is PAGED instead: a compact index + on-demand inspect tools, so
    # we degrade gracefully instead of blindly truncating.
    arranger_resident_char_budget: int = 180_000
    # Reasoning passes in resident mode (2 = draft, then critique + revise).
    arranger_passes: int = 2
    # Effort for the draft pass; the final (critique) pass uses autoedit_effort.
    arranger_draft_effort: str = "medium"
    # Paged mode: effort per turn and a hard cap on tool/reason turns.
    arranger_paged_effort: str = "medium"
    arranger_max_turns: int = 12

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
