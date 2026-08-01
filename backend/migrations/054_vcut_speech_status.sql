-- vcut_moment_energy.plan.md section 8: make the speech channel's silent
-- fail-open fallback observable. Today _run_speech_channel_with_fallback
-- catches ANY exception from run_speech_channel and falls back to
-- copy_prior_speech_cuts -- so a broken speech pipeline looks identical to
-- a working one in GET /cuts. This column records which path actually ran.
-- Additive only -- the old l3 pipeline never writes or reads it.

alter table public.ingest_runs add column if not exists speech_channel_status jsonb;

comment on column public.ingest_runs.speech_channel_status is
    'vcut pipeline only: {source: "pipeline"|"copy_prior", error: str|null} -- which speech path actually ran on this ingest (vcut_moment_energy.plan.md section 8). Null on a v3 run or a pre-migration vcut run.';
