-- vcut_pass2_rich.plan.md section 3.3/7: the shared Gemini CachedContent
-- handle both Pass 1 and Pass 2 ride, persisted so the background
-- vcut_enrich task (and a later energy-slider re-enqueue) can reuse it
-- without re-uploading frames. Additive only -- the old pipeline ignores
-- this column.

alter table public.ingest_runs add column if not exists vcut_cache jsonb;

comment on column public.ingest_runs.vcut_cache is
    'vcut pipeline only: {name, model, created_at, ttl_s} -- the Gemini CachedContent resource both Pass 1 and Pass 2 ride (vcut_pass2_rich.plan.md). Null when creation failed/degraded -- both passes fall back to uncached behavior.';
