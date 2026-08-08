-- seam_cut_pipeline.plan.md section 4: two additive JSON artifacts on
-- ingest_runs so the vcut energy slider re-derives cuts from persisted
-- state alone -- no model call, no R2 download, no motion recompute on a
-- slider drag. The old pipeline (app.services.l3.ingest) never writes or
-- reads either column.

alter table public.ingest_runs add column if not exists seam_cache jsonb;
alter table public.ingest_runs add column if not exists loose_plan jsonb;

comment on column public.ingest_runs.seam_cache is
    'vcut pipeline only: {file_id: {hop_ms, S, action_energy, frame_diff}} -- per-file seam-quality curve + component tracks, persisted so POST .../cuts/energy can re-run resolve_cuts without recomputing anything.';
comment on column public.ingest_runs.loose_plan is
    'vcut pipeline only: {file_id: [{span_ms, peaks:[{t_ms, tag}], meaning}]} -- the VLM Pass-1 loose cut plan, persisted so the energy slider re-derives boundaries from the SAME plan every time.';
