-- =============================================
-- 048: Per-stage wall-clock timings on ingest_runs (scale_architecture.plan.md
-- Pillar 7 -- observability baked into the code, not bolted on later).
--
-- `timings_ms` is a flat {stage: milliseconds} map written once at the end
-- of a run (success or failure) by ingest.py's run_ingest -- pass1, extract,
-- pass2 (plus pass2's slowest single batch, since wall clock there is
-- max(batches) not sum), identity, post, total. Same "write it down instead
-- of re-deriving it later" rationale as pass1_output/identity_map on this
-- same table. scripts/timing_report.py reads this back for a per-run/
-- per-project breakdown.
-- =============================================

alter table public.ingest_runs
    add column if not exists timings_ms jsonb;

comment on column public.ingest_runs.timings_ms is
    'scale_architecture.plan.md Pillar 7: {stage: ms} wall-clock breakdown for this run (pass1/extract/pass2/pass2_max_batch/identity/post/total).';
