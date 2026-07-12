-- =============================================
-- Cumulative identity map (identity_map.plan.md): one reconciled-cast
-- payload per ingest run -- persons clustered across files, each file's
-- bound voice, and the derived oncam/alias maps footage_map.py renders
-- into the brain's index. Null on older runs (or a run where reconciliation
-- found nothing to bind/cluster) -- footage_map falls back to today's
-- behavior byte-identical when this is null.
--
-- Idempotent, additive-safe.
-- =============================================

alter table ingest_runs add column if not exists identity_map jsonb null;
