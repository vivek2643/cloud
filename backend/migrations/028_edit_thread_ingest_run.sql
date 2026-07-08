-- =============================================
-- 028: Pin an edit thread to the Cuts v3 ingest run it was opened against.
--
-- Edit threads are FILE-scoped (edit_threads.file_ids); Cuts v3 cut_records are
-- INGEST-RUN-scoped. The brain's footage map resolves "the run the editor was
-- looking at" from a thread's files. Resolving that LIVE on every turn means a
-- re-ingest MID-THREAD silently swaps the beat universe under an active edit --
-- moment ids (`{fid8}:m{idx}`) are positional, so the same ref can start
-- pointing at a different beat and already-placed refs go stale.
--
-- Pinning the covering run at thread creation makes a thread STABLE and
-- reproducible for its whole life. Nullable + no FK: older threads (and any
-- created before a Cuts v3 ingest exists) stay null and fall back to live
-- "latest run" resolution -- exactly today's behavior. No FK so deleting an
-- ingest run never tears down a thread (same loose ownership as file_ids).
-- =============================================

alter table public.edit_threads
    add column if not exists ingest_run_id uuid;

comment on column public.edit_threads.ingest_run_id is
    'Cuts v3 ingest_run this thread is pinned to (the footage snapshot the editor opened). Null => resolve the latest covering run live each turn.';
