-- =============================================
-- Phase 1.5: async chat turns
--
-- A chat_turn is one round-trip of the conversational editor: user message ->
-- (retrieval, one-or-more Claude calls, EDL persist, render enqueue) -> result.
--
-- Why persist this when the live progress comes from an in-memory broker:
--   * Durability/observability: the final ChatResponse + lineage (edl_version,
--     render) survive process restarts and are queryable for debugging.
--   * Reconnect: a client that missed the live SSE stream (refresh, flaky
--     network) can GET the turn and read its terminal state.
--   * Cancellation: cancel_requested is a durable flag the runner checks
--     between steps, independent of whether the SSE socket is still open.
--
-- The granular live event stream itself is NOT stored here (it lives in the
-- broker); only the latest snapshot (phase/progress) and final result are.
--
-- Run in Supabase SQL editor.
-- =============================================

create extension if not exists "uuid-ossp";

create table if not exists public.chat_turns (
    id               uuid primary key default uuid_generate_v4(),
    user_id          uuid not null,
    -- Resolved during the run (find-or-create default project), so nullable
    -- at creation time.
    project_id       uuid references public.projects(id) on delete set null,

    status           text not null default 'queued'
                     check (status in ('queued', 'running', 'done', 'failed', 'cancelled')),
    -- Coarse phase label for UI: retrieving | reasoning | persisting | rendering | done
    phase            text,
    progress_pct     int not null default 0,

    -- The full ChatRequestBody we ran with (for replay/debugging).
    request_json     jsonb,
    -- The full ChatResponse on success (so reconnecting clients get the result).
    result_json      jsonb,
    error            text,

    -- Durable cooperative-cancel flag. The runner checks this between steps.
    cancel_requested boolean not null default false,

    -- Lineage produced by this turn (mirrors what the response carries).
    edl_version_id   uuid references public.edl_versions(id) on delete set null,
    render_id        uuid references public.renders(id) on delete set null,

    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

create index if not exists idx_chat_turns_user_created
    on public.chat_turns(user_id, created_at desc);
create index if not exists idx_chat_turns_project_created
    on public.chat_turns(project_id, created_at desc);
create index if not exists idx_chat_turns_status on public.chat_turns(status);

drop trigger if exists chat_turns_updated_at on public.chat_turns;
create trigger chat_turns_updated_at
    before update on public.chat_turns
    for each row execute function public.set_updated_at();
