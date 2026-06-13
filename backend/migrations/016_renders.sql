-- =============================================
-- 016: Renders for the L3 edit documents.
--
-- A render is one async ffmpeg job that turns a specific edit-document VERSION
-- (spine + operations -> resolved layers) into a downloadable MP4 at one
-- preset. We key directly on (thread_id, document_version) because the L3
-- document is already append-only/versioned (014) -- no separate EDL table.
--
-- Lifecycle: API inserts status='queued' + enqueues a procrastinate task ->
-- worker flips to 'running', streams progress_pct, then writes output_r2_key +
-- duration_ms + status='done' (or 'failed' + error). Frontend polls
-- GET /api/renders/:id until terminal.
--
-- No RLS: backend-only with the service key, same as 013/014.
-- =============================================

create extension if not exists "uuid-ossp";

create table if not exists public.renders (
    id                uuid primary key default uuid_generate_v4(),
    thread_id         uuid not null references public.edit_threads(id) on delete cascade,
    -- The exact document snapshot this render was built from.
    document_version  int not null,
    preset            text not null default 'preview',
    status            text not null default 'queued'
                      check (status in ('queued', 'running', 'done', 'failed', 'cancelled')),
    progress_pct      int not null default 0,
    -- A hash of the resolved layer set; lets the API short-circuit to an
    -- existing 'done' render of the identical timeline+preset.
    resolved_hash     text,
    output_r2_key     text,
    duration_ms       int,
    error             text,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create index if not exists idx_renders_thread   on public.renders(thread_id, created_at desc);
create index if not exists idx_renders_status    on public.renders(status);
create index if not exists idx_renders_dedup     on public.renders(thread_id, document_version, preset);

drop trigger if exists renders_updated_at on public.renders;
create trigger renders_updated_at
    before update on public.renders
    for each row execute function public.set_updated_at();

comment on table public.renders is 'L3: async MP4 renders of a specific edit_documents version at one preset.';
