-- =============================================
-- Phase 1: EDL canonical model + async renders
--
-- Three additive tables. Nothing in 001/.../005 changes.
--
-- Design principle: an EDL (edit decision list) is the source of truth
-- for the timeline. Every actor -- Claude, manual edits, the renderer --
-- reads and writes the same versioned EDL. Renders are downstream
-- consumers, run async via procrastinate.
--
-- Run in Supabase SQL editor.
-- =============================================

create extension if not exists "uuid-ossp";

-- ---- projects ------------------------------------------------------------
-- A project owns a chat session and a chain of EDL versions. For v1 we
-- create one "default" project per (user, source_file_ids set) so the chat
-- UI has somewhere to land without a dedicated project picker. Phase 2 adds
-- explicit naming/listing UI.

create table if not exists public.projects (
    id               uuid primary key default uuid_generate_v4(),
    user_id          uuid not null,
    name             text not null default 'Untitled',
    -- The set of files in scope for the editor. Stored denormalized so we
    -- can quickly find-or-create a default project keyed on the same set.
    source_file_ids  uuid[] not null default '{}',
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

create index if not exists idx_projects_user on public.projects(user_id);
-- Quick lookup for find-or-create-default by sorted file_ids fingerprint.
create index if not exists idx_projects_source_set on public.projects using gin (source_file_ids);

drop trigger if exists projects_updated_at on public.projects;
create trigger projects_updated_at
    before update on public.projects
    for each row execute function public.set_updated_at();

-- ---- edl_versions --------------------------------------------------------
-- Immutable. Every edit -- AI or manual -- creates a new row pointing to
-- the version it was based on via parent_id. This gives us undo/redo and
-- branching for free.
--
-- edl_json shape (Phase 1, cut-only):
-- {
--   "version": 1,
--   "fps": 30,
--   "resolution": [1920, 1080],
--   "clips": [
--     {"id": "c1", "shot_id": "<uuid>",
--      "source_in_ms": 1200, "source_out_ms": 4800,
--      "timeline_in_ms": 0, "timeline_out_ms": 3600}
--   ]
-- }

create table if not exists public.edl_versions (
    id           uuid primary key default uuid_generate_v4(),
    project_id   uuid not null references public.projects(id) on delete cascade,
    parent_id    uuid references public.edl_versions(id) on delete set null,
    edl_json     jsonb not null,
    author_kind  text not null check (author_kind in ('user', 'claude', 'system')),
    commit_msg   text,
    created_at   timestamptz not null default now()
);

create index if not exists idx_edl_versions_project_created
    on public.edl_versions(project_id, created_at desc);
create index if not exists idx_edl_versions_parent
    on public.edl_versions(parent_id);

-- ---- renders -------------------------------------------------------------
-- Async render jobs. Each row is one render attempt for one EDL version,
-- at one preset. Frontend polls GET /api/renders/:id; worker fills it in.

create table if not exists public.renders (
    id                uuid primary key default uuid_generate_v4(),
    edl_version_id   uuid not null references public.edl_versions(id) on delete cascade,
    preset            text not null default 'preview',
    status            text not null default 'queued'
                      check (status in ('queued', 'running', 'done', 'failed', 'cancelled')),
    progress_pct      int not null default 0,
    output_r2_key     text,
    duration_ms       int,
    error             text,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create index if not exists idx_renders_edl_version on public.renders(edl_version_id);
create index if not exists idx_renders_status on public.renders(status);

drop trigger if exists renders_updated_at on public.renders;
create trigger renders_updated_at
    before update on public.renders
    for each row execute function public.set_updated_at();

-- =============================================
-- Notes
-- =============================================
-- * No FK from projects.user_id to auth.users -- dev mode (002) dropped that
--   pattern. Re-add when real auth is enabled.
-- * RLS not enabled here. The backend uses the service role key so RLS is
--   irrelevant; the frontend never reads these tables directly.
-- * The EDL JSON is intentionally tiny (cut-only). Future polish phases add
--   optional fields without breaking existing rows: transitions, captions,
--   multi-track, per-clip volume/speed/effects.
