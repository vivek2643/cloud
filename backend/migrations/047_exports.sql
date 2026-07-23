-- =============================================
-- 047: Exports for the L3 edit documents (export_options.plan.md).
--
-- An export is one async job that turns a specific edit-document VERSION
-- into ONE of three deliverables: 'mp4' (delegates to the existing render
-- stack at a chosen quality preset), 'srt' (sidecar subtitles), or
-- 'rough_cut' (a self-relinking FCPXML+SRT ZIP for an NLE). Mirrors 016's
-- shape: keyed on (thread_id, document_version), same lifecycle (API
-- inserts status='queued' + enqueues a procrastinate task on the "export"
-- queue -> worker flips to 'running', writes output_r2_key + status='done'
-- or 'failed' + error). Frontend polls GET /api/exports/:id until terminal.
--
-- export_options.plan.md's "Note on document source": exports are
-- THREAD-scoped, not project-scoped -- there is no project_id -> thread_id
-- link anywhere in this schema (edit_threads has no project_id column, and
-- the Cuts-ingest `projects` table has no FK to edit_threads either), and
-- every sibling feature this plan builds on (renders, captions, color
-- grade) is already thread-scoped the same way. Inventing a project-scoped
-- resolution path here would be new, unfounded design work the plan itself
-- flags as an open decision -- thread-scoping needs none.
--
-- No RLS: backend-only with the service key, same as 016.
-- =============================================

create table if not exists public.exports (
    id                uuid primary key default uuid_generate_v4(),
    thread_id         uuid not null references public.edit_threads(id) on delete cascade,
    -- The exact document snapshot this export was built from.
    document_version  int not null,
    kind              text not null check (kind in ('mp4', 'rough_cut', 'srt')),
    -- Quality preset (render.compositor.PRESETS keys: 2160/1080/720/source);
    -- meaningless for kind='srt' (no media rendered), left at the default.
    quality           text not null default '1080',
    include_media     boolean not null default false,
    status            text not null default 'queued'
                      check (status in ('queued', 'running', 'done', 'failed')),
    output_r2_key     text,
    error             text,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create index if not exists idx_exports_thread  on public.exports(thread_id, created_at desc);
create index if not exists idx_exports_status  on public.exports(status);

drop trigger if exists exports_updated_at on public.exports;
create trigger exports_updated_at
    before update on public.exports
    for each row execute function public.set_updated_at();

comment on table public.exports is 'L3: async export jobs (mp4/srt/rough_cut) for a specific edit_documents version.';
