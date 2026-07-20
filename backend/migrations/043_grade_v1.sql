-- =============================================
-- color_grading_upgrade.plan.md Step 1.0: the v1 grade pipeline runs as a
-- background job that measures + matches + resolves + pre-bakes, then
-- PERSISTS the result -- layers.resolve reads it instead of recomputing on
-- every document resolve. Step 1.2 adds a small per-span measurement cache
-- so repeated resolves don't re-decode frames.
--
-- No RLS: backend-only with the service key, same as renders (016)/L3 tables.
-- =============================================

create table if not exists public.resolved_grades (
    id           uuid primary key default uuid_generate_v4(),
    thread_id    uuid not null references public.edit_threads(id) on delete cascade,
    -- seg_id (main-line) or op_id (place_video coverage) -- the same id
    -- vocabulary layers.py already uses as a VideoLayer.layer_id.
    shot_key     text not null,
    -- hash(ordered shot spans + look + grade flags + schema_version, see
    -- grade/job.py::compute_input_hash) -- includes TIMELINE SPANS, not just
    -- the look: trimming a cut changes both its own span stats and its
    -- neighbors' sequence-match window, so a span-only change must still
    -- invalidate and re-enqueue.
    input_hash   text not null,
    -- {cdl, creative_lut_ref, working_space, grade_hash, soft_local} -- the
    -- exact shape grade.resolver.resolve_clip_grade already returns.
    grade_json   jsonb not null,
    -- Pre-baked .cube handle (a cache-dir path or storage key), nullable
    -- until the bake step runs.
    cube_ref     text,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),
    unique (thread_id, shot_key, input_hash)
);

-- The READ path (layers.resolve under v1) wants the FRESHEST row per
-- (thread_id, shot_key) regardless of input_hash -- that's what makes a
-- pending/stale job degrade gracefully to "last known grade" instead of
-- blocking preview (see grade/job.py::fetch_latest_grades).
create index if not exists idx_resolved_grades_latest
    on public.resolved_grades(thread_id, shot_key, updated_at desc);

drop trigger if exists resolved_grades_updated_at on public.resolved_grades;
create trigger resolved_grades_updated_at
    before update on public.resolved_grades
    for each row execute function public.set_updated_at();

comment on table public.resolved_grades is
    'v1 grade pipeline: one persisted, pre-baked grade per (thread, shot, input_hash) -- layers.resolve reads this under grade_pipeline=="v1" instead of computing inline.';


-- One row per thread: the current/last run_grade_job's progress, polled by
-- GET /api/edit/threads/{id}/grade-status (Phase 4's progress bar).
create table if not exists public.grade_jobs (
    thread_id   uuid primary key references public.edit_threads(id) on delete cascade,
    state       text not null default 'idle'
               check (state in ('idle', 'grading', 'done', 'error')),
    total       int not null default 0,
    done        int not null default 0,
    input_hash  text,
    error       text,
    updated_at  timestamptz not null default now()
);

drop trigger if exists grade_jobs_updated_at on public.grade_jobs;
create trigger grade_jobs_updated_at
    before update on public.grade_jobs
    for each row execute function public.set_updated_at();

comment on table public.grade_jobs is
    'One row per thread: run_grade_job''s live status (state/total/done/input_hash), polled for the grading-progress indicator.';


-- Step 1.2: per-(file,span) color measurement cache, keyed on the EXACT
-- source window a shot plays (not the whole file) -- so matching/correcting
-- reflects the lighting actually on screen, not a 40s file's whole-file mean.
-- File-scoped, not thread-scoped: the same clip+span measured from two
-- different edit threads shares one cache entry.
create table if not exists public.cut_color_stats (
    file_id        uuid not null references public.files(id) on delete cascade,
    in_ms          int not null,
    out_ms         int not null,
    schema_version int not null default 1,
    stats_json     jsonb not null,
    created_at     timestamptz not null default now(),
    primary key (file_id, in_ms, out_ms)
);

comment on table public.cut_color_stats is
    'Step 1.2: color_stats-shaped measurement over one USED span [in_ms,out_ms) of a file, not the whole file -- cached so repeated grade-job runs don''t re-decode.';
