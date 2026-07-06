-- =============================================
-- Cuts v3, Step A: LLM-grouped cuts storage.
--
-- Two additive tables (nothing existing changes). This is the ingest-run
-- ledger + the final per-cut record the LLM ingest pipeline (pass1 -> images
-- -> pass2 -> post) writes once and every product surface reads forever.
-- See cuts_v3.plan.md.
-- =============================================

-- ---- ingest_runs -----------------------------------------------------------
-- One row per project ingest attempt. Tracks pipeline status, which models
-- ran, and full token/cost accounting (accumulated across pass 1 + every
-- pass-2 shard) so cost is auditable per run, not just estimated.
create table if not exists public.ingest_runs (
    id                  uuid primary key default uuid_generate_v4(),
    project_id          uuid not null references public.projects(id) on delete cascade,
    status              text not null default 'pending'
        check (status in ('pending', 'pass1', 'images', 'pass2', 'post', 'ready', 'failed')),
    pass1_model         text,
    pass2_model         text,
    input_tokens        bigint not null default 0,
    output_tokens       bigint not null default 0,
    cache_read_tokens   bigint not null default 0,
    cache_write_tokens  bigint not null default 0,
    cost_usd            numeric(10, 4) not null default 0,
    project_summary     text,
    -- Raw pass-1 output (speech_cuts/take_candidates/video_tentative_groups/
    -- junk_suspects/clip_summaries), persisted so pass 2 + audit can read it
    -- back without re-deriving it.
    pass1_output        jsonb,
    error               text,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);

create index if not exists idx_ingest_runs_project on public.ingest_runs(project_id, created_at desc);

drop trigger if exists ingest_runs_updated_at on public.ingest_runs;
create trigger ingest_runs_updated_at
    before update on public.ingest_runs
    for each row execute function public.set_updated_at();

comment on table public.ingest_runs is
    'Cuts v3: one row per project LLM-ingest attempt -- status, models, token/cost accounting, raw pass-1 output, error detail. See cuts_v3.plan.md.';

-- ---- cut_records -------------------------------------------------------------
-- One row per FINAL cut -- the complete, LLM-judged + deterministically-
-- assembled record every product surface reads. Boundaries (src_in_ms/
-- src_out_ms) are always snapped to word edges (speech) or atom edges
-- (video) by code, never an LLM-emitted millisecond (see plan North Star #1).
create table if not exists public.cut_records (
    id              uuid primary key default uuid_generate_v4(),
    ingest_run_id   uuid not null references public.ingest_runs(id) on delete cascade,
    file_id         uuid not null references public.files(id) on delete cascade,
    src_in_ms       int not null,
    src_out_ms      int not null,
    kind            text not null check (kind in ('speech', 'video')),
    -- Speech: the inclusive [start,end] word index this cut spans (its
    -- boundaries are exactly those words' edges). Video: the atom ids merged
    -- into this cut (its boundaries are exactly those atoms' outer edges).
    word_span       jsonb,
    atom_ids        jsonb,
    label           text,
    summary         text,
    speaker         text,
    on_camera       boolean,
    -- Cross-clip take grouping (pass 2). NULL take_group_id = not part of any
    -- take comparison. take_role distinguishes the shown winner from a stacked
    -- take (same words/setting) or an outlook (same words, different setting).
    take_group_id   uuid,
    take_role       text check (take_role in ('take', 'outlook', 'winner')),
    junk            boolean not null default false,
    junk_reason     text,
    -- {subject_box, crop_16x9, crop_9x16, crop_1x1, rotation_deg}
    framing         jsonb,
    -- {graded, palette, exposure}
    look            jsonb,
    -- Normalized boxes clear of the subject on both the hero AND drift frame.
    caption_zones   jsonb,
    -- {min_ms, natural_ms, max_ms, energy_grade, levels[5], natural_sound}
    pace            jsonb,
    hero_ts_ms      int,
    hero_key        text,
    transition_in   text,
    transition_out  text,
    created_at      timestamptz not null default now(),

    constraint cut_records_span_valid check (src_out_ms > src_in_ms)
);

create index if not exists idx_cut_records_ingest_run on public.cut_records(ingest_run_id);
create index if not exists idx_cut_records_file on public.cut_records(file_id, src_in_ms);
create index if not exists idx_cut_records_take_group on public.cut_records(take_group_id) where take_group_id is not null;

comment on table public.cut_records is
    'Cuts v3: one row per final cut -- LLM-judged meaning + deterministically-assembled framing/pace/hero-frame, over code-enforced word/atom-snapped boundaries. See cuts_v3.plan.md.';
