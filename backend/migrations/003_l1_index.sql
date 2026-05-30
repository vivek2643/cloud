-- =============================================
-- Phase 1: L1 Video Indexing schema
--   + Delta 1.B: future-proof nullable columns for L2/L3
--
-- Run this in Supabase SQL Editor.
-- =============================================

-- pgvector for embeddings (halfvec lives here too, since pgvector 0.7+)
create extension if not exists vector;
create extension if not exists pg_trgm;

-- =============================================
-- Stage tracking: replaces the single 'status' column for fine-grained
-- per-stage retry. (file_id, stage) unique => orchestrator can safely
-- skip stages already marked 'done' on retry.
-- =============================================

create table public.processing_jobs (
    id          uuid primary key default uuid_generate_v4(),
    file_id     uuid not null references public.files(id) on delete cascade,
    stage       text not null,
    status      text not null default 'queued',
    attempts    int  not null default 0,
    error       text,
    started_at  timestamptz,
    finished_at timestamptz,
    created_at  timestamptz not null default now(),
    unique (file_id, stage)
);
create index idx_processing_jobs_file on public.processing_jobs(file_id);

-- =============================================
-- Shots: one row per scene boundary detected by PySceneDetect
-- (form-factor-aware constraints applied in services/l1/shots.py).
--
-- Nullable L2 columns populated lazily by Phase 2 stages:
--   - dinov2_embedding      <- L2 Stage A
--   - framing_scale         <- L2 Stage A (rule pass)
--   - camera_dynamics       <- L2 Stage A (rule pass)
--   - tracked_character_ids <- L2 Stage B
--   - narrative_*           <- L2 Stage D
-- =============================================

create table public.shots (
    id                    uuid primary key default uuid_generate_v4(),
    file_id               uuid not null references public.files(id) on delete cascade,
    shot_index            int  not null,
    start_ms              int  not null,
    end_ms                int  not null,
    keyframe_r2_key       text,
    focus_score           real,
    brightness            real,
    motion_magnitude      real,
    -- L2 enrichment (nullable, populated on demand)
    framing_scale         text,
    camera_dynamics       text,
    tracked_character_ids uuid[],
    dinov2_embedding      halfvec(768),
    narrative_description text,
    narrative_role        text,
    emotional_valence     real,
    created_at            timestamptz not null default now(),
    unique (file_id, shot_index)
);
create index idx_shots_file on public.shots(file_id);

-- =============================================
-- L1 SigLIP 2 embeddings -> 768-d halfvec
-- halfvec_cosine_ops + m=16, ef_construction=128 per 2026 pgvector best practice
-- =============================================

create table public.shot_embeddings (
    shot_id   uuid primary key references public.shots(id) on delete cascade,
    embedding halfvec(768) not null
);
create index idx_shot_embeddings_hnsw
    on public.shot_embeddings using hnsw (embedding halfvec_cosine_ops)
    with (m = 16, ef_construction = 128);

-- =============================================
-- Transcripts: full text + segments + filler marks
-- =============================================

create table public.transcripts (
    file_id  uuid primary key references public.files(id) on delete cascade,
    language text,
    text     text not null,
    segments jsonb not null,
    fillers  jsonb not null default '[]'::jsonb,
    tsv      tsvector generated always as (to_tsvector('simple', text)) stored
);
create index idx_transcripts_tsv on public.transcripts using gin(tsv);

-- =============================================
-- Whole-file audio features
--
-- L2 columns (nullable, populated by Phase 2 Stage C):
--   - acoustic_tags   <- YAMNet top tags
--   - event_segments  <- per-second timeline
-- =============================================

create table public.audio_features (
    file_id           uuid primary key references public.files(id) on delete cascade,
    integrated_lufs   real,
    true_peak_db      real,
    is_musical        boolean not null default false,
    bpm               real,
    onsets_ms         jsonb not null default '[]'::jsonb,
    silence_intervals jsonb not null default '[]'::jsonb,
    acoustic_tags     text[],
    event_segments    jsonb
);

-- =============================================
-- files extensions: L1 status + L2 status
-- =============================================

alter table public.files
    add column if not exists l1_status text not null default 'pending';
    -- pending | running | ready | failed | skipped

alter table public.files
    add column if not exists l2_status text;
    -- null = never run | running | ready | partial | failed
