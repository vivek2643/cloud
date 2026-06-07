-- =============================================
-- Phase 2 Layer A: adaptive keyframe coverage
--
-- The fixed anchor/motion/variance triple (mig 005) gives every shot exactly
-- three frames regardless of length or how much it actually changes. That
-- under-covers long, visually busy shots (a 30s handheld walk-through) and
-- over-covers short static ones.
--
-- This table holds a VARIABLE number of keyframes per shot, chosen at index
-- time by duration + intra-shot visual variance (see l1/keyframes.extract_adaptive).
-- It is purely additive: shots.keyframe_r2_key and shot_embeddings stay the
-- primary retrieval path, and anything reading keyframes falls back to the
-- legacy triple when a shot has no rows here (every pre-migration shot).
--
-- Run this in the Supabase SQL Editor.
-- =============================================

create table if not exists public.shot_keyframes (
    id          uuid primary key default uuid_generate_v4(),
    shot_id     uuid not null references public.shots(id) on delete cascade,
    frame_index int  not null,            -- 0..N-1, time-ordered within the shot
    kind        text not null,            -- anchor | motion | variance | coverage
    ts_ms       int  not null,            -- absolute timestamp in the source video
    r2_key      text not null,            -- 224x224 JPEG in R2
    embedding   halfvec(768),             -- SigLIP image vector (nullable until L1 stage 4)
    blur        real,                     -- Laplacian variance (lower = blurrier)
    created_at  timestamptz not null default now(),
    unique (shot_id, frame_index)
);

create index if not exists idx_shot_keyframes_shot
    on public.shot_keyframes(shot_id);

-- HNSW so Layer B retrieval can run cosine over the *frame* set, not just the
-- per-shot anchor. m=16/ef_construction=128 matches shot_embeddings (mig 003).
create index if not exists idx_shot_keyframes_hnsw
    on public.shot_keyframes using hnsw (embedding halfvec_cosine_ops)
    with (m = 16, ef_construction = 128);

comment on table public.shot_keyframes is
    'Variable-count adaptive keyframes per shot (Phase 2 Layer A). Additive over '
    'the legacy anchor/motion/variance triple; readers fall back to the triple '
    'when a shot has no rows here.';
comment on column public.shot_keyframes.kind is
    'anchor|motion|variance carry the same meaning as the legacy triple; '
    'coverage = extra farthest-point frames added for long/high-variance shots.';
