-- =============================================
-- Phase 2: L2 character / face ReID
--
-- Per-shot L2 columns (framing_scale, camera_dynamics, tracked_character_ids,
-- dinov2_embedding, narrative_*, etc.) were added in migration 003 as
-- nullable columns. This migration only adds the `characters` table that
-- L2 Stage B (faces) populates.
-- =============================================

-- Each row is a clustered identity for a single user's drive. SCRFD detects
-- faces, ArcFace produces 512-d embeddings, the clustering pass either
-- assigns the embedding to an existing character (cosine > 0.55) or inserts
-- a new "Person_N" row.

create table public.characters (
    id          uuid primary key default uuid_generate_v4(),
    user_id     uuid not null,
    label       text not null,
    embedding   halfvec(512) not null,
    created_at  timestamptz not null default now()
);

create index idx_characters_user on public.characters(user_id);

create index idx_characters_hnsw
    on public.characters
    using hnsw (embedding halfvec_cosine_ops)
    with (m = 16, ef_construction = 128);
