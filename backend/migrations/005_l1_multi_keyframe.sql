-- =============================================
-- L1 multi-keyframe enrichment (Phase 1.C)
--
-- Extends every shot with anchor / peak-motion / peak-variance keyframes
-- and the SigLIP vectors for each, plus an intra-shot variance score and
-- a Laplacian-variance blur signal.
--
-- All columns are additive + nullable so existing L1 rows keep working.
-- =============================================

-- ---- shots ----------------------------------------------------------------

alter table public.shots
    add column if not exists peak_motion_ms       int,
    add column if not exists peak_variance_ms     int,
    add column if not exists intra_shot_variance  real,
    add column if not exists blur_min             real,
    add column if not exists r2_keyframe_motion_key   text,
    add column if not exists r2_keyframe_variance_key text;

comment on column public.shots.peak_motion_ms is
    'Absolute timestamp (ms) of the peak-motion keyframe within the shot. Used by L3 sub-clip trimming.';
comment on column public.shots.intra_shot_variance is
    '1 - cosine(anchor_embedding, motion_embedding). Higher = shot changes more visually internally.';
comment on column public.shots.blur_min is
    'Minimum Laplacian variance across the 3 keyframes (lower = blurrier).';
comment on column public.shots.r2_keyframe_motion_key is
    'R2 key for the peak-motion keyframe JPEG (224x224, low-quality).';
comment on column public.shots.r2_keyframe_variance_key is
    'R2 key for the peak-variance keyframe JPEG (224x224, low-quality).';

-- The legacy `keyframe_r2_key` column is reused as the ANCHOR frame from now on.
-- Old rows keep their midpoint frame; new rows store the anchor (mid-shot sample).

-- ---- shot_embeddings ------------------------------------------------------

alter table public.shot_embeddings
    add column if not exists embedding_motion   halfvec(768),
    add column if not exists embedding_variance halfvec(768);

-- We deliberately do NOT add HNSW indexes on motion/variance vectors yet:
-- L3 retrieval still uses anchor; motion/variance are for sub-clip logic only.
-- Add indexes when query patterns demand them.

comment on column public.shot_embeddings.embedding is
    'SigLIP vector of the anchor (midpoint) keyframe. Used for primary L3 retrieval.';
comment on column public.shot_embeddings.embedding_motion is
    'SigLIP vector of the peak-motion keyframe.';
comment on column public.shot_embeddings.embedding_variance is
    'SigLIP vector of the peak-variance keyframe.';
