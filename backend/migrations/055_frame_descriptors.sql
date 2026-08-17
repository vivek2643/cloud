-- =============================================
-- brain_mirror Phase 1: per-frame perceptual descriptor (pHash).
--
-- The missing STATE signal. scene_cuts/motion_dynamics/frame-diff are all
-- temporal derivatives over the source timeline (how fast the picture is
-- changing); a seam in a finished edit needs to compare two arbitrary,
-- non-adjacent source instants for "same shot?". A per-frame 64-bit DCT pHash
-- answers that for any pair via Hamming distance.
--
-- A dedicated per-file table, additive and easy to drop -- mirrors scene_cuts
-- (022). One jsonb array per file rather than a row-per-frame table: the
-- read-side loads ALL of a file's hashes at once (nearest-sample lookup around a
-- seam instant), so one indexed PK lookup beats thousands of point rows.
-- =============================================

create table if not exists public.frame_descriptors (
    file_id        uuid primary key references public.files(id) on delete cascade,
    hop_ms         int   not null default 0,
    -- Sampled perceptual hashes in ascending t_ms: [{t_ms, h}]. h is a 16-char
    -- hex string encoding the 64-bit DCT pHash (never a raw image).
    phashes        jsonb not null default '[]'::jsonb,
    schema_version int   not null default 1,
    created_at     timestamptz not null default now()
);

comment on table public.frame_descriptors is
    'L1 brain_mirror signal: per-frame 64-bit DCT perceptual hash (pHash) sampled from the proxy, for same-shot seam comparison by Hamming distance (see l1.frame_descriptors).';
comment on column public.frame_descriptors.hop_ms is
    'Sampling hop in ms (1000/SAMPLE_FPS) between consecutive pHash samples.';
comment on column public.frame_descriptors.phashes is
    'Ascending-t_ms sampled pHashes [{t_ms, h}]. h = 16-hex 64-bit DCT perceptual hash.';
comment on column public.frame_descriptors.schema_version is
    'l1.frame_descriptors.SCHEMA_VERSION at write time, so a hash/sampling change can be told apart from a stale row.';
