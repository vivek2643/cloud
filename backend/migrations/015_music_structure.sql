-- =============================================
-- L1 (music-only): deep musical-structure analysis.
--
-- A standalone audio/music upload runs an audio-only L1 path (no proxy, no
-- motion, no speech). Beyond the existing audio_features (loudness, BPM,
-- onsets, beat cut grid), a music file also gets the musical scaffolding an
-- editor cuts a montage TO: a bar/downbeat grid, sections/phrases, an energy
-- envelope, key, and a PHRASE cut-cost grid that dips at downbeats + section
-- boundaries.
--
-- Own per-file table (mirrors motion_dynamics): only music files populate it.
-- =============================================

create table if not exists public.music_structure (
    file_id            uuid primary key references public.files(id) on delete cascade,
    bpm                double precision not null default 0,
    music_key          text,
    -- Beat + bar grids (absolute ms).
    beat_times_ms      jsonb not null default '[]'::jsonb,
    downbeat_times_ms  jsonb not null default '[]'::jsonb,
    -- Sections / phrases: [{start_ms,end_ms,label}] (repeated material shares a letter).
    sections           jsonb not null default '[]'::jsonb,
    -- Intensity envelope (0..1) sampled every energy_hop_ms -- cut on the build/drop.
    energy_hop_ms      int   not null default 0,
    energy             jsonb not null default '[]'::jsonb,
    -- PHRASE cut grid: cut ON musically-strong instants (downbeats, section starts).
    phrase_cut_hop_ms  int   not null default 0,
    phrase_cut_cost    jsonb not null default '[]'::jsonb,
    phrase_cut_points  jsonb not null default '[]'::jsonb,
    created_at         timestamptz not null default now()
);

comment on table public.music_structure is
    'L1 musical-structure analysis for standalone music uploads: bar/downbeat grid, sections/phrases, energy envelope, key, and a phrase cut-cost grid. Only populated for audio files.';
comment on column public.music_structure.downbeat_times_ms is
    'Bar-start times (ms). 4/4 assumption: phase chosen by max onset energy on bar-start beats.';
comment on column public.music_structure.sections is
    'Spectral-segmentation phrases [{start_ms,end_ms,label}]; recurring material shares a label letter.';
comment on column public.music_structure.energy is
    'Normalized intensity envelope (0..1) every energy_hop_ms -- locate builds, drops, breakdowns.';
comment on column public.music_structure.phrase_cut_cost is
    'Dense per-hop phrase cut cost (0=on a downbeat/section boundary/ideal .. 1=mid-phrase/avoid). "Safe to cut" = 1 - cost.';
comment on column public.music_structure.phrase_cut_points is
    'Discrete musically-strong cut instants [{ts_ms,kind,score}]. kind: downbeat | section.';
