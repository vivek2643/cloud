-- =============================================
-- L1 cut-grid channels: BEAT + MOTION (action, camera/distortion)
--
-- Phase 1 of the multi-channel cut grid, extending the dialogue grid (010).
--
--   BEAT   -- free derivation from audio_features (librosa onsets/bpm). Stored
--             alongside the other audio signals, on audio_features.
--   MOTION -- one cheap optical-flow pass over the proxy yields action (subject
--             motion) + camera/distortion (global motion + blur). Video-derived,
--             so it gets its own per-file table mirroring audio_features.
--
-- All additive + nullable/defaulted. Silent files leave beat empty; still or
-- audio-only files leave motion empty.
-- =============================================

-- --- BEAT (audio-derived) -> audio_features --------------------------------
alter table public.audio_features
    add column if not exists beat_cut_cost   jsonb not null default '[]'::jsonb,
    add column if not exists beat_cut_hop_ms int   not null default 0,
    add column if not exists beat_cut_points jsonb not null default '[]'::jsonb;

comment on column public.audio_features.beat_cut_cost is
    'Dense per-hop beat cut cost (0=on a beat/ideal .. 1=off-beat/avoid) every beat_cut_hop_ms. "Safe to cut" = 1 - cost. Empty for non-musical files.';
comment on column public.audio_features.beat_cut_hop_ms is
    'Sample hop (ms) for beat_cut_cost (default 100ms).';
comment on column public.audio_features.beat_cut_points is
    'Discrete beat onsets to cut ON [{ts_ms,kind,score}]. kind: beat.';

-- --- MOTION (video-derived) -> dedicated table -----------------------------
create table if not exists public.motion_dynamics (
    file_id          uuid primary key references public.files(id) on delete cascade,
    hop_ms           int   not null default 0,
    -- Raw file-normalized signals (0..1), kept for inspection / future tuning.
    action_energy    jsonb not null default '[]'::jsonb,
    camera_motion    jsonb not null default '[]'::jsonb,
    camera_coherence jsonb not null default '[]'::jsonb,
    camera_stability jsonb not null default '[]'::jsonb,
    blur             jsonb not null default '[]'::jsonb,
    -- Derived cut-cost channels (0=ideal seam .. 1=avoid).
    action_cut_cost  jsonb not null default '[]'::jsonb,
    camera_cut_cost  jsonb not null default '[]'::jsonb,
    -- Discrete subject-motion impacts to cut ON.
    action_points    jsonb not null default '[]'::jsonb,
    created_at       timestamptz not null default now()
);

comment on table public.motion_dynamics is
    'L1 motion cut grids from one optical-flow pass over the proxy: ACTION (cut on subject-motion impacts) + CAMERA/DISTORTION (avoid cutting during chaotic/transient camera moves/blur, NOT smooth deliberate moves).';
comment on column public.motion_dynamics.camera_motion is
    'Per-hop camera-motion magnitude from a fitted similarity model (pan/tilt+zoom+roll), file-normalized 0..1.';
comment on column public.motion_dynamics.camera_coherence is
    'Per-hop RANSAC inlier ratio (0..1): how rigidly the whole frame moves as one. ~1 = clean global move (pan/zoom), ~0 = shake or moving subject.';
comment on column public.motion_dynamics.camera_stability is
    'Per-hop temporal steadiness of camera velocity (0..1): ~1 = sustained move (dolly), ~0 = jerky transient (whip/bump).';
comment on column public.motion_dynamics.action_cut_cost is
    'Dense per-hop action cut cost (0=on an impact/ideal .. 1=mid-motion/avoid). "Safe to cut" = 1 - cost.';
comment on column public.motion_dynamics.camera_cut_cost is
    'Dense per-hop camera/distortion cut cost (0=settled OR smooth coherent move .. 1=chaotic/jerky/blurred). Smooth deliberate moves are NOT penalized.';
comment on column public.motion_dynamics.action_points is
    'Discrete subject-motion impacts to cut ON [{ts_ms,kind,score}]. kind: action_impact.';
