-- =============================================
-- L1 perception signals (universal-editor phase 1)
--
-- Cheap, CPU-only signals the editor uses to cut better:
--   - shots.motion_dx / motion_dy: dominant screen-space motion direction of
--     the shot (the optical-flow vector we already compute but used to discard).
--     Enables motion-continuous match cuts + direction-reversal avoidance.
--
-- All columns are additive + nullable so existing L1 rows keep working.
-- =============================================

-- ---- shots: motion direction ---------------------------------------------

alter table public.shots
    add column if not exists motion_dx real,
    add column if not exists motion_dy real;

comment on column public.shots.motion_dx is
    'Dominant horizontal motion of the shot (magnitude-weighted mean optical-flow dx, px/frame @320x180). +right / -left.';
comment on column public.shots.motion_dy is
    'Dominant vertical motion of the shot (magnitude-weighted mean optical-flow dy, px/frame @320x180). +down / -up.';

-- ---- audio_features: prosody / rhythm ------------------------------------

alter table public.audio_features
    add column if not exists energy_peaks_ms jsonb not null default '[]'::jsonb,
    add column if not exists pause_map       jsonb not null default '[]'::jsonb,
    add column if not exists rms_db          jsonb not null default '[]'::jsonb,
    add column if not exists pitch_hz        jsonb not null default '[]'::jsonb,
    add column if not exists prosody_hop_ms  int  not null default 0,
    add column if not exists sync_env        jsonb not null default '[]'::jsonb,
    add column if not exists sync_hop_ms     int  not null default 0;

comment on column public.audio_features.energy_peaks_ms is
    'Timestamps (ms) of RMS-energy emphasis peaks. Cut-sync points for non-musical audio.';
comment on column public.audio_features.pause_map is
    'Contiguous low-energy pauses [{start_ms,end_ms}] -- natural, breath-safe cut boundaries.';
comment on column public.audio_features.rms_db is
    'Coarse energy envelope (dB) sampled every prosody_hop_ms, bounded length.';
comment on column public.audio_features.pitch_hz is
    'Coarse f0 contour (Hz, 0=unvoiced) sampled every prosody_hop_ms. Prosody / emphasis.';
comment on column public.audio_features.prosody_hop_ms is
    'Sample hop (ms) for rms_db and pitch_hz series.';
comment on column public.audio_features.sync_env is
    'Fixed-hop normalized energy envelope (0..1) for cross-file simultaneity (multicam) alignment.';
comment on column public.audio_features.sync_hop_ms is
    'Sample hop (ms) for sync_env (shared time grid across files).';
