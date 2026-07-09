-- =============================================
-- Camera-movement descriptor.
--
-- L1 already fits a global camera model (pan/tilt + zoom + roll) per hop but
-- only kept the MAGNITUDE. Persist the SIGNED per-hop velocity components so a
-- cut can be tagged with a simple, human-readable camera move (static, pan
-- left/right, tilt up/down, zoom in/out, follow subject, shaky) -- one phrase
-- the brain can read directly. Absolute (not file-normalized) on purpose.
--
-- Idempotent, additive-safe.
-- =============================================

alter table motion_dynamics add column if not exists camera_dx   jsonb not null default '[]'::jsonb;
alter table motion_dynamics add column if not exists camera_dy   jsonb not null default '[]'::jsonb;
alter table motion_dynamics add column if not exists camera_zoom jsonb not null default '[]'::jsonb;

-- Per-cut camera-movement label, computed deterministically at ingest
-- (post._classify_camera_move). 'unknown' for cuts ingested before this / when
-- no motion signal exists.
alter table cut_records add column if not exists camera text not null default 'unknown';
