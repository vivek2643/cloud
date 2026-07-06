-- =============================================
-- Cuts v3, Step B: transition_points (premium natural cut instants).
--
-- Extends motion_dynamics (011) -- no new table, same file, same optical-flow
-- pass. Two kinds, computed cheaply alongside the existing channels:
--   wipe       -- a large near-field blob sweeps the frame (classic pass-by
--                 transition editors hunt for).
--   degenerate -- the frame collapses to one texture (over-zoom, lens
--                 blocked) -- "must cut by here".
-- See cuts_v3.plan.md, section 1a.
-- =============================================

alter table public.motion_dynamics
    add column if not exists transition_points jsonb not null default '[]'::jsonb;

comment on column public.motion_dynamics.transition_points is
    'cuts-v3 premium natural cut instants [{ts_ms, kind: wipe|degenerate, strength}]. See l1.motion_dynamics._wipe_points / _degenerate_points.';
