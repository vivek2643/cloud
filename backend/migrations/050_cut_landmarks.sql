-- =============================================
-- 050: Per-cut signal landmarks (brain_perception_upgrade.plan.md Change 1,
-- Mechanism A -- the beat-line "sig:" breadcrumb).
--
-- Compact per-cut signal landmarks distilled at ingest from L1 arrays
-- (motion_dynamics.action_energy/action_points, audio_features.rms_db/
-- silence_intervals, scene_cuts.shot_points/composition_points): interior
-- action hits, audio change points, silence gaps, internal shot cuts --
-- counts + capped offset lists. Powers the beat-line sig: breadcrumb and
-- warms inspect_cut. {} on a pre-migration / no-signal cut.
-- =============================================

alter table public.cut_records
    add column if not exists landmarks jsonb not null default '{}'::jsonb;

comment on column public.cut_records.landmarks is
    'Compact per-cut signal landmarks distilled at ingest from L1 arrays '
    '(motion_dynamics.action_energy/action_points, audio_features.rms_db/'
    'silence_intervals, scene_cuts.shot_points/composition_points): interior '
    'action hits, audio change points, silence gaps, internal shot cuts -- '
    'counts + capped offset lists. Powers the beat-line sig: breadcrumb and '
    'warms inspect_cut. {} on a pre-migration / no-signal cut.';
