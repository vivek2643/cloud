-- =============================================
-- 044: apply the two long-pending cleanup drops from 018/019 that never
-- actually ran (found via a full migration-vs-live-schema audit -- the
-- deploy pipeline has no automated migration step, so these were skipped
-- by hand at some point and never revisited).
--
-- Both are grep-confirmed dead: zero references anywhere in app/.
--
--   * public.recommendations -- the old hero-cuts "Recommended" filtration
--     pass (018's own intent); the module that created/read it is long
--     deleted. 018 tried to drop this via a bare `drop table`; this
--     migration is the idempotent (`if exists`) version.
--   * audio_features.{acoustic_tags, energy_peaks_ms, event_segments,
--     pause_map, pitch_hz, sync_env, sync_hop_ms} -- unconsumed L1 signals
--     (018's original target list). `pause_map`/`pitch_hz` were already
--     gone live (018 was evidently partially hand-run at some point); the
--     other five are re-attempted here, idempotently.
--
-- Deliberately NOT touched: public.projects. Migration 012 lists it as a
-- drop target, but 012 predates cuts_v3 (024) and sync_groups (036), both
-- of which added NEW, currently-live foreign keys to projects afterward.
-- 012's drop of projects is now obsolete/wrong and must never be applied.
--
-- Idempotent; safe to re-run.
-- =============================================

drop table if exists public.recommendations;

alter table public.audio_features
    drop column if exists acoustic_tags,
    drop column if exists energy_peaks_ms,
    drop column if exists event_segments,
    drop column if exists pause_map,
    drop column if exists pitch_hz,
    drop column if exists sync_env,
    drop column if exists sync_hop_ms;
