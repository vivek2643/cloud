-- =============================================
-- Cleanup: drop the legacy hero-cut / cuts-v2-partition / speech-thoughts
-- infra and the L1 signals nothing reads anymore now that the brain and UI
-- run entirely off Cuts v3 (cut_records).
--
-- hero_cuts_cache   -- precomputed hero-cut substrate (hero_cuts.py/hero_store.py,
--                       deleted in cleanup.plan.md Bucket B2).
-- speech_thoughts   -- thought_segments.py output (deleted in Bucket C1).
-- music_structure   -- deep musical-structure pass; only reader was the L1
--                       debug snapshot (deleted in Bucket C3).
-- audio_features.{f0_hz,dialogue_cut_*,beat_cut_*} -- partition/hero/
--                       clip_timeline-only grids (cut_cost.py/beat_cost.py,
--                       deleted in Bucket C3); f0_hz was partition-only.
--                       rms_db/prosody_hop_ms/silence_intervals are untouched.
--
-- Idempotent (if exists) so it's safe to re-run. See cleanup.plan.md.
-- =============================================

drop table if exists public.hero_cuts_cache;
drop table if exists public.speech_thoughts;
drop table if exists public.music_structure;

alter table public.audio_features
    drop column if exists f0_hz,
    drop column if exists dialogue_cut_cost,
    drop column if exists dialogue_cut_hop_ms,
    drop column if exists dialogue_cut_points,
    drop column if exists beat_cut_cost,
    drop column if exists beat_cut_hop_ms,
    drop column if exists beat_cut_points;
