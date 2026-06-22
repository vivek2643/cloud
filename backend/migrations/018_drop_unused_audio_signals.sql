-- =============================================
-- 018: drop unused audio_features columns
--
-- Cleanup of L1 signals that nothing reads:
--   * energy_peaks_ms / pause_map / pitch_hz  -- prosody signals that were
--       computed but never consumed by any editing logic or the audit log.
--   * sync_env / sync_hop_ms                  -- the cross-file multicam sync
--       fingerprint; its only consumer was the L3 agent sync module, removed.
--   * acoustic_tags / event_segments          -- planned YAMNet audio-events
--       columns (see 003) that were never populated.
--
-- The producing code (audio_features.compute / pipeline _stage5_audio) and the
-- snapshot reader have been updated to stop touching these. Idempotent.
-- =============================================

alter table public.audio_features
    drop column if exists energy_peaks_ms,
    drop column if exists pause_map,
    drop column if exists pitch_hz,
    drop column if exists sync_env,
    drop column if exists sync_hop_ms,
    drop column if exists acoustic_tags,
    drop column if exists event_segments;
