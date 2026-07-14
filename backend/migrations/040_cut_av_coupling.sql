-- =============================================
-- Cut-level A/V coupling with authoritative audio
-- (av_coupling_authoritative.plan.md).
--
-- A cut is a coupled (video, authoritative-audio) unit. The coupling is a
-- (audio_file_id, audio_offset_ms) pair baked onto the cut at ASSEMBLY time
-- (post.assemble_cut_records) instead of re-derived lazily at render/resolve
-- time (the old sync.audio_route.resolve_audio_routes path, kept as a
-- legacy fallback for pre-migration edit documents).
--
--   - audio_file_id: the coupled audio source. NULL means "same as
--     file_id, offset 0" -- so existing rows are correct with no backfill.
--   - audio_offset_ms: add to the video's src ms to get the audio's src ms.
--     0 for same-source cuts (the ~90% common case); a per-cut refined
--     delta (cross-correlated against the authoritative file's own loudness
--     envelope) for a synced-group cut whose picture and authoritative
--     audio come from different files.
--   - audio_align_confidence: the refinement's normalized-correlation peak
--     (0..1-ish), for diagnostics. NULL when the cut is same-source, or when
--     the refinement's guard rejected a weak/ambiguous peak and fell back
--     to the group's unrefined global delta.
--
-- Idempotent, additive-safe.
-- =============================================

alter table cut_records add column if not exists audio_file_id text null;
alter table cut_records add column if not exists audio_offset_ms integer not null default 0;
alter table cut_records add column if not exists audio_align_confidence real null;
