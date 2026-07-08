-- =============================================
-- Drop the L2 / VLM perception layer entirely. The Gemini clip_perception pass
-- and its consumers (cast/relations/takes/valence) are removed; Cuts v3 (pass2
-- images) now carries per-cut characteristics/shot_size + quality scores.
-- Idempotent (if exists), additive-safe.
-- =============================================
drop table if exists public.clip_perception;
alter table public.files drop column if exists l2_status;
