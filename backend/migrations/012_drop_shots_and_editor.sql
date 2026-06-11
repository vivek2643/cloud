-- =============================================
-- Remove the shot/embedding model and the shot-based editor stack.
--
-- The pipeline no longer detects shots, extracts keyframes, or computes SigLIP
-- embeddings, and the L3 editor / semantic search / EDL + render stack that
-- consumed them has been deleted from the codebase. This migration drops the
-- now-orphaned tables so the schema matches the L1-only producer.
--
-- DESTRUCTIVE: this permanently deletes all shots, embeddings, saved edits,
-- EDL versions, renders, and chat-turn history. There is no down migration.
-- `cascade` handles foreign-key dependents regardless of drop order.
--
-- Kept: files, folders, transcripts, audio_features (incl. dialogue/beat cut
-- grids), motion_dynamics, processing_jobs.
-- =============================================

-- Shot model (shot_embeddings + shot_keyframes FK -> shots FK -> files).
drop table if exists public.shot_embeddings cascade;
drop table if exists public.shot_keyframes  cascade;
drop table if exists public.shots           cascade;

-- Editor output (renders FK -> edl_versions FK -> projects).
drop table if exists public.renders      cascade;
drop table if exists public.edl_versions cascade;
drop table if exists public.projects     cascade;

-- Conversational editor turn history.
drop table if exists public.chat_turns cascade;

-- Dead L2 characters table (L2 was removed earlier).
drop table if exists public.characters cascade;

-- The files.l2_status column is now unused (L2 + the editor are gone) but is
-- left in place: dropping a populated column is destructive and harmless to
-- keep. Remove it in a later migration if desired.
