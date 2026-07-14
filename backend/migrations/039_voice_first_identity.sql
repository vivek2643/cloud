-- =============================================
-- Voice-first identity layer (voice_first_identity.plan.md).
--
-- Phase A: one voiceprint (embedding vector) per file-local diarized
-- speaker, captured during L1 diarization -- the cross-clip identity spine
-- `identity/voices.py` clusters on. Null/empty on an older file (diarization
-- never ran, or the pyannote build in use couldn't produce embeddings) --
-- identity/voices.py falls back to per-file, unclustered voices.
--
-- Phase G/H: per-cut identity facts, now code-derived (never LLM-echoed):
--   - voice_ids: the global voice(s) heard in this cut (from Pass 1's
--     word-level speaker_ids, mapped through voice clustering).
--   - speaker_person: the global person id (Px) the speaker pass bound the
--     speaking voice to, when a binding was confident enough. Null when the
--     voice is unbound (narration, an unclustered voice, no agreement
--     across binding windows) -- honest ignorance, never a guess.
--   - visible_persons: every global person id visible on screen in this
--     cut (from the redesigned per-cut-occurrence face clustering, Phase D)
--     -- replaces the old one-person-per-file assumption.
-- `cut_records.on_camera`/`speaker` (existing columns) keep their shape;
-- their SOURCE changes (rewritten by the new identity/apply.py) but nothing
-- here needs to touch them.
--
-- Idempotent, additive-safe.
-- =============================================

alter table transcripts add column if not exists speaker_embeddings jsonb not null default '{}'::jsonb;

alter table cut_records add column if not exists voice_ids jsonb not null default '[]'::jsonb;
alter table cut_records add column if not exists speaker_person text null;
alter table cut_records add column if not exists visible_persons jsonb not null default '[]'::jsonb;
