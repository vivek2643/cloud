-- =============================================
-- audio_and_audit.plan.md Phase 4: coarse musical structure on audio_features.
--
-- Beyond BPM + onsets, a musical source (an A2 bed or a video clip doubling
-- as one) also gets a coarse structure -- section/phrase boundaries and the
-- single strongest ("drop") moment -- so "shift so a strong beat lands on
-- the climax" (guidance_doc.md Working with music) can target something
-- specific instead of any onset. Approximate (a prior, not truth): both
-- default empty/null when detection didn't run or was unreliable, never a
-- fabricated boundary.
-- =============================================

alter table public.audio_features
    add column if not exists sections jsonb not null default '[]'::jsonb,
    add column if not exists drop_ms  int;

comment on column public.audio_features.sections is
    'Coarse phrase/section boundaries [{start_ms,end_ms}], snapped to the bar grid -- a prior for pacing, not verified song structure.';
comment on column public.audio_features.drop_ms is
    'The single strongest musical moment (largest local onset-strength jump, snapped to the nearest beat), in this source''s OWN time -- null when undetected.';
