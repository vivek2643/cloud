-- =============================================
-- 017: Dialogue segments (the "Dialogues" lens).
--
-- One row per file holding the precomputed dialogue selects at BOTH
-- granularities in a single document:
--   segments = { "sentence": [DialogueSegment, ...],
--                "topic":    [DialogueSegment, ...] }
--
-- Derived deterministically from transcripts.segments (word timings +
-- speaker) with audio-snapped cut points, so the Sentence/Topic switch in
-- the UI is a zero-recompute read. file_id is the PK so re-running the
-- stage upserts. Best-effort: absence of a row just means "not computed yet".
-- =============================================

create table if not exists public.dialogue_segments (
    file_id        uuid primary key references public.files(id) on delete cascade,
    schema_version int  not null default 1,
    segments       jsonb not null default '{}'::jsonb,
    created_at     timestamptz not null default now()
);

comment on table public.dialogue_segments is
    'Dialogues lens: precomputed speech selects per file at sentence + topic granularity.';
