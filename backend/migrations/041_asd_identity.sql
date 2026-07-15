-- asd_identity.plan.md: per-file face tracks (detect+track+embed+ASD),
-- persisted once at L1 and reused across every ingest/project -- the
-- deterministic replacement for the old per-project Gemini voice-ID pass.
-- One row per file, jsonb + schema_version, same best-effort/nullable shape
-- as transcripts.speaker_embeddings and color_stats.

create table if not exists public.face_tracks (
    file_id         uuid primary key references public.files(id) on delete cascade,
    schema_version  int  not null default 1,
    tracks          jsonb not null default '[]'::jsonb,
    created_at      timestamptz not null default now()
);

comment on table public.face_tracks is
    'L1 active-speaker pass: per-file face tracks (embedding + sampled boxes + ASD-speaking intervals). See app/services/l1/active_speaker.py. Never a gate -- an empty tracks list just means identity/faces.py has nothing to cluster for this file.';
comment on column public.face_tracks.tracks is
    'List of {track_id, embedding, frames: [{t_ms, box}], speaking: [{start_ms, end_ms, score}], best_crop_ms}, proxy pixel space.';

-- processing_jobs.stage gains "active_speaker" as a row value -- no column/
-- constraint change needed, stage is free-text (see 003_l1_index.sql).
