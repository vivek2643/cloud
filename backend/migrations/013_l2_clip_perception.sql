-- =============================================
-- 013: L2 perception store.
--
-- One row per clip holding the Gemini "footage log" (ClipPerception) as JSONB.
-- The whole artifact is stored as one document because it is read/written
-- atomically (one VLM pass per clip) and queried structurally downstream
-- rather than relationally. file_id is the PK so re-running L2 upserts.
--
-- files.l2_status (added in 003) is reused as the lifecycle flag:
--   null -> never eligible/run, 'queued', 'running', 'ready', 'failed',
--   'skipped' (clip longer than L2_MAX_DURATION_SECONDS or L2 disabled).
-- =============================================

create table if not exists public.clip_perception (
    file_id        uuid primary key references public.files(id) on delete cascade,
    schema_version int  not null default 1,
    model          text,
    perception     jsonb not null default '{}'::jsonb,
    usage          jsonb,
    created_at     timestamptz not null default now()
);

comment on table public.clip_perception is
    'L2 VLM (Gemini) perception artifact: one ClipPerception document per short clip.';
