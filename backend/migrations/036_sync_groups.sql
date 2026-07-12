-- =============================================
-- Multicam / dual-system audio sync (audio_sync.plan.md SS6).
--
-- A sync group is a set of files (camera angles + optionally an external
-- mic) the user has declared to be the same recorded moment, aligned on one
-- shared clock. `offset_ms` on a member is the group-clock position of that
-- file's own t=0 (group_ms = file_own_ms + offset_ms), computed by
-- deterministic cross-correlation (or set manually via the nudge UI).
--
-- Level 1 (this plan): ONE authoritative audio source per group, picked by
-- code (external file > best-sounding camera), overridable by the user.
-- Level 2 (per-speaker routing) is a later phase -- not modeled here.
--
-- `cut_records.sync_group_id` links a cut back to the group it was ingested
-- under (SS6's "pinning": a later re-sync must not mutate an existing cut's
-- group membership -- the ingest snapshots which group/offsets it used at
-- ingest time, same spirit as `ingest_run_id`).
--
-- Idempotent, additive-safe.
-- =============================================

create table if not exists sync_groups (
    id                            uuid primary key default uuid_generate_v4(),
    project_id                    uuid not null references projects(id) on delete cascade,
    authoritative_audio_file_id   uuid references files(id) on delete set null,
    created_by                    text not null default 'user' check (created_by in ('auto', 'user')),
    created_at                    timestamptz not null default now()
);

create index if not exists idx_sync_groups_project on sync_groups(project_id);

create table if not exists sync_group_members (
    group_id      uuid not null references sync_groups(id) on delete cascade,
    file_id       uuid not null references files(id) on delete cascade,
    -- Group-clock position of this file's own t=0: group_ms = file_own_ms + offset_ms.
    offset_ms     int not null,
    role          text not null check (role in ('video_angle', 'audio')),
    -- Cross-correlation peak height (0..1-ish, not a probability) that
    -- produced `offset_ms`; null when `aligned_by = 'manual'`.
    confidence    real,
    aligned_by    text not null check (aligned_by in ('auto', 'manual')),
    primary key (group_id, file_id)
);

create index if not exists idx_sync_group_members_file on sync_group_members(file_id);

alter table cut_records add column if not exists sync_group_id uuid references sync_groups(id) on delete set null;
create index if not exists idx_cut_records_sync_group on cut_records(sync_group_id);
