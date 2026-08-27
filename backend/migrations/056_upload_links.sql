-- =============================================
-- 056: Anonymous upload links (frontend_project_ux.plan.md Stage 2).
--
-- A bearer-token link that lets anyone (no account) upload files into ONE
-- project (folder). Files land owned by the folder's owner, not the
-- anonymous uploader -- user_id is copied from the folder at creation time.
--
-- token is a secrets.token_urlsafe(32) bearer credential: unguessable is the
-- entire security model (see the public router's _resolve_link guard), since
-- anyone holding it can write into the project.
--
-- used_count is incremented at completion, not at presign, so an abandoned
-- or failed upload never burns the quota -- see app/routers/upload_links.py.
-- =============================================

create table if not exists public.upload_links (
    id          uuid primary key default uuid_generate_v4(),
    token       text not null unique,
    folder_id   uuid not null references public.folders(id) on delete cascade,
    user_id     uuid not null references auth.users(id) on delete cascade,
    expires_at  timestamptz,          -- null = never expires
    max_files   int,                  -- null = unlimited
    used_count  int  not null default 0,
    revoked     bool not null default false,
    created_at  timestamptz not null default now()
);

create index if not exists upload_links_token_idx  on public.upload_links (token);
create index if not exists upload_links_folder_idx on public.upload_links (folder_id);

comment on table public.upload_links is
    'frontend_project_ux.plan.md Stage 2: anonymous, tokened upload links scoped to one project (folder). Revoked (not deleted) on teardown so a used link can report why it stopped working.';
comment on column public.upload_links.token is
    'Bearer credential (secrets.token_urlsafe(32)) -- unguessable, since possession alone grants write access to the project.';
comment on column public.upload_links.user_id is
    'The PROJECT OWNER (copied from folders.user_id at creation), not the anonymous uploader -- uploaded files are owned by them.';
comment on column public.upload_links.used_count is
    'Incremented at upload COMPLETION, not at presign -- an abandoned or failed presign must not burn the quota.';
