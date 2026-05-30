-- =============================================
-- AeroDrive Phase 0: Drive Schema
-- Run this in Supabase SQL Editor
-- =============================================

-- Enable UUID generation
create extension if not exists "uuid-ossp";

-- =============================================
-- Folders
-- =============================================

create table public.folders (
    id         uuid primary key default uuid_generate_v4(),
    user_id    uuid not null references auth.users(id) on delete cascade,
    name       text not null,
    parent_id  uuid references public.folders(id) on delete cascade,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index idx_folders_user_parent on public.folders(user_id, parent_id);

-- Auto-update updated_at
create or replace function public.set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger folders_updated_at
    before update on public.folders
    for each row execute function public.set_updated_at();

-- RLS
alter table public.folders enable row level security;

create policy "Users can view their own folders"
    on public.folders for select
    using (auth.uid() = user_id);

create policy "Users can insert their own folders"
    on public.folders for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own folders"
    on public.folders for update
    using (auth.uid() = user_id);

create policy "Users can delete their own folders"
    on public.folders for delete
    using (auth.uid() = user_id);

-- Service role bypass for backend operations
create policy "Service role full access to folders"
    on public.folders for all
    using (auth.role() = 'service_role');

-- =============================================
-- Files
-- =============================================

create table public.files (
    id                uuid primary key default uuid_generate_v4(),
    user_id           uuid not null references auth.users(id) on delete cascade,
    folder_id         uuid references public.folders(id) on delete set null,
    name              text not null,
    filename          text not null,
    mime_type         text not null,
    file_size         bigint not null default 0,
    file_type         text not null default 'other',
    r2_key            text not null,
    r2_proxy_key      text,
    r2_thumbnail_key  text,
    duration_seconds  double precision,
    width             integer,
    height            integer,
    status            text not null default 'uploading',
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create index idx_files_user_folder on public.files(user_id, folder_id);
create index idx_files_status on public.files(status);

create trigger files_updated_at
    before update on public.files
    for each row execute function public.set_updated_at();

-- RLS
alter table public.files enable row level security;

create policy "Users can view their own files"
    on public.files for select
    using (auth.uid() = user_id);

create policy "Users can insert their own files"
    on public.files for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own files"
    on public.files for update
    using (auth.uid() = user_id);

create policy "Users can delete their own files"
    on public.files for delete
    using (auth.uid() = user_id);

create policy "Service role full access to files"
    on public.files for all
    using (auth.role() = 'service_role');
