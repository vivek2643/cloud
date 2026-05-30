-- =============================================
-- Dev mode: disable login/signup
-- Drop the FK to auth.users so the backend can use a hardcoded dev user UUID
-- without requiring a real Supabase auth user to exist.
--
-- Run this in Supabase SQL Editor.
-- To re-enable auth later, restore the constraints (see bottom of file).
-- =============================================

alter table public.folders
    drop constraint if exists folders_user_id_fkey;

alter table public.files
    drop constraint if exists files_user_id_fkey;

-- Optional: drop the existing RLS policies that reference auth.uid().
-- The backend already uses the service role key (which bypasses RLS),
-- so leaving the policies in place is harmless. They simply have no effect
-- when auth is disabled because the frontend never makes direct PostgREST calls.
-- If you still want a clean slate:
-- drop policy if exists "Users can view their own folders" on public.folders;
-- drop policy if exists "Users can insert their own folders" on public.folders;
-- drop policy if exists "Users can update their own folders" on public.folders;
-- drop policy if exists "Users can delete their own folders" on public.folders;
-- drop policy if exists "Users can view their own files" on public.files;
-- drop policy if exists "Users can insert their own files" on public.files;
-- drop policy if exists "Users can update their own files" on public.files;
-- drop policy if exists "Users can delete their own files" on public.files;

-- =============================================
-- To re-enable later:
-- alter table public.folders
--     add constraint folders_user_id_fkey
--     foreign key (user_id) references auth.users(id) on delete cascade;
--
-- alter table public.files
--     add constraint files_user_id_fkey
--     foreign key (user_id) references auth.users(id) on delete cascade;
-- =============================================
