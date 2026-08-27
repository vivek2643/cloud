-- =============================================
-- 057: Drop upload_links.user_id's FK to auth.users.
--
-- Live-verified bug in 056: dev mode (see 002_dev_disable_auth_fk.sql) runs
-- with a synthetic user id that has no auth.users row, and 002 already
-- dropped this exact FK from folders.user_id / files.user_id for that
-- reason. 056 re-added it on upload_links.user_id, which broke link
-- creation immediately -- "insert or update on table upload_links violates
-- foreign key constraint upload_links_user_id_fkey" against the live dev
-- user. Match the established convention: no FK to auth.users, same loose
-- ownership the rest of the app relies on (see 014_l3_edit_threads.sql).
-- =============================================

alter table public.upload_links
    drop constraint if exists upload_links_user_id_fkey;

-- =============================================
-- To re-enable later (once real Supabase auth users exist):
-- alter table public.upload_links
--     add constraint upload_links_user_id_fkey
--     foreign key (user_id) references auth.users(id) on delete cascade;
-- =============================================
