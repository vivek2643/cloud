-- =============================================
-- 045: migrations-tracking table for the new applier/guard system
-- (migration_runner.plan.md).
--
-- Root cause this closes: there was previously no record anywhere of which
-- of the 44 prior .sql files had actually been run against a given
-- database -- migrations were applied by hand, pasted into the Supabase SQL
-- Editor one at a time (see README.md's old "Database" section). That gap
-- is what let 042/043 sit unapplied against production while code that
-- depended on them shipped anyway, and is the likely explanation for 018
-- having been applied only partially (2 of its 7 target column drops
-- landed, 5 did not) -- manual, non-atomic application, more than once.
--
-- `checksum` is sha256 of the file's NORMALIZED text (trailing whitespace
-- stripped per line, single trailing newline -- see
-- app/services/db_migrations.py::checksum()), not raw bytes, so an
-- innocuous re-save of an old file doesn't flip its hash and trip the
-- drift guard. A checksum mismatch means "this file's content changed
-- since it was recorded as applied" -- flagged as drift, never silently
-- ignored.
--
-- This migration's own row is included in the one-time 45-row backfill
-- (001..045) that accompanies it -- see migration_runner.plan.md Step 1.
-- =============================================

create table if not exists public.schema_migrations (
    filename    text primary key,
    checksum    text not null,
    applied_at  timestamptz not null default now()
);
