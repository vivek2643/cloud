#!/usr/bin/env python3
"""Tests for LOCAL-DEV DATA ISOLATION (DB_SCHEMA + R2_KEY_PREFIX).

Proves the two properties that keep production safe:
  1. UNSET == PRODUCTION: with neither env var set, every isolation hook is a
     no-op -- no search_path options, no R2 key prefix, ledger stays
     public.schema_migrations, migration SQL is passed through untouched.
  2. OPT-IN isolation: with DB_SCHEMA/R2_KEY_PREFIX set, connections pin the
     dev schema, keys get the prefix, the ledger + rewritten tables target the
     dev schema, and the `public.`->`dev.` rewrite is correct.

No DB / no network -- pure config + string logic (see test_grade.py's "no DB"
convention).

Run:  .venv/bin/python scripts/test_data_isolation.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.config import Settings  # noqa: E402
from app.services import db_migrations, r2  # noqa: E402


def _base_env() -> dict:
    """Minimal required Settings fields so Settings() constructs without a
    real .env -- values are placeholders, never used to connect here."""
    return {
        "supabase_url": "https://x.supabase.co",
        "supabase_service_key": "k",
        "r2_account_id": "a",
        "r2_access_key_id": "b",
        "r2_secret_access_key": "c",
    }


# --------------------------------------------------------------------------
# UNSET == PRODUCTION
# --------------------------------------------------------------------------

def test_unset_is_production_no_op(monkey_prefix=""):
    os.environ.pop("DB_SCHEMA", None)
    os.environ.pop("R2_KEY_PREFIX", None)
    s = Settings(**_base_env())
    assert s.db_schema == ""
    assert s.effective_db_schema == "public"
    assert s.pg_options == ""
    assert s.pg_connect_kwargs() == {}, "prod must pass NO connect kwargs"
    assert db_migrations._target_schema() == "public"
    assert db_migrations._migrations_table() == "public.schema_migrations"
    # Migration SQL passed through byte-for-byte when targeting public.
    sql = "create table public.files();\n-- comment public.x\n"
    assert db_migrations._rewrite_schema(sql, "public") == sql
    print("ok  test_unset_is_production_no_op")


def test_unset_r2_key_unchanged():
    os.environ.pop("R2_KEY_PREFIX", None)
    r2.get_settings.cache_clear()
    assert r2._full_key("raw/u/f/name.mp4") == "raw/u/f/name.mp4"
    assert r2._full_key("renders/abc.mp4") == "renders/abc.mp4"
    print("ok  test_unset_r2_key_unchanged")


# --------------------------------------------------------------------------
# OPT-IN ISOLATION
# --------------------------------------------------------------------------

def test_db_schema_pins_search_path():
    s = Settings(**_base_env(), db_schema="dev")
    assert s.effective_db_schema == "dev"
    assert s.pg_options == "-c search_path=dev,public"
    assert s.pg_connect_kwargs() == {"options": "-c search_path=dev,public"}
    print("ok  test_db_schema_pins_search_path")


def test_migrations_table_and_rewrite_target_dev():
    os.environ["DB_SCHEMA"] = "dev"
    try:
        assert db_migrations._target_schema() == "dev"
        assert db_migrations._migrations_table() == "dev.schema_migrations"
        sql = (
            "create table if not exists public.files(\n"
            "  id uuid references public.folders(id) on delete cascade);\n"
        )
        out = db_migrations._rewrite_schema(sql, "dev")
        assert "public." not in out, out
        assert "dev.files" in out and "dev.folders" in out
    finally:
        os.environ.pop("DB_SCHEMA", None)
    print("ok  test_migrations_table_and_rewrite_target_dev")


def test_r2_prefix_scopes_keys():
    os.environ["R2_KEY_PREFIX"] = "dev"
    r2.get_settings.cache_clear()
    try:
        assert r2._full_key("raw/u/f/name.mp4") == "dev/raw/u/f/name.mp4"
        # A prefixed delete can never name a bare production key.
        assert r2._full_key("renders/x.mp4").startswith("dev/")
    finally:
        os.environ.pop("R2_KEY_PREFIX", None)
        r2.get_settings.cache_clear()
    print("ok  test_r2_prefix_scopes_keys")


def test_prefix_slash_normalization():
    for raw in ("dev", "dev/", "/dev/"):
        os.environ["R2_KEY_PREFIX"] = raw
        r2.get_settings.cache_clear()
        try:
            assert r2._full_key("raw/x") == "dev/raw/x", raw
        finally:
            os.environ.pop("R2_KEY_PREFIX", None)
            r2.get_settings.cache_clear()
    print("ok  test_prefix_slash_normalization")


def main():
    test_unset_is_production_no_op()
    test_unset_r2_key_unchanged()
    test_db_schema_pins_search_path()
    test_migrations_table_and_rewrite_target_dev()
    test_r2_prefix_scopes_keys()
    test_prefix_slash_normalization()
    print("\nall data-isolation tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
