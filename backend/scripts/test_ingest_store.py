"""
Pure unit tests for the parts of ``app.services.l3.ingest_store`` that don't
need a database -- the SQL-touching functions are exercised implicitly by an
actual ingest run (real Postgres), not here.

Run:  .venv/bin/python scripts/test_ingest_store.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import ingest_store as store  # noqa: E402
from app.services.l3.post import CutRecord, PaceEnvelope  # noqa: E402


def _pace():
    return PaceEnvelope(min_ms=100, natural_ms=200, max_ms=300, levels=[1.0] * 5,
                       energy_grade="calm", natural_sound=False)


def _record(file_id="f1", take_group_id=None):
    return CutRecord(file_id=file_id, src_in_ms=0, src_out_ms=100, kind="speech",
                     word_span=(0, 1), atom_ids=None, label="x", summary="y",
                     speaker=None, on_camera=None, junk=False, junk_reason="",
                     junk_confidence="low",
                     framing={}, look={}, caption_zones=[], hero_ts_ms=50,
                     pace=_pace(), take_group_id=take_group_id, take_role=None)


def test_take_group_uuid_map_assigns_one_uuid_per_distinct_string_id():
    records = [_record(take_group_id="tg1"), _record(take_group_id="tg1"), _record(take_group_id="tg2")]
    mapping = store.take_group_uuid_map(records)
    assert set(mapping.keys()) == {"tg1", "tg2"}
    assert mapping["tg1"] != mapping["tg2"]
    print("ok  test_take_group_uuid_map_assigns_one_uuid_per_distinct_string_id")


def test_take_group_uuid_map_skips_records_without_a_group():
    records = [_record(take_group_id=None), _record(take_group_id="tg1")]
    mapping = store.take_group_uuid_map(records)
    assert set(mapping.keys()) == {"tg1"}
    print("ok  test_take_group_uuid_map_skips_records_without_a_group")


def test_take_group_uuid_map_empty_for_no_records():
    assert store.take_group_uuid_map([]) == {}
    print("ok  test_take_group_uuid_map_empty_for_no_records")


def main():
    test_take_group_uuid_map_assigns_one_uuid_per_distinct_string_id()
    test_take_group_uuid_map_skips_records_without_a_group()
    test_take_group_uuid_map_empty_for_no_records()
    print("\nall ingest_store tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
