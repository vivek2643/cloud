"""
Measure (color_grading.plan.md SS3, first stage of the grade stack): batch-
fetch L1 `color_stats` for every source file a document references, once per
turn -- same pattern as `observe.build_context`'s `durations` fetch (see
`render/tasks._durations`). The correct/match layers only ever see this
already-fetched dict; they never touch the database themselves, keeping
`grade.resolver` a pure function over its inputs (cheap to call once per
clip, easy to test without a live DB).
"""
from __future__ import annotations

from typing import Dict, List

import psycopg

from app.config import get_settings

_COLS = (
    "file_id", "black_point", "white_point", "mid_gray", "rgb_mean",
    "rgb_median", "rgb_std", "lab_ab_cast", "wb_gray_world", "wb_white_patch",
    "clip_shadow_pct", "clip_highlight_pct", "is_log_flat", "skin_lab",
)


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def fetch_color_stats(file_ids: List[str]) -> Dict[str, dict]:
    """file_id -> color_stats row (as a plain dict), for every id that has
    one. Missing entries (L1 hasn't run color_stats for that file yet, or
    the file predates this stage) are simply absent -- callers treat that
    as "no measurement, correct layer stays identity" (never-worse: no
    measurement means no basis to change anything)."""
    ids = list({f for f in file_ids if f})
    if not ids:
        return {}
    cols_sql = ", ".join(_COLS)
    with _pg() as conn:
        rows = conn.execute(
            f"select {cols_sql} from color_stats where file_id = any(%s::uuid[])",
            (ids,),
        ).fetchall()
    out: Dict[str, dict] = {}
    for row in rows:
        d = dict(zip(_COLS, row))
        # psycopg returns a Postgres `uuid` column as a `uuid.UUID`, but every
        # caller looks this dict up with a STRING file_id (from the document
        # JSON: `color_stats.get(seg["file_id"])`). Keying by the raw UUID makes
        # every `.get(str)` silently miss -> the correct + match layers get no
        # data and collapse to identity (grading degrades to a flat global look
        # on uncorrected footage). Normalize to str so the lookup actually hits.
        out[str(d.pop("file_id"))] = d
    return out
