"""
Per-span color measurement (color_grading_upgrade.plan.md Step 1.2): matching/
correcting on a whole-file mean is wrong when the timeline only plays a 2s
window of a 40s clip -- this measures the span actually USED.

Runs INSIDE `grade/job.py::run_grade_job`, never inline on a document
resolve (it downloads + decodes a few frames, the same cost class L1's own
`color_stats` pays once at ingest). Cached per `(file_id, in_ms, out_ms)` in
`cut_color_stats` so repeated job runs over an unchanged span never re-decode.

Reuses `l1/color_stats.py`'s decode + aggregate primitives verbatim (same
frame shape, same statistics) so a span measurement and a whole-file
measurement are directly comparable -- `resolve_clip_grade`'s callers can
treat either one as "a color_stats row" without caring which."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import psycopg

from app.config import get_settings
from app.services.l1 import color_stats as color_stats_mod
from app.services.processing import _download_from_r2

logger = logging.getLogger(__name__)

# Bump when the cached shape changes so stale cache rows recompute.
SCHEMA_VERSION = 1
# Cheap: a handful of frames within the span is plenty for mean/percentile
# stats (vs. L1's whole-file COLOR_STATS_MAX_FRAMES=12 over a much longer
# window) -- span durations are typically a few seconds, not tens of.
SPAN_MAX_FRAMES = 4


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _cached(file_id: str, in_ms: int, out_ms: int) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        row = conn.execute(
            """
            select stats_json from cut_color_stats
             where file_id = %s and in_ms = %s and out_ms = %s and schema_version = %s
            """,
            (file_id, in_ms, out_ms, SCHEMA_VERSION),
        ).fetchone()
    return row[0] if row else None


def _store_cache(file_id: str, in_ms: int, out_ms: int, stats: Dict[str, Any]) -> None:
    with _pg() as conn:
        conn.execute(
            """
            insert into cut_color_stats (file_id, in_ms, out_ms, schema_version, stats_json)
            values (%s, %s, %s, %s, %s::jsonb)
            on conflict (file_id, in_ms, out_ms) do update set
                schema_version = excluded.schema_version,
                stats_json = excluded.stats_json,
                created_at = now()
            """,
            (file_id, in_ms, out_ms, SCHEMA_VERSION, json.dumps(stats)),
        )


def _fetch_proxy_path(file_id: str, tmp_dir: str) -> Optional[str]:
    """Download this file's proxy (or original) to a local temp path -- the
    same `r2_proxy_key`-preferred lookup `render/tasks.py::_file_lookup`
    uses, since L1's own ingest-time proxy download is long gone by the time
    this runs (well after ingest, inside the grade job)."""
    with _pg() as conn:
        row = conn.execute(
            "select r2_proxy_key, r2_key from files where id = %s", (file_id,)
        ).fetchone()
    if not row:
        return None
    key = row[0] or row[1]
    if not key:
        return None
    ext = os.path.splitext(key)[1] or ".mp4"
    local = os.path.join(tmp_dir, f"span_{file_id}{ext}")
    _download_from_r2(key, local)
    return local


def measure_span(
    file_id: str, in_ms: int, out_ms: int, *, hero_ts_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """A `color_stats`-shaped dict measured over `[in_ms, out_ms)` of `file_id`
    only -- biases one sample to `hero_ts_ms` when it falls inside the span.
    None on any failure (missing file, decode error, empty span) -- never-
    worse: the caller falls back to whole-file `color_stats`, exactly like a
    file with no L1 measurement at all."""
    try:
        in_ms, out_ms = int(in_ms), int(out_ms)
    except (TypeError, ValueError):
        return None
    if out_ms <= in_ms or not file_id:
        return None

    cached = _cached(file_id, in_ms, out_ms)
    if cached is not None:
        return cached

    try:
        with tempfile.TemporaryDirectory(prefix="edso_span_") as tmp:
            path = _fetch_proxy_path(file_id, tmp)
            if not path:
                return None
            span_s = (out_ms - in_ms) / 1000.0
            offsets = color_stats_mod._sample_timestamps(span_s, SPAN_MAX_FRAMES)
            timestamps = [in_ms / 1000.0 + o for o in offsets]
            if hero_ts_ms is not None and in_ms <= int(hero_ts_ms) <= out_ms:
                timestamps[0] = int(hero_ts_ms) / 1000.0

            frames: List[Any] = []
            for ts in timestamps:
                frame = color_stats_mod._decode_rgb_frame_at(
                    path, ts, color_stats_mod.COLOR_STATS_W, color_stats_mod.COLOR_STATS_H,
                )
                if frame is not None:
                    frames.append(frame)
            if not frames:
                return None
            stats = color_stats_mod._aggregate(frames).to_dict()
    except Exception:
        logger.exception("measure_span failed for %s [%d,%d)", file_id, in_ms, out_ms)
        return None

    _store_cache(file_id, in_ms, out_ms, stats)
    return stats
