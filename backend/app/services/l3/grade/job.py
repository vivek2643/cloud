"""
The `v1` grade pipeline as a background job (color_grading_upgrade.plan.md
Step 1.0 -- the foundational seam every later Phase-1 layer plugs into).

Why a job at all: the full `v1` stack (per-shot SPAN measurement, one
sequence-match pass, resolve, pre-bake) is too heavy to run inline on every
document resolve for a long timeline. So it runs ONCE per meaningful change,
as a Procrastinate task (`run_grade_job`), and PERSISTS the result
(`resolved_grades`) + live progress (`grade_jobs`) -- `layers.resolve` then
just READS under `grade_pipeline=="v1"` instead of computing (see
`fetch_latest_grades`, and `layers.py::resolve`'s `grade_lookup` param).
`legacy` never touches this file at all.

The read path never blocks on the job: `fetch_latest_grades` returns the
FRESHEST persisted row per shot regardless of `input_hash`, so a shot whose
job hasn't finished for the CURRENT edit simply keeps showing whatever grade
a PREVIOUS run already computed (or plain identity if none exists yet) --
preview stays responsive while the worker catches up.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg

from app.config import get_settings
from app.services.jobs import app as jobs_app
from app.services.l3.grade.cache import ensure_cube_file
from app.services.l3.grade.cdl import Grade
from app.services.l3.grade.match import ShotStats, solve_sequence_match
from app.services.l3.grade.measure import fetch_color_stats
from app.services.l3.grade.measure_span import measure_span
from app.services.l3.grade.resolver import resolve_clip_grade

logger = logging.getLogger(__name__)

# Bump when compute_input_hash's payload shape changes so old hashes never
# collide with a differently-computed new one.
INPUT_HASH_SCHEMA_VERSION = 1

# Same local-disk, content-addressed cube cache the preview cube endpoint and
# the render compositor already use (grade/cache.py's documented "not shared
# across instances" limitation applies here too -- see that module).
_CUBE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "edso_grade_cubes")


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


# --------------------------------------------------------------------------
# Shot enumeration + input_hash (shared by the enqueue trigger and the job)
# --------------------------------------------------------------------------

@dataclass
class Shot:
    key: str                       # seg_id (spine) or op_id (place_video)
    file_id: str
    in_ms: int
    out_ms: int
    hero_ts_ms: Optional[int]
    item: Dict[str, Any]           # the raw seg/op dict (grade/arc_intent/working_space overrides)


def ordered_shots(document: Dict[str, Any]) -> List[Shot]:
    """Every gradeable shot (main-line segs + `place_video` ops) in PROGRAM
    order -- the same population `layers.resolve` grades, and the order
    `solve_sequence_match` needs for neighbor grouping (Step 1.4)."""
    shots: List[Shot] = []
    for seg in document.get("timeline") or []:
        if not seg.get("seg_id") or not seg.get("file_id"):
            continue
        shots.append(Shot(
            key=str(seg["seg_id"]), file_id=str(seg["file_id"]),
            in_ms=int(seg.get("in_ms", 0)), out_ms=int(seg.get("out_ms", 0)),
            hero_ts_ms=seg.get("hero_ts_ms"), item=seg,
        ))
    for op in document.get("operations") or []:
        if op.get("type") != "place_video" or not op.get("op_id") or not op.get("source_file_id"):
            continue
        shots.append(Shot(
            key=str(op["op_id"]), file_id=str(op["source_file_id"]),
            in_ms=int(op.get("src_in_ms", 0)), out_ms=int(op.get("src_out_ms", 0)),
            hero_ts_ms=op.get("hero_ts_ms"), item=op,
        ))
    return shots


def compute_input_hash(document: Dict[str, Any]) -> str:
    """Stable hash over everything that can change what `run_grade_job` would
    produce: EVERY shot's identity + SPAN + per-clip overrides (an ORDERED
    list, not a set -- position matters, since `solve_sequence_match` groups
    by adjacency), the sequence-level look, and the active grade flags.
    Trimming a cut moves ITS OWN span stats AND can change which neighbors
    its matching window sees, so span (not just `look`) MUST be part of this
    payload -- a look-only hash would silently serve a stale match/measure
    after a trim. Two documents that would resolve to byte-identical v1
    grades hash the same; anything else changes it, which is exactly what
    `maybe_enqueue` diffs against to decide whether to re-run the job."""
    settings = get_settings()
    shots = ordered_shots(document)
    payload = {
        "v": INPUT_HASH_SCHEMA_VERSION,
        "shots": [
            {"key": s.key, "file_id": s.file_id, "in_ms": s.in_ms, "out_ms": s.out_ms,
             "grade": s.item.get("grade"), "arc_intent": s.item.get("arc_intent"),
             "working_space": s.item.get("working_space"), "subject_box": s.item.get("subject_box")}
            for s in shots
        ],
        "look": document.get("look") or {},
        "flags": {
            "grade_pipeline": settings.grade_pipeline,
            "grade_even_lighting": settings.grade_even_lighting,
            "grade_semantic": settings.grade_semantic,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def fetch_latest_grades(thread_id: str, shot_keys: List[str]) -> Dict[str, dict]:
    """The FRESHEST `resolved_grades` row per shot_key for this thread,
    regardless of `input_hash` -- under `v1`, `layers.resolve` reads
    EXCLUSIVELY from here. This single "most recent wins" query is what
    implements the graceful fallback (Step 1.0 §5): when the job for the
    CURRENT input_hash hasn't finished, the freshest row is simply whatever
    a PREVIOUS run last computed for that shot -- preview never blocks on
    the job. A shot never persisted at all is just absent (caller uses
    identity)."""
    if not shot_keys:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            """
            select distinct on (shot_key) shot_key, grade_json
              from resolved_grades
             where thread_id = %s and shot_key = any(%s)
             order by shot_key, updated_at desc
            """,
            (thread_id, list(shot_keys)),
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_job_state(thread_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        row = conn.execute(
            "select state, total, done, input_hash, error from grade_jobs where thread_id = %s",
            (thread_id,),
        ).fetchone()
    if not row:
        return None
    return {"state": row[0], "total": row[1], "done": row[2], "input_hash": row[3], "error": row[4]}


def _upsert_job_status(
    thread_id: str, *, state: Optional[str] = None, total: Optional[int] = None,
    done: Optional[int] = None, input_hash: Optional[str] = None, error: Optional[str] = None,
) -> None:
    """Dynamic partial upsert (mirrors `render/store.py::update_status`):
    ensures a row exists, then sets only the given fields -- so e.g. bumping
    `done` doesn't require re-passing `total`/`input_hash` every call."""
    with _pg() as conn:
        conn.execute(
            "insert into grade_jobs (thread_id) values (%s) on conflict (thread_id) do nothing",
            (thread_id,),
        )
        sets: List[str] = []
        params: List[Any] = []
        for col, val in (("state", state), ("total", total), ("done", done),
                        ("input_hash", input_hash), ("error", error)):
            if val is not None:
                sets.append(f"{col} = %s")
                params.append(val)
        if state == "grading":
            sets.append("error = null")   # a fresh pass clears any stale error
        if not sets:
            return
        params.append(thread_id)
        conn.execute(f"update grade_jobs set {', '.join(sets)} where thread_id = %s", params)


def _upsert_grade_row(thread_id: str, shot_key: str, input_hash: str,
                      grade_json: Dict[str, Any], cube_ref: Optional[str]) -> None:
    with _pg() as conn:
        conn.execute(
            """
            insert into resolved_grades (thread_id, shot_key, input_hash, grade_json, cube_ref)
            values (%s, %s, %s, %s::jsonb, %s)
            on conflict (thread_id, shot_key, input_hash) do update set
                grade_json = excluded.grade_json,
                cube_ref = excluded.cube_ref
            """,
            (thread_id, shot_key, input_hash, json.dumps(grade_json), cube_ref),
        )


# --------------------------------------------------------------------------
# Enqueue trigger (called from the edit-thread save paths)
# --------------------------------------------------------------------------

def maybe_enqueue(thread_id: str, document: Dict[str, Any]) -> None:
    """Enqueue `run_grade_job` iff the CURRENT `input_hash` differs from
    whatever this thread last ran/is running (idempotent -- Step 1.0 §4).
    A no-op entirely under `legacy` (the job is v1-only) or with no
    gradeable shots yet."""
    settings = get_settings()
    if settings.grade_pipeline != "v1" or not (document.get("timeline") or document.get("operations")):
        return
    h = compute_input_hash(document)
    existing = get_job_state(thread_id)
    if existing and existing.get("input_hash") == h and existing.get("state") in ("grading", "done"):
        return
    _upsert_job_status(thread_id, state="idle", input_hash=h)
    try:
        jobs_app.configure_task("run_grade_job", queue="grade").defer(thread_id=thread_id)
    except Exception:
        logger.exception("grade.job: failed to enqueue run_grade_job for thread %s", thread_id)


# --------------------------------------------------------------------------
# The task
# --------------------------------------------------------------------------

@jobs_app.task(name="run_grade_job", queue="grade", retry=False)
def run_grade_job(thread_id: str) -> None:
    """Step 1.0 §3: measure each shot's OWN span (1.2), match neighbors once
    (1.4), resolve + persist each shot's grade (1.1/1.3/1.5), pre-bake each
    distinct grade's `.cube`. Re-derives `input_hash` from the LATEST
    document at run time (rather than trusting a hash captured when this was
    enqueued) so a worker that was busy for a while still grades the current
    state, not a stale one -- if a newer enqueue already landed by the time
    this runs, whichever run finishes LAST wins (upserts are idempotent).
    Never crashes the worker: any exception is recorded as `grade_jobs.error`
    (same best-effort semantics as `color_stats`/`render_edit`) -- the read
    path's graceful fallback means a stuck job never blocks preview, it just
    stays on the last-good grade."""
    from app.services.l3 import store as edit_store

    document, _version = edit_store.latest_document(thread_id)
    if not document:
        return
    h = compute_input_hash(document)
    existing = get_job_state(thread_id)
    if existing and existing.get("state") == "done" and existing.get("input_hash") == h:
        return   # already satisfied by the time this run actually started

    shots = ordered_shots(document)
    _upsert_job_status(thread_id, state="grading", total=len(shots), done=0, input_hash=h)
    try:
        file_ids = list({s.file_id for s in shots})
        color_stats = fetch_color_stats(file_ids)
        sequence_look = document.get("look")

        # Step 1.2: measure the span actually used; never-worse fallback to
        # whole-file color_stats when span measurement fails/unavailable.
        shot_stats: Dict[str, ShotStats] = {}
        for s in shots:
            stats = measure_span(s.file_id, s.in_ms, s.out_ms, hero_ts_ms=s.hero_ts_ms)
            if stats is None:
                stats = color_stats.get(s.file_id)
            shot_stats[s.key] = ShotStats(key=s.key, file_id=s.file_id, stats=stats)

        # Step 1.4: ONE sequence-match pass over the whole ordered timeline.
        match_deltas: Dict[str, Grade] = solve_sequence_match(
            [shot_stats[s.key] for s in shots]
        )

        cube_cache: Dict[str, Optional[str]] = {}
        done = 0
        for s in shots:
            stats = shot_stats[s.key].stats
            grade_json = resolve_clip_grade(
                s.item, color_stats=stats, sequence_look=sequence_look,
                match_delta=match_deltas.get(s.key), pipeline="v1",
            )
            gh = grade_json.get("grade_hash")
            if gh not in cube_cache:
                try:
                    cube_cache[gh] = ensure_cube_file(grade_json, _CUBE_CACHE_DIR)
                except Exception:
                    logger.exception("run_grade_job: cube bake failed for hash %s (thread %s)", gh, thread_id)
                    cube_cache[gh] = None
            _upsert_grade_row(thread_id, s.key, h, grade_json, cube_cache.get(gh))
            done += 1
            _upsert_job_status(thread_id, done=done)

        _upsert_job_status(thread_id, state="done", done=done, input_hash=h)
        logger.info("run_grade_job: thread %s graded %d shots (hash %s)", thread_id, done, h)
    except Exception as e:  # noqa: BLE001 -- never crash the worker
        logger.exception("run_grade_job failed for thread %s", thread_id)
        _upsert_job_status(thread_id, state="error", error=str(e)[:1000])
