"""
The grade pipeline as a background job (color_grading_upgrade.plan.md
Step 1.0 -- the foundational seam every later Phase-1 layer plugs into;
standardized on as the SOLE pipeline by grade_pipeline_standardize.plan.md).

Why a job at all: the full stack (per-shot SPAN measurement, one
sequence-match pass, resolve, pre-bake) is too heavy to run inline on every
document resolve for a long timeline. So it runs ONCE per meaningful change,
as a Procrastinate task (`run_grade_job`), and PERSISTS the result
(`resolved_grades`) + live progress (`grade_jobs`) -- `layers.resolve` then
just READS (see `fetch_latest_grades`, and `layers.py::resolve`'s
`grade_lookup` param), never computes inline.

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
from app.services.l3.grade.balance import solve_balance
from app.services.l3.grade.cache import ensure_cube_file
from app.services.l3.grade.cdl import Grade
from app.services.l3.grade.leveling import ShotLevelInput, solve_leveling
from app.services.l3.grade.match import ShotStats, group_neighbors, solve_sequence_match
from app.services.l3.grade.measure import fetch_color_stats
from app.services.l3.grade.measure_span import measure_span
from app.services.l3.grade.reference import GroupReference, compute_group_reference
from app.services.l3.grade.resolver import resolve_clip_grade
from app.services.l3.grade.scene_group import ShotSceneMeta as SceneMeta
from app.services.l3.grade.scene_group import group_shots_semantically
from app.services.l3.grade.tone import WORKING_SPACE_V1, from_working_scalar, to_working_scalar

logger = logging.getLogger(__name__)

# Bump when compute_input_hash's payload shape changes so old hashes never
# collide with a differently-computed new one -- OR when the grade MATH itself
# changes (the produced grade for the same payload differs), so every thread's
# cached `done` grades are invalidated and re-graded on next run.
# v2: Correct/Match now solve in the v1 working space + composite guardrails
# (was solving in display space then applying in linear -> crushed shadows).
# v3 (color_shot_matching.plan.md): two-stage group->balance->match redesign
# -- graceful RGB grouping fallback, a new Balance layer, a robust
# median-member group reference replacing the arbitrary anchor, stronger
# Match strengths, a wider Leveling window. The grade math changed, so every
# cached grade must be recomputed.
# v4 (color_scene_grouping.plan.md): semantic grouping now joins each shot to
# its covering cut_record for real speaker/on_camera/label/summary/take/sync
# metadata (was always empty on the raw timeline seg), plus an always-on RGB
# base so grouping itself never degrades to all-singletons. Grouping feeds
# Balance/Match directly, so this changes the grade math too.
# v5: resolver._clamp_composite_v1's negative-offset floor now also anchors
# at a genuine shadow probe (COMPOSITE_SHADOW_PROBE), not just mid-gray --
# fixes a real shadow crush (a shot needing only a modest negative offset
# could crush a display ~0.15 shadow to pure black while mid-gray stayed
# safely above the old floor; verified live on 7 of 53 real shots). The
# grade math changed, so every cached grade must be recomputed.
# v6 (color_subject_exposure.plan.md): subject_box now gets populated from
# cut_records.framing.subject_box (was always None -- the whole
# subject_luma -> Leveling chain existed but was dormant), and Leveling
# converges a grouped shot's subject_luma toward the group's median subject
# luma instead of the group's whole-frame mid_gray. Grade math changed.
# v7 (color_skin_vibrance.plan.md): the Correct layer gains a skin-anchored
# tint vote (blended into the WB solve) and a bounded global saturation
# floor (chroma_mean -> sat), both gated on grade_skin_vibrance. Grade math
# changed whenever the flag is on.
# v8 (color_tone_contrast.plan.md): a filmic contrast S-curve is baked into
# tone.from_working (gated on grade_tone_contrast), changing the baked cube
# (not the CDL itself) whenever the flag is on.
# v9 (color_response_engine.plan.md): the Look layer gains a new "engine"
# mode -- a LookSpec baked into the creative LUT grid (gated on
# grade_look_engine) -- changing the baked cube whenever a document's
# document["look"] selects it and the flag is on.
# v10 (halation_grain.plan.md): an active engine look's halation/grain
# params now route into soft_local (gated on grade_film_texture, requires
# grade_look_engine too), changing the render -vf chain / preview pass
# whenever both flags are on and the look carries texture.
# v11 (grade_pipeline_standardize.plan.md): collapsed every dev flag this
# history tracked (grade_pipeline/even_lighting/semantic/shot_match_v2/
# scene_join/subject_exposure/skin_vibrance/look_engine) to permanently-on
# behavior, made film_texture purely look-scoped (no flag), and hardwired
# the global tone_contrast S-curve permanently OFF (0.0) -- the "flags"
# payload below is gone entirely, so this version bump alone is what forces
# every project's stored grade to be recomputed against the new pipeline.
INPUT_HASH_SCHEMA_VERSION = 11

# Same local-disk, content-addressed cube cache the preview cube endpoint and
# the render compositor already use (grade/cache.py's documented "not shared
# across instances" limitation applies here too -- see that module).
_CUBE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "edso_grade_cubes")


def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _to_working_scalar(value: Optional[float], default: Optional[float]) -> float:
    """Thin wrapper over the shared `tone.to_working_scalar` (kept for this
    module's call sites) -- projects one DISPLAY-encoded scalar (a measured
    mid_gray/black_point/white_point/subject_luma) into the v1 working space,
    the same transform correct.py/match.py now use to solve their levels
    CDL in the space it's applied. `value is None` -> `default`."""
    return to_working_scalar(value, default, WORKING_SPACE_V1)


def _has_real_groups(groups: Optional[List[List[int]]]) -> bool:
    """color_shot_matching.plan.md Phase 1: True iff at least one group has
    2+ members (i.e. grouping actually found structure to match on). All-
    singletons means the semantic signals were absent/unhelpful for this
    document -- fall back to RGB adjacency (`match.group_neighbors`)
    instead of silently letting matching do nothing."""
    return bool(groups) and any(len(g) >= 2 for g in groups)


def _ws_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """color_shot_matching.plan.md Phase 2: project one shot's DISPLAY-space
    color_stats dict into WORKING-space scalars -- the shape both
    `reference.compute_group_reference` and `balance.solve_balance` operate
    on (Balance's math has to live in the same space the CDL is applied,
    same reasoning as correct.py/match.py's `_project`/`_proj`)."""
    stats = stats or {}
    mid = stats.get("mid_gray")
    return {
        "mid_gray": _to_working_scalar(mid, None) if mid is not None else None,
        "black_point": _to_working_scalar(stats.get("black_point"), 0.0),
        "white_point": _to_working_scalar(stats.get("white_point"), 1.0),
        "rgb_mean": [_to_working_scalar(c, 0.5) for c in (stats.get("rgb_mean") or [0.5, 0.5, 0.5])],
    }


def _apply_working_scalar(value: Optional[float], grade: Grade, channel: int = 1) -> Optional[float]:
    """Apply one channel of an already-solved `grade`'s slope/offset to a
    WORKING-space scalar. Channel 1 (green) by construction is never
    touched by Balance's white-balance term (balance.py always normalizes
    so the green multiplier is exactly 1.0), so this isolates the
    exposure/contrast contribution -- the right channel for projecting an
    ACHROMATIC scalar (black/white/mid_gray) through a per-channel grade."""
    if value is None:
        return None
    return value * grade.slope[channel] + grade.offset[channel]


def _corrected_display_stats(ws: Dict[str, Any], grade: Grade) -> Dict[str, Any]:
    """color_shot_matching.plan.md Phase 2c: project a shot's WORKING-space
    stats (`_ws_stats`) through an already-solved `grade` (its Balance
    delta) and back to DISPLAY space, so a LATER stage (Match) solves
    against the image AS CORRECTED, not the raw source -- the same "solve
    against the corrected image" discipline resolver.py's
    `_corrected_source_stats` already uses for the Look layer.

    This is NOT optional polish: composing two independently-solved "move
    toward the same group reference" deltas (Balance's, then Match's, both
    aimed at the raw source) does not converge -- it overshoots, because
    Match's own delta was solved assuming the shot was still at its RAW
    position. Verified empirically (an adversarial synthetic 5-shot group,
    color_shot_matching.plan.md's own suggested mids [0.30, 0.55, 0.32,
    0.50, 0.35]): composing Match-on-raw-stats on top of Balance left
    cross-shot mid_gray stdev at 0.101 (vs 0.101 pre-redesign -- a virtual
    no-op) and actually WORSENED contrast stdev (0.138 -> 0.218); Match
    solved against THESE corrected stats instead converges mid_gray stdev
    to 0.011 and contrast stdev to 0.081 on the same fixture."""
    mid = ws.get("mid_gray")
    rgb = ws.get("rgb_mean") or [0.5, 0.5, 0.5]
    corrected_mid = _apply_working_scalar(mid, grade, 1)
    return {
        "black_point": from_working_scalar(_apply_working_scalar(ws.get("black_point"), grade, 1), WORKING_SPACE_V1),
        "white_point": from_working_scalar(_apply_working_scalar(ws.get("white_point"), grade, 1), WORKING_SPACE_V1),
        "mid_gray": from_working_scalar(corrected_mid, WORKING_SPACE_V1) if corrected_mid is not None else None,
        "rgb_mean": [
            from_working_scalar(_apply_working_scalar(rgb[c], grade, c), WORKING_SPACE_V1) for c in range(3)
        ],
    }


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
    by adjacency) and the sequence-level look. Trimming a cut moves ITS OWN
    span stats AND can change which neighbors its matching window sees, so
    span (not just `look`) MUST be part of this payload -- a look-only hash
    would silently serve a stale match/measure after a trim. Two documents
    that would resolve to byte-identical grades hash the same; anything else
    changes it, which is exactly what `maybe_enqueue` diffs against to decide
    whether to re-run the job. (No "flags" payload -- the pipeline has no
    more dev flags; `INPUT_HASH_SCHEMA_VERSION` alone forces a re-grade when
    the grade MATH changes.)"""
    shots = ordered_shots(document)
    payload = {
        "v": INPUT_HASH_SCHEMA_VERSION,
        "shots": [
            {"key": s.key, "file_id": s.file_id, "in_ms": s.in_ms, "out_ms": s.out_ms,
             "grade": s.item.get("grade"), "arc_intent": s.item.get("arc_intent"),
             "working_space": s.item.get("working_space"), "subject_box": s.item.get("subject_box"),
             # Phase 3.2's semantic grouping keys off these -- a speaker
             # re-identification or a relabel can change which shots match.
             "speaker_person": s.item.get("speaker_person"), "on_camera": s.item.get("on_camera"),
             "label": s.item.get("label"), "summary": s.item.get("summary")}
            for s in shots
        ],
        "look": document.get("look") or {},
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
    A no-op with no gradeable shots yet."""
    if not (document.get("timeline") or document.get("operations")):
        return
    h = compute_input_hash(document)
    existing = get_job_state(thread_id)
    if existing and existing.get("input_hash") == h and existing.get("state") in ("grading", "done"):
        return
    _upsert_job_status(thread_id, state="idle", input_hash=h)
    try:
        # Short-lived, per-call connector opened around the defer -- mirrors
        # upload.py::_enqueue_l1 / renders.py::_enqueue. The API process never
        # opens the shared `jobs_app`, so `jobs_app.configure_task(...).defer()`
        # raised AppNotOpen, got swallowed by the except below, and NO grade job
        # was ever enqueued (every look/trim silently no-op'd the preview).
        from procrastinate import App, PsycopgConnector

        enqueue_app = App(connector=PsycopgConnector(
            conninfo=get_settings().database_url, min_size=1, max_size=2))
        with enqueue_app.open():
            enqueue_app.configure_task("run_grade_job", queue="grade").defer(thread_id=thread_id)
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

        # color_subject_exposure.plan.md: the cut_records join (scene_meta's
        # metadata + subject_box) must resolve BEFORE the measure loop below
        # -- measure_span needs the subject box up front to measure
        # subject_luma on the hero frame. One join feeds two downstream
        # consumers: the METADATA (label/summary/speaker/etc, used for
        # grouping below) and the SUBJECT BOX (used for measurement).
        from app.services.l3.grade.scene_meta import lookup_shot_cut_meta

        cut_meta = lookup_shot_cut_meta(
            [(s.key, s.file_id, s.in_ms, s.out_ms) for s in shots]
        )
        subject_boxes: Dict[str, List[float]] = {
            k: m.subject_box for k, m in cut_meta.items() if m.subject_box
        }

        # Step 1.2 (+ Step 3.1, + color_subject_exposure.plan.md): measure
        # the span actually used; never-worse fallback to whole-file
        # color_stats when span measurement fails/unavailable. subject_box
        # (the joined cut_records.framing.subject_box; an item-level
        # subject_box -- always None today, no caller ever writes one --
        # wins if somehow present, for forward-compat) also measures
        # subject_luma, but ONLY on the hero frame -- and no real timeline
        # seg carries its own hero_ts_ms (verified live: 0/many documents),
        # so without a fallback the whole subject_luma chain stays inert
        # even with a valid box. cut_records.hero_ts_ms is 100% populated
        # and on the SAME source-time axis as in_ms/out_ms (the exact axis
        # scene_meta's own overlap join already assumes) -- use it as a
        # fallback, but ONLY when we're actually resolving a subject box,
        # so a document/shot not using this feature never has its color-
        # stats SAMPLE POINTS shifted (measure_span reuses hero_ts_ms to
        # pick timestamps[0] even without a box -- scoping this narrowly
        # keeps that pre-existing behavior byte-identical when unused).
        shot_stats: Dict[str, ShotStats] = {}
        for s in shots:
            subject_box = s.item.get("subject_box") or subject_boxes.get(s.key)
            hero_ts_ms = s.hero_ts_ms
            if hero_ts_ms is None and subject_box is not None:
                cm = cut_meta.get(s.key)
                if cm is not None and cm.hero_ts_ms is not None:
                    hero_ts_ms = cm.hero_ts_ms
            stats = measure_span(s.file_id, s.in_ms, s.out_ms, hero_ts_ms=hero_ts_ms,
                                 subject_box=subject_box)
            if stats is None:
                stats = color_stats.get(s.file_id)
            shot_stats[s.key] = ShotStats(key=s.key, file_id=s.file_id, stats=stats)

        # Step 1.4 (+ Step 3.2, + color_scene_grouping.plan.md): ONE
        # sequence-match pass over the whole ordered timeline -- semantic
        # grouping (when the shots carry the structural facts it needs)
        # overrides the default RGB-adjacency grouping so matching aligns
        # shots that are the same scene BY MEANING, not merely RGB-close.
        # The raw timeline seg never carries speaker_person/on_camera/label/
        # summary (they live on cut_records, one join away) -- use the
        # (already-joined, see above) real values, plus its own span
        # rgb_mean as scene_group's graceful RGB base (see scene_group.py's
        # trust hierarchy): a shot with no covering cut_record simply has no
        # metadata, so group_shots_semantically falls back to the RGB base
        # for it rather than forcing an all-singleton result.
        scene_meta = []
        for s in shots:
            cm = cut_meta.get(s.key)
            span_rgb = (shot_stats[s.key].stats or {}).get("rgb_mean")
            scene_meta.append(SceneMeta(
                key=s.key, file_id=s.file_id,
                speaker_person=(cm.speaker_person if cm else None),
                on_camera=(cm.on_camera if cm else None),
                label=(cm.label if cm else ""),
                summary=(cm.summary if cm else ""),
                voice_ids=(cm.voice_ids if cm else []),
                take_group_id=(cm.take_group_id if cm else None),
                sync_group_id=(cm.sync_group_id if cm else None),
                rgb_mean=list(span_rgb) if span_rgb else None,
            ))
        semantic_groups = group_shots_semantically(scene_meta)

        ordered = [shot_stats[s.key] for s in shots]
        # color_shot_matching.plan.md Phase 1: an all-singleton semantic
        # result means the signals were absent/unhelpful for THIS document
        # (the common real-data case: speaker_person/on_camera/label/summary
        # all unset) -- degrade to the RGB fallback instead of silently
        # letting Match do nothing (SS2 "primary cause": singleton groups are
        # skipped, and a non-None `groups` bypasses group_neighbors
        # entirely).
        if not _has_real_groups(semantic_groups):
            semantic_groups = None
        groups = semantic_groups if semantic_groups is not None else group_neighbors(ordered)

        # Phase 2a: Balance's robust reference, solved on WORKING-space
        # stats (Balance's math has to live in the space the CDL is
        # applied).
        balance_references: Dict[int, GroupReference] = {}
        display_stats = [o.stats or {} for o in ordered]
        ws_stats = [_ws_stats(o.stats) for o in ordered]
        for gi, idxs in enumerate(groups):
            b_ref = compute_group_reference([ws_stats[i] for i in idxs])
            if b_ref is not None:
                balance_references[gi] = b_ref

        balance_deltas: Dict[str, Grade] = solve_balance(
            ws_stats, groups, balance_references, [s.key for s in shots],
        )

        # Phase 2c: Match solves against the AS-BALANCED image, not the
        # raw span stats -- same "corrected source" discipline
        # resolver.py's _corrected_source_stats already uses for the
        # Look layer. Composing two independently-solved "toward the
        # same reference" deltas on the RAW stats does not converge, it
        # overshoots (see _corrected_display_stats's docstring for the
        # verified numbers) -- this is what makes Balance and Match
        # actually reinforce instead of fight.
        balanced_display_stats = [
            display_stats[i] if o.key not in balance_deltas
            else _corrected_display_stats(ws_stats[i], balance_deltas[o.key])
            for i, o in enumerate(ordered)
        ]
        match_references: Dict[int, GroupReference] = {}
        for gi, idxs in enumerate(groups):
            m_ref = compute_group_reference([balanced_display_stats[i] for i in idxs])
            if m_ref is not None:
                match_references[gi] = m_ref
        balanced_shots = [
            ShotStats(key=o.key, file_id=o.file_id, stats=balanced_display_stats[i])
            for i, o in enumerate(ordered)
        ]
        match_deltas: Dict[str, Grade] = solve_sequence_match(
            balanced_shots, groups=groups, working_space=WORKING_SPACE_V1,
            references=match_references,
        )

        # Phase 2: ONE leveling pass (exposure + tonal placement) --
        # working-space-projected scalars, same projection Step 1.5's
        # _corrected_source_stats already uses.
        leveling_deltas: Dict[str, Grade] = {}
        from app.services.l3.grade.cdl import compose as _compose

        group_idx_by_shot = [
            next((gi for gi, idxs in enumerate(groups) if i in idxs), None)
            for i in range(len(shots))
        ]

        level_inputs: List[ShotLevelInput] = []
        for i, s in enumerate(shots):
            stats = shot_stats[s.key].stats or {}
            subj = stats.get("subject_luma")
            subject_luma = _to_working_scalar(subj, None) if subj is not None else None

            # Phase 4b: for a shot that's a member of a real (2+) Balance
            # /Match group, level its CURRENT value from the AS-CORRECTED
            # (post Balance+Match) stats, toward an EXPLICIT target (the
            # SAME group reference Balance/Match already converged on)
            # instead of the local smooth-target average -- otherwise
            # Leveling's own window average re-diverges what they just
            # aligned. Ungrouped shots keep the original local-smoothing
            # behavior exactly (target_* stays None).
            group_idx = group_idx_by_shot[i]
            ref = balance_references.get(group_idx) if group_idx is not None else None
            if ref is not None and ws_stats:
                bm_grade = _compose(balance_deltas.get(s.key, Grade()), match_deltas.get(s.key, Grade()), 1.0)
                mid_gray = _apply_working_scalar(ws_stats[i].get("mid_gray"), bm_grade, 1)
                if mid_gray is None:
                    mid_gray = 0.5   # same uncorrected fallback constant as the no-group path
                black = _apply_working_scalar(ws_stats[i]["black_point"], bm_grade, 1)
                white = _apply_working_scalar(ws_stats[i]["white_point"], bm_grade, 1)
                level_inputs.append(ShotLevelInput(
                    key=s.key, mid_gray=mid_gray, black_point=black, white_point=white,
                    subject_luma=subject_luma,
                    target_mid_gray=ref.mid_gray, target_black_point=ref.black_point,
                    target_white_point=ref.white_point,
                ))
            else:
                mid_gray = _to_working_scalar(stats.get("mid_gray"), 0.5)
                black = _to_working_scalar(stats.get("black_point"), 0.0)
                white = _to_working_scalar(stats.get("white_point"), 1.0)
                level_inputs.append(ShotLevelInput(
                    key=s.key, mid_gray=mid_gray, black_point=black, white_point=white,
                    subject_luma=subject_luma,
                ))

        # color_subject_exposure.plan.md Phase 2: a grouped shot's usable
        # subject_luma should converge toward the GROUP's median SUBJECT
        # luma (working space), not the group's whole-frame mid_gray
        # reference (`target_mid_gray` above) -- a face's own brightness
        # has no reason to equal the frame average. Computed inline here
        # (not in reference.py) to keep the matching/balance math
        # untouched, per this plan's own non-goals -- Balance/Match never
        # see this value. <2 usable members in a group -> leave
        # target_subject_luma unset (falls back to target_mid_gray).
        from statistics import median

        from app.services.l3.grade.leveling import _usable_subject_luma

        subject_by_group: Dict[int, List[float]] = {}
        for i, gi in enumerate(group_idx_by_shot):
            if gi is None:
                continue
            usable = _usable_subject_luma(level_inputs[i].subject_luma, level_inputs[i].mid_gray)
            if usable is not None:
                subject_by_group.setdefault(gi, []).append(usable)
        for i, gi in enumerate(group_idx_by_shot):
            members = subject_by_group.get(gi) if gi is not None else None
            if members and len(members) >= 2:
                level_inputs[i].target_subject_luma = median(members)

        leveling_deltas = solve_leveling(level_inputs)

        cube_cache: Dict[str, Optional[str]] = {}
        done = 0
        for s in shots:
            stats = shot_stats[s.key].stats
            grade_json = resolve_clip_grade(
                s.item, color_stats=stats, sequence_look=sequence_look,
                balance_delta=balance_deltas.get(s.key),
                match_delta=match_deltas.get(s.key), leveling_delta=leveling_deltas.get(s.key),
                tone_contrast=0.0,
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
