"""
vcut_pass2_video_specifics.plan.md -- Pass 2 ("specifics"): answer each
moment FLAG's planner-selected question_ids (+ custom probes, qplan.py)
against real video, one call PER FILE. Writes answers onto the matching
MomentFlag.specifics (never onto a cut id -- resolve.py composes flag
specifics onto whatever cut ends up containing them, section 7), so an
energy re-resolve can never wipe them again.

Two entry points, per section 6.1's "run Pass 2 inline while the cache is
warm" decision:

  run_enrich_inline(plan, video_by_file) -- VIDEO mode. Called directly
    from orchestrate.run_vcut_ingest, in the SAME run, while the per-file
    video CachedContents (+ uploads) built for Pass 1 are still open --
    riding each file's cache (or sending the video inline when a file has
    no cache: creation failed / below the min-cache-token floor).

  run_enrich(project_id, ingest_run_id) -- FRAMES mode (today's default,
    and video mode's own history before this plan). Still the deferred
    background task (orchestrate.vcut_enrich); loads the plan (already
    carrying question_ids from run_vcut_ingest's own inline qplan step),
    answers each file's flagged moments from a fresh hero-still frame per
    moment (the ORIGINAL fallback path, now per-moment not per-cut),
    persists the plan, then re-resolves + rewrites cut_records so they
    carry the composed specifics (section 5.4/7.3 -- no more per-cut
    update_cut_scene_specifics call for video cuts at all).

Both are per-file fan-out (mirrors pass1.run_pass1's own pool) and FAIL-
OPEN per file -- unlike Pass 1's NO-FALLBACK contract, a file's enrich
failure here just logs and leaves that file's moments without specifics
(summary-only); it never aborts the run (section 1: "no hard failures for
cost optimizations").
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from app.services.llm.base import Block, image_block, text_block, video_file_block
from app.services.vcut.pass1 import NEUTRAL_SYSTEM, VideoHandle
from app.services.vcut.resolve import FilePlan, MomentFlag, MomentPlan

logger = logging.getLogger(__name__)

_ENRICH_MAX_WORKERS = 6
# Per-FILE budget (not a project's) -- a file's flagged moments are a small,
# bounded amount of JSON even for a busy clip.
_MAX_TOKENS_PER_FILE = 12000


class _CustomAnswer(BaseModel):
    key: str = ""
    value: str = ""


class _MomentAnswerOut(BaseModel):
    """One flagged moment's answer, keyed by its ``moment_index`` (the
    flag's own position in FilePlan.flags -- flags have no separate stable
    id; section 5.2's own "per-file moment index" choice). Carries every
    bank field any moment might have selected; the caller trims to that
    moment's OWN selected ids (closed-bank/"selected fields only" contract,
    unchanged from the original Pass 2)."""
    moment_index: int
    subject: str = ""
    action: str = ""
    moment_type: str = ""
    setting: str = ""
    on_screen_text: str = ""
    notable_object: str = ""
    count: Optional[int] = None
    motion_quality: str = ""
    continuity_cue: str = ""
    shot_size: str = ""
    camera_move: str = ""
    motion_direction: str = ""
    subject_entry_exit: str = ""
    headroom_lookroom: str = ""
    usable: str = ""
    energy_emotion: str = ""
    hook_potential: str = ""
    tags: List[str] = []
    # Answers to that moment's own custom_questions probes, matched by key
    # (a flat list, not Dict[str,str] -- a dict-shaped field doesn't survive
    # gemini_schema()'s additionalProperties-stripping sanitizer, the same
    # "invented shorthand ids" trap already hit and fixed in pass1.py/
    # segment_llm.py this engagement).
    custom: List[_CustomAnswer] = []
    # reframe_vcut_geometry.plan.md section 1: the subject anchor, normalized
    # [x, y, w, h] in the shown frame -- NOT a bank question (see _task_text
    # below), requested for EVERY moment regardless of question_ids, since
    # reframe applies to every video cut. Tuple, not a bank field, matches
    # the proven l3/pass2.Framing.subject_box shape.
    subject_box: Optional[Tuple[float, float, float, float]] = None


class Pass2Schema(BaseModel):
    answers: List[_MomentAnswerOut] = []


def _check_nonempty(parsed: Pass2Schema) -> Optional[str]:
    if not parsed.answers:
        return "answers must not be empty -- answer for every flagged moment shown."
    return None


def _flagged(flags: List[MomentFlag]) -> List[Tuple[int, MomentFlag]]:
    return [(i, f) for i, f in enumerate(flags) if f.question_ids]


def _moments_listing(indexed_flags: List[Tuple[int, MomentFlag]]) -> str:
    lines = []
    for i, flag in indexed_flags:
        custom_part = ""
        if flag.custom_questions:
            probes = "; ".join(f"{c.get('key', '')}: {c.get('prompt', '')}" for c in flag.custom_questions)
            custom_part = f" custom_probes=[{probes}]"
        # subject_box is appended to the DISPLAYED fields list on every line
        # (not to flag.question_ids itself -- _specifics_from_answer still
        # only extracts the real bank ids) so the model gives it the same
        # weight as this moment's own selected fields, per moment, instead
        # of a single easy-to-miss instruction in the preamble.
        fields = list(flag.question_ids) + ["subject_box"]
        lines.append(
            f"- moment={i} t={flag.t_ms}ms summary={flag.summary!r} "
            f"fields=[{', '.join(fields)}]{custom_part}"
        )
    return "\n".join(lines)


def _task_text(indexed_flags: List[Tuple[int, MomentFlag]], *, video: bool) -> str:
    preamble = (
        "Answer the requested fields for each moment below -- focus on what happens "
        "AROUND its own t_ms (its heart). Only answer the fields listed for that "
        "moment (plus its own custom probes, matched by key); leave every other "
        "field at its default (empty string/array, or null for count). "
        "subject_box is listed for every moment (it's not one of the bank fields "
        "above, but it works the same way) -- the normalized [x, y, w, h] bounding "
        "box of the main subject in frame, or null only if there's genuinely no "
        "single identifiable subject (e.g. a wide establishing shot). Return it for "
        "every moment that has a clear subject -- don't skip it."
    )
    if video:
        preamble = "You are shown this file's video. " + preamble
    return f"{preamble}\n\nMoments:\n{_moments_listing(indexed_flags)}"


def _specifics_from_answer(
    ans: _MomentAnswerOut, question_ids: List[str], custom_questions: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Trim an answer to only ITS OWN requested bank fields, plus its own
    custom probe answers matched by key (an answer key not among this
    moment's own custom_questions is ignored -- the model must not invent
    a probe nobody asked)."""
    out: Dict[str, Any] = {qid: getattr(ans, qid) for qid in question_ids if hasattr(ans, qid)}
    wanted_keys = {c.get("key", "") for c in custom_questions}
    for c in ans.custom:
        if c.key and c.key in wanted_keys and c.value:
            out[c.key] = c.value
    return out


def _valid_normalized_box(
    box: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Defensive guard, added after a live smoke test caught the model
    occasionally returning subject_box in ABSOLUTE PIXEL coordinates (e.g.
    [0.0, 280.0, 1000.0, 720.0]) instead of the requested normalized [0,1]
    box, despite the prompt/schema both saying "normalized" -- reframe.
    solve_crops assumes normalized inputs and has no way to detect the
    mistake itself, so a bad box doesn't error, it silently produces a
    plausible-looking but WRONG crop (clamped toward whichever corner the
    huge coordinate points at). Reject (fall back to None -- solve_crops'
    own documented centered-crop fallback) anything with a non-positive
    width/height or a coordinate meaningfully outside [0, 1]; a small
    tolerance (0.05) absorbs ordinary floating overshoot right at an edge."""
    if box is None:
        return None
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    if any(v < -0.05 or v > 1.05 for v in (x, y, x + w, y + h)):
        return None
    return box


def _write_answers_onto_flags(
    flags: List[MomentFlag], answers_by_index: Dict[int, _MomentAnswerOut],
) -> List[MomentFlag]:
    """A NEW flags list (functional, no mutation -- matches qplan.py's own
    style) with each answered flag's specifics filled in; an unanswered or
    unflagged moment passes through unchanged.

    subject_box (reframe_vcut_geometry.plan.md section 1) is captured onto
    every ANSWERED flag regardless of question_ids -- it's not a bank
    field, so it must not be gated by the same "has question_ids" check
    specifics uses (in practice every flag reaching here already has
    question_ids, qplan.py never leaves one empty -- but this keeps the
    two concerns independent rather than accidentally coupled). Passed
    through _valid_normalized_box, which drops an occasional malformed
    (non-normalized) box rather than trusting it blindly."""
    new_flags = list(flags)
    for i, flag in enumerate(flags):
        ans = answers_by_index.get(i)
        if ans is None:
            continue
        specifics = flag.specifics
        if flag.question_ids:
            specifics = _specifics_from_answer(ans, flag.question_ids, flag.custom_questions)
        new_flags[i] = replace(flag, specifics=specifics, subject_box=_valid_normalized_box(ans.subject_box))
    return new_flags


# --------------------------------------------------------------------------
# VIDEO mode -- section 5.1/5.3: per-file video (cached when available)
# --------------------------------------------------------------------------

def _answer_file_video(
    file_id: str, flags: List[MomentFlag], video_handle: VideoHandle, model: str,
    fps: float, media_resolution: str,
) -> Tuple[List[MomentFlag], Dict[str, int]]:
    """One file's flagged moments, answered against its video -- riding
    video_handle.cache_name when set (section 3.2's shared per-file cache),
    else sending the video inline (section 5.3's no-cache fallback, needs
    no re-upload -- the SAME Files API handle Pass 1 already has). Raises
    on failure; the caller (run_enrich_inline) catches per file."""
    from app.services.llm.ingest_gemini import complete_gemini

    indexed = _flagged(flags)
    if not indexed:
        return flags, {}

    task_text = _task_text(indexed, video=True)
    if video_handle.cache_name:
        system: str = NEUTRAL_SYSTEM
        blocks: List[Block] = [text_block(task_text)]
        cached_content: Optional[str] = video_handle.cache_name
    else:
        system = task_text
        blocks = [video_file_block(video_handle.file_uri, fps=fps)]
        cached_content = None

    completion = complete_gemini(
        system, blocks, Pass2Schema, model=model, max_tokens=_MAX_TOKENS_PER_FILE, thinking="low",
        extra_check=_check_nonempty, media_resolution=media_resolution, cached_content=cached_content,
    )
    parsed = Pass2Schema.model_validate(completion.data)
    answers_by_index = {a.moment_index: a for a in parsed.answers}
    return _write_answers_onto_flags(flags, answers_by_index), completion.usage


def run_enrich_inline(
    plan: MomentPlan, video_by_file: Dict[str, VideoHandle],
) -> Tuple[MomentPlan, Dict[str, int]]:
    """vcut_pass2_video_specifics.plan.md section 6.1: video-mode enrich,
    run INLINE (same run_vcut_ingest call, before _cleanup_video_inputs
    tears the caches/uploads down) -- a deferred call risks the
    CachedContent's TTL expiring between passes. Per-file fan-out on a
    bounded pool, FAIL-OPEN per file: one file's enrich failure logs and
    leaves that file's moments without specifics, never aborts the run.
    Uses settings.vcut_pass1_model for every call, cached or not -- section
    1's "one model everywhere" principle (a CachedContent is bound to its
    creation model, so Pass 2 MUST match Pass 1's model to read it).
    Returns (new plan with specifics written on, usage) -- the CALLER
    persists the plan and re-resolves/rewrites cut_records (section 5.4)."""
    from app.config import get_settings

    settings = get_settings()
    model = settings.vcut_pass1_model

    plans_by_file: Dict[str, List[MomentFlag]] = {fp.file_id: fp.flags for fp in plan.files}
    usages: List[Dict[str, int]] = []
    with ThreadPoolExecutor(max_workers=min(_ENRICH_MAX_WORKERS, max(1, len(plan.files)))) as pool:
        future_to_file = {}
        for fp in plan.files:
            video_handle = video_by_file.get(fp.file_id)
            if video_handle is None or not _flagged(fp.flags):
                continue
            future = pool.submit(
                _answer_file_video, fp.file_id, fp.flags, video_handle, model,
                settings.vcut_video_fps, settings.vcut_video_media_resolution,
            )
            future_to_file[future] = fp.file_id
        for future, file_id in future_to_file.items():
            try:
                new_flags, usage = future.result()
                plans_by_file[file_id] = new_flags
                usages.append(usage)
            except Exception:
                logger.exception("vcut pass2 (inline/video): enrich failed for file %s -- "
                                 "its moments stay summary-only", file_id)

    files = [FilePlan(file_id=fp.file_id, flags=plans_by_file[fp.file_id]) for fp in plan.files]
    total_usage: Dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            total_usage[key] = total_usage.get(key, 0) + value
    return MomentPlan(files=files, genre=plan.genre), total_usage


# --------------------------------------------------------------------------
# FRAMES mode -- section 5.3's other fallback: hero-still frames, unchanged
# in spirit from the original implementation, now per-moment not per-cut.
# --------------------------------------------------------------------------

def _answer_file_frames(
    file_id: str, flags: List[MomentFlag], proxy_key: str, model: str,
) -> Tuple[List[MomentFlag], Dict[str, int]]:
    """One file's flagged moments, answered from a fresh hero-still frame
    PER MOMENT (frames-mode's only path; also video-mode's degraded
    fallback is inline video, not this -- this function is frames-mode
    only). Reuses app.services.l3.frames (a generic ffmpeg/R2 primitive,
    sanctioned for reuse by seam_cut_pipeline.plan.md section 6, not L3
    business logic). Raises on failure; the caller (run_enrich) catches
    per file."""
    from app.services.l3.frames import extract_for_planned_frames
    from app.services.l3.image_plan import PlannedFrame
    from app.services.llm.ingest_gemini import complete_gemini
    from app.services.vcut.params import SAMPLE_WIDTH_PX

    indexed = _flagged(flags)
    if not indexed:
        return flags, {}

    planned = [
        PlannedFrame(file_id=file_id, ts_ms=flag.t_ms, reason="vcut_enrich",
                    ref=f"vcut_enrich[{file_id}][{i}]")
        for i, flag in indexed
    ]
    by_ts = extract_for_planned_frames(planned, {file_id: proxy_key}, width=SAMPLE_WIDTH_PX)

    blocks: List[Block] = []
    have: List[Tuple[int, MomentFlag]] = []
    for i, flag in indexed:
        b64 = by_ts.get((file_id, flag.t_ms))
        if not b64:
            continue
        blocks.append(text_block(f"moment={i} frame:"))
        blocks.append(image_block(b64))
        have.append((i, flag))
    if not blocks:
        return flags, {}

    task_text = _task_text(have, video=False)
    completion = complete_gemini(
        task_text, blocks, Pass2Schema, model=model, max_tokens=_MAX_TOKENS_PER_FILE, thinking="low",
        extra_check=_check_nonempty,
    )
    parsed = Pass2Schema.model_validate(completion.data)
    answers_by_index = {a.moment_index: a for a in parsed.answers}
    return _write_answers_onto_flags(flags, answers_by_index), completion.usage


def _pg_conn():
    from app.services import db
    return db.connection_dict_row()


def _proxy_keys_for_run(ingest_run_id: str) -> Dict[str, str]:
    """{file_id: proxy_key} for this run's kind='video' cut_records --
    reuses the already-inserted cut_records (run_vcut_ingest inserts video
    cuts before this deferred task ever fires) rather than re-deriving the
    project's file set."""
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select distinct cr.file_id::text as file_id, coalesce(f.r2_proxy_key, f.r2_key) as proxy_key
              from cut_records cr
              join files f on f.id = cr.file_id
             where cr.ingest_run_id = %s and cr.kind = 'video'
            """,
            (ingest_run_id,),
        ).fetchall()
    return {r["file_id"]: r["proxy_key"] for r in rows if r["proxy_key"]}


def run_enrich(project_id: str, ingest_run_id: str) -> None:
    """THIS FUNCTION SPENDS REAL MONEY (one Gemini call per file with
    flagged moments), same disclosure as pass1.run_pass1's own docstring.
    Called by orchestrate.vcut_enrich -- FRAMES mode only now (video-mode
    enrich runs inline, run_enrich_inline above, section 6.1). Loads the
    plan (already carrying planner-assigned question_ids from run_vcut_
    ingest's own inline qplan step -- a plan with none planned at all, e.g.
    a not-yet-re-ingested pre-planner run, is a silent no-op), answers each
    file's flagged moments, writes specifics onto the plan, persists it,
    then re-resolves at DEFAULT_ENERGY and rewrites cut_records so they
    carry the composed specifics (section 5.4/7.3 -- no more per-cut
    update_cut_scene_specifics call for video cuts). Never raises upward in
    practice (already wrapped by orchestrate.vcut_enrich), but stays
    defensive internally too: any problem here never affects whether the
    run's cuts are usable, only whether they carry specifics."""
    from app.config import get_settings
    from app.services.l3 import ingest_store as l3store
    from app.services.vcut import store as vstore
    from app.services.vcut.params import DEFAULT_ENERGY
    from app.services.vcut.resolve import resolve_cuts

    seam_cache, plan_dict = vstore.load_seam_and_plan(ingest_run_id)
    if not seam_cache or not plan_dict:
        return
    plan = MomentPlan.from_dict(plan_dict)
    if not any(_flagged(fp.flags) for fp in plan.files):
        logger.info("vcut pass2: run %s has no planned moments -- nothing to enrich", ingest_run_id)
        return

    proxy_key_by_file = _proxy_keys_for_run(ingest_run_id)
    settings = get_settings()

    plans_by_file: Dict[str, List[MomentFlag]] = {fp.file_id: fp.flags for fp in plan.files}
    total_usage: Dict[str, int] = {}
    for fp in plan.files:
        proxy_key = proxy_key_by_file.get(fp.file_id)
        if not proxy_key or not _flagged(fp.flags):
            continue
        try:
            new_flags, usage = _answer_file_frames(fp.file_id, fp.flags, proxy_key, settings.vcut_pass2_model)
            plans_by_file[fp.file_id] = new_flags
            for key, value in usage.items():
                total_usage[key] = total_usage.get(key, 0) + value
        except Exception:
            logger.exception("vcut pass2: enrich failed for file %s -- its moments stay summary-only",
                             fp.file_id)

    new_plan = MomentPlan(
        files=[FilePlan(file_id=fp.file_id, flags=plans_by_file[fp.file_id]) for fp in plan.files],
        genre=plan.genre,
    )
    if total_usage:
        l3store.accumulate_pass2_usage(ingest_run_id, total_usage)
    vstore.persist_seam_and_plan(ingest_run_id, seam_cache, new_plan.to_dict())

    resolved = resolve_cuts(new_plan, seam_cache, energy=DEFAULT_ENERGY)
    n_written = len(vstore.insert_video_cuts(ingest_run_id, resolved, seam_cache))
    logger.info("vcut pass2: project %s run %s enriched, %d video cut(s) rewritten",
               project_id, ingest_run_id, n_written)
