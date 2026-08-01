"""
vcut_pass2_rich.plan.md -- Pass 2 (background): answer each cut's Pass-1-
selected question_ids (questions.py) richly, riding the SAME shared Gemini
CachedContent Pass 1 created (all sampled non-speech frames -- see
orchestrate._create_vcut_cache), so the model sees the CUT's full temporal
context rather than one hero thumbnail. Writes onto cut_records.
scene_specifics via the EXISTING ingest_store.update_cut_scene_specifics.
Runs AFTER cuts are shown (vcut_enrich, orchestrate.py) -- never changes cut
boundaries, never blocks ingest, fail-open per call (mirrors l3_scene_
enrich's own contract: a failure here never affects whether the run's cuts
are usable).

Falls back to re-extracting one hero frame per cut (the ORIGINAL
implementation's own path, kept as the degraded fallback, not the primary)
when there is no cache handle at all, or the cached call itself fails (e.g.
the cache expired between Pass 1 and this later background task) -- the
fallback still answers the cut's own SELECTED question_ids, just with less
context. Cache teardown (delete_pass2_cache) always runs in a finally,
best-effort.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.services.llm.base import Block, image_block, text_block
from app.services.vcut import questions
from app.services.vcut.pass1 import NEUTRAL_SYSTEM

logger = logging.getLogger(__name__)

_CACHED_PREAMBLE = (
    "Using the cached frames for each listed file, answer the requested fields "
    "for each cut_id below -- focus on the frames that fall within that cut's "
    "time window (its hero frame is the anchor). Only answer the fields listed "
    "for that cut_id; leave every other field at its default (empty string, or "
    "null for count)."
)

_HERO_PREAMBLE = (
    "For each cut_id below you are shown one representative frame (labeled "
    "with its cut_id). Answer the requested fields for each cut_id from its "
    "own frame. Only answer the fields listed for that cut_id; leave every "
    "other field at its default (empty string, or null for count)."
)


class _AnswerOut(BaseModel):
    cut_id: str
    subject: str = ""
    action: str = ""
    moment_type: str = ""
    setting: str = ""
    on_screen_text: str = ""
    notable_object: str = ""
    count: Optional[int] = None
    motion_quality: str = ""
    continuity_cue: str = ""


class Pass2Schema(BaseModel):
    answers: List[_AnswerOut] = []


def _pg_conn():
    from app.services import db
    return db.connection_dict_row()


def _video_cuts_for_run(ingest_run_id: str) -> List[dict]:
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select cr.id::text as id, cr.file_id::text as file_id,
                   cr.src_in_ms as src_in_ms, cr.src_out_ms as src_out_ms,
                   cr.hero_ts_ms as hero_ts_ms, cr.summary as summary,
                   coalesce(f.r2_proxy_key, f.r2_key) as proxy_key
              from cut_records cr
              join files f on f.id = cr.file_id
             where cr.ingest_run_id = %s and cr.kind = 'video'
            """,
            (ingest_run_id,),
        ).fetchall()
    return [dict(r) for r in rows if r["proxy_key"]]


def _question_ids_by_file(loose_plan_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """Phase 2 interim (vcut_moment_energy.plan.md section 9): Pass 1 now
    plants moment FLAGS only -- it no longer selects per-clip question_ids
    (the new FilePlan shape has no such field at all). Until Pass 2 is
    reworked to derive questions from each cut's own summary (Phase 2,
    deferred), every file gets the same small default set -- "scene_
    specifics can temporarily fall back to summary-only" per the plan's own
    interim instruction, and this guards the read rather than crashing on
    the new plan shape."""
    return {file_id: list(questions.DEFAULT_QUESTION_IDS) for file_id in (loose_plan_dict or {}).keys()}


def _cuts_with_meta(cuts: List[dict], question_ids_by_file: Dict[str, List[str]]) -> List[dict]:
    out = []
    for c in cuts:
        qids = question_ids_by_file.get(c["file_id"]) or list(questions.DEFAULT_QUESTION_IDS)
        out.append({**c, "question_ids": qids})
    return out


def _cuts_listing(cuts_with_meta: List[dict]) -> str:
    lines = []
    for c in cuts_with_meta:
        lines.append(
            f"- cut_id={c['id']} file={c['file_id']} window=[{c['src_in_ms']},{c['src_out_ms']}]ms "
            f"hero={c['hero_ts_ms']}ms meaning={c['summary']!r} fields=[{', '.join(c['question_ids'])}]"
        )
    return "\n".join(lines)


def _task_text(cuts_with_meta: List[dict], cached: bool) -> str:
    preamble = _CACHED_PREAMBLE if cached else _HERO_PREAMBLE
    return f"{preamble}\n\n{questions.bank_prompt_lines()}\n\nCuts:\n{_cuts_listing(cuts_with_meta)}"


def _hero_frames(cuts: List[dict]) -> Dict[str, str]:
    """{cut_id: base64 jpeg} -- one still per cut at its own hero_ts_ms, the
    fallback frame source when no (working) cache is available. Reuses
    app.services.l3.frames (a generic ffmpeg/R2 primitive, sanctioned for
    reuse by seam_cut_pipeline.plan.md section 6, not L3 business logic)."""
    from app.services.l3.frames import extract_for_planned_frames
    from app.services.l3.image_plan import PlannedFrame
    from app.services.vcut.params import SAMPLE_WIDTH_PX

    proxy_key_by_file = {c["file_id"]: c["proxy_key"] for c in cuts}
    planned = [
        PlannedFrame(file_id=c["file_id"], ts_ms=c["hero_ts_ms"], reason="vcut_enrich",
                    ref=f"vcut_enrich[{c['id']}]")
        for c in cuts
    ]
    by_file_ts = extract_for_planned_frames(planned, proxy_key_by_file, width=SAMPLE_WIDTH_PX)
    return {
        c["id"]: by_file_ts[(c["file_id"], c["hero_ts_ms"])]
        for c in cuts if (c["file_id"], c["hero_ts_ms"]) in by_file_ts
    }


def _fallback_blocks(cuts_with_meta: List[dict], hero_by_cut: Dict[str, str]) -> List[Block]:
    blocks: List[Block] = []
    for c in cuts_with_meta:
        b64 = hero_by_cut.get(c["id"])
        if not b64:
            continue
        blocks.append(text_block(f"cut_id={c['id']} frame:"))
        blocks.append(image_block(b64))
    return blocks


def scene_specifics_from_answers(
    parsed: Pass2Schema, known: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Pure (no I/O): {cut_id: {selected_field: value}} -- only cut_ids
    present in ``known`` (this run's own cuts) survive; only that cut's OWN
    selected question_ids are copied out of its answer, even though the
    answer schema itself always carries every bank field (the closed-bank/
    "selected fields only" contract, section 6.4)."""
    out: Dict[str, Dict[str, Any]] = {}
    for ans in parsed.answers:
        qids = known.get(ans.cut_id)
        if qids is None:
            continue
        out[ans.cut_id] = {qid: getattr(ans, qid) for qid in qids if hasattr(ans, qid)}
    return out


def _check_nonempty(parsed: Pass2Schema) -> Optional[str]:
    if not parsed.answers:
        return "answers must not be empty -- answer for every cut_id shown."
    return None


def _call_cached(cuts_with_meta: List[dict], cache_handle: Dict[str, Any]):
    from app.services.llm.ingest_gemini import complete_gemini

    task_text = _task_text(cuts_with_meta, cached=True)
    return complete_gemini(
        NEUTRAL_SYSTEM, [text_block(task_text)], Pass2Schema,
        model=cache_handle["model"], max_tokens=12000, thinking="low",
        extra_check=_check_nonempty, cached_content=cache_handle["name"],
    )


def _call_fallback(cuts_with_meta: List[dict], model: str):
    from app.services.llm.ingest_gemini import complete_gemini

    hero_by_cut = _hero_frames(cuts_with_meta)
    blocks = _fallback_blocks(cuts_with_meta, hero_by_cut)
    if not blocks:
        return None
    task_text = _task_text(cuts_with_meta, cached=False)
    return complete_gemini(
        task_text, blocks, Pass2Schema,
        model=model, max_tokens=12000, thinking="low",
        extra_check=_check_nonempty,
    )


def run_enrich(project_id: str, ingest_run_id: str) -> None:
    """THIS FUNCTION SPENDS REAL MONEY (one Gemini call), same disclosure as
    pass1.run_pass1. Called by orchestrate.vcut_enrich; never raises upward
    to that caller in practice since it's already wrapped there, but stays
    defensive internally too (an empty cut list or frame set is a silent
    no-op, not an error)."""
    from app.config import get_settings
    from app.services.l3 import ingest_store as l3store
    from app.services.llm.ingest_gemini import delete_pass2_cache
    from app.services.vcut import store as vstore

    cuts = _video_cuts_for_run(ingest_run_id)
    if not cuts:
        return

    _seam_cache, loose_plan_dict = vstore.load_seam_and_plan(ingest_run_id)
    question_ids_by_file = _question_ids_by_file(loose_plan_dict)
    cuts_with_meta = _cuts_with_meta(cuts, question_ids_by_file)

    cache_handle = vstore.load_vcut_cache(ingest_run_id)
    completion = None
    try:
        if cache_handle:
            try:
                completion = _call_cached(cuts_with_meta, cache_handle)
            except Exception:
                logger.exception(
                    "vcut pass2: cached call failed for run %s (cache likely expired) -- "
                    "falling back to hero frames", ingest_run_id)
        if completion is None:
            settings = get_settings()
            completion = _call_fallback(cuts_with_meta, settings.vcut_pass2_model)
    finally:
        if cache_handle:
            delete_pass2_cache(cache_handle.get("name"))

    if completion is None:
        logger.warning("vcut pass2: no frames available for run %s -- nothing to enrich", ingest_run_id)
        return

    l3store.accumulate_pass2_usage(ingest_run_id, completion.usage)
    parsed = Pass2Schema.model_validate(completion.data)

    known = {c["id"]: c["question_ids"] for c in cuts_with_meta}
    specifics_by_cut = scene_specifics_from_answers(parsed, known)
    n_written = 0
    for cut_id, fields in specifics_by_cut.items():
        try:
            l3store.update_cut_scene_specifics(cut_id, fields)
            n_written += 1
        except Exception:
            logger.exception("vcut pass2: failed to persist scene_specifics for cut %s", cut_id)
    logger.info("vcut pass2: project %s run %s wrote %d/%d cut(s) (cache=%s)",
               project_id, ingest_run_id, n_written, len(cuts), bool(cache_handle))
