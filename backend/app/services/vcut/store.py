"""
seam_cut_pipeline.plan.md section 8 -- turn ResolvedCuts into CutRecords and
write them through the EXISTING ingest_store persistence layer (create_
ingest_run / delete_cut_records_for_run / insert_cut_records / set_status),
plus the two vcut-only JSON artifacts (seam_cache/loose_plan, migration
052_vcut_artifacts.sql) that let the energy endpoint re-derive without any
I/O. Nothing from app.services.l3 is imported here except post.CutRecord/
PaceEnvelope and ingest_store itself (principle 7's isolation constraint) --
even the energy-grade bucketing below is vcut's own small copy rather than a
reuse of post.py's private ``_energy_grade``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.post import CutRecord, PaceEnvelope
from app.services.vcut.resolve import ResolvedCut

# A vcut cut never speed-ramps (pace/reframe is out of scope, plan section
# 0) -- min/natural/max all pinned to the cut's own length. 5 == the old
# pipeline's PACE_LEVEL_TARGETS length (post_params.py); duplicated as a
# bare literal rather than imported, per this module's isolation contract.
_PACE_LEVELS = [1.0] * 5

_ENERGY_GRADE_BANDS = (("calm", 0.33), ("medium", 0.66))  # else "high"

# Columns copied byte-for-byte from a prior run's speech cut_records (every
# CutRecord column except id/ingest_run_id/created_at/updated_at/hero_key/
# scene_specifics/transition_in/transition_out/junk_confidence -- those stay
# NULL/default on the copy, matching a fresh insert_cut_records() row).
_SPEECH_COPY_COLUMNS = (
    "file_id", "src_in_ms", "src_out_ms", "kind", "word_span", "atom_ids",
    "label", "summary", "on_camera", "take_group_id", "take_role", "junk", "junk_reason",
    "framing", "look", "caption_zones", "pace", "hero_ts_ms", "channel", "continuity",
    "speech_quality", "total_quality", "characteristics", "camera", "sync_group_id",
    "screen_text", "salience", "landmarks", "voice_ids", "speaker_person", "visible_persons",
    "audio_file_id", "audio_offset_ms", "audio_align_confidence",
)


def _pg_conn():
    from app.services import db
    return db.connection()


def _energy_grade(mean_action_energy: float) -> str:
    for grade, upper in _ENERGY_GRADE_BANDS:
        if mean_action_energy < upper:
            return grade
    return "high"


def _mean_in_range(a_ms: float, b_ms: float, hop_ms: int, track: List[float]) -> float:
    if not track or hop_ms <= 0:
        return 0.0
    lo_i = max(0, int(a_ms // hop_ms))
    hi_i = min(len(track) - 1, int(b_ms // hop_ms))
    if lo_i > hi_i:
        return 0.0
    vals = track[lo_i:hi_i + 1]
    return sum(vals) / len(vals) if vals else 0.0


def _short_label(meaning: str, max_words: int = 6) -> str:
    words = (meaning or "").split()
    label = " ".join(words[:max_words])
    return label or "clip"


def _pace_for(duration_ms: int) -> PaceEnvelope:
    natural = max(1, duration_ms)
    return PaceEnvelope(min_ms=natural, natural_ms=natural, max_ms=natural,
                        levels=list(_PACE_LEVELS), energy_grade="calm", natural_sound=True)


def build_cut_records(resolved: List[ResolvedCut], seam: Dict[str, dict]) -> List[CutRecord]:
    """One minimal, non-speech CutRecord per ResolvedCut (vcut_moment_
    energy.plan.md section 7.1): kind="video", channel="shown".
    ``total_quality`` is the cut's own mean S(t) (clamped) -- by this
    pipeline's own definition of quality, not borrowed from the old
    pipeline's judgment model. scene_specifics is left at its column
    default; Pass 2 (pass2.py/vcut_enrich) fills it in later, same contract
    as the old pipeline's l3_scene_enrich."""
    records: List[CutRecord] = []
    for cut in resolved:
        file_seam = seam.get(cut.file_id) or {}
        hop_ms = int(file_seam.get("hop_ms") or 0)
        S = file_seam.get("S") or []
        action_energy = file_seam.get("action_energy") or []
        mean_s = _mean_in_range(cut.in_ms, cut.out_ms, hop_ms, S)
        mean_ae = _mean_in_range(cut.in_ms, cut.out_ms, hop_ms, action_energy)
        total_quality = max(0.0, min(1.0, mean_s))
        pace = _pace_for(cut.out_ms - cut.in_ms)
        pace.energy_grade = _energy_grade(mean_ae)

        records.append(CutRecord(
            file_id=cut.file_id, src_in_ms=cut.in_ms, src_out_ms=cut.out_ms,
            kind="video", word_span=None, atom_ids=None,
            label=_short_label(cut.summary), summary=cut.summary,
            on_camera=None, junk=False, junk_reason="",
            framing={}, look={}, caption_zones=[],
            hero_ts_ms=cut.peak_ms, pace=pace, take_group_id=None, take_role=None,
            channel="shown", speech_quality=None, total_quality=total_quality,
        ))
    return records


def delete_video_cuts_for_run(ingest_run_id: str) -> None:
    """Scoped to kind='video' only -- unlike ingest_store.
    delete_cut_records_for_run (unconditional per run), this leaves the
    run's kind='speech' rows (written by run_speech_channel in this same
    run) alone. Used both by the initial build and by the energy re-derive
    endpoint, which must never touch speech cuts."""
    with _pg_conn() as conn:
        conn.execute(
            "delete from cut_records where ingest_run_id = %s and kind = 'video'",
            (ingest_run_id,),
        )


def persist_seam_and_plan(ingest_run_id: str, seam_cache: Dict[str, dict], loose_plan_dict: Dict[str, Any]) -> None:
    """The two artifacts (section 4) the energy endpoint needs to re-derive
    with zero I/O: {file_id: {hop_ms, S, action_energy, frame_diff}} and the
    moment-flags plan (vcut_moment_energy.plan.md; resolve.MomentPlan.
    to_dict())."""
    with _pg_conn() as conn:
        conn.execute(
            "update ingest_runs set seam_cache = %s, loose_plan = %s where id = %s",
            (json.dumps(seam_cache), json.dumps(loose_plan_dict), ingest_run_id),
        )


def load_seam_and_plan(ingest_run_id: str) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select seam_cache, loose_plan from ingest_runs where id = %s", (ingest_run_id,)
        ).fetchone()
    if not row:
        return {}, {}
    seam_cache = row[0] if isinstance(row[0], dict) else (json.loads(row[0]) if row[0] else {})
    loose_plan_dict = row[1] if isinstance(row[1], dict) else (json.loads(row[1]) if row[1] else {})
    return seam_cache or {}, loose_plan_dict or {}


def persist_speech_channel_status(ingest_run_id: str, status: Dict[str, Any]) -> None:
    """vcut_moment_energy.plan.md section 8: {source: "pipeline"|
    "copy_prior", error: str|None} -- makes a silently-degrading speech
    pipeline observable (GET /cuts, logs) instead of looking identical to a
    working one."""
    with _pg_conn() as conn:
        conn.execute(
            "update ingest_runs set speech_channel_status = %s where id = %s",
            (json.dumps(status), ingest_run_id),
        )


def persist_vcut_cache(ingest_run_id: str, handle: Optional[Dict[str, Any]]) -> None:
    """The shared Gemini CachedContent handle (vcut_pass2_rich.plan.md
    section 3.3): {name, model, created_at, ttl_s}, or None when creation
    failed/degraded -- both passes then fall back to uncached behavior."""
    with _pg_conn() as conn:
        conn.execute(
            "update ingest_runs set vcut_cache = %s where id = %s",
            (json.dumps(handle) if handle else None, ingest_run_id),
        )


def load_vcut_cache(ingest_run_id: str) -> Optional[Dict[str, Any]]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select vcut_cache from ingest_runs where id = %s", (ingest_run_id,)
        ).fetchone()
    if not row or not row[0]:
        return None
    return row[0] if isinstance(row[0], dict) else json.loads(row[0])


def _latest_other_run_id(project_id: str, exclude_run_id: str) -> Optional[str]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select id::text from ingest_runs where project_id = %s and id != %s "
            "order by created_at desc limit 1",
            (project_id, exclude_run_id),
        ).fetchone()
    return row[0] if row else None


def copy_prior_speech_cuts(project_id: str, new_ingest_run_id: str) -> int:
    """Principle 1/section 2: speech is a separate channel, reused verbatim
    -- vcut never generates its own speech cuts (no L3 pass1/lattice call;
    principle 7's isolation constraint rules that out). Copies the most
    recent OTHER ingest_run's kind='speech' cut_records as-is (new id, new
    ingest_run_id, every other column byte-identical) into this run. A
    project with no prior ingest at all simply contributes zero speech cuts
    -- graceful degradation, not an error."""
    prior_run_id = _latest_other_run_id(project_id, new_ingest_run_id)
    if not prior_run_id:
        return 0
    cols = ", ".join(_SPEECH_COPY_COLUMNS)
    with _pg_conn() as conn:
        result = conn.execute(
            f"insert into cut_records (ingest_run_id, {cols}) "
            f"select %s, {cols} from cut_records where ingest_run_id = %s and kind = 'speech'",
            (new_ingest_run_id, prior_run_id),
        )
        return result.rowcount or 0
