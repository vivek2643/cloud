"""
speech_cuts_pipeline.plan.md section 13 -- orchestration: run_speech_channel
(project_id, ingest_run_id, seam_cache), called from app.services.vcut.
orchestrate.run_vcut_ingest (after the video cuts are already inserted,
same run). Raises on any failure, matching l3.ingest's own "no silent
partial success" contract. NO FALLBACK: the caller no longer copies a
prior run's speech cuts on failure -- a speech-channel bug propagates and
fails the whole run rather than being masked by degraded/old cuts.

THIS FUNCTION SPENDS REAL MONEY: one Gemini-pro text call (segment_llm) +
at most one Gemini flash-lite vision call (frames), the latter skipped
entirely when there are zero on-camera winning cuts.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from app.services.vcut.speech import boundaries, delivery, frames, inputs, outlooks, segment_llm, select, store

logger = logging.getLogger(__name__)


def _project_file_rows(project_id: str) -> List[Tuple[str, str, int]]:
    """(file_id, filename, duration_ms) for every video file in the
    project. Mirrors vcut.orchestrate._project_file_rows' own query shape
    (not imported -- that one also carries proxy_key, which this module
    doesn't need; inputs.py/frames.py read their own R2/DB data directly)."""
    from app.services import db
    with db.connection_dict_row() as conn:
        rows = conn.execute(
            """
            select f.id::text as id, f.filename as filename,
                   coalesce(f.duration_seconds, 0) as duration_seconds
              from files f
             where f.id = any(select unnest(source_file_ids) from projects where id = %s)
               and f.file_type = 'video'
            """,
            (project_id,),
        ).fetchall()
    return [(r["id"], r["filename"], int(round((r["duration_seconds"] or 0) * 1000))) for r in rows]


def _resolve_take_group(
    group_beats: List["segment_llm._BeatOut"], collapsed_inputs: Dict[str, "inputs.FileSpeechInputs"],
) -> List[store.ResolvedBeat]:
    """boundaries.py -> delivery.py -> select.py for ONE take group (section
    9/10). A beat whose file has no loaded inputs (shouldn't happen -- every
    beat came FROM a collapsed take instance's own transcript -- but guarded
    defensively) is dropped from the group rather than crashing the run.

    speech_noise_gate.plan.md section 3: a beat the noise gate catches (the
    segment LLM's own is_speech=false judgment, OR the self-calibrating
    energy gate) is no longer silently dropped -- it's routed to its own
    junk=True ResolvedBeat (auditable/recoverable) instead of a real
    candidate. GUARDRAIL: junk beats are built separately from, and never
    added to, the metrics/candidates fed to group_delivery_scores/
    select_winner -- a phantom can never dilute group normalization, win a
    take group, or suppress a real take. A group whose beats are ALL junk
    yields only junk records (no winner) -- acceptable; they're hidden by
    the default junk filter same as video junk."""
    take_group_key = f"speech[{sorted(b.id for b in group_beats)[0]}]"
    by_beat_id = {b.id: b for b in group_beats}
    metrics_by_beat: Dict[str, Tuple[int, int, delivery.TakeMetrics]] = {}
    junk_beats: List[Tuple["segment_llm._BeatOut", int, int, str]] = []

    for beat in group_beats:
        fi = collapsed_inputs.get(beat.file_id)
        if fi is None:
            continue
        in_ms, out_ms = boundaries.compute_boundaries_ms(beat.word_span, fi.words, fi.silences, fi.duration_ms)
        metrics = delivery.compute_take_metrics(fi.words, in_ms, out_ms, fi.rms_db, fi.rms_hop_ms)

        if not beat.is_speech:
            junk_reason = "non_speech_llm"
        elif not delivery.is_voiced_beat(metrics.energy, fi.rms_voiced_db, fi.rms_floor_db):
            junk_reason = "non_speech_energy"
        else:
            junk_reason = ""

        if junk_reason:
            logger.info("run_speech_channel: gated beat %s (file %s, %d-%dms) as junk (%s) -- "
                        "energy=%.1fdB floor=%.1f voiced=%.1f", beat.id, beat.file_id,
                        in_ms, out_ms, junk_reason, metrics.energy, fi.rms_floor_db, fi.rms_voiced_db)
            junk_beats.append((beat, in_ms, out_ms, junk_reason))
            continue
        metrics_by_beat[beat.id] = (in_ms, out_ms, metrics)

    resolved: List[store.ResolvedBeat] = [
        store.ResolvedBeat(
            beat=beat, in_ms=in_ms, out_ms=out_ms, take_group_key=take_group_key,
            take_role="take", speech_quality=0.0, junk=True, junk_reason=junk_reason,
        )
        for beat, in_ms, out_ms, junk_reason in junk_beats
    ]

    if metrics_by_beat:
        beat_ids = list(metrics_by_beat.keys())
        scores = delivery.group_delivery_scores([metrics_by_beat[bid][2] for bid in beat_ids])
        candidates = [
            select.TakeCandidate(beat_id=bid, fluency_llm=by_beat_id[bid].fluency,
                                 flags=list(by_beat_id[bid].flags), delivery=score)
            for bid, score in zip(beat_ids, scores)
        ]
        result = select.select_winner(candidates)
        resolved.extend(
            store.ResolvedBeat(
                beat=by_beat_id[bid], in_ms=metrics_by_beat[bid][0], out_ms=metrics_by_beat[bid][1],
                take_group_key=take_group_key, take_role=result.roles[bid], speech_quality=result.finals[bid],
            )
            for bid in beat_ids
        )

    return resolved


def run_speech_channel(project_id: str, ingest_run_id: str, seam_cache: Dict[str, dict]) -> int:
    """Returns the number of kind='speech' cut_records written (0 if the
    project has no transcribed speech at all -- not an error)."""
    from app.services.l3 import ingest_store as l3store

    file_rows = _project_file_rows(project_id)
    file_ids = [r[0] for r in file_rows]
    inputs_by_file = inputs.load_project_inputs(file_rows)
    if not inputs_by_file:
        logger.info("run_speech_channel: no transcripts for project %s -- nothing to do", project_id)
        return 0

    sync_groups = outlooks.load_sync_groups(file_ids)
    take_instance_ids = outlooks.collapse_to_take_instances(file_ids, sync_groups)
    collapsed_inputs = {fid: inputs_by_file[fid] for fid in take_instance_ids if fid in inputs_by_file}
    if not collapsed_inputs:
        logger.info("run_speech_channel: project %s's take instances have no transcript -- nothing to do",
                   project_id)
        return 0

    take_groups, seg_usage = segment_llm.run_segment_llm(collapsed_inputs)
    l3store.accumulate_pass2_usage(ingest_run_id, seg_usage)

    resolved_beats: List[store.ResolvedBeat] = []
    for group_beats in take_groups:
        resolved_beats.extend(_resolve_take_group(group_beats, collapsed_inputs))
    if not resolved_beats:
        return 0

    expanded = store.expand_outlook_cuts(resolved_beats, sync_groups)
    angle_file_ids = sorted({ec.file_id for ec in expanded})
    face_tracks_by_file = inputs.load_face_tracks_for_files(angle_file_ids)
    action_energy_by_file = {
        fid: (seam_cache[fid]["action_energy"], seam_cache[fid]["hop_ms"])
        for fid in angle_file_ids if fid in seam_cache
    }
    visual_by_key, frame_usage = frames.run_frame_analysis(expanded, face_tracks_by_file, action_energy_by_file)
    if frame_usage:
        l3store.accumulate_pass2_usage(ingest_run_id, frame_usage)

    records, specifics = store.build_speech_cut_records(resolved_beats, sync_groups, face_tracks_by_file,
                                                          visual_by_key)
    record_ids = l3store.insert_cut_records(ingest_run_id, records)
    for cut_id, spec in zip(record_ids, specifics):
        if spec:
            try:
                l3store.update_cut_scene_specifics(cut_id, spec)
            except Exception:
                logger.exception("run_speech_channel: failed to persist scene_specifics for cut %s", cut_id)

    logger.info("run_speech_channel: project %s run %s wrote %d speech cut(s) from %d take group(s)",
               project_id, ingest_run_id, len(records), len(take_groups))
    return len(records)
