"""
brain_perception_blindness.plan.md B1 -- continuity for vcut cut_records
(video AND speech). `_clip_continuity` (app.services.l3.post) is wired to
the OLD pipeline's atom lattice / Pass2Cut, neither of which exists here, so
this reuses the pipeline-agnostic PIECE of that logic -- app.services.l3.
seam's Seam/classify_seam, the pure weld-vs-hard classifier both pipelines'
continuity should agree on -- rather than the old orchestration wrapper.

Runs as a POST-PROCESSING pass over one ingest run's already-inserted
cut_records (both kinds), not inline in either channel's own build_*_
records: video cuts can be re-inserted mid-run (video-input-mode's inline
enrich re-resolve) and the speech channel writes independently, so any
single channel's build step never has visibility into the OTHER channel's
final rows. Reading the run back once, after both channels have landed, is
simpler than threading cross-channel state through either write path.

Signal mapping from the old pipeline's Seam inputs to what vcut actually
has:
  - has_scene_or_transition: a `scene_cuts.shot_points` timestamp landing in
    the gap -- shot_points is the vcut-available analogue of the old
    pipeline's atom "shot_cut" boundary reason (both ultimately read the
    same scene_cuts table; the old pipeline just pre-bakes it into atom
    boundaries). composition_points are deliberately excluded -- a soft
    framing change, not a break, matching the hard/soft split
    observe._shots_channel already draws between the two.
  - has_flagged_break: always False. vcut's Pass 1 marks every moment
    without filtering (junk_span_suppression.plan.md's diagnosis); there is
    no production-break-suspect list analogous to the old pipeline's
    per-file JunkSuspects to test against. An honest absence, not a
    fabricated one.
  - same_speaker: voice_ids on both sides, same as the old pipeline. Every
    vcut CutRecord (video AND speech) currently carries voice_ids=[] (never
    populated in this pipeline), so this always reads True today -- a
    permissive default, not a bug; other seam signals still decide.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.services.l3.seam import Seam, classify_seam

logger = logging.getLogger(__name__)


def compute_continuity_for_file(
    cuts: List[Dict[str, Any]], shot_points: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """``cuts``: one file's OWN cut_records rows (both kind='video' and
    kind='speech'), already sorted by src_in_ms -- cuts_read.rows_for_run's
    own ORDER BY does this; each row needs at least id/src_in_ms/
    src_out_ms/voice_ids. Junk cuts are NOT filtered out -- they stay in the
    numbering (a gap in cut_no is the signal a junk beat sits there), same
    contract as the old pipeline. ``shot_points``: this file's own
    scene_cuts.shot_points ([{"ts_ms": ...}, ...]). Returns {cut_id:
    continuity_dict}, the exact shape post._clip_continuity produces --
    footage_map._weld_mark/_continuity_tag read this shape either way."""
    n = len(cuts)
    shot_ts = sorted(int(p["ts_ms"]) for p in shot_points if p.get("ts_ms") is not None)
    out: Dict[str, Dict[str, Any]] = {
        c["id"]: {
            "clip": c["file_id"], "cut_no": pos + 1, "of": n,
            "prev_contiguous": False, "next_contiguous": False,
            "seam_reason_prev": None, "seam_reason_next": None,
        } for pos, c in enumerate(cuts)
    }
    for pos in range(n - 1):
        cur, nxt = cuts[pos], cuts[pos + 1]
        cur_s, cur_e = int(cur["src_in_ms"]), int(cur["src_out_ms"])
        nxt_s, nxt_e = int(nxt["src_in_ms"]), int(nxt["src_out_ms"])
        gap_lo, gap_hi = min(cur_e, nxt_s), max(cur_e, nxt_s)
        has_break = any(gap_lo <= t <= gap_hi for t in shot_ts)
        verdict = classify_seam(Seam(
            same_clip=True,
            same_speaker=(set(cur.get("voice_ids") or []) == set(nxt.get("voice_ids") or [])),
            gap_ms=max(0, nxt_s - cur_e),
            bridged_speech_ms=(cur_e - cur_s) + (nxt_e - nxt_s),
            has_scene_or_transition=has_break,
            has_flagged_break=False,
        ))
        out[cur["id"]]["next_contiguous"] = verdict.weldable
        out[cur["id"]]["seam_reason_next"] = verdict.reason
        out[nxt["id"]]["prev_contiguous"] = verdict.weldable
        out[nxt["id"]]["seam_reason_prev"] = verdict.reason
    return out


def write_continuity_for_run(ingest_run_id: str, seam_cache: Dict[str, dict]) -> int:
    """Read this run's cut_records back (both kinds, already inserted by
    both channels), compute continuity per file over ALL of that file's
    cuts together, write it onto each row. Returns the number of rows
    updated. ``seam_cache``: the SAME in-memory dict run_vcut_ingest already
    threads through both channels -- shot_points rides on it (vcut.
    orchestrate._add_landmark_signals_to_seam_cache), so this needs no new
    DB read for that part. Self-guarding (fail-open, matching vcut.identity.
    reconcile_and_store's own contract): never raises. A single file's
    continuity failure is logged and skipped; a total failure (e.g. the
    read-back itself) leaves every cut's continuity at its column default
    ({}) rather than failing a run whose actual cuts already landed."""
    try:
        from app.services.l3 import cuts_read, ingest_store

        rows = cuts_read.rows_for_run(ingest_run_id)
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_file.setdefault(r["file_id"], []).append(r)

        n_written = 0
        for file_id, cuts in by_file.items():
            try:
                shot_points = (seam_cache.get(file_id) or {}).get("shot_points") or []
                continuity_by_id = compute_continuity_for_file(cuts, shot_points)
            except Exception:
                logger.exception(
                    "vcut.continuity: computation failed for file %s (leaving its cuts without continuity)",
                    file_id)
                continue
            for cut_id, continuity in continuity_by_id.items():
                ingest_store.update_cut_continuity(cut_id, continuity)
                n_written += 1
        return n_written
    except Exception:
        logger.exception("vcut.continuity: run %s failed entirely (cuts already landed, continuing)",
                         ingest_run_id)
        return 0
