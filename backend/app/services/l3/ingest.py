"""
Cuts v3 orchestrator: pass1 -> image_plan -> frame extraction -> pass2
(identity + full visual judgment in one call per batch, batches run
concurrently) -> post -> persist + hero frames. One call per project;
re-runnable (each call is a fresh ``ingest_run`` row -- nothing is patched
in place). See cuts_v3.plan.md section 6 and the build-order table (step E).

Pass 2's batches run IN PARALLEL (ThreadPoolExecutor), not sequentially:
they only share a read-only cached prefix, so nothing stops them firing at
the same time -- this is a pure wall-clock win. Batching is pure size-based
chunking, no co-location constraint at all (pass2_merge.plan.md): the one
thing that used to need cross-cut pixels -- resolving a take group's members
into take/winner/outlook -- is now deterministic code
(``pass2.apply_take_groups``, fed by pass 1's own ``take_candidates``), so
there is no reason for a take's members to share a batch or for images to be
sent to the model more than once.

THIS MODULE SPENDS REAL MONEY once invoked: ``run_ingest`` makes one real
pass-1 API call and one real pass-2 call per batch. Every deterministic step
it wires together (pass1's prompt building, image_plan, pass2's prompt
building, post's assembly) already has its own zero-cost test coverage;
``scripts/test_ingest.py`` mocks every one of those seams so the
ORCHESTRATION ITSELF (stage sequencing, status transitions, error handling)
is verified without spending anything. Actually calling ``run_ingest``
against a real project is a separate, explicit decision.
"""
from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from app.config import get_settings
from app.services.jobs import app
from app.services.l1.snapshot import build_l1_snapshot
from app.services.l3 import frames as fr
from app.services.l3 import image_plan as ip
from app.services.l3 import ingest_store as store
from app.services.l3 import pass1
from app.services.l3 import pass2
from app.services.l3 import post
from app.services.l3.identity import apply as identity_apply
from app.services.l3.lattice import Lattice
from app.services.l3.sync import store as sync_store
from app.services.l3.sync.lattice_merge import apply_outlook_groups, outlook_groups
from app.services.l3.pass2_params import MAX_PARALLEL_PASS2_BATCHES, STILL_WIDTH_PX
from app.services.processing import _download_from_r2, _upload_to_r2

logger = logging.getLogger(__name__)


def _load_signals(
    file_ids: List[str],
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, list], Dict[str, dict]]:
    motion_by_file: Dict[str, dict] = {}
    scene_by_file: Dict[str, dict] = {}
    silences_by_file: Dict[str, list] = {}
    audio_by_file: Dict[str, dict] = {}
    for fid in file_ids:
        snap = build_l1_snapshot(fid)
        motion_by_file[fid] = snap.get("motion_dynamics") or {}
        scene_by_file[fid] = snap.get("scene_cuts") or {}
        af = snap.get("audio_features") or {}
        silences_by_file[fid] = af.get("silence_intervals") or []
        # rms_db (dB energy envelope) + its hop feed the speech_quality loudness
        # term in post.assemble_cut_records; empty when a clip has no audio_features.
        audio_by_file[fid] = {"rms_db": af.get("rms_db") or [], "hop_ms": af.get("prosody_hop_ms") or 0}
    return motion_by_file, scene_by_file, silences_by_file, audio_by_file


def _proxy_keys_for_files(file_ids: List[str]) -> Dict[str, str]:
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select id::text, r2_proxy_key from files where id = any(%s::uuid[])", (file_ids,),
        ).fetchall()
    return {fid: key for fid, key in rows if key}


def _extract_and_upload_heroes(records: List[post.CutRecord], record_ids: List[str],
                               proxy_keys: Dict[str, str]) -> None:
    by_file: Dict[str, List[Tuple[str, post.CutRecord]]] = {}
    for rid, rec in zip(record_ids, records):
        by_file.setdefault(rec.file_id, []).append((rid, rec))

    for file_id, items in by_file.items():
        proxy_key = proxy_keys.get(file_id)
        if not proxy_key:
            logger.warning("ingest: no proxy for %s -- skipping %d hero frame(s)", file_id, len(items))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            proxy_path = os.path.join(tmp, "proxy.mp4")
            _download_from_r2(proxy_key, proxy_path)
            for rid, rec in items:
                still_path = os.path.join(tmp, f"{rid}.jpg")
                fr.extract_still(proxy_path, rec.hero_ts_ms, still_path, width=STILL_WIDTH_PX)
                hero_key = f"heroes/{file_id}/{rid}.jpg"
                _upload_to_r2(still_path, hero_key, "image/jpeg")
                store.set_hero_key(rid, hero_key)


def run_ingest(project_id: str) -> str:
    """Run the full cuts-v3 ingest for one project. Returns the new
    ingest_run id. On any failure the run is marked ``failed`` with the
    reason and the exception re-raised -- no partial result is ever left
    looking like a success (the plan's "no fallback" rule)."""
    settings = get_settings()
    ingest_run_id = store.create_ingest_run(project_id, settings.ingest_pass1_model, settings.ingest_pass2_model)
    try:
        file_rows = pass1.load_project_file_rows(project_id)
        if not file_rows:
            raise ValueError(f"project {project_id} has no ingest-ready files")

        # Pin whatever OUTLOOK groups exist for this project's files RIGHT NOW
        # (a later re-align must not mutate this run) and swap each angle's
        # Lattice speech side onto its group's authoritative source. A project
        # with no declared groups is untouched here (`outlook_by_file` empty ->
        # `file_rows`/hints/ids all pass through as-is): the no-regression
        # guarantee.
        pre_file_ids = [fid for fid, _name, _dur, _lat in file_rows]
        outlook_by_file = sync_store.sync_groups_for_files(pre_file_ids)
        # EVERY member of a declared group is an outlook of the others (a
        # different camera on one moment) -- speech-swapped, beat-mirrored, and
        # grouped as outlooks. Alignment confidence is metadata only, never an
        # exclusion: demoting a low-confidence member to an independent clip is
        # exactly what let pass 1 mis-group it as a TAKE with its own audio.
        groups = outlook_groups(outlook_by_file)
        file_rows, outlook_hints, outlook_file_ids = apply_outlook_groups(file_rows, outlook_by_file, groups)
        outlook_group_by_file = {fid: gid for gid, grp in groups.items() for fid in grp["members"]}

        lattices: Dict[str, Lattice] = {fid: lat for fid, _name, _dur, lat in file_rows}
        file_ids = list(lattices.keys())
        motion_by_file, scene_by_file, silences_by_file, audio_by_file = _load_signals(file_ids)
        proxy_keys = _proxy_keys_for_files(file_ids)

        store.set_status(ingest_run_id, "pass1")
        pass1_completion = pass1.run_pass1(file_rows, outlook_hints)
        pass1_output = pass1.Pass1Output.model_validate(pass1_completion.data)
        # Outlook angles: mirror the authoritative angle's speech beats onto
        # every angle in the group BEFORE enforcement, so each angle carries the
        # identical spans. Enforcement then runs per angle (synced=True) and
        # yields byte-identical final speech cuts.
        pass1_output = pass1.replicate_outlook_speech(pass1_output, lattices, groups)
        # Deterministic boundary repair (split cuts crossing atom-owned gaps,
        # realign take members, and drop any take-candidate that is really one
        # outlook group's angles) -- the model owns meaning, the lattice owns
        # boundaries. The ENFORCED output is what gets persisted.
        pass1_output = pass1.enforce_lattice_partition(
            pass1_output, lattices, outlook_file_ids, outlook_group_by_file)
        # Now that each angle's cuts are aligned, link them into take_candidates
        # -- pass 2a resolves each as an outlook (no winner), feeding
        # footage_map/observe's alt-PIC angle switching.
        pass1_output = pass1.group_outlooks(pass1_output, groups)
        store.record_pass1_result(ingest_run_id, pass1_output.model_dump(), pass1_completion.usage,
                                  pass1_output.project_summary)

        store.set_status(ingest_run_id, "images")
        planned_frames = ip.build_image_plan(pass1_output, lattices, motion_by_file, scene_by_file,
                                             silences_by_file)
        images_b64 = fr.extract_for_planned_frames(planned_frames, proxy_keys)

        # Still reported as "pass2" -- the DB status column's check
        # constraint only knows the original stage name, and adding a new
        # one isn't worth a migration for what's purely a progress label.
        store.set_status(ingest_run_id, "pass2")
        batches = pass2.build_pass2_batches(pass1_output, planned_frames)
        batch_args = []
        for batch_refs in batches:
            ref_set = set(batch_refs)
            batch_frames = [f for f in planned_frames if f.ref in ref_set]
            batch_file_ids = {f.file_id for f in batch_frames}
            batch_file_rows = [row for row in file_rows if row[0] in batch_file_ids]
            batch_args.append((batch_file_rows, batch_frames))

        all_cuts: List[pass2.Pass2Cut] = []
        # Batches only share a read-only cached prefix -- no reason to make
        # one wait on another's response. Concurrency here is a pure
        # wall-clock win (see MAX_PARALLEL_PASS2_BATCHES); a failure in any
        # batch still propagates and fails the whole run, same as sequential did.
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_PASS2_BATCHES, len(batch_args)) or 1) as pool:
            futures = [
                pool.submit(pass2.run_pass2_batch, rows, pass1_output, frames, images_b64)
                for rows, frames in batch_args
            ]
            for future in futures:
                completion = future.result()
                store.accumulate_pass2_usage(ingest_run_id, completion.usage)
                batch_output = pass2.Pass2BatchOutput.model_validate(completion.data)
                # locators (word_span/atom_ids) are code-derived from pass 1,
                # not echoed by the model -- see pass2.backfill_locators.
                batch_output = pass2.backfill_locators(batch_output, pass1_output)
                all_cuts.extend(pass2.to_pass2_cuts(batch_output.cuts))

        pass2_output = pass2.Pass2Output(cuts=all_cuts)
        pass2_output = pass2.apply_junk_suspects(pass2_output, pass1_output)
        # take_group_id/take_role are entirely code-owned now (pass2_merge.
        # plan.md D1/D2): the model never resolves takes, so this is the
        # ONLY place they're ever set -- both declared-outlook members
        # (alternate angles, never retakes) and genuine same-setting takes
        # (post._enforce_take_winner crowns the winner right after post.py's
        # assemble_cut_records call below).
        pass2_output = pass2.apply_take_groups(pass2_output, pass1_output)

        # Cumulative identity map (identity_map.plan.md): reconcile WHO's
        # talking (voice<->camera-motion binding, Phase 1) and WHO's shown
        # (cross-file appearance clustering, Phase 2) once, deterministically,
        # then rewrite on_camera from pass 2's per-still guess into a derived
        # fact (Phase 3a). Must run before assemble_cut_records below --
        # on_camera also feeds total_quality there, so the rewrite only
        # counts if it lands first. `groups` is the SAME outlook grouping the
        # speech lattice merge already used (line ~132), so binding and
        # identity share one notion of "these cameras are one moment."
        pass2_output, identity_map = identity_apply.run(pass2_output, lattices, motion_by_file, groups)
        if identity_map.get("persons"):
            store.set_identity_map(ingest_run_id, identity_map)

        store.set_status(ingest_run_id, "post")
        records = post.assemble_cut_records(pass2_output, lattices, motion_by_file, silences_by_file,
                                            junk_suspects=pass1_output.junk_suspects,
                                            audio_by_file=audio_by_file,
                                            synced_file_ids=outlook_file_ids,
                                            sync_group_by_file=outlook_group_by_file)

        store.delete_cut_records_for_run(ingest_run_id)
        record_ids = store.insert_cut_records(ingest_run_id, records)
        _extract_and_upload_heroes(records, record_ids, proxy_keys)

        store.set_status(ingest_run_id, "ready")
    except Exception as e:
        logger.exception("ingest run %s failed", ingest_run_id)
        store.set_status(ingest_run_id, "failed", error=str(e))
        raise
    return ingest_run_id


@app.task(name="l3_cuts_v3_ingest", queue="ingest", retry=False)
def l3_cuts_v3_ingest(project_id: str) -> None:
    run_ingest(project_id)


def defer_ingest(project_id: str) -> None:
    """Enqueue a cuts-v3 ingest run on the network-bound ``ingest`` procrastinate
    queue (its own worker, decoupled from GPU ingest). Not auto-retried: each
    attempt is a real, costed API call, and a failure here is almost always a
    schema/prompt problem worth looking at, not a transient one."""
    from procrastinate import App, PsycopgConnector

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        enqueue_app.configure_task("l3_cuts_v3_ingest", queue="ingest").defer(project_id=project_id)


def run_many(project_ids: List[str], max_workers: int = 4) -> Dict[str, Any]:
    """Run cuts-v3 ingest for several projects concurrently. Different
    projects share no state -- separate API calls, separate DB rows,
    separate R2 keys -- so this is a pure wall-clock win with none of the
    prompt-cache tradeoff that parallelizing shards WITHIN one project would
    have (those deliberately stay sequential, see run_ingest). Never raises
    itself: returns {project_id: ingest_run_id} on success or
    {project_id: the raised Exception} on failure, so one project's failure
    doesn't stop the others from being reported."""
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(project_ids))) as pool:
        futures = {pool.submit(run_ingest, pid): pid for pid in project_ids}
        for future in futures:
            project_id = futures[future]
            try:
                results[project_id] = future.result()
            except Exception as e:
                results[project_id] = e
    return results
