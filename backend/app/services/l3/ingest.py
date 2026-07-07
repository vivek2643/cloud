"""
Cuts v3 orchestrator: pass1 -> image_plan -> frame extraction -> pass2a
(identity + take resolution, shards run concurrently) -> pass2b (visual
judgment, batches run concurrently) -> merge -> post -> persist + hero
frames. One call per project; re-runnable (each call is a fresh
``ingest_run`` row -- nothing is patched in place). See cuts_v3.plan.md
section 6 and the build-order table (step E).

Pass 2a and pass 2b both run their calls IN PARALLEL (ThreadPoolExecutor),
not sequentially: within each stage the calls only share a read-only cached
prefix, so nothing stops them firing at the same time -- this is a pure
wall-clock win (pass 2a's take-comparison shards still stay take-group-aware
via co-location; pass 2b's batches have no such constraint at all and can
run with even more parallelism). See pass2a.py / pass2b.py for why the
identity/take half and the pure visual-judgment half are split into two
calls instead of one large one.

THIS MODULE SPENDS REAL MONEY once invoked: ``run_ingest`` makes one real
pass-1 API call, one real pass-2a call per identity shard, and one real
pass-2b call per visual batch. Every deterministic step it wires together
(pass1's prompt building, image_plan, pass2a/pass2b's prompt building,
post's assembly) already has its own zero-cost test coverage;
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
from app.services.l3 import pass2a
from app.services.l3 import pass2b
from app.services.l3 import post
from app.services.l3.lattice import Lattice
from app.services.l3.pass2_params import (
    MAX_CUTS_PER_VISUAL_BATCH, MAX_PARALLEL_SHARDS, MAX_PARALLEL_VISUAL_BATCHES, STILL_WIDTH_PX,
)
from app.services.processing import _download_from_r2, _upload_to_r2

logger = logging.getLogger(__name__)


def _load_signals(file_ids: List[str]) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, list]]:
    motion_by_file: Dict[str, dict] = {}
    scene_by_file: Dict[str, dict] = {}
    silences_by_file: Dict[str, list] = {}
    for fid in file_ids:
        snap = build_l1_snapshot(fid)
        motion_by_file[fid] = snap.get("motion_dynamics") or {}
        scene_by_file[fid] = snap.get("scene_cuts") or {}
        silences_by_file[fid] = (snap.get("audio_features") or {}).get("silence_intervals") or []
    return motion_by_file, scene_by_file, silences_by_file


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
        lattices: Dict[str, Lattice] = {fid: lat for fid, _name, _dur, lat in file_rows}
        file_ids = list(lattices.keys())
        motion_by_file, scene_by_file, silences_by_file = _load_signals(file_ids)
        proxy_keys = _proxy_keys_for_files(file_ids)

        store.set_status(ingest_run_id, "pass1")
        pass1_completion = pass1.run_pass1(file_rows)
        pass1_output = pass1.Pass1Output.model_validate(pass1_completion.data)
        # Deterministic boundary repair (split cuts crossing atom-owned gaps,
        # realign take members) -- the model owns meaning, the lattice owns
        # boundaries. The ENFORCED output is what gets persisted: pass 2's
        # cached prefix and the image plan must see the same refs.
        pass1_output = pass1.enforce_lattice_partition(pass1_output, lattices)
        store.record_pass1_result(ingest_run_id, pass1_output.model_dump(), pass1_completion.usage,
                                  pass1_output.project_summary)

        store.set_status(ingest_run_id, "images")
        planned_frames = ip.build_image_plan(pass1_output, lattices, motion_by_file, scene_by_file,
                                             silences_by_file)
        images_b64 = fr.extract_for_planned_frames(planned_frames, proxy_keys)

        # pass2a/pass2b are both still reported as "pass2" -- the DB status
        # column's check constraint only knows the original stage names,
        # and splitting it further isn't worth a migration for what's
        # purely a finer-grained progress label.
        store.set_status(ingest_run_id, "pass2")
        identity_shards = pass2a.build_identity_shards(pass1_output, planned_frames)
        shard_args = []
        for shard_files in identity_shards:
            shard_set = set(shard_files)
            shard_frames = [f for f in planned_frames if f.file_id in shard_set]
            shard_file_rows = [row for row in file_rows if row[0] in shard_set]
            shard_args.append((shard_file_rows, shard_frames))

        all_identity_cuts: List[pass2a.IdentityCut] = []
        # Shards only share a read-only cached prefix -- no reason to make
        # one wait on another's response. Concurrency here is a pure
        # wall-clock win (see MAX_PARALLEL_SHARDS); a failure in any shard
        # still propagates and fails the whole run, same as sequential did.
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_SHARDS, len(shard_args)) or 1) as pool:
            futures = [
                pool.submit(pass2a.run_identity_shard, rows, pass1_output, frames, images_b64)
                for rows, frames in shard_args
            ]
            for future in futures:
                completion = future.result()
                store.accumulate_pass2_usage(ingest_run_id, completion.usage)
                shard_output = pass2a.IdentityOutput.model_validate(completion.data)
                # locators (word_span/atom_ids) are code-derived from pass 1,
                # not echoed by the model -- see pass2a.backfill_locators.
                shard_output = pass2a.backfill_locators(shard_output, pass1_output)
                all_identity_cuts.extend(shard_output.cuts)
        identity_output = pass2a.IdentityOutput(cuts=all_identity_cuts)

        batches = pass2b.build_visual_batches(identity_output, MAX_CUTS_PER_VISUAL_BATCH)
        visual_by_index: Dict[int, pass2b.VisualJudgment] = {}
        # No take-style co-location constraint here at all (see pass2b.py),
        # so batches can run with even more parallelism than pass 2a's shards.
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_VISUAL_BATCHES, len(batches)) or 1) as pool:
            futures = [
                pool.submit(pass2b.run_visual_batch, identity_output, batch, planned_frames, images_b64)
                for batch in batches
            ]
            for future in futures:
                completion = future.result()
                store.accumulate_pass2_usage(ingest_run_id, completion.usage)
                batch_output = pass2b.VisualOutput.model_validate(completion.data)
                for judgment in batch_output.judgments:
                    visual_by_index[judgment.cut_index] = judgment

        pass2_output = pass2.merge_identity_and_visual(identity_output, visual_by_index)
        pass2_output = pass2.apply_junk_suspects(pass2_output, pass1_output)

        store.set_status(ingest_run_id, "post")
        records = post.assemble_cut_records(pass2_output, lattices, motion_by_file, silences_by_file)

        store.delete_cut_records_for_run(ingest_run_id)
        record_ids = store.insert_cut_records(ingest_run_id, records)
        _extract_and_upload_heroes(records, record_ids, proxy_keys)

        store.set_status(ingest_run_id, "ready")
    except Exception as e:
        logger.exception("ingest run %s failed", ingest_run_id)
        store.set_status(ingest_run_id, "failed", error=str(e))
        raise
    return ingest_run_id


@app.task(name="l3_cuts_v3_ingest", queue="l2", retry=False)
def l3_cuts_v3_ingest(project_id: str) -> None:
    run_ingest(project_id)


def defer_ingest(project_id: str) -> None:
    """Enqueue a cuts-v3 ingest run on the l2 procrastinate queue (same
    queue L2 perception / hero-cut precompute run on). Not auto-retried:
    each attempt is a real, costed API call, and a failure here is almost
    always a schema/prompt problem worth looking at, not a transient one."""
    from procrastinate import App, PsycopgConnector

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        enqueue_app.configure_task("l3_cuts_v3_ingest", queue="l2").defer(project_id=project_id)


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
