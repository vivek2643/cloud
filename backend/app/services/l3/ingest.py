"""
Cuts v3 orchestrator: pass1 -> image_plan -> frame extraction -> pass2
(shards, back-to-back for cache) -> post -> persist + hero frames. One call
per project; re-runnable (each call is a fresh ``ingest_run`` row -- nothing
is patched in place). See cuts_v3.plan.md section 6 and the build-order
table (step E).

THIS MODULE SPENDS REAL MONEY once invoked: ``run_ingest`` makes one real
pass-1 API call and one real pass-2 API call per shard. Every deterministic
step it wires together (pass1's prompt building, image_plan, frame
extraction, pass2's prompt building, post's assembly) already has its own
zero-cost test coverage; ``scripts/test_ingest.py`` mocks every one of those
seams so the ORCHESTRATION ITSELF (stage sequencing, status transitions,
error handling) is verified without spending anything. Actually calling
``run_ingest`` against a real project is a separate, explicit decision.
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
from app.services.l3.lattice import Lattice
from app.services.l3.pass2_params import STILL_WIDTH_PX
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
        store.record_pass1_result(ingest_run_id, pass1_completion.data, pass1_completion.usage,
                                  pass1_output.project_summary)

        store.set_status(ingest_run_id, "images")
        planned_frames = ip.build_image_plan(pass1_output, lattices, motion_by_file, scene_by_file,
                                             silences_by_file)
        images_b64 = fr.extract_for_planned_frames(planned_frames, proxy_keys)

        store.set_status(ingest_run_id, "pass2")
        shards = pass2.build_shards(pass1_output, planned_frames)
        all_cuts: List[pass2.Pass2Cut] = []
        for shard_files in shards:
            shard_set = set(shard_files)
            shard_frames = [f for f in planned_frames if f.file_id in shard_set]
            shard_file_rows = [row for row in file_rows if row[0] in shard_set]
            completion = pass2.run_pass2_shard(shard_file_rows, pass1_output, shard_frames, images_b64)
            store.accumulate_pass2_usage(ingest_run_id, completion.usage)
            shard_output = pass2.Pass2Output.model_validate(completion.data)
            all_cuts.extend(shard_output.cuts)

        store.set_status(ingest_run_id, "post")
        records = post.assemble_cut_records(pass2.Pass2Output(cuts=all_cuts), lattices,
                                            motion_by_file, silences_by_file)

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
