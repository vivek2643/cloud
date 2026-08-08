"""
seam_function.plan.md §6 -- the "FOR ALL CLIPS" driver.

Enumerates every video file with L1 done (files.l1_status = 'ready'),
computes S(t) for each via build_seam_signals -> compute_seam_curve, and
optionally caches the result as JSON under backend/logs/seam/<file_id>.json
-- a convenience for the cutviz validation UI / a later harness, NOT a DB
table (v1 is compute-on-demand + optional cache only, per the plan's own
"no schema change" non-goal; backend/logs/ is already gitignored runtime
output, same as the rest of this repo's local caches).

One clip per worker (ProcessPoolExecutor -- the work is ffmpeg/opencv/
librosa-bound per clip, so process-level parallelism, not threads). Pool
size is the real cross-process concurrency bound (app.services.limits.
ffmpeg_slot() only bounds concurrency WITHIN one process's own thread
semaphore, inherited "for free" here since signals.py's own ffmpeg calls
already wrap themselves in it -- each worker process gets its own copy).
Failures are best-effort: logged and skipped, never fatal, mirroring
motion_dynamics' own non-fatal contract.

Run:
  .venv/bin/python scripts/seam/compute_all_seam.py
  .venv/bin/python scripts/seam/compute_all_seam.py --limit 20 --workers 4
  .venv/bin/python scripts/seam/compute_all_seam.py --file-id <uuid> --no-cache
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_all_seam")

CACHE_DIR = os.path.join(BACKEND, "logs", "seam")


def _list_file_ids(limit: Optional[int]) -> List[str]:
    from app.services import db

    sql = "select id::text from files where file_type = 'video' and l1_status = 'ready' order by created_at desc"
    if limit:
        sql += f" limit {int(limit)}"
    with db.connection_dict_row() as conn:
        rows = conn.execute(sql).fetchall()
    return [r["id"] for r in rows]


def _cache_path(file_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{file_id}.json")


def process_one(file_id: str, write_cache: bool, force: bool) -> Dict[str, Any]:
    """Runs in a worker process -- imports are deferred to inside the
    function (rather than module top-level) so each spawned worker builds
    its own fresh app/DB state, not a copy inherited from the parent."""
    cache_path = _cache_path(file_id)
    if write_cache and not force and os.path.exists(cache_path):
        return {"file_id": file_id, "status": "skipped (cached)"}

    try:
        from app.services.seam import build_seam_signals, compute_seam_curve

        signals = build_seam_signals(file_id)
        curve = compute_seam_curve(signals)
    except Exception as exc:  # noqa: BLE001 -- best-effort batch job, never fatal
        logger.exception("seam compute failed for %s", file_id)
        return {"file_id": file_id, "status": f"error: {exc}"}

    if write_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        payload = {
            "file_id": file_id, "hop_ms": curve.hop_ms, "t_ms": curve.t_ms,
            "S": curve.S, "g_sharp": curve.g_sharp, "g_gest": curve.g_gest,
            "still": curve.still, "audio": curve.audio, "w_aud": curve.w_aud,
            "beats_ms": signals.beats_ms, "onsets_ms": signals.onsets_ms,
            "onset_strength": signals.onset_strength, "is_musical": signals.is_musical,
            "meta": curve.meta,
        }
        with open(cache_path, "w") as fh:
            json.dump(payload, fh)

    return {"file_id": file_id, "status": "ok", "n": len(curve.S)}


def run(limit: Optional[int], workers: int, write_cache: bool, force: bool,
        single_file_id: Optional[str]) -> None:
    file_ids = [single_file_id] if single_file_id else _list_file_ids(limit)
    logger.info("processing %d file(s) (workers=%d, cache=%s)", len(file_ids), workers, write_cache)

    results: List[Dict[str, Any]] = []
    if workers <= 1:
        for fid in file_ids:
            results.append(process_one(fid, write_cache, force))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, fid, write_cache, force): fid for fid in file_ids}
            for fut in as_completed(futures):
                fid = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001 -- a worker crash must not kill the batch
                    logger.exception("worker crashed for %s", fid)
                    results.append({"file_id": fid, "status": f"error: {exc}"})

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"].startswith("skipped"))
    failed = len(results) - ok - skipped
    logger.info("done: %d ok, %d skipped, %d failed (of %d)", ok, skipped, failed, len(results))
    for r in results:
        if r["status"] != "ok" and not r["status"].startswith("skipped"):
            logger.info("  %s: %s", r["file_id"], r["status"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of files processed")
    ap.add_argument("--workers", type=int,
                    default=int(os.getenv("SEAM_WORKERS", str(os.cpu_count() or 4))),
                    help="pool size (env SEAM_WORKERS, default = CPU count)")
    ap.add_argument("--file-id", default=None, help="process a single file (testing/debugging)")
    ap.add_argument("--cache", dest="write_cache", action="store_true", default=True)
    ap.add_argument("--no-cache", dest="write_cache", action="store_false",
                    help="compute only, don't write logs/seam/<file_id>.json")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a cache file already exists")
    args = ap.parse_args()
    run(args.limit, args.workers, args.write_cache, args.force, args.file_id)


if __name__ == "__main__":
    main()
