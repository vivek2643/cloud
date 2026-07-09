"""One-off driver: bring EVERY project up to the latest cuts-v3 pipeline.

Phase 1 -- L1 motion refresh. The camera-move tag needs the signed camera
velocity series (camera_dx/dy/zoom) that only exists for files whose motion
stage ran after migration 033. Re-run motion_dynamics for every file still on
the old row, off its CHEAPEST existing source (proxy_b if present, else the
1080p editing proxy) so we never pull a multi-GB raw. The stage upsert
overwrites the whole row deterministically, so all other L1 signals are
unchanged -- it just fills in the camera series.

Phase 2 -- re-ingest every distinct project so quality scores, framing, and
the camera tag land on every cut.

Idempotent-ish: re-running Phase 1 just recomputes (harmless); Phase 2 always
creates a fresh ingest_run (the UI reads the latest ready one).
"""
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from app.config import get_settings
from app.services.processing import _download_from_r2
from app.services.l1.pipeline import _pg_conn, _stage9_motion_dynamics
from app.services.l3 import ingest


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _distinct_projects_and_need():
    """(project_ids in source order, [(file_id, source_key, duration_s)] to refresh)."""
    with psycopg.connect(get_settings().database_url) as c:
        rows = c.execute("select id::text, source_file_ids from projects").fetchall()
        seen = {}
        for pid, sf in rows:
            key = tuple(sorted(str(x) for x in (sf or [])))
            if key:
                seen.setdefault(key, []).append(pid)
        project_ids = [pids[0] for _key, pids in seen.items()]
        all_files = sorted(set(f for key in seen for f in key))

        have = set(
            r[0] for r in c.execute(
                "select file_id::text from motion_dynamics where camera_dx <> '[]'::jsonb"
            ).fetchall()
        )
        frows = c.execute(
            "select id::text, r2_proxy_b_key, r2_proxy_key, duration_seconds "
            "from files where id = any(%s::uuid[]) and file_type = 'video'",
            (all_files,),
        ).fetchall()
        need = []
        for fid, pb, pk, dur in frows:
            if fid in have:
                continue
            src = pb or pk
            if not src:
                _log(f"  WARN {fid[:8]} has no proxy source; skipping motion refresh")
                continue
            need.append((fid, src, float(dur or 0.0)))
    return project_ids, need


def _refresh_one(file_id: str, source_key: str, duration_s: float) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "motion_src.mp4")
        _download_from_r2(source_key, path)
        with _pg_conn() as conn:
            _stage9_motion_dynamics(file_id, path, duration_s, conn)
    return file_id


def phase1_refresh(need, workers: int = 3) -> None:
    _log(f"PHASE 1: refreshing L1 motion (camera series) for {len(need)} files "
         f"({workers} workers)")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_refresh_one, f, s, d): f for f, s, d in need}
        for fut in as_completed(futs):
            fid = futs[fut]
            done += 1
            try:
                fut.result()
                _log(f"  [{done}/{len(need)}] refreshed {fid[:8]}")
            except Exception as e:  # noqa: BLE001
                _log(f"  [{done}/{len(need)}] FAILED {fid[:8]}: {type(e).__name__}: {e}")
                traceback.print_exc()
    _log("PHASE 1 complete")


def phase2_ingest(project_ids, workers: int = 3) -> None:
    _log(f"PHASE 2: re-ingesting {len(project_ids)} projects ({workers} workers)")
    results = ingest.run_many(project_ids, max_workers=workers)
    ok, fail = 0, 0
    for pid, res in results.items():
        if isinstance(res, Exception):
            fail += 1
            _log(f"  FAILED {pid[:8]}: {type(res).__name__}: {res}")
        else:
            ok += 1
            _log(f"  ok {pid[:8]} -> run {str(res)[:8]}")
    _log(f"PHASE 2 complete: {ok} ok, {fail} failed")


def main() -> None:
    t0 = time.time()
    project_ids, need = _distinct_projects_and_need()
    _log(f"plan: {len(project_ids)} projects, {len(need)} files need motion refresh")
    phase1_refresh(need)
    phase2_ingest(project_ids)
    _log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
