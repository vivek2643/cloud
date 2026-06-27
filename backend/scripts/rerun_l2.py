"""
One-off: re-run L2 perception so every ready clip is logged under the current
SCHEMA_VERSION (the typed relation graph + roles + ids). The L3 hero-cut cache
recomputes lazily on the next feed read (PARAMS_VERSION is in its signature), so
this script only needs to refresh L2.

Usage:
    python3 scripts/rerun_l2.py --dry-run     # inventory only, no Gemini calls
    python3 scripts/rerun_l2.py               # re-run every stale/ready clip
    python3 scripts/rerun_l2.py --only <uuid> # one file
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l2.perception import l2_perception  # noqa: E402
from app.services.l2.schema import SCHEMA_VERSION  # noqa: E402


def _conn():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _ready_files(conn) -> list[tuple[str, str, int, str]]:
    """(file_id, name, perception_schema_version, l1_status) for L1-ready clips,
    ordered shortest-first so cheap clips validate the run before long ones."""
    rows = conn.execute(
        """
        select f.id::text,
               coalesce(f.name, f.id::text),
               coalesce(cp.schema_version, 0),
               coalesce(f.l1_status, ''),
               coalesce(f.duration_seconds, 0)
          from files f
          left join clip_perception cp on cp.file_id = f.id
         where f.l1_status = 'ready'
           and f.file_type in ('video', 'audio')
         order by coalesce(f.duration_seconds, 0) asc
        """
    ).fetchall()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="single file_id")
    ap.add_argument("--force", action="store_true",
                    help="re-run even files already at the current schema")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel L2 calls (I/O-bound: R2 download + Gemini)")
    args = ap.parse_args()

    settings = get_settings()
    if not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY not set; cannot run L2.", file=sys.stderr)
        return 2

    with _conn() as conn:
        rows = _ready_files(conn)

    if args.only:
        rows = [r for r in rows if r[0] == args.only]

    stale = [r for r in rows if args.force or r[2] < SCHEMA_VERSION]
    current = [r for r in rows if not args.force and r[2] >= SCHEMA_VERSION]

    print(f"SCHEMA_VERSION = {SCHEMA_VERSION}")
    print(f"ready clips: {len(rows)} | to re-run: {len(stale)} | "
          f"already current: {len(current)}")
    for fid, name, sv, _l1, dur in stale:
        print(f"  RERUN  {fid}  schema={sv}  {dur:.0f}s  {name[:48]}")

    if args.dry_run:
        print("\n(dry run -- no Gemini calls made)")
        return 0

    total = len(stale)
    workers = max(1, min(args.workers, total))
    print(f"\nrunning {total} clips on {workers} parallel workers ...\n", flush=True)
    counter = {"ok": 0, "fail": 0, "done": 0}
    lock = threading.Lock()

    def run_one(item) -> None:
        fid, name, sv, _l1, dur = item
        t0 = time.time()
        try:
            l2_perception(fid)          # procrastinate task is directly callable
            status, detail = "ok", f"{time.time() - t0:.1f}s"
        except Exception as e:           # noqa: BLE001 -- keep going on one failure
            status, detail = "fail", repr(e)
        with lock:
            counter[status] += 1
            counter["done"] += 1
            n = counter["done"]
            tag = "done " + detail if status == "ok" else "FAILED " + detail
            print(f"[{n}/{total}] {dur:.0f}s {name[:44]:44} -> {tag}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, it) for it in stale]
        for _ in as_completed(futures):
            pass

    print(f"\nL2 rerun complete: {counter['ok']} ok, {counter['fail']} failed, "
          f"{total} attempted. L3 will recompute lazily on next feed read.")
    return 0 if counter["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
