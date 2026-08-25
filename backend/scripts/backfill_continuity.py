"""
Backfill the B1 continuity block (brain_perception_blindness.plan.md) onto
cut_records that were ingested before B1 existed.

A full re-ingest is NOT required. Continuity is the plan's only change to
STORED data -- A1/A2/A3 are render-time (footage_map/observe) and apply to old
rows the moment they're read. And continuity itself is already a post-pass over
already-inserted rows (vcut.continuity.write_continuity_for_run), so the only
thing separating an old run from a backfilled one is where `shot_points` comes
from: the live path takes it off the in-memory seam_cache, which is gone once a
run ends, but the same values are persisted in `scene_cuts.shot_points`.

So this script rebuilds that seam_cache from `scene_cuts` and calls the
PRODUCTION function unchanged. Deliberately no second implementation of the
seam logic: a backfill that computed continuity its own way could silently
disagree with what new ingests produce, which is exactly the class of bug this
plan exists to remove.

Files with no `scene_cuts` row get shot_points=[], which is the identical
degradation the live path already applies when seam_cache lacks a file
(`(seam_cache.get(file_id) or {}).get("shot_points") or []`) -- those cuts
still get cut_no/of and gap-based weld verdicts, just without shot-break
detection. Honest absence, same as production.

Idempotent: re-running recomputes and overwrites, never appends.

    python scripts/backfill_continuity.py --dry-run     # report, write nothing
    python scripts/backfill_continuity.py --limit 1     # smoke-test one run
    python scripts/backfill_continuity.py               # all runs
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

sys.path.insert(0, ".")

from app.services import db  # noqa: E402
from app.services.vcut import continuity as vcut_continuity  # noqa: E402


def _runs() -> List[str]:
    with db.connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "select distinct ingest_run_id::text from public.cut_records "
            "where ingest_run_id is not null order by 1"
        )
        return [r[0] for r in cur.fetchall()]


def _seam_cache_for_run(run_id: str) -> Dict[str, dict]:
    """Rebuild the shot_points half of the run's seam_cache from scene_cuts.

    Keys are file_id TEXT to match what cuts_read.rows_for_run returns
    (`file_id::text`), since write_continuity_for_run groups by that value and
    then looks it up in this dict -- a uuid-keyed dict would miss every file
    and silently produce continuity with no shot breaks.
    """
    with db.connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "select s.file_id::text, s.shot_points from public.scene_cuts s "
            "where s.file_id in ("
            "  select distinct c.file_id from public.cut_records c "
            "  where c.ingest_run_id = %s)",
            (run_id,),
        )
        return {fid: {"shot_points": pts or []} for fid, pts in cur.fetchall()}


def _stats() -> tuple[int, int]:
    with db.connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "select count(*), count(*) filter (where continuity::text <> '{}') "
            "from public.cut_records"
        )
        return tuple(cur.fetchone())  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N runs")
    args = ap.parse_args()

    total_before, with_before = _stats()
    runs = _runs()
    if args.limit:
        runs = runs[: args.limit]
    print(f"cut_records: {total_before} rows, {with_before} already have continuity")
    print(f"runs to process: {len(runs)}\n")

    written = 0
    no_scene: List[str] = []
    t0 = time.time()
    for i, run_id in enumerate(runs, 1):
        seam_cache = _seam_cache_for_run(run_id)
        if not seam_cache:
            no_scene.append(run_id)
        if args.dry_run:
            print(f"[{i}/{len(runs)}] {run_id}  files_with_shot_points="
                  f"{len(seam_cache)}  (dry run, nothing written)")
            continue
        n = vcut_continuity.write_continuity_for_run(run_id, seam_cache)
        written += n
        print(f"[{i}/{len(runs)}] {run_id}  files_with_shot_points="
              f"{len(seam_cache)}  rows_written={n}")

    elapsed = time.time() - t0
    print(f"\n{'DRY RUN -- ' if args.dry_run else ''}rows written: {written} "
          f"in {elapsed:.1f}s")
    if no_scene:
        print(f"runs whose files had NO scene_cuts row: {len(no_scene)} "
              f"(cuts still get cut_no/of + gap-based welds, no shot breaks)")
    if not args.dry_run:
        total_after, with_after = _stats()
        print(f"cut_records: {total_after} rows, {with_after} now have continuity "
              f"(was {with_before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
