"""Read-only post-re-ingest verification (no model calls, no writes):

1. ENERGY DIAL across cuts -- for each project's latest run, re-resolve the
   persisted flags plan at energy 0.0/0.35/0.7/1.0 (exactly what POST
   /cuts/energy does, minus the HTTP + DB write) and report cut count + median
   length. Asserts the dial's contract: as energy rises, cut COUNT is
   non-decreasing and median LENGTH is non-increasing (loose->tight).

2. PASS 2 specifics -- for each latest run, how many video cuts carry
   scene_specifics, with a few concrete examples (summary + specifics) so we
   can eyeball whether the analysis makes sense.
"""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import db  # noqa: E402
from app.services.vcut import resolve as rv, store  # noqa: E402

_ENERGIES = [0.0, 0.35, 0.7, 1.0]


def _latest_runs():
    with db.connection() as conn:
        return conn.execute(
            """
            select distinct on (project_id) project_id::text, id::text
              from ingest_runs
             where created_at > current_date
             order by project_id, created_at desc
            """
        ).fetchall()


def _median_len(cuts):
    if not cuts:
        return 0
    return int(statistics.median((c.out_ms - c.in_ms) for c in cuts))


def energy_dial(runs):
    print("\n===== ENERGY DIAL (re-resolve persisted flags) =====")
    print(f"{'project':10} " + "  ".join(f"e={e:<4}" for e in _ENERGIES) + "   monotonic?")
    mono_ok = mono_bad = noplan = 0
    for pid, rid in runs:
        seam, plan_dict = store.load_seam_and_plan(rid)
        if not plan_dict or not seam:
            noplan += 1
            continue
        plan = rv.MomentPlan.from_dict(plan_dict)
        cols = []
        counts, medians = [], []
        for e in _ENERGIES:
            cuts = rv.resolve_cuts(plan, seam, energy=e)
            counts.append(len(cuts))
            medians.append(_median_len(cuts))
            cols.append(f"{len(cuts):>2}/{_median_len(cuts)//1000}s")
        count_ok = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
        len_ok = all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))
        ok = count_ok and len_ok
        mono_ok += ok
        mono_bad += (not ok)
        verdict = "ok" if ok else f"VIOLATED (count_ok={count_ok} len_ok={len_ok})"
        print(f"{pid[:8]:10} " + "  ".join(f"{c:<6}" for c in cols) + f"   {verdict}")
    print(f"\n  monotonic: {mono_ok} ok, {mono_bad} violated, {noplan} without a persisted plan")
    print("  (each cell = #cuts / median-cut-seconds at that energy; "
          "count should rise & length shrink as energy increases)")


def specifics(runs):
    print("\n===== PASS 2 scene_specifics (video cuts) =====")
    with db.connection_dict_row() as conn:
        total_v = total_spec = 0
        examples = []
        for pid, rid in runs:
            rows = conn.execute(
                """
                select label, summary, scene_specifics
                  from cut_records
                 where ingest_run_id = %s and kind = 'video'
                 order by src_in_ms
                """,
                (rid,),
            ).fetchall()
            v = len(rows)
            s = sum(1 for r in rows if r["scene_specifics"])
            total_v += v
            total_spec += s
            for r in rows:
                if r["scene_specifics"] and len(examples) < 12:
                    examples.append((pid[:8], r["summary"] or r["label"], r["scene_specifics"]))
        pct = (100.0 * total_spec / total_v) if total_v else 0.0
        print(f"video cuts: {total_v}   with scene_specifics: {total_spec} ({pct:.0f}%)\n")
        print("-- examples (project | summary | scene_specifics) --")
        for proj, summ, spec in examples:
            print(f"  {proj}  {(summ or '')[:38]:38}  {spec}")


def main():
    runs = _latest_runs()
    print(f"latest-today runs: {len(runs)} project(s)")
    energy_dial(runs)
    specifics(runs)


if __name__ == "__main__":
    main()
