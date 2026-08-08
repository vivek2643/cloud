"""Read-only: for each project re-ingested today that has speech, compare the
NEW run's speech cut spans to the PRIOR run's -- ignoring take_group_id (always
a fresh uuid) -- to see whether the boundaries actually moved or the runs are
basically the same cuts with new ids.

For each project prints: today#, prior#, exact-span matches (file_id+in+out
within +/-100ms), and how many are genuinely new/changed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import db  # noqa: E402

TOL_MS = 100


def _spans(conn, run_id):
    return conn.execute(
        "select file_id::text, src_in_ms, src_out_ms from cut_records "
        "where ingest_run_id=%s and kind='speech' order by file_id, src_in_ms",
        (run_id,),
    ).fetchall()


def _match_count(today, prior):
    """Greedy: how many of today's spans have a prior span on the same file
    within TOL_MS on both edges."""
    used = [False] * len(prior)
    hits = 0
    for f, a, b in today:
        for j, (pf, pa, pb) in enumerate(prior):
            if used[j] or pf != f:
                continue
            if abs(pa - a) <= TOL_MS and abs(pb - b) <= TOL_MS:
                used[j] = True
                hits += 1
                break
    return hits


def main():
    with db.connection() as conn:
        today_runs = conn.execute(
            """
            select distinct on (project_id) project_id::text, id::text
              from ingest_runs where created_at > current_date
             order by project_id, created_at desc
            """
        ).fetchall()

        print(f"{'project':10} {'today':>5} {'prior':>5} {'match':>5} {'moved/new':>9}  note")
        for pid, rid in today_runs:
            today = _spans(conn, rid)
            if not today:
                continue
            prior = conn.execute(
                "select id::text from ingest_runs where project_id=%s and id!=%s "
                "order by created_at desc limit 1", (pid, rid)
            ).fetchone()
            if not prior:
                print(f"{pid[:8]:10} {len(today):5d} {'-':>5} {'-':>5} {'-':>9}  no prior run (all new)")
                continue
            prior_spans = _spans(conn, prior[0])
            hits = _match_count(today, prior_spans)
            note = "identical boundaries" if hits == len(today) == len(prior_spans) else \
                   ("mostly same" if hits >= 0.7 * max(len(today), 1) else "substantially re-cut")
            print(f"{pid[:8]:10} {len(today):5d} {len(prior_spans):5d} {hits:5d} "
                  f"{len(today)-hits:9d}  {note}")


if __name__ == "__main__":
    main()
