"""Read-only: for every project whose LATEST ingest_run is from today's
re-ingest, decide whether its speech cuts came from the NEW pipeline
(run_speech_channel) or the fail-open FALLBACK (copy_prior_speech_cuts).

Discriminators:
- copy_prior copies a prior run's speech rows verbatim and leaves
  scene_specifics NULL -> if any speech cut in the run has scene_specifics,
  the new pipeline ran.
- copy_prior reproduces the prior run's exact (file_id, src_in_ms, src_out_ms,
  take_group_id) set -> if today's set == the prior run's set, it's a copy;
  if it differs (or there is no prior run), it was freshly generated.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import db  # noqa: E402


def _digest(conn, run_id):
    rows = conn.execute(
        """
        select file_id::text, src_in_ms, src_out_ms, coalesce(take_group_id::text,'')
          from cut_records where ingest_run_id = %s and kind = 'speech'
         order by file_id, src_in_ms, src_out_ms
        """,
        (run_id,),
    ).fetchall()
    return tuple(rows)


def main():
    with db.connection() as conn:
        runs = conn.execute(
            """
            select distinct on (project_id)
                   project_id::text, id::text, created_at
              from ingest_runs
             order by project_id, created_at desc
            """
        ).fetchall()
        # keep only runs created today (the re-ingest)
        today_runs = conn.execute(
            """
            select distinct on (project_id) project_id::text, id::text
              from ingest_runs
             where created_at > current_date
             order by project_id, created_at desc
            """
        ).fetchall()

        print(f"{'project':10} {'speech':>6} {'spec':>5} {'outlk':>5} {'win':>4} {'prior':>6}  verdict")
        n_fresh = n_copy = n_nospeech = 0
        for pid, rid in today_runs:
            n_speech = conn.execute(
                "select count(*) from cut_records where ingest_run_id=%s and kind='speech'", (rid,)
            ).fetchone()[0]
            n_spec = conn.execute(
                "select count(*) from cut_records where ingest_run_id=%s and kind='speech' "
                "and scene_specifics is not null", (rid,)
            ).fetchone()[0]
            n_outlook = conn.execute(
                "select count(*) from cut_records where ingest_run_id=%s and kind='speech' "
                "and take_role='outlook'", (rid,)
            ).fetchone()[0]
            n_win = conn.execute(
                "select count(*) from cut_records where ingest_run_id=%s and kind='speech' "
                "and take_role='winner'", (rid,)
            ).fetchone()[0]

            prior = conn.execute(
                "select id::text from ingest_runs where project_id=%s and id!=%s "
                "order by created_at desc limit 1", (pid, rid)
            ).fetchone()
            prior_id = prior[0] if prior else None

            if n_speech == 0:
                verdict = "no-speech"
                n_nospeech += 1
            else:
                today_dig = _digest(conn, rid)
                if prior_id is None:
                    verdict = "FRESH (no prior to copy from)"
                    n_fresh += 1
                else:
                    prior_dig = _digest(conn, prior_id)
                    if today_dig == prior_dig and prior_dig:
                        verdict = "COPIED (== prior run)"
                        n_copy += 1
                    else:
                        verdict = "FRESH (differs from prior)"
                        n_fresh += 1

            print(f"{pid[:8]:10} {n_speech:6d} {n_spec:5d} {n_outlook:5d} {n_win:4d} "
                  f"{(prior_id[:6] if prior_id else '-'):>6}  {verdict}")

        print(f"\nSUMMARY: fresh(new pipeline)={n_fresh}  copied(fallback)={n_copy}  "
              f"no-speech={n_nospeech}  total_today={len(today_runs)}")


if __name__ == "__main__":
    main()
