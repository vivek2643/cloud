"""Read-only post-re-ingest onboarding check: proves EVERY project's latest run
is on the single NEW cuts pipeline -- video (non-speech) + new speech (with the
noise gate) -- and that nothing fell back.

Per project's latest run it reports:
  SPEECH  : speech_channel_status.source (want 'pipeline'; flags copy_prior/failed/none)
  GATE    : # speech cuts marked junk by the noise gate, split non_speech_llm /
            non_speech_energy (proves the gate actually fired, not just shipped)
  VIDEO   : # video cuts + how many carry Pass-2 scene_specifics
  ENERGY  : re-resolves the persisted flags plan at 0/0.35/0.7/1.0 and asserts
            the dial's contract (cut COUNT non-decreasing, median LENGTH
            non-increasing as energy rises)

No model calls, no writes, no money.
"""
import os
import statistics
import sys

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
             order by project_id, created_at desc
            """
        ).fetchall()


def _speech_source(conn, rid):
    row = conn.execute(
        "select speech_channel_status from ingest_runs where id = %s", (rid,)
    ).fetchone()
    return ((row[0] if row else None) or {}).get("source")


def _counts(conn, rid):
    return conn.execute(
        """
        select
          count(*) filter (where kind='video') as video,
          count(*) filter (where kind='video' and scene_specifics is not null) as video_spec,
          count(*) filter (where kind='speech') as speech,
          count(*) filter (where kind='speech' and junk) as speech_junk,
          count(*) filter (where kind='speech' and junk_reason='non_speech_llm') as junk_llm,
          count(*) filter (where kind='speech' and junk_reason='non_speech_energy') as junk_energy,
          count(*) filter (where take_role='outlook') as outlook,
          count(*) filter (where take_role='winner') as winner
        from cut_records where ingest_run_id=%s
        """,
        (rid,),
    ).fetchone()


def _median_len(cuts):
    return int(statistics.median((c.out_ms - c.in_ms) for c in cuts)) if cuts else 0


def _energy_dial(rid):
    seam, plan_dict = store.load_seam_and_plan(rid)
    if not plan_dict or not seam:
        return None
    plan = rv.MomentPlan.from_dict(plan_dict)
    counts, medians = [], []
    for e in _ENERGIES:
        cuts = rv.resolve_cuts(plan, seam, energy=e)
        counts.append(len(cuts))
        medians.append(_median_len(cuts))
    count_ok = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    len_ok = all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))
    return counts, medians, (count_ok and len_ok)


def main():
    runs = _latest_runs()
    print(f"latest run per project: {len(runs)}\n")
    print(f"{'project':9} {'speech_src':11} {'video/spec':10} {'speech':7} "
          f"{'junk(llm/en)':13} {'out/win':8} {'energy dial (cnt@0..1)':24} mono")
    bad_speech = no_gate = mono_bad = noplan = 0
    for pid, rid in runs:
        with db.connection() as conn:
            src = _speech_source(conn, rid) or "none"
            c = _counts(conn, rid)
        video, video_spec, speech, sjunk, jllm, jen, outlook, winner = c
        dial = _energy_dial(rid)
        if dial is None:
            noplan += 1
            dial_str, mono = "(no plan)", "-"
        else:
            counts, medians, ok = dial
            dial_str = "/".join(str(x) for x in counts)
            mono = "ok" if ok else "BAD"
            mono_bad += (not ok)
        if src != "pipeline":
            bad_speech += 1
        if sjunk == 0:
            no_gate += 1
        print(f"{pid[:8]:9} {src:11} {f'{video}/{video_spec}':10} {speech:<7} "
              f"{f'{jllm}/{jen}':13} {f'{outlook}/{winner}':8} {dial_str:24} {mono}")

    print(f"\nSUMMARY  projects={len(runs)}")
    print(f"  speech NOT pipeline (copy_prior/failed/none): {bad_speech}  (want 0)")
    print(f"  runs with 0 gated speech cuts               : {no_gate}  "
          f"(informational -- clean audio can legitimately gate nothing)")
    print(f"  energy dial NON-monotonic                   : {mono_bad}  (want 0)")
    print(f"  runs without a persisted flags plan         : {noplan}")


if __name__ == "__main__":
    main()
