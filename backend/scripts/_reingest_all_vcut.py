"""One-off: re-ingest EVERY project on the NEW vcut cuts pipeline, synchronously
(no procrastinate workers). Per project: run_vcut_ingest (spans -> seam -> Pass1
-> resolve -> video+speech cut_records -> 'ready'). The latest ingest_run wins,
so the frontend shows these cuts immediately.

Pass 2 (scene_specifics): in VIDEO mode, run_vcut_ingest already runs the
question planner + Pass-2 enrich INLINE on the still-warm per-file video cache
(vcut_pass2_video_specifics.plan.md section 6.1), so this script must NOT call
run_enrich again -- doing so would run the FRAMES-mode hero-still enrich and
overwrite the good cached-video specifics. Only in FRAMES mode (input mode !=
'video') does this script call run_enrich itself (the deferred vcut_enrich task
is otherwise bypassed by running synchronously here).

SPENDS REAL MONEY: one Gemini Pass-1 vision + one speech Gemini-pro text + one
Pass-2 enrich (cache-discounted) per project. Sequential + per-project retry so a
transient API overload never aborts the whole batch. Best-effort enrich: a Pass-2
failure never hides the already-visible cuts.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402
from app.services import db  # noqa: E402
from app.services.vcut import pass2 as p2  # noqa: E402
from app.services.vcut.orchestrate import run_vcut_ingest  # noqa: E402

_RETRIES = 3
_COOLDOWN_S = 60.0


def _project_ids():
    with db.connection() as conn:
        rows = conn.execute(
            """
            select p.id::text
              from projects p
             where exists (
                     select 1 from files f
                      where f.id = any(p.source_file_ids) and f.file_type = 'video'
                   )
             order by p.created_at desc nulls last
            """
        ).fetchall()
    return [r[0] for r in rows]


def _counts(rid):
    with db.connection() as conn:
        return conn.execute(
            """
            select count(*),
                   count(*) filter (where kind = 'video'),
                   count(*) filter (where kind = 'speech'),
                   count(*) filter (where take_role = 'outlook'),
                   count(*) filter (where take_role = 'winner')
              from cut_records where ingest_run_id = %s
            """,
            (rid,),
        ).fetchone()


def _speech_source(rid):
    """vcut_moment_energy.plan.md section 8/12.3: which speech path this
    run actually took ("pipeline" | "copy_prior" | None for a pre-migration
    row) -- watched here so a batch re-ingest reports how many projects'
    speech channel silently fell back."""
    with db.connection_dict_row() as conn:
        row = conn.execute(
            "select speech_channel_status from ingest_runs where id = %s", (rid,)
        ).fetchone()
    status = row["speech_channel_status"] if row else None
    return (status or {}).get("source")


def _run_one(pid):
    last = None
    for attempt in range(_RETRIES + 1):
        try:
            rid = run_vcut_ingest(pid)
            # VIDEO mode already enriched inline inside run_vcut_ingest -- only
            # the FRAMES-mode path needs an explicit run_enrich here (see module
            # docstring: a second call in video mode would overwrite the good
            # cached-video specifics with hero-still ones).
            if get_settings().vcut_pass1_input_mode != "video":
                try:
                    p2.run_enrich(pid, rid)
                except Exception as e:  # noqa: BLE001 - enrich is best-effort
                    print(f"    enrich failed for {pid} (cuts still visible): {e!r}", flush=True)
            return rid, _counts(rid)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == _RETRIES:
                raise
            print(f"    RETRY {pid} attempt {attempt + 1}/{_RETRIES} after {exc!r}; "
                  f"cooldown {_COOLDOWN_S:.0f}s", flush=True)
            time.sleep(_COOLDOWN_S)
    raise last  # unreachable


def main():
    # Optional: explicit project ids as CLI args (resume a partial batch without
    # recutting already-done projects); else every project.
    pids = sys.argv[1:] or _project_ids()
    total = len(pids)
    print(f"REINGEST_START total={total}", flush=True)
    ok = fail = 0
    speech_source_counts = {"pipeline": 0, "copy_prior": 0, "none": 0}
    for i, pid in enumerate(pids, 1):
        t0 = time.time()
        try:
            rid, c = _run_one(pid)
            ok += 1
            dt = time.time() - t0
            source = _speech_source(rid) or "none"
            speech_source_counts[source] = speech_source_counts.get(source, 0) + 1
            print(f"PROJECT_DONE [{i}/{total}] {pid} run={rid} "
                  f"total/video/speech/outlook/winner={c} speech_source={source} {dt:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"PROJECT_FAIL [{i}/{total}] {pid} error={e!r}", flush=True)
    print(f"REINGEST_DONE ok={ok} fail={fail} total={total} "
          f"speech_pipeline={speech_source_counts['pipeline']} "
          f"speech_copy_prior={speech_source_counts['copy_prior']} "
          f"speech_none={speech_source_counts['none']}", flush=True)


if __name__ == "__main__":
    main()
