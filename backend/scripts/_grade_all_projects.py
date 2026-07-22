"""Run run_grade_job (grade_pipeline_standardize.plan.md: the pipeline has
no more dev flags -- every capability is always on) for the current edit
thread of every project. Required once after the standardization deploy
(INPUT_HASH_SCHEMA_VERSION bumped, so every stored grade is stale until
this reruns). Idempotent: re-running just re-grades."""
import psycopg  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services.l3 import store as edit_store  # noqa: E402
from app.services.l3.grade import job as grade_job  # noqa: E402

s = get_settings()

with psycopg.connect(s.database_url, autocommit=True) as c:
    projects = c.execute("select id::text, source_file_ids from projects").fetchall()
    # folder names for readability
    folders = c.execute("select id::text, name from folders").fetchall()
    file_folder = {r[0]: {} for r in []}
    fname = {}
    frows = c.execute("select id::text, folder_id::text from files").fetchall()
    f2folder = {r[0]: r[1] for r in frows}
    folder_name = {r[0]: r[1] for r in folders}

    def label_for(fids):
        names = {folder_name.get(f2folder.get(f)) for f in fids}
        names.discard(None)
        return ", ".join(sorted(n for n in names if n)) or "?"

    def latest_thread(fids):
        rows = c.execute(
            "select id::text from edit_threads where file_ids && %s::uuid[] order by updated_at desc",
            (list(fids),),
        ).fetchall()
        return [r[0] for r in rows]

results = []
for pid, sfids in projects:
    fids = [str(x) for x in (sfids or [])]
    if not fids:
        continue
    name = label_for(fids)
    with psycopg.connect(s.database_url, autocommit=True) as c2:
        pass
    tids = None
    with psycopg.connect(s.database_url, autocommit=True) as c:
        tids = [r[0] for r in c.execute(
            "select id::text from edit_threads where file_ids && %s::uuid[] order by updated_at desc",
            (fids,)).fetchall()]
    chosen = None
    for tid in tids:
        try:
            doc, _ = edit_store.latest_document(tid)
        except Exception:
            doc = None
        if doc and (doc.get("timeline") or doc.get("operations")):
            chosen = (tid, doc)
            break
    if not chosen:
        results.append((name, pid[:8], None, 0, "no-doc", 0, 0))
        print(f"[skip] {name} ({pid[:8]}): no gradeable thread")
        continue
    tid, doc = chosen
    shots = grade_job.ordered_shots(doc)
    print(f"[run ] {name} ({pid[:8]}) thread={tid[:8]} shots={len(shots)} ...", flush=True)
    fn = getattr(grade_job.run_grade_job, "func", grade_job.run_grade_job)
    try:
        fn(tid)
    except Exception as e:
        results.append((name, pid[:8], tid[:8], len(shots), f"EXC:{e}"[:40], 0, 0))
        print(f"       !! exception: {e}")
        continue
    st = grade_job.get_job_state(tid) or {}
    with psycopg.connect(s.database_url, autocommit=True) as c:
        baked = c.execute("select count(*), count(cube_ref) from resolved_grades where thread_id=%s and input_hash=%s",
                          (tid, st.get("input_hash"))).fetchone()
    results.append((name, pid[:8], tid[:8], len(shots), st.get("state"), baked[0], baked[1]))
    print(f"       -> {st.get('state')} rows={baked[0]} baked={baked[1]}", flush=True)

print("\n==== SUMMARY ====")
print(f"{'project':22} {'proj':8} {'thread':8} {'shots':>5} {'state':>8} {'rows':>5} {'baked':>5}")
for name, pid, tid, n, state, rows, baked in results:
    print(f"{name[:22]:22} {pid:8} {str(tid):8} {n:5} {str(state):>8} {rows:5} {baked:5}")
