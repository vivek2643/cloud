"""Grade validation (color_grading_upgrade.plan.md Finish step;
grade_pipeline_standardize.plan.md: the pipeline is un-flagged now -- there
is no more separate "legacy" path to diff against, so the "legacy" column
below is just a second resolve_clip_grade call over whole-file stats,
kept as a sanity cross-check against the job's per-SPAN result).

Runs run_grade_job synchronously for one thread, then verifies the job
completed, resolved_grades populated + cubes baked.

Writes to resolved_grades/grade_jobs. Safe, reversible (re-running just
re-grades)."""
import sys

import psycopg  # noqa: E402
from app.config import get_settings  # noqa: E402

s = get_settings()

from app.services.l3 import store as edit_store  # noqa: E402
from app.services.l3.grade import job as grade_job  # noqa: E402
from app.services.l3.grade.resolver import resolve_clip_grade  # noqa: E402

thread_id = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else thread_id[:8]

doc, version = edit_store.latest_document(thread_id)
if not doc:
    print(f"{label}: no document for thread")
    sys.exit(1)
shots = grade_job.ordered_shots(doc)
print(f"{label}: thread={thread_id} version={version} shots={len(shots)}")

# --- run the real job synchronously (call underlying func) ---
fn = getattr(grade_job.run_grade_job, "func", grade_job.run_grade_job)
print(f"{label}: running run_grade_job ...")
fn(thread_id)

# --- verify status + persisted rows ---
st = grade_job.get_job_state(thread_id)
print(f"{label}: job state = {st}")
rows = grade_job.fetch_latest_grades(thread_id, [sh.key for sh in shots])
print(f"{label}: resolved_grades rows = {len(rows)} / {len(shots)} shots")
with psycopg.connect(s.database_url, autocommit=True) as c:
    baked = c.execute(
        "select count(*), count(cube_ref) from resolved_grades where thread_id=%s and input_hash=%s",
        (thread_id, st.get("input_hash") if st else None),
    ).fetchone()
    print(f"{label}: rows for current hash = {baked[0]}, with baked cube = {baked[1]}")

# --- job's per-span result vs a whole-file cross-check, per shot ---
def cdl_of(g):
    return g.get("cdl") if g else None

def mag(cdl):
    if not cdl:
        return 0.0
    sl = cdl.get("slope", [1, 1, 1]); of = cdl.get("offset", [0, 0, 0])
    po = cdl.get("power", [1, 1, 1]); sat = cdl.get("sat", 1.0)
    return (sum(abs(x - 1) for x in sl) + sum(abs(x) for x in of)
            + sum(abs(x - 1) for x in po) + abs(sat - 1))

from app.services.l3.grade.measure import fetch_color_stats  # noqa: E402
cstats = fetch_color_stats(list({sh.file_id for sh in shots}))
print(f"\n{'shot':10} {'job_mag':>8} {'wholefile_mag':>13} {'identity?':>10}")
nontrivial = 0
for sh in shots[:12]:
    job_cdl = cdl_of(rows.get(sh.key))
    whole_file = cdl_of(resolve_clip_grade(sh.item, color_stats=cstats.get(sh.file_id), sequence_look=doc.get("look")))
    jm, wm = mag(job_cdl), mag(whole_file)
    if jm > 1e-6:
        nontrivial += 1
    print(f"{sh.key[:10]:10} {jm:8.4f} {wm:13.4f} {'yes' if jm<1e-6 else 'no':>10}")
print(f"\n{label}: {nontrivial}/{min(len(shots),12)} shown shots have a non-identity grade")
