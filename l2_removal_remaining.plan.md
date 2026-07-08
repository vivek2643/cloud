# L2 / VLM removal — remaining work (execution doc)

**Goal:** finish removing the L2 / VLM perception layer (`clip_perception` +
Gemini video pass) *completely*. The backend app already imports cleanly with L2
gone; what's left is tests, frontend, the DB migration, env/docs, and the final
verify + commit.

**Context (read first):** Cuts v3 (pass1/pass2a/pass2b) never used L2. Person
characteristics + `shot_size` now come from the Cuts v3 pass-2 image LLM; two
deterministic quality scores (`speech_quality`, `total_quality`) live on
`cut_records`. L2 was a *separate* Gemini pass that wrote the `clip_perception`
table, consumed only by some L3 brain features. All those consumers were
fail-open, so removing L2 degrades (never crashes) the brain.

---

## Already DONE (do NOT redo — verify only if curious)

Backend, committed to the working tree (uncommitted, no DB change yet):

- **Queue re-homed:** `l3_cuts_v3_ingest` moved from `queue="l2"` → `queue="ingest"`
  in `backend/app/services/l3/ingest.py`; `backend/run_workers.sh` now launches an
  `ingest-worker` on `queue=ingest` (env `INGEST_CONCURRENCY`, was `L2_CONCURRENCY`).
- **L2 production stopped:** removed `_enqueue_l2` + its call sites and L2 comments
  from `backend/app/services/l1/pipeline.py`; dropped `from app.services.l2 import
  perception` from `register_tasks()` in `backend/app/services/jobs.py`.
- **Deleted** the whole package `backend/app/services/l2/` (`perception.py`,
  `gemini_video.py`, `prompt.py`, `schema.py`, `__init__.py`) and scripts
  `backend/scripts/rerun_l2.py`, `backend/scripts/rediarize.py`,
  `backend/scripts/test_cast.py`, `backend/scripts/test_relations.py`.
- **Config:** removed the `enable_l2_perception` + all `l2_*` settings from
  `backend/app/config.py` (kept `gemini_api_key`/`gemini_model` — shared LLM backbone).
- **API:** removed `/reanalyze` + `/reanalyze-stale` routes and the `l2_status` /
  `perceiving` phase logic from `backend/app/routers/files.py`; removed `l2_status`
  from `FileResponse` in `backend/app/models/schemas.py`.
- **`clip_perception` consumers:** deleted `cast.py`, `relations.py`, `takes.py`
  (pure L2). Stripped valence from `observe.py` (removed `_valence_by_file`, the
  `valence_by_file` + `relations` fields on `EditContext`, and the relations build
  block) and from `feel.py` (removed `CutFeel.valence`, `dominant_valence`,
  `_valence_shift`, the `valence_by_file` arg). Rewrote `framing.py` to be
  motion-only (removed all perception/`orientation_rotate`/`_load_perceptions`;
  `focus_for_range(action_points, src_in, src_out)` is the new signature). Reduced
  `auto_edit._clip_cards` to file name+duration (no `clip_perception`). Removed the
  `relations`/alias plumbing call from `footage_map.assemble_map` and the
  `clip_perception` read from `footage_map._span_detail` (keeps L1 transcript).
  Removed the `relations=` arg in `converse.py`'s `assemble_map` call.
- **Import smoke PASSED** across `app.services.jobs`, `l1.pipeline`, `routers.files`,
  and every `l3.*` module.

---

## REMAINING TASKS

### 1. Fix the backend test suite

`backend/scripts/test_framing.py` — framing is now **motion-only**. Delete the
Phase-2 perception tests and the perception mocks:
- Delete function `test_focus_priority_and_centers` (it calls
  `framing.focus_for_range(perc, [], ...)` with a perception dict + asserts
  `speaking`/`person` sources that no longer exist).
- Delete function `test_orientation_mapping_and_annotate` (calls the removed
  `framing.orientation_rotate` and monkeypatches the removed `_load_perceptions`).
- In `test_motion_annotate_idempotent`: remove the two lines that monkeypatch
  `framing._load_perceptions` (`orig = framing._load_perceptions` /
  `framing._load_perceptions = lambda fids: {}` and the `finally: framing._load_perceptions = orig`);
  just call `annotate_document` directly (no perception rows to stub anymore).
- Add ONE small motion-only focus test to replace coverage, e.g.:
  ```python
  def test_focus_from_motion_centroid_only():
      from app.services.l3 import framing
      pts = [{"ts_ms": 500, "centroid": [0.2, 0.8]}, {"ts_ms": 1500, "centroid": [0.4, 0.6]}]
      f = framing.focus_for_range(pts, 0, 2000)
      assert f["source"] == "motion" and abs(f["cx"] - 0.3) < 1e-6, f
      assert framing.focus_for_range([], 0, 1000) is None  # nothing -> centered
      print("  OK  focus from motion centroid; empty -> None")
  ```
- Update `main()`: drop the two deleted calls, add `test_focus_from_motion_centroid_only()`.

`backend/scripts/test_feel.py`:
- Delete function `test_valence_dominant_and_shift` (lines ~73-80) and its call in
  `main()` (line ~89). `feel.simulate` no longer accepts `valence_by_file` and
  `FeelReport.dominant_valence` is gone.

`backend/scripts/test_observe_act.py`:
- In `_ctx` (line ~50): remove `valence_by_file={"ffffffff-1111": "tense"}` from the
  `observe.EditContext(...)` call (the field no longer exists).
- Line ~535: remove `valence_by_file={}` from that `observe.EditContext(...)` call.

Then run and confirm green:
```bash
cd backend && for t in test_framing test_feel test_observe_act test_post test_ingest test_pass2 test_pass2b test_footage_map; do echo "=== $t ==="; .venv/bin/python scripts/$t.py 2>&1 | tail -3; done
```

### 2. Frontend

`frontend/src/lib/api.ts`:
- Delete the `l2_status?: ...` line from the `FileRecord` interface (line ~86).
- Delete the two exported functions `reanalyzeFile` and `reanalyzeStale`
  (lines ~123-137, including their doc comments). They are not referenced anywhere
  else (verified by grep) so this is safe.

`frontend/src/components/drive-content.tsx`:
- Line ~79: change the delete-confirm copy `"...all of its L1/L2 analysis..."`
  → `"...all of its analysis..."`.
- (Optional) line ~597-599 comment mentions "perception finishes" — reword to
  "analysis finishes". The generic `analysis_progress`/`analysis_phase` UI stays
  as-is (it just never shows the "perceiving" phase anymore).

Then typecheck:
```bash
cd frontend && npx tsc --noEmit
```

### 3. Migration 032 — drop the table + column

Create `backend/migrations/032_drop_l2_perception.sql`:
```sql
-- =============================================
-- Drop the L2 / VLM perception layer entirely. The Gemini clip_perception pass
-- and its consumers (cast/relations/takes/valence) are removed; Cuts v3 (pass2
-- images) now carries per-cut characteristics/shot_size + quality scores.
-- Idempotent (if exists), additive-safe.
-- =============================================
drop table if exists public.clip_perception;
alter table public.files drop column if exists l2_status;
```
Apply it (idempotent, plain psycopg — same pattern used for migration 031):
```bash
cd backend && .venv/bin/python -c "
import psycopg
from app.config import get_settings
sql = open('migrations/032_drop_l2_perception.sql').read()
with psycopg.connect(get_settings().database_url, autocommit=True) as c:
    c.execute(sql)
print('migration 032 applied')
"
```
> NOTE: the old `characters` table (migration 004) is already dropped. Do not
> touch migrations 003/013 (historical). Nothing reads `clip_perception` after
> the code changes above, so dropping it is safe.

### 4. `.env.example` + stale docs

`.env.example` (repo root): delete the whole `# --- L2: VLM perception (Gemini) ---`
block, lines ~20-32 (`ENABLE_L2_PERCEPTION`, `L2_MAX_DURATION_SECONDS`,
`L2_GEMINI_MODEL`, `L2_VIDEO_FPS`, `L2_MEDIA_RESOLUTION`, `L2_MAX_OUTPUT_TOKENS`,
`L2_FILE_ACTIVE_TIMEOUT_SECONDS`). Keep the `GEMINI_API_KEY`/`GEMINI_MODEL` lines
above it (shared LLM backbone).

(Optional, non-blocking) stale prose mentions of L2/VLM in plan/`docs/*.md`
(`cleanup.plan.md`, `cuts_v2*.plan.md`, `docs/master_strategy.md`,
`docs/color_grading.md`, `framing_transforms.plan.md`, `client_proxy.plan.md`).
These are docs only — safe to leave, or note "L2 removed" if editing.

### 5. Final sweep + verify + commit

Grep for any leftover references (should return nothing but historical migrations
013/003/004 and doc prose):
```bash
cd /Users/vivekgandhari/Documents/cloud && rg -n "clip_perception|enqueue_l2|reenqueue_l2|stale_perception|l2_status|valence_by_file|enable_l2_perception|app\.services\.l2|build_relations|build_cast|build_take_groups" backend/app frontend/src
```
Confirm the backend imports still load:
```bash
cd backend && .venv/bin/python -c "import app.services.jobs, app.services.l1.pipeline, app.routers.files; from app.services.l3 import observe, feel, framing, footage_map, converse, auto_edit; print('ok')"
```
Then commit + push (only after tests + tsc are green):
```bash
cd /Users/vivekgandhari/Documents/cloud && git add -A && git commit -m "Remove L2/VLM perception layer completely" && git push
```

---

## Risk notes (expected, acceptable behaviour changes)

- **Cross-clip person identity** (global `Gx` ids, PIC/SND aliasing across clips)
  is gone. Speakers now read as each clip's own diarized handle. Multicam grouping
  still works via `take_group_id` on `cut_records`.
- **Clip headers** in the footage map lose logline/mood/topics/people; the Cuts v3
  per-cut labels/summaries carry the meaning instead.
- **Reframe focus** is motion-centroid only (no perception regions); per-cut crops
  + rotation come from pass2b `framing` on `cut_records`.
- **Feel** loses the `valence` (emotional-tone) dimension; pace/rhythm/jump-cut
  reads are unchanged.
- Nothing in the running app crashes from any of the above (all were fail-open).
