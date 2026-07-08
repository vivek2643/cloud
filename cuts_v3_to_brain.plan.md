# Plan: Wire Cuts v3 (`cut_records`) into the Brain (agentic editor)

## Goal
Make the agentic editor ("brain") read its footage/beat context from the **Cuts v3
`cut_records`** table (the current source of truth the UI already uses) instead of the
legacy **hero-cut** substrate (`hero_cuts_cache` / `footage_trees`). After this, the
beat the brain places is *the same beat the editor sees in the Cuts tab* — one cut
universe, end to end.

## Current state (why this is needed)
The agent and Cuts v3 are **parallel pipelines** today. The agent never reads
`cut_records`.

- Agent turn: `POST /api/edit/threads/{id}/messages` → `converse.respond()` →
  `observe.build_context()` → `footage_map.assemble_map()` → **`hero_store.get_anchor_cuts()`**
  (reads `hero_cuts_cache`, band 2). The LLM loop (`tools.run_edit_loop`) places beats by
  `(moment_ref, level)` resolved through `arrange._MapIndex`.
- Cuts v3: `POST /api/projects/{id}/ingest` → LLM pass1/pass2a/pass2b → `post.assemble_cut_records()`
  → `cut_records` rows. Read for UI via `cuts_v3_read.py` / `GET /api/projects/{id}/cuts-v3`.

Key files:
- `backend/app/services/l3/footage_map.py` — `assemble_map`, `get_trees`, `_build_trees`, `build_clip_tree`
- `backend/app/services/l3/arrange.py` — `_MapIndex.resolve()` (the `(ref, level)` contract)
- `backend/app/services/l3/observe.py` — `build_context()` (single DB-touch for a turn)
- `backend/app/services/l3/hero_store.py` — `get_anchor_cuts()`, `signatures_for()` (to be bypassed)
- `backend/app/services/l3/cuts_v3_read.py` — the `cut_records` reader (the model to mirror)
- `backend/app/services/l3/store.py` — edit threads store `file_ids` only (no `project_id`)

## The two real design forks

### Fork A — the `(ref, level)` variant contract
`_MapIndex.resolve()` (arrange.py:135–162) expects each moment to carry a **ladder of
zoom/tightness variants** keyed by level name (`broad/calm/balanced/tight/sharp`); `place`
and `tighten` pick a level. `cut_records` have **no ladder** — they carry one span
(`src_in_ms`,`src_out_ms`), a `hero_ts_ms` anchor, and `pace` bounds
(`min_ms`/`natural_ms`/`max_ms`, `levels[5]`). The frontend energy dial synthesizes
tightness client-side via **anchor-protected negative padding** toward `hero_ts_ms`.

**DECISION (LOCKED) — Fork A: synthesize the ladder deterministically in code** from each
`cut_record`, mirroring the frontend dial math — inset the span symmetrically toward
`hero_ts_ms`, clamped to `pace.min_ms`, never trimming past any anchor. Emit 5 rungs mapped
to the existing level names (`broad/calm/balanced/tight/sharp`) so `_MapIndex` / `place` /
`tighten` keep working unchanged. No LLM numbers involved (respects "code owns numbers").
Every cut is presented ladder-style; the flat-single-variant option is **rejected**.

### Fork B — thread ↔ project linkage
Edit threads are **file-scoped** (`store` keeps `file_ids`); `cut_records` are
**project/ingest-run-scoped**. Need a resolver: **given the thread's `file_ids`, find the
latest `ingest_run` whose `cut_records` cover them** (join `cut_records.file_id` → pick the
most recent `ingest_run_id` that has rows for those files). Encapsulate as
`cuts_v3_read.latest_run_for_files(file_ids) -> ingest_run_id | None`.

## Design principles (carry over from the rest of the project)
- **Code owns numbers, LLM owns categories.** The ladder synthesis and any thresholds are
  derived from each clip's own signals (span, `hero_ts_ms`, `pace`), never hardcoded.
- **No fallbacks that silently mask a miss.** If a file has no `cut_records`, it is simply
  absent from the map (same fail-open contract `get_trees` already uses) — do not fabricate.
- **Preserve the moment contract.** Keep `moment_id` shape (`{fid8}:m{idx}`), `variants`,
  `channel`, `gist`, `run_id`, so `converse`/`tools`/`act` stay byte-stable.

---

## Implementation

### Phase 1 — `cut_records` → clip-tree projection (core)
New module `backend/app/services/l3/cutrecord_map.py`:
- `def cut_dicts_for_files(file_ids) -> dict[file_id, list[cut_dict]]`
  - Resolve run via `latest_run_for_files`, read rows with `cuts_v3_read` (or a thin shared query).
  - Drop `junk == true` rows (they are hidden in the UI; keep `junk_reason` available if needed later).
  - Map each `cut_record` → the **cut dict shape `build_clip_tree` already consumes**
    (footage_map.py:243–288): `hero_id`←`id`, `channel`, `subject`←derive from `channel`/`kind`
    (`said`→person, `shown`→object/place, `done`→person), `summary`, `speaker`, `label`→`gist`,
    `flags`←`[]` (+ `"muted"` when `pace.natural_sound` false for video), `audio`, `mute`←video
    default-mute rule, `score`←derive from duration/anchor (deterministic), `people`, `framing`,
    `quality`←from `look`, and a synthesized **`ladder`** (Fork A).
  - `take_group_id` / `take_role` carried through for Phase 3.
- `def synth_ladder(cut) -> list[rung]` — 5 rungs via anchor-protected negative padding toward
  `hero_ts_ms`, clamped to `pace.min_ms`; speech cuts stay full-span (their words are the point),
  matching current dial behavior. Each rung: `{level, in_ms, out_ms, play_ms, keep_spans?}`.
  For speech, thread `pace.remove_spans` into `keep_spans` so filler/gap trims survive placement.

### Phase 2 — swap the tree source behind a flag
In `footage_map._build_trees()` (footage_map.py:399–412), source cuts from
`cutrecord_map.cut_dicts_for_files()` instead of `hero_store.get_anchor_cuts()`, gated by a
config flag `settings.footage_source` (`"cut_records"` default, `"hero"` legacy). Reuse
`build_clip_tree` unchanged so `_assign_runs`, variants, gist suppression all keep working.
- Update `get_trees` cache signature: key the `footage_trees` cache on the **ingest_run_id +
  a cut_records content hash** when in `cut_records` mode (so re-ingest busts the tree cache),
  instead of `hero_store.signatures_for`.
- `observe.build_context` needs no change (it just consumes the struct).

### Phase 3 — take groups from `cut_records` (replace recompute)
In `footage_map._annotate_dups()` (688–765), when in `cut_records` mode, build `dup_groups`
from persisted `take_group_id` / `take_role` (winner/take/outlook) instead of
`takes.build_take_groups()` token-overlap recompute. One winner per group is already enforced
in `post._enforce_one_winner_per_take_group`, so this is a direct read.

### Phase 4 — continuous source (`source_awareness`)  [SUPERSEDED → cuts_v3_continuity.plan.md]
Originally: reconcile the "CONTINUOUS SOURCE" digest (still hero-cut/atom lineage via
`clip_timeline_store`) with the v3 lattice. **Decision changed:** rather than migrate the raw
whole-clip view, we are **removing** it from the brain and replacing it with a deterministic
per-cut **continuity** parameter (clip + cut number + weldable-neighbor flag via
`seam.classify_seam`), keeping junk visible-but-labeled. See `cuts_v3_continuity.plan.md`.

### Phase 5 — thread creation carries the run  [IMPLEMENTED]
Store `ingest_run_id` on the thread at `auto_edit.start_thread` (resolve once via
`latest_run_for_files`) so the map is pinned to the run the editor was looking at, rather than
re-resolving "latest" each turn — a re-ingest mid-thread can no longer swap the beat universe
under an active edit (`moment_id`s are positional). Additive nullable column; null → resolve live
(older threads / pre-ingest projects behave exactly as before).

Delivered:
- `migrations/028_edit_thread_ingest_run.sql` — nullable `ingest_run_id` on `edit_threads` (no FK).
- `store.create_thread(..., ingest_run_id=None)` persists it; `store.get_thread` returns it.
- `auto_edit.start_thread` resolves the covering run at creation (cut_records mode only; fail-open).
- The pinned run threads through `converse.respond` → `observe.build_context(run_id=)` →
  `footage_map.assemble_map/get_trees/_build_trees/_source_signatures` →
  `cutrecord_map.cut_dicts_for_files/signatures_for`. The effective run is resolved **once** in
  `build_context` and carried on `EditContext.run_id`, so the map struct and the re-assembled
  BEAT INDEX text (`converse._assemble_source_context`) read the same snapshot within a turn.
- Tests: `test_cutrecord_map.py` — pin honored without live-resolve; unpinned falls back to latest.
  Live-verified on Reel 4: pin round-trips through the store and pinning an older run (48 moments)
  vs latest (51) proves the pin is honored, not a no-op.

---

## Files to touch
- **New:** `backend/app/services/l3/cutrecord_map.py` (projection + ladder synth + run resolver, or put resolver in `cuts_v3_read.py`).
- `backend/app/services/l3/footage_map.py` — `_build_trees` source swap; `get_trees` cache-key change.
- `backend/app/services/l3/cuts_v3_read.py` — add `latest_run_for_files()` + a reusable row fetch.
- `backend/app/config.py` — add `footage_source` flag (default `cut_records`).
- (Phase 5) `backend/app/services/l3/store.py` + `auto_edit.py` — optional `ingest_run_id` on thread.
- **No change** to `converse.py`, `tools.py`, `act.py`, `arrange.py` (contract preserved).

## Testing / verification
- Unit: `cutrecord_map` — projection maps a known `cut_record` row to the exact dict keys
  `build_clip_tree` reads; `synth_ladder` respects `pace.min_ms`, never trims past `hero_ts_ms`,
  and leaves speech full-span; `latest_run_for_files` picks the newest covering run.
- Golden: for one project (Reel 4, `project a294f9da…`, run `3f2b8b41…`, 54 cuts) assert
  `assemble_map(file_ids)` in `cut_records` mode yields one moment per non-junk cut, with
  `channel`/`gist`/`variants` populated and `dup_groups` matching persisted take groups.
- Loop smoke: run `converse.respond` on a thread over those files; confirm `read_state` /
  `place` / `tighten` resolve refs (no `_MapIndex` misses) and the placed span equals the
  cut's balanced-rung span.
- Flag flip: `footage_source="hero"` reproduces today's behavior exactly (safety switch).

## Rollout
1. Land Phases 1–3 behind `footage_source` (default `cut_records`), keep `hero` path intact.
2. Verify on Reel 4 + one podcast/multi-clip project (take groups exercised).
3. Phase 4/5 as follow-ups. Once stable, deprecate hero-cut precompute for the agent path.

## Open questions for the executor
- Ladder synthesis: **DECIDED — Fork A (5 synthesized rungs), ladder-style for every cut.**
  Mirror the frontend dial math (anchor-protected negative padding toward `hero_ts_ms`,
  clamped to `pace.min_ms`) so the brain's tightening matches what the editor sees.
- `subject` tag: `cut_records` has no explicit `subject`; derive from `channel`+`kind`, or add a
  lightweight `subject` to the ingest LLM output later. For v1, derive deterministically.
- Whether to also retire the `HeroCutsView`/hero-cut endpoints once the agent no longer needs them.
