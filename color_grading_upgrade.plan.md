# Plan: Color grading & lighting — parity, then the semantic leap

## Goal
Take color grading from "mild auto-clean" to (Phase 1) **parity on the six
fundamentals** the pro tools have, (Phase 2) **bounded even-lighting**, (Phase 3)
**the semantic leap only we can do** (subject-, scene-, and story-aware grading),
and (Phase 4) a **small frontend progress indicator** while grading runs.

## Non-negotiable principles
- **Parity math lives in one place and stays parity-safe.** Preview (WebGL, `frontend/src/components/preview/lut-gl.ts` + `grade-cube-client.ts`) and export (ffmpeg `lut3d`, `backend/app/services/render/compositor.py`) BOTH consume the same baked `.cube` from `grade/lut_bake.py`. Any new stage (working-space transform, tone map) must be baked INTO that cube so both sides inherit it automatically. Never apply a grade step on only one side.
- **Never-worse still holds** for the auto layers: bounded, conservative, reversible.
- **Measurement, not description, drives lighting/exposure** — never the Pass-2 visual summary.
- **Switchability:** every new behavior sits behind a settings flag so we can revert to today's stack instantly.
- **Libraries deferred:** design the *seams and stages*; leave the exact color-science library (OCIO/ACES config vs. hand-rolled transfer functions) as a pluggable slot to decide later. Do NOT block on library choice.
- **Grading runs as a background job (decided).** The `v1` grade pass (per-shot span measurement, sequence matching, leveling, LUT pre-bake) is too heavy to run inline on every document resolve for long timelines. It runs as a Procrastinate worker task that computes + persists resolved grades and pre-bakes cubes; `layers.resolve` READS the persisted result instead of computing it. This is what makes Phase 4's progress bar real. Today's inline path stays as the `legacy` fallback.

## Current state (grounded, so steps are concrete)
- `Grade` = ASC CDL (per-channel slope/offset/power + sat scalar), applied by `grade/cdl.py::apply_cdl` directly on 0..1 **display** values. No scene-referred transform, no tone map — `working_space` is a label only.
- Stack `Measure → Correct → Match → Look → Arc → Soft-local → bake`, resolved per clip in `grade/resolver.py::resolve_clip_grade`, called once per spine seg and per `place_video` op inside `layers.py::resolve` (~L613, L682).
- **Correct** (`grade/correct.py`): per-file, never-worse levels stretch (slope cap 1.5) + gray-world/white-patch WB (clamp 1.5). Skin-anchored WB intentionally absent.
- **Match** (`grade/match.py`): clusters ALL referenced **source files** by whole-file `rgb_mean` distance (`RGB_DIST_MAX=0.12`), nudges members 40% toward the highest-quality file's mean. Whole-file, document-wide, timeline-agnostic.
- **Measurement** (`l1/color_stats.py`): per-FILE aggregate over ~12 evenly-spaced frames — `black/white_point`, `mid_gray`, `rgb_mean/median/std`, `lab_ab_cast`, `wb_*`, `clip_*_pct`, `is_log_flat`, `skin_lab` (center proxy, no face detector), `palette`, `luma_hist`. Stored in `color_stats` table, fetched by `grade/measure.py::fetch_color_stats`.
- Available L1/L3 signals for later phases: `cut_records.framing.subject_box` (subject region), `on_camera`/`speaker_person` (ASD), `hero_ts_ms`, `salience`, `label`/`summary`/`channel` (Pass-2 visual summary — coarse, 2-frame), transcript/`dialogue_segments`.
- Frontend grading (`color-grade-view.tsx`): selecting a preset / dropping a reference / uploading `.cube` / dragging arc → `applyLook` → debounced `saveEditDocument`. Grade is recomputed inline on document resolve (cheap today). A `Saving…` spinner exists; there is NO grading-progress surface.

---

# PHASE 1 — Reach parity on the six fundamentals

Feature flag: add `settings.grade_pipeline = "v1" | "legacy"` in `backend/app/config.py` (default `legacy` until Phase 1 verified, then flip to `v1`). Every step below checks this flag; `legacy` = today's exact behavior.

## Step 1.0 — Grading runs as a background job (foundational architecture)
Build this seam FIRST (after the flag), so every later `v1` layer plugs into the job rather than into inline resolve.

**Files:** new `backend/app/services/l3/grade/job.py`, `backend/worker.py` (task registration), a migration for a `resolved_grades` table + a `grade_jobs` status row, `backend/app/services/l3/layers.py`, and the edit routes (find the router that owns `/api/edit/threads/{id}/...`; `grade-export` already lives there per `color-grade-view.tsx`).

1. **Persistence — `resolved_grades` table.** Columns: `thread_id`, `shot_key` (`seg_id`/`op_id`), `input_hash` (see §4), `grade_json` (the `{cdl, creative_lut_ref, working_space, grade_hash, soft_local}` descriptor `resolve_clip_grade` already returns), `cube_ref` (pre-baked `.cube` handle, nullable), `updated_at`. Unique on `(thread_id, shot_key, input_hash)`.
2. **Status — `grade_jobs`.** Columns: `thread_id` (unique/pk), `state` (`idle|grading|done|error`), `total` (shot count), `done` (shots completed), `input_hash`, `error` (nullable), `updated_at`. One row per thread, upserted as the job runs.
3. **The task — `grade/job.py::run_grade_job(thread_id)`** (registered as a Procrastinate task in `worker.py`, so the already-running local worker picks it up):
   - Load the document; enumerate shots in program order (spine spans + `place_video` ops).
   - Set `grade_jobs` → `grading`, `total = len(shots)`, `done = 0`.
   - Run the `v1` pipeline: per shot `measure_span` (1.2), then `solve_sequence_match` once (1.4), then `resolve_clip_grade` per shot (1.1/1.3/1.5) and leveling (Phase 2 if flag), incrementing `done` after each shot (or each cheap batch) so progress advances.
   - Pre-bake each distinct `grade_hash` to a `.cube` via `lut_bake.bake_cube_text`, store `cube_ref`.
   - Upsert `resolved_grades` rows; set `grade_jobs` → `done`. On exception → `error` (never crash the worker; mirror `color_stats` best-effort semantics).
4. **Triggering + invalidation (idempotent).** Define `input_hash = hash(ordered shot spans + look + grade flags + schema_version)`. Enqueue `run_grade_job` when that hash changes: on look change and on timeline-structure change (cuts added/removed/trimmed shift spans AND neighbors). Debounce on the enqueue side (the frontend already debounces saves ~300ms). If a job for the current `input_hash` is already `done`, do nothing; if `grading`, don't double-enqueue.
5. **`layers.resolve` reads, never computes (under `v1`).** For each shot, look up `resolved_grades` by `(thread_id, shot_key, input_hash)`:
   - hit → use the persisted `grade_json`.
   - miss (job not finished yet) → **graceful fallback**: use the previous `input_hash`'s grade for that shot if present, else identity. Preview must always render; it just shows the pre-grade (or last) look until the job lands. `legacy` flag keeps the current inline `resolve_clip_grade` path unchanged.
6. **Status endpoint.** `GET /api/edit/threads/{thread_id}/grade-status` → `{state, progress: done/total (0..1), error}`. Add a `POST .../grade` (or reuse the save path) to explicitly (re)enqueue when needed.

**Acceptance:** changing the look enqueues one job; the worker computes + persists grades and pre-bakes cubes; `grade-status` reports `grading` then `done` with advancing progress; `layers.resolve` renders instantly from persisted grades (or a graceful fallback while pending); `legacy` flag bypasses the job entirely.

## Step 1.1 — Scene-referred working space + tone mapping (parity-safe, baked into the cube)
**Files:** `grade/lut_bake.py`, new `grade/tone.py`, `grade/cdl.py` (no behavior change, just call order), `grade/resolver.py` (pass `working_space` through — already does).

1. Create `grade/tone.py` with two pure functions over `(...,3)` float32 arrays in 0..1, plus their inverses:
   - `to_working(rgb_display, working_space) -> rgb_working`: linearize the display-encoded input into the chosen scene-referred working space. For `v1`, implement a **single documented transfer**: inverse sRGB/Rec.709 EOTF → linear (this is the "slot"; a fuller ACES input transform can replace it later without touching callers).
   - `from_working(rgb_working, working_space) -> rgb_display`: the output transform = a **filmic tone map** (implement a standard, well-documented curve, e.g. a Reinhard/Hable-style shoulder) → re-encode to display. This is what turns "slope/offset filter" into "graded."
   - Keep both behind `working_space`; `working_space=="rec709_legacy"` returns input unchanged (identity) so `legacy` flag = today.
2. In `grade/lut_bake.py::bake_cube_text`, when `working_space` is a `v1` space, change the pipeline from `apply_cdl(grid, grade)` to:
   `working = to_working(grid, ws)` → `graded = apply_cdl(working, grade)` (CDL now operates in scene-referred space) → `display = from_working(graded, ws)` → optional creative LUT → clip. Bake `display`.
3. Because the cube is what both preview and export sample, **no frontend or compositor change is needed** for the math — verify parity with a test that samples the same RGB through `apply_cdl`+tone vs. the baked cube (trilinear) within tolerance.
4. Add `working_space` selection: `resolve_clip_grade` sets it from `settings.grade_pipeline` (`"rec709_v1"` vs `"rec709_legacy"`). It's already part of `grade_hash`, so caches invalidate correctly.

**Acceptance:** with `v1`, a flat/log clip and a contrasty clip both come back with pleasing, non-clipped tone; preview and a rendered frame match within tolerance; `legacy` flag reproduces today's bytes exactly.

## Step 1.2 — Measure the shot actually USED, not the whole file
Matching/correcting on whole-file means is wrong (you keep a 2s window of a 40s clip). Add per-span measurement.

**Files:** new `grade/measure_span.py`, `grade/resolver.py`, `layers.py`.

1. Add `grade/measure_span.py::measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None) -> dict`: decode a few frames (reuse `l1/color_stats.py::_decode_rgb_frame_at` + `_aggregate`) sampled WITHIN `[in_ms,out_ms]` (bias one sample to `hero_ts_ms` when given). Return the same shape as a `color_stats` row (so downstream code is unchanged) but for the used span. Cheap: 3–5 frames.
2. Cache per `(file_id,in_ms,out_ms)` in a small table `cut_color_stats` (or a keyed cache) so repeated resolves don't re-decode. Key includes a schema version.
3. This runs INSIDE `run_grade_job` (Step 1.0), not inline in `layers.resolve`: the job resolves each shot's grade against its **span stats** (`measure_span(seg.file_id, seg.in_ms, seg.out_ms, hero_ts_ms=…)`) instead of the whole-file `color_stats.get(file_id)`, and persists the result. Fall back to whole-file stats if span measurement fails (never-worse).

**Acceptance:** two cuts from the same file but different lighting windows get different correction; a smoke test asserts span stats differ from file stats on a clip with a mid-clip lighting change.

## Step 1.3 — Auto-balance to parity (percentile-based, still unbiased WB)
**Files:** `grade/correct.py`.

1. Keep the never-worse philosophy but base exposure on **luma percentiles from `luma_hist`/`black_point`/`white_point`/`mid_gray`** (already measured) rather than only the black/white stretch: target a mid-gray placement and a bounded black/white anchoring, all clamped (reuse `LEVELS_SLOPE_MAX`).
2. Keep gray-world+white-patch WB (unbiased; skin-anchored still deliberately out — see `correct.py` docstring). Optionally raise trust when both agree (already does).
3. All new math runs in the working space (Step 1.1) so it composes correctly with tone mapping.
4. Gate entirely on `grade_pipeline=="v1"`; `legacy` calls the current `solve_correct_grade` untouched.

**Acceptance:** a batch of test clips (dim, bright, flat, color-cast) all come back to a consistent neutral mid-gray without clipping; never-worse holds (already-correct footage barely moves).

## Step 1.4 — Timeline-aware, percentile shot/sequence matching (replaces whole-file clustering)
**Files:** `grade/match.py`, `layers.py`.

1. New `solve_sequence_match(ordered_shots) -> {shot_key: Grade}` where `ordered_shots` is the timeline in program order, each with its **span stats** (Step 1.2). "shot_key" = `seg_id`/`op_id`.
2. Grouping: default to grouping **adjacent** shots (and shots sharing a `file_id`/scene) rather than global RGB clustering — Phase 3 will swap in semantic grouping; for Phase 1, group by span-stat similarity among neighbors within a window. Keep `RGB_DIST_MAX`-style threshold but on span stats.
3. Match on **percentiles, not just mean**: nudge each member's `black_point`, `mid_gray`, `white_point`, and `lab_ab_cast` toward the group anchor (highest-quality span), each bounded (`MATCH_STRENGTH`-style, keep conservative). Convert the deltas into a `Grade` (slope/offset per channel + a small lift), in working space.
4. INSIDE `run_grade_job` (Step 1.0), when `v1`: build `ordered_shots` from `spine_spans` + `place_video` ops in program order, call `solve_sequence_match` ONCE, and pass each shot's delta into `resolve_clip_grade` as `match_delta` (replacing `solve_match_deltas`), persisting the result. `legacy` keeps the inline `solve_match_deltas` path in `layers.resolve`.

**Acceptance:** a two-camera interview (two files, different exposure) that today doesn't cluster now matches across the cut; matching only fires between shots that actually neighbor/share a scene; a test asserts non-neighbor dissimilar shots are NOT dragged together.

## Step 1.5 — Reference / look transfer composes in working space
**Files:** `grade/reference_transfer.py`, `grade/resolver.py` (already wires `_solve_look` against corrected stats).

1. Ensure `solve_reference_transfer` operates on **working-space** corrected span stats (extend `resolver.py::_corrected_source_stats` to project through the working-space transform when `v1`).
2. No new UI needed — `color-grade-view.tsx` already sends `reference_stats` + `match_strength`.

**Acceptance:** dropping a reference still shifts the whole sequence toward it without double-stretching exposure (the `_corrected_source_stats` guard already exists; just verify under `v1`).

## Step 1.6 — Temporal stability (guarantee, not feature)
**Files:** `layers.py` (doc), a test.

1. Today each shot gets ONE grade for its whole span → inherently flicker-free within a shot. Formalize: assert one resolved grade per timeline segment; forbid per-frame grade variance in `v1`. (Per-frame/time-varying grades are explicitly out — they need compositor work; see Phase 2 note.)
2. Add a test that a single shot resolves to exactly one `grade_hash` across its duration.

**Acceptance:** no intra-shot grade variation; documented as an invariant.

## Step 1.7 — Region/subject masking foundation (minimal; we "start ahead")
**Files:** `grade/softlocal.py`, `grade/resolver.py`.

1. Do NOT build heavy masking now. Just widen the existing `soft_local` seam so a future subject/region mask can be attached (it's already a separate spatial-parameter pass applied both sides — see `lut_bake.py` docstring). Add an optional `subject_box` field to the `soft_local` descriptor, plumbed from `cut_records.framing.subject_box` when available, but leave the actual spatial op as identity for Phase 1 (wired for real in Phase 3).

**Acceptance:** `soft_local` can carry a `subject_box` end-to-end (resolve → hash → bake seam) with no visual change yet.

---

# PHASE 2 — Even lighting (bounded photometric leveling)

Feature flag: `settings.grade_even_lighting = bool` (default off). Only the two generic-safe ideas; everything spatial/temporal/subject is deferred (needs render work or semantics).

## Step 2.1 — Across-shot exposure leveling to a smooth target
**Files:** new `grade/leveling.py`, `layers.py`.

1. From the ordered per-shot span stats (Step 1.2), build a per-shot **exposure value** (e.g. `mid_gray` in working space).
2. Compute a **smooth target curve** across the timeline (low-pass the per-shot exposure sequence — a moving average / gentle spline). This preserves an intended slow bright→dark arc while flattening shot-to-shot flicker.
3. Per shot, compute a **bounded** gain that moves its exposure toward the smoothed target — cap the correction (e.g. ≤ some stops) so large intended differences survive. Emit as a `Grade` offset/slope in working space, composed as a new stage between Match and Look in `resolver.py` (gated on `grade_even_lighting`).

**Acceptance:** a montage of shots with jittery brightness reads even; an intentional day→night sequence keeps its arc (the smooth target follows it); the bound prevents any shot from being pushed more than the cap.

## Step 2.2 — Tonal-placement alignment (shadow/mid/highlight)
**Files:** `grade/leveling.py`.

1. Extend leveling to also align each shot's **shadow point and highlight point** (from span stats) toward the group/sequence placement, bounded, in working space — so lighting reads even, not merely equally bright.
2. Guardrails: never push black/white points into clipping; skip when a shot is a statistical outlier (likely a genuinely different scene) beyond a threshold.

**Acceptance:** low-contrast and punchy shots in one scene converge in tonal feel; cross-scene outliers are left alone.

**Explicit non-goals (documented in the file):** temporal within-shot stabilization and local/intra-frame relight are OUT — both need time-varying / spatially-varying grades the compositor can't do today (same limitation as unrendered video speed), and both need a flaw-vs-intent signal we don't have. Revisit in a render-capabilities effort.

---

# PHASE 3 — The semantic leap (our specialty)

Remember the constraint: **Pass 2 gives only a coarse visual summary per cut; the reliable signals are L1 photometry + `cut_records` structural fields (`subject_box`, `on_camera`, `speaker_person`, `salience`, `hero_ts_ms`) + transcript.** Feature flag: `settings.grade_semantic = bool` (default off).

## Step 3.1 — Subject-aware exposure evening (uses ASD / subject_box)
**Files:** `grade/measure_span.py`, `grade/leveling.py`, `grade/resolver.py`.

1. In `measure_span`, when `cut_records.framing.subject_box` exists for the cut, ALSO measure luma **inside the subject box** on the hero frame → `subject_luma`.
2. In leveling (Phase 2), when a valid `subject_luma` exists, level the **subject's** luma across shots (target the sequence's subject-luma), not the whole-frame mid-gray — the perceptually correct evenness for people content.
3. Gate: only when a subject box is present AND the subject isn't a deliberate silhouette (bounded correction + skip if subject is extremely dark/bright vs. frame, which signals intent). No-op for b-roll/no-subject shots (fall back to Phase 2 whole-frame leveling).

**Acceptance:** in a multi-shot talking-head/interview, the speaker's face brightness is consistent across cuts even when backgrounds differ; b-roll shots are unaffected.

## Step 3.2 — Semantic scene grouping for matching (uses labels + structure)
**Files:** `grade/match.py`, new `grade/scene_group.py`.

1. Build `group_shots_semantically(ordered_shots, cut_meta) -> groups` using signals we trust: same `file_id`, temporal adjacency, same `speaker_person`/`on_camera`, and coarse `label`/`summary` similarity (as a weak tiebreaker only — never the sole signal, given it's a 2-frame summary).
2. Feed these groups into `solve_sequence_match` (Step 1.4) so matching aligns shots that are the *same scene by meaning*, not merely RGB-close — this is the reliability edge over pixel-only tools (they mis-match unrelated shots).

**Acceptance:** shots from one setup grade together even if a transient (a bright object entering) skews their RGB; shots from different scenes that happen to be RGB-close are NOT force-matched.

## Step 3.3 — Narrative-driven arc (uses transcript / purpose)
**Files:** `grade/arc.py` (table exists), `grade/resolver.py`, `l3/converse.py`/`tools.py` (let EDSO set arc intent), guidance doc.

1. Today `arc_intent` is a manual tag (`tag_arc_intent` verb) scaled by `arc_intensity` (default 0 = invisible). Add an **automatic, still-invisible-by-default** path: EDSO may set per-section `arc_intent` from the transcript/purpose (tense beat cooler, resolution warmer) — a categorical choice from the existing `arc.py` table, never raw CDL.
2. Keep intensity a user dial (default 0), so it never surprises; semantic arc only shows when the user opts in.
3. Add a short guidance-doc note (generic): the grade can follow the story's arc when it serves the purpose; it's a categorical intent, not a look.

**Acceptance:** with arc intensity > 0, a piece with a clear emotional arc gets a subtle, coherent tonal arc driven by content; at intensity 0 nothing changes.

---

# PHASE 4 — Frontend: grading progress indicator

Phase 1 makes grading a background job (Step 1.0), so the progress bar reflects the REAL job, not a fake spinner.

**Files:** `frontend/src/components/color-grade-view.tsx`, `frontend/src/lib/api.ts`, `frontend/src/stores/timeline-view.ts` (small state).

1. **API client** (`api.ts`): add `getGradeStatus(threadId, token) -> {state, progress, error}` hitting `GET /api/edit/threads/{id}/grade-status` (Step 1.0 §6).
2. **Polling:** in `color-grade-view.tsx`, after any grade action fires (preset select, reference drop, LUT upload, arc drag → all already call `applyLook`; also on a timeline change that re-enqueues), start polling `getGradeStatus` (e.g. every ~750ms) while `state === "grading"`; stop on `done`/`error`. Debounce so a dragged dial doesn't spam polls.
3. **UI:** a small **determinate** progress bar (uses `progress = done/total`) in the panel header, styled per the design system (grey track, orange `--accent` fill); show "Grading… {n}/{total}" then clear on `done`. On `error`, surface via the existing `error` state and clear the bar. Optionally mirror a thin bar over the preview.
4. **Refresh on done:** when the job completes, re-resolve/refetch so the preview picks up the newly persisted grades (bump the store's version or refetch the document).

**Acceptance:** any grade control (or a timeline edit that changes grades) shows a determinate bar that advances with the job and clears on completion; errors surface and clear the bar; the preview updates to the new grade when the job finishes.

---

## Cross-cutting

### Switchability / flags
- `settings.grade_pipeline` (`legacy` default → `v1`), `settings.grade_even_lighting` (off), `settings.grade_semantic` (off). Each flag independently revertible. `legacy` across the board must reproduce today's exact output (hash-level parity test).

### Tests (add/extend)
- `backend/scripts/test_grade_*` (create if absent): tone/working-space parity (apply vs baked cube), never-worse invariants, span-vs-file stats differ, sequence-match neighbor-only, leveling smooth-target preserves arc + bound, subject-luma evening no-ops without a box, semantic grouping doesn't force unrelated shots, legacy-flag byte parity.
- Job tests: `run_grade_job` populates `resolved_grades` + `grade_jobs` for a fixture thread; `input_hash` stability (same inputs → no re-enqueue, changed span/look → re-enqueue); `layers.resolve` reads persisted grades and falls back gracefully when a job is pending.
- Frontend: a lightweight test/manual check that the determinate progress bar appears, advances, and clears on done/error.

### Order & risk
Build order: **1.0 (job seam) → 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7**, then verify parity and flip `grade_pipeline=v1`; then **Phase 2**, then **Phase 3** (3.1 → 3.2 → 3.3), then **Phase 4** (depends on Step 1.0's status endpoint). Two linchpins: **1.0 (the job + persistence + read-path seam)** — everything `v1` plugs into it — and **1.1 (scene-referred + tone map)** — get its parity test green before anything builds on it. Note Phase 2/3 layers run *inside* `run_grade_job`, composed into the persisted per-shot grade.

### Libraries
Do not hard-commit to OCIO/ACES vs. hand-rolled transforms now. Implement `grade/tone.py` with clean `to_working`/`from_working` seams so the transfer functions can be swapped for a proper color-management library later without touching any caller.

## Finish
Run the grade test suites (and a quick re-resolve of one project to eyeball preview), then **commit and push**.
