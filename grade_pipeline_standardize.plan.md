# Grade Pipeline Standardization + Look-Thumbnail Fix — Implementation Plan

**Goal.** Collapse the color-grading pipeline to ONE canonical, un-flagged path. The dev flags in `backend/app/config.py` are the debt: some flag *competing versions* of the same stage (keep the winner, delete the loser), and some flag *additive capabilities* that the user wants kept (hardwire them on / scope them to the look, just remove the dev flag — never delete the feature). Exactly one feature's *flag* is removed while its *code is kept dormant*: `grade_tone_contrast` (the global filmic S-curve) is hardwired **OFF** — it caused the "too dark / bad looks" regression and double-applies contrast on top of engine looks — but the S-curve plumbing stays in place so it can be re-enabled in code later (no UI/flag). Separately, Part B root-causes and fixes the Look-gallery thumbnails that render blank.

> **Scope: color grading ONLY.** Nothing outside `backend/app/services/l3/grade/`, its callers, the `/api/grade/*` router, the render compositor's grade seam, and the frontend grade view is in scope.

---

## Framing: two kinds of flags

The user's instruction "remove the versions, standardize, keep the best" maps onto two distinct categories. **Get this distinction right** — it is the whole point of the task.

- **COMPETING VERSIONS** — an old implementation and a new one of the *same* stage, gated so we can A/B. Standardization = keep the new one, **delete the old branch**, remove the flag.
- **ADDITIVE CAPABILITIES** — independent features that *layer on top* of the pipeline. They do not replace anything. Standardization = **keep the feature**, fold it permanently into the single pipeline (hardwire on, or scope to the look), remove the dev flag. **Never delete the code.**

---

## Part A — Flag inventory + decisions

Current defaults read from `backend/app/config.py` (lines cited):

| Flag | Current default | Category | What it gates | DECISION |
|---|---|---|---|---|
| `grade_pipeline` (`:146`) | `"v1"` | **Competing version** | `layers.resolve` reads pre-baked grades from the v1 job vs. computing inline (`legacy`) | **Keep v1, DELETE legacy path**, remove flag |
| `grade_shot_match_v2` (`:161`) | `True` | **Competing version** | `match.solve_sequence_match` reference-based matching (v2) vs. pre-redesign anchor-only | **Keep v2, DELETE pre-v2 branch**, remove flag |
| `grade_scene_join` (`:168`) | `True` | **Competing version** | cut_record metadata join for grouping vs. RGB-adjacency-only fallback | **Keep join, DELETE old grouping fallback**, remove flag |
| `grade_even_lighting` (`:150`) | `True` | **Additive capability** | Bounded across-shot exposure leveling (`leveling.solve_leveling`) | **Keep, hardwire ON**, remove flag |
| `grade_semantic` (`:153`) | `True` | **Additive capability** | Subject-aware leveling + semantic scene grouping inputs | **Keep, hardwire ON**, remove flag |
| `grade_look_engine` (`:195`) | `True` | **Additive capability** | Parametric Look engine (`mode=="engine"` → `build_look_grid`) | **Keep, hardwire ON**, remove flag |
| `grade_subject_exposure` (`:176`) | `False` | **Additive capability** | Converge subject-luma (not whole-frame) per group | **Keep, hardwire ON**, remove flag |
| `grade_skin_vibrance` (`:184`) | `False` | **Additive capability** | Skin-anchored tint correction + saturation floor | **Keep, hardwire ON**, remove flag |
| `grade_film_texture` (`:201`) | `False` | **Additive capability (look-scoped)** | Halation + grain in `soft_local`, declared by engine looks | **Keep, scope to the look** (apply whenever the active look declares `halation`/`grain`), remove flag |
| `grade_tone_contrast` (`:189`) + `grade_tone_contrast_strength` (`:190`) | `False`, `0.9` | **Flag removed, code kept OFF** | Global filmic S-curve baked into `tone.from_working` | **Remove both flags, hardwire the call site to `0.0` (off); keep the S-curve plumbing dormant** |

> Note: `grade_subject_exposure`, `grade_skin_vibrance`, `grade_film_texture` are currently **OFF** by default. The user has explicitly stated these are wanted additive capabilities, so the decision is to turn them **ON permanently** as part of the standard pipeline (not to delete). This is a behavior change from today's defaults — see Risks.

---

### A.1 — COMPETING VERSIONS (delete the old branch)

#### `grade_pipeline` → hardwire `"v1"`, delete the `legacy` inline-resolve path
The v1 grade job (`grade/job.py::run_grade_job`) is the go-forward pipeline. `legacy` re-computes grades inline in `layers.resolve` at resolve time; v1 reads pre-baked results from `resolved_grades`.

Delete / hardwire:
- `backend/app/services/l3/layers.py::resolve` (`:564-637`): drop the `grade_pipeline` kwarg and the `v1_grades` branch. Keep ONLY the v1 read path:
  - `grade=(grade_lookup.get(seg["seg_id"]) or identity_grade_json(WORKING_SPACE_V1))` (`:632`).
  - Delete the `else resolve_clip_grade(...)` inline branch (`:633-636`) and the `match_deltas = ... solve_match_deltas(color_stats) ...` inline call (`:617`) — `solve_match_deltas` at resolve time is legacy-only. (Apply the same removal to every op/overlay layer in `resolve` that has the same `v1_grades ? ... : resolve_clip_grade(...)` ternary — search the function for `resolve_clip_grade(` and `grade_lookup.get(`.)
- Callers that branch on `grade_pipeline == "v1"` — simplify each to always take the v1 branch and stop passing `grade_pipeline`:
  - `backend/app/services/render/tasks.py` (`:88, :100, :126, :129`).
  - `backend/app/services/l3/observe.py` (`:233, :235, :244`).
  - `backend/app/routers/edit_threads.py` (`:169`).
- **Keep** `resolve_clip_grade` itself (`grade/resolver.py:206`) — it is still called by the v1 job (`job.py`), not dead. Just remove its `pipeline="legacy"` default's *legacy* behavior by hardwiring the v1 branch inside it (the `pipeline` param and its `"legacy"` code paths at `:238-242` and wherever `pipeline == "v1"` is checked become unconditional). Do this carefully and keep byte-parity for the v1 path.
- `backend/migrations/043_grade_v1.sql` — no change (schema stays; only the code flag goes).

#### `grade_shot_match_v2` → hardwire ON, delete pre-v2 matching
The v2 path passes group `references` into `match.solve_sequence_match`; the legacy path (`references=None`) uses the single-shot `max(quality)` anchor with "anchor is exempt" (`match.py:243-267`).

Verify + hardwire:
- Confirm it is ON (`config.py:161` = `True`) and exercised: `job.py:509` `if settings.grade_shot_match_v2:` builds and passes `references`.
- In `grade/job.py` (`:503-509`), remove the `if settings.grade_shot_match_v2:` guard and always build/pass `references` + `groups`.
- In `grade/match.py::solve_sequence_match` (`:234-263`), the `references`-present branch becomes the only supported path. The `references or {}` / `anchor = max(members, ...)` fallback (`:255-263`, `anchor_key`) is the pre-v2 branch — delete it, since `run_grade_job` will always pass `references`. Keep `group_neighbors` as the grouping primitive (it is used regardless).
- Update the docstring (`:234-242`) which currently documents the `None`-default legacy behavior.

#### `grade_scene_join` → hardwire ON, delete old RGB-only grouping fallback
Joining each shot's `(file_id, span)` to its covering `cut_record` gives real scene metadata for grouping. `False` = empty metadata → RGB-adjacency-only fallback.

Verify + hardwire in `grade/job.py`:
- `:427` `if settings.grade_semantic and (settings.grade_scene_join or settings.grade_subject_exposure):` — the cut_record join.
- `:487-488` `cm = cut_meta.get(s.key) if settings.grade_scene_join else None` and `span_rgb = ... if settings.grade_scene_join else None`.
- Remove the `grade_scene_join` guards so the join always runs and its metadata always feeds grouping. Keep the RGB base as the *graceful fallback when a shot has no covering cut_record* (that is a real runtime condition, not the flag) — do not delete the RGB path, only the flag that forced it.
- `grade/scene_group.py` docstring (`:11`) references `settings.grade_semantic`; update wording.

---

### A.2 — ADDITIVE CAPABILITIES (keep, remove flag, DO NOT delete)

For each: remove the `settings.<flag>` read and make the capability unconditional (its own internal "is this a no-op for this clip?" guards stay — e.g. `spec.is_identity()`, `texture_spec.halation > 0.0`, a missing `subject_box`).

- **`grade_even_lighting`** → always compute leveling. `job.py:578` `if settings.grade_even_lighting:` → unconditional. `leveling.py` docstring (`:5`) update.
- **`grade_semantic`** → always use semantic grouping + subject inputs. Remove `settings.grade_semantic` guards at `job.py:427, :435, :484, :633` and `resolver`/`scene_group` references. (Semantic inputs still degrade gracefully when metadata is absent — keep those runtime guards.)
- **`grade_look_engine`** → always honor `mode=="engine"`. In `resolver.py::resolve_clip_grade`, drop `look_engine_enabled` (`:218`) and make the engine block unconditional (`:308-313` — keep the `spec.is_identity()` skip). Update `job.py:665` (`look_engine_enabled=...`) to stop passing the flag.
- **`grade_subject_exposure`** → always converge on subject luma when a subject box exists. Remove guards at `job.py:427, :435, :633`. Keep the "box present?" runtime check. **(Turns ON a currently-OFF capability.)**
- **`grade_skin_vibrance`** → always apply skin-anchored tint + sat floor. `job.py:660` (`skin_vibrance=settings.grade_skin_vibrance`) and `resolver.py:216` `skin_vibrance` param → hardwire `True`. **(Turns ON a currently-OFF capability.)**
- **`grade_film_texture`** → **look-scoped, not a global toggle.** Halation/grain are declared per-look by the engine `LookSpec` (`look_engine.py:70-71`; e.g. `kodak_2383` has `halation=0.25, grain=0.04`). In `resolver.py:337` (`if film_texture_enabled and look_engine:`) drop `film_texture_enabled` so the branch fires purely on the active look declaring nonzero `halation`/`grain`. Remove the `film_texture_enabled` param (`:219`) and `job.py:666` (`film_texture_enabled=...`). Net effect: film looks automatically get their grain/halation; non-film looks never do — no global flag needed. **(Turns ON a currently-OFF capability, but only for looks that ask for it.)**

The frontend already surfaces this correctly: `look-card.tsx` shows a `grain` badge on `family === "film"` cards, and `color_grade-view.tsx`'s comment notes texture is look-declared. No frontend change required for these.

---

### A.3 — `grade_tone_contrast`: remove the FLAG, keep the CODE dormant (OFF)

**Decision (user-confirmed): remove both flags and hardwire the global S-curve OFF, but KEEP the S-curve plumbing in the codebase.** The feature is not deleted — it is made permanently inert (always fed `0.0`) with no flag and no UI, so a future dev can re-enable it in code by passing a non-zero value.

Why off:
- It is the global filmic S-curve that **caused the "too dark / bad looks" regression** the user called out.
- It **double-applies contrast**: engine looks already carry their own per-look `contrast` (`LookSpec.contrast`, applied via `_contrast_pivot` in `look_engine.build_look_grid`). A *second* global S-curve stacked on top over-darkens and muddies exactly the looks we just validated.

Why keep the code:
- The user wants it retained (not deleted) in case a well-tuned global contrast pass is wanted later. Keeping it dormant costs nothing at runtime (a `0.0` contrast is an exact identity in `from_working`) and avoids ripping plumbing out of `tone.py`/`lut_bake.py`/`resolver.py`/`cache.py`/`cdl.py`/`compositor.py`/frontend.

Change (minimal — flag removal only):
- `config.py:186-190` — **remove both settings** (`grade_tone_contrast`, `grade_tone_contrast_strength`).
- `grade/job.py:253-254` (hash `flags`) — drop the two `tone_contrast` entries from the flags payload. `:661-663` — **hardwire the `tone_contrast=` argument to `resolve_clip_grade` to `0.0`** (delete the `settings.grade_tone_contrast ...` conditional; pass a literal `0.0`).
- **Keep everything else as-is** — the `tone_contrast` parameter threaded through `resolve_clip_grade` → `grade_hash` → `bake_cube_text`/`from_working`, `cache.py`, `compositor.py`'s identity-collapse, and the frontend `tone_contrast?` field / `grade-cube-client.ts` guard. All of it stays; because the value is now always `0.0`, these paths are inert (`from_working` with `contrast=0.0` is identity, `_grade_key`'s `tone_contrast <= 0.0` collapses to identity, and the frontend `if (grade.tone_contrast)` guard never fires). No behavior change beyond the S-curve being off.
- `_contrast_pivot` and `TONE_PIVOT` in `grade/tone.py` stay regardless (also reused by `look_engine.build_look_grid` for per-look contrast at `look_engine.py:312`).

**Trade-off:** there is no *global* contrast knob in the product (no flag, no UI). Per-look contrast (via each `LookSpec.contrast`) is unaffected and remains the intended way to add/soften contrast. Re-enabling the global curve later is a one-line code change (pass a non-zero `tone_contrast`), ideally re-introduced look-scoped rather than as a global stack.

---

### A.4 — Hash bump + re-grade ALL projects (required)

Removing flags changes the `compute_input_hash` payload in `grade/job.py` (`:245-257`, the `"flags"` dict), so every project's stored `input_hash` changes.

1. Bump `INPUT_HASH_SCHEMA_VERSION` in `grade/job.py:91` from `10` → `11`.
2. Prune the `"flags"` dict (`:245-257`) to only what still exists after the flag removals. If ALL grade behavior becomes unconditional, the `flags` block can be dropped entirely (the `INPUT_HASH_SCHEMA_VERSION` bump alone forces the re-grade); prefer keeping a minimal/empty marker consistent with the new pipeline.
3. **Re-grade every project** after deploying, using the existing `backend/scripts/_grade_all_projects.py` (it currently asserts `grade_pipeline == "v1" and grade_even_lighting and grade_semantic` at `:15` — update that assert to match the new no-flag world). Until re-graded, `layers.resolve` will serve the freshest prior `resolved_grades` row (stale) or identity for shots with no row — no errors, just stale color until the job reruns.

---

### A.5 — Test files to update (enumerate only; do NOT rewrite here)

These pass the flags explicitly and will not compile/behave once the signatures change:

- `backend/scripts/test_grade.py` — heaviest. Many `run_grade_job(...)` and `layers.resolve(...)` calls pass the flags directly:
  - `resolve(..., grade_pipeline="v1", grade_lookup=...)` at `:500, :511`.
  - `run_grade_job(... grade_pipeline=, grade_even_lighting=, grade_semantic=, grade_shot_match_v2=, grade_scene_join=, grade_subject_exposure=, grade_skin_vibrance=, grade_tone_contrast=, grade_tone_contrast_strength=, grade_look_engine=, grade_film_texture=)` at `:1081-1085, :1217-1222, :1261-1266, :1306-1311, :1352-1357, :1530-1535, :1581-1587, :1635-1640, :1700-1705`.
  - The kill-switch / flag-off tests (`test_run_grade_job_*` for `grade_shot_match_v2=False`, `grade_scene_join=False`, subject-exposure gating at `:1064, :1332, :1417, :1556, :1621`) test *deleted* branches — remove or repurpose.
  - The `tone_contrast` tests (`:1892-2022`, `:2573-2581`): the code path is KEPT (just off by default), so tests that exercise the S-curve directly via `bake_cube_text`/`resolve_clip_grade` with an explicit `tone_contrast` value can stay. Only the ones that pass the now-removed `grade_tone_contrast`/`grade_tone_contrast_strength` kwargs to `run_grade_job` need those kwargs dropped (and should assert the default is off, i.e. contrast `0.0`).
  - The `resolve` no-arg / legacy tests (`:485-511`) — update to the v1-only signature.
- `backend/scripts/_validate_grade_v1.py` — asserts/prints `grade_pipeline`/`even_lighting`/`semantic` (`:9, :24-25`).
- `backend/scripts/_validate_engine_looks.py` — reads many `settings.grade_*` (`:73-243`); update to the no-flag pipeline.
- `backend/scripts/_grade_all_projects.py` — the flag assert at `:15-16` (see A.4).

Signature changes to propagate: `layers.resolve(...)` loses `grade_pipeline`/`grade_lookup` legacy semantics; `resolve_clip_grade(...)` loses `pipeline`, `look_engine_enabled`, `film_texture_enabled`, and `skin_vibrance` becomes unconditional (but **keeps** its `tone_contrast` parameter, now always passed `0.0`); `run_grade_job(...)` loses all the flag kwargs (including `grade_tone_contrast`/`grade_tone_contrast_strength`). `bake_cube_text`/`grade_hash`/`ensure_cube_file` **keep** their `tone_contrast` parameter (inert at `0.0`).

---

### A.6 — Presets / backward-compat (no UI change)

- `backend/app/services/l3/grade/presets.py::list_presets` returns the 12 legacy CDL presets (`mode:"preset"`). These are **already hidden** from the default picker — `color-grade-view.tsx:589` filters to `p.mode === "engine" && p.look_id !== "engine_identity"`. **Recommendation: keep `presets.py` as a silent, resolvable backward-compat path** (documents that already reference a `preset_id` still resolve), but keep it out of the UI (already the case). Do not delete it.
- Confirm the gallery is engine-only: `color-grade-view.tsx` `groupByFamily(presets.filter((p) => p.mode === "engine" ...))` (`:589`) — yes, engine-only. No change.
- The combined listing endpoint `routers/grade.py::get_grade_presets` (`:41-49`) returns `list_presets() + list_engine_looks()` — keep as-is (frontend already filters).

---

## Part B — Look-gallery thumbnails render blank (ROOT CAUSE + FIX)

### B.1 — The originally-suspected cause is DISPROVEN (with evidence)

The hypothesis was: the `/api/grade/cube` endpoint gates the engine grid behind `grade_look_engine` server-side, so with the flag OFF it returned identity cubes for look_engine requests. **This is false.** Evidence:

1. **No flag gate exists in the endpoint.** `rg grade_look_engine backend/app/routers/grade.py` → no matches. The cube path is `routers/grade.py::get_grade_cube` (`:108-161`) → `cache.ensure_cube_file` (`cache.py:25-74`) → `build_look_grid` (`look_engine.py:294`). None of these read `grade_look_engine`. Whenever a `look_engine` query param is present, the engine grid is built unconditionally (`cache.py:50-58`).
2. **Live endpoint returns a correct, non-identity cube.** Calling the running backend exactly as the frontend does:
   ```
   GET /api/grade/cube?cdl={identity}&working_space=rec709_v1&look_engine={punchy_vibrant spec}
   → HTTP 200, text/plain, 35937 rows; 35936/35937 rows differ from the identity cube.
   ```
   (Auth is dev-bypassed via `config.dev_user_id`, so this is exactly the dev path the browser hits.)
3. **The gallery listing includes `look_params`.** `GET /api/grade/presets` → 17 engine entries, each with a non-empty `look_params` (`look_engine.py::list_engine_looks` `:562-568`), 12 legacy `mode:"preset"` entries. So the frontend has the data it needs to request each thumbnail cube.
4. **The reference still is served.** `GET http://localhost:3000/look-thumb-ref.jpg` → 200, image/jpeg, 45257 bytes (valid 768×512 JPEG).
5. **The exact thumbnail math produces correct, distinct thumbnails offline.** `backend/scripts/_diag_look_thumbs.py` bakes each look's cube through the *real* `bake_cube_text` path (identity CDL + look grid, `rec709_v1`) and samples the reference still through it — the identical operation the WebGL renderer performs. Output `backend/logs/look_thumbs_sheet.png` shows 16 visibly distinct, correct looks (B&W Film is monochrome, Kodak 2383 is warm/contrasty, Moody Cinematic is teal-shadowed, etc.).

**Conclusion:** every layer that can be exercised without a logged-in browser is healthy — backend endpoint, flag independence, `look_params` payload, reference image, and the cube→sample math. The blank thumbnails are therefore a **client-side WebGL/DOM problem in `frontend/src/components/preview/look-thumbnail.ts`**, not a backend or flag problem.

### B.2 — Root cause (client-side) + fix

The thumbnail renderer builds ONE shared, module-level renderer and caches its result **permanently**:

```97:167:frontend/src/components/preview/look-thumbnail.ts
let rendererPromise: Promise<Renderer | null> | null = null;
...
async function getRenderer(): Promise<Renderer | null> {
  if (!rendererPromise) {
    rendererPromise = (async () => {
      ...
      const gl = canvas.getContext("webgl2", ...);
      if (!gl) return null;
      ...
      try { image = await loadImage(REF_IMAGE_SRC); } catch { return null; }
      ...
    })();
  }
  return rendererPromise;
}
```

`rendererPromise` is assigned once and never reset. If the **first** `getRenderer()` call resolves to `null` for *any* reason, that `null` is cached for the entire page session and **every** `LookCard` then falls into its neutral/failed swatch — i.e. *all* thumbnails blank until a hard reload. `requestLookThumbnail` returns `null` on that path (`:207-209`), and `LookCard`'s `LookThumbnail` renders the flat `--sidebar`/`--border` swatch (`look-card.tsx:38-43`) — exactly the "shows none" symptom.

The realistic triggers for that first-call `null`, given the current repo state:
- The reference still `public/look-thumb-ref.jpg` was **just replaced** (uncommitted; 11155 → 45257 bytes). If the first render fired before the new file was in place / while the dev server had a stale 404 cached for that path, `loadImage` rejects → `getRenderer` caches `null` for the session.
- A transient WebGL2 context/compile hiccup (e.g. context-count pressure alongside the live preview player's own WebGL2 context from `lut-gl.ts`) on the first call, cached forever.

**Concrete fix (primary): make `getRenderer()` non-poisoning.** Do not cache a `null` outcome — only cache a successful `Renderer`. On failure, clear `rendererPromise` so the next card retries:

```ts
async function getRenderer(): Promise<Renderer | null> {
  if (!rendererPromise) {
    rendererPromise = buildRenderer(); // the existing async IIFE body
  }
  const r = await rendererPromise;
  if (!r) rendererPromise = null; // don't poison the singleton; let the next call retry
  return r;
}
```
Apply the same "don't cache the failure" discipline to `requestLookThumbnail`'s per-look path if desired (it already dedupes via `pending` and clears in `finally`, so a failed cube fetch already retries on the next mount — no change strictly needed there).

**Verification once fixed will require the browser (see B.3).** Because the entire non-browser path is proven healthy, the fix is the singleton hardening above plus a browser confirmation that pins whether anything else remains.

### B.3 — How to verify (browser)

1. Open the app, start/open an edit thread, open the Colour grading panel so the Look gallery mounts.
2. **Network tab:** confirm one `GET /api/grade/cube?...&look_engine=...` per look, each `200` with a `text/plain` `.cube` body (not 304-empty, not 401). Confirm `GET /look-thumb-ref.jpg` is `200`.
3. **Console:** confirm no WebGL2 "context lost / too many contexts" warnings and no shader-compile errors from `look-thumbnail.ts`.
4. **Expected result:** each card shows the reference still put through that look — thumbnails **differ per look** and match `backend/logs/look_thumbs_sheet.png` (and match what the live preview paints when that look is selected). A hard reload should no longer be able to leave every card blank.

If, after the singleton fix, thumbnails still fail: the Network/Console evidence from step 2–3 will point at the exact remaining cause (auth on the cube fetch in a *non-dev* build, a CORS header on the `Response` in `routers/grade.py:152-161`, or a WebGL capability gap), each of which has a targeted fix — but none of these are reproducible in the current dev environment, where the full path is verified working.

---

## Risks / things to confirm with the user

1. **`grade_tone_contrast` — DECIDED (user-confirmed): keep the code, hardwire OFF, remove the flag.** Not deleted. The S-curve plumbing stays inert (always `0.0`) so it can be re-enabled in code later; both settings are removed and the `job.py` call site passes a literal `0.0`. **Trade-off:** no *global* contrast knob in the product; per-look contrast (each `LookSpec.contrast`) is unaffected. No further sign-off needed.
2. **Additive features currently OFF get turned ON.** `grade_subject_exposure`, `grade_skin_vibrance`, and `grade_film_texture` are `False` today; the plan hardwires them on (film_texture look-scoped). The user has said these are wanted, so this is settled — but flag that it *is* a behavior change from today's default output and will be part of the mandatory re-grade.
3. **Hardwire-on vs. keep a user-facing UI toggle** — per additive feature, recommendation:
   - `grade_even_lighting` — **hardwire on globally.** It's a correctness/consistency baseline; no user reason to disable per project.
   - `grade_semantic` — **hardwire on globally.** Infrastructure for grouping/subject signals; degrades gracefully when metadata is absent.
   - `grade_look_engine` — **hardwire on globally.** It's the go-forward look system; "no look" is already expressed by selecting None, not by a flag.
   - `grade_subject_exposure` — **hardwire on globally.** Rides existing subject boxes; no per-project need.
   - `grade_skin_vibrance` — **hardwire on globally**, *but* this is the one additive feature with fairness sensitivity (skin tint). Recommend on-by-default; if the user wants an escape hatch, a per-look/per-project intensity in the Look UI is more appropriate than a dev flag.
   - `grade_film_texture` — **scope to the look (recommended), no global toggle.** Grain/halation apply exactly when the selected engine look declares them; the film-family cards already advertise this with the `grain` badge.
4. **Mandatory re-grade.** After the hash bump, all projects show stale/identity color until `_grade_all_projects.py` reruns. Confirm timing / that it's acceptable to run the batch re-grade on deploy.
5. **`presets.py` retained.** Legacy CDL presets stay resolvable (backward-compat) but hidden from the UI. Confirm no desire to hard-delete them (would break any saved document referencing a `preset_id`).

---

## Ordered implementation checklist

1. **Decisions (settled):** `grade_tone_contrast` → keep code, hardwire OFF, remove flag (step 5). `subject_exposure`/`skin_vibrance`/`film_texture` → turned ON (step 4). No open blockers.
2. **Part B first (quick win, low risk):** harden `getRenderer()` in `look-thumbnail.ts` (don't cache `null`); verify thumbnails in-browser per B.3.
3. **Competing versions — delete legacy branches:**
   a. `grade_pipeline`: hardwire v1 in `layers.resolve` + callers (`render/tasks.py`, `observe.py`, `edit_threads.py`); collapse `resolve_clip_grade`'s `pipeline` to v1-only; delete legacy inline-resolve.
   b. `grade_shot_match_v2`: always pass `references`/`groups` in `job.py`; delete the anchor-only fallback in `match.py`.
   c. `grade_scene_join`: always run the cut_record join in `job.py`; keep the RGB *runtime* fallback, delete the flag guards.
4. **Additive capabilities — remove flags, keep code:** `grade_even_lighting`, `grade_semantic`, `grade_look_engine`, `grade_subject_exposure`, `grade_skin_vibrance` unconditional; `grade_film_texture` → look-declared (drop `film_texture_enabled`, keep the `halation/grain > 0` runtime checks).
5. **`grade_tone_contrast` — remove flag, keep code OFF:** delete both settings from `config.py`; in `job.py` drop the two `tone_contrast` entries from the hash `flags` and hardwire the `tone_contrast=` argument to `resolve_clip_grade` to a literal `0.0`. Leave the S-curve plumbing (`tone.py`, `lut_bake.py`, `cache.py`, `cdl.py`, `resolver.py`, `compositor.py`, frontend `api.ts`/`grade-cube-client.ts`/`look-thumbnail.ts`) untouched — inert at `0.0`.
6. **Remove all removed settings from `config.py`** (`grade_pipeline`, `grade_even_lighting`, `grade_semantic`, `grade_shot_match_v2`, `grade_scene_join`, `grade_subject_exposure`, `grade_skin_vibrance`, `grade_look_engine`, `grade_film_texture`, `grade_tone_contrast`, `grade_tone_contrast_strength`).
7. **Hash:** bump `INPUT_HASH_SCHEMA_VERSION` 10 → 11 in `job.py`; prune/remove the `"flags"` payload.
8. **Update tests/scripts** enumerated in A.5 to the new signatures (do NOT keep dead flag-off tests).
9. **Re-grade all projects** via updated `_grade_all_projects.py`; confirm preview color for a sample of projects.
10. **Confirm** the Look gallery renders live, per-look thumbnails that match the applied grade (final Part B check).
