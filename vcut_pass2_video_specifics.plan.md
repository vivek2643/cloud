# vcut_pass2_video_specifics.plan.md

Rework Pass 2 ("specifics") into a **cheap, high-quality, video-grounded**
enrichment that (a) reuses a **per-file video cache** shared with Pass 1,
(b) asks **content-aware questions** chosen by a text-only planner, and
(c) attaches specifics to **moment flags** (not ephemeral cut ids) so the
energy dial composes them for free and never wipes them.

Everything stays on **gemini-3.1-flash-lite**. The quality lift comes from
better questions + real video, not a pricier model. Cost is minimized by
paying for each file's video exactly once (cache creation) and reading it
from cache for both Pass 1 and Pass 2.

---

## 0. Why (the three problems this fixes)

1. **Specifics are generic.** Question selection is disabled: every cut gets
   `questions.DEFAULT_QUESTION_IDS` (`subject/action/moment_type`). Pass 2
   asks the same thing whether it's a piano recital or a product demo.
   (`pass2._question_ids_by_file` returns the default set for every file --
   see its own docstring: "Phase 2 interim ... deferred".)

2. **Specifics aren't grounded in video.** In video mode there is **no Pass 2
   cache at all** (`orchestrate.run_vcut_ingest` leaves `cache_handle=None`
   in the `video` branch), so `pass2.run_enrich` falls back to
   **re-extracting one hero still per cut** and guessing motion from a single
   frame. `motion_quality`/`continuity_cue`/"what happens" are not answerable
   from one frame.

3. **The cut is the wrong home for a specific.** `scene_specifics` is written
   per **cut id** (`l3store.update_cut_scene_specifics`). Cuts are energy-
   dependent (a loose cut fragments into N as energy rises), so every energy
   re-resolve deletes+reinserts video rows with new ids and **NULLs the
   specifics**. The current mitigation (`_snapshot_video_specifics` /
   `_resolve_energy_preserving` in `routers/projects.py`) re-maps by hero
   containment -- a workable band-aid, but it is lossy at high energy and only
   exists because specifics live on the wrong object.

---

## 1. Principles / invariants

- **One model everywhere: `settings.vcut_pass1_model` (flash-lite).** Pass 1,
  the video cache, and Pass 2 all use it. A Gemini `CachedContent` is bound to
  the model that creates it; keeping one model is what lets Pass 1 **and** Pass
  2 share the same per-file video cache. (A stronger Pass-2 model is an
  explicit future knob -- section 12 -- and would require its own cache.)
- **Pay for each file's video once.** The per-file video is uploaded once
  (already done) and turned into a `CachedContent` once. Pass 1 reads it, Pass
  2 reads it. Cache reads are billed at the discounted cached-input rate; the
  variable cost is output tokens.
- **Pass 1 stays a lean segmenter.** It only plants moment flags
  (`t_ms/shape/summary`). Do NOT add description/questions to Pass 1 -- keep
  its output bounded and its segmentation reliable.
- **Specifics live on moment flags, composed onto cuts by `resolve.py`.** This
  retires the preservation band-aid: energy re-resolve recomputes cuts from
  the plan, so composed specifics come along automatically and can never be
  wiped.
- **No hard failures for cost optimizations.** Cache creation, planner, and
  Pass 2 all degrade gracefully (fall back to inline video / default
  questions / summary-only specifics) -- never abort a run over enrichment.
- **No band-aids.** Fix the object model (specifics on flags), don't paper
  over it.

---

## 2. Architecture at a glance

```
prep (video mode):
  spans -> non-speech per file
  seam cache
  cut + upload non-speech subclip per file            [exists: _prepare_video_inputs]
  NEW: create per-file video CachedContent (flash-lite) from each upload
       -> video_cache_by_file: {file_id: cache_name}

Pass 1 (per file):
  reference file's video cache -> moment flags          [rework: cached instead of inline]

resolve @ DEFAULT_ENERGY -> video cut_records            [exists]

NEW question planner (text-only, one cheap call):
  all moment summaries + local transcript + L1 hints
  -> genre + per-moment question_ids (+ optional custom probes)
  -> written onto plan flags, plan re-persisted

Pass 2 / enrich (per file, INLINE while cache is warm):
  reference file's video cache + that file's moments&their question_ids
  -> per-moment specifics
  -> written onto plan flags, plan re-persisted
  -> re-resolve @ DEFAULT_ENERGY, rewrite cut_records (specifics composed)

teardown: delete video caches + uploaded files
```

---

## 3. Per-file video cache (shared by Pass 1 + Pass 2)

**File:** `backend/app/services/vcut/orchestrate.py`,
`backend/app/services/llm/ingest_gemini.py`,
`backend/app/services/vcut/store.py`.

### 3.1 New adapter: create a cache from an uploaded video
`ingest_gemini.py` already has `create_pass2_cache(system, blocks, model,
ttl_seconds)` (builds a `CachedContent` from `Block`s via
`_parts_for_content`) and `upload_video(path) -> {uri, name}`. Add a thin
helper that builds the cache from a **video file block**:

```python
def create_video_cache(system: str, file_uri: str, *, model: str,
                       ttl_seconds: int) -> Optional[str]:
    """CachedContent whose body is one uploaded video (Files API uri),
    created with `model` so both Pass 1 and Pass 2 can read it with that
    same model. None on failure / below the model's min-cache-token floor
    -> caller falls back to sending the video inline."""
```
Implement it exactly like `create_pass2_cache` but with
`blocks=[video_file_block(file_uri)]` (reuse `create_pass2_cache` if a
`video_file_block` flows through `_parts_for_content` cleanly -- verify;
otherwise add `create_video_cache` next to it). NEVER raise: return `None`.

> **Min-token floor:** Gemini requires a minimum token count to cache. A very
> short sub-clip may be below it and `create_video_cache` returns `None`.
> That's fine -- Pass 1 sends that file's video inline (today's path) and Pass
> 2 sends it inline too (section 5.3). Only the *cost optimization* is lost.

### 3.2 Build caches in `_prepare_video_inputs`
Extend the returned handle so each file carries its cache name. Simplest:
after `upload_video` succeeds, call `create_video_cache(p1.NEUTRAL_SYSTEM,
handle["uri"], model=settings.vcut_pass1_model,
ttl_seconds=settings.vcut_cache_ttl_s)` and stash it. Return a
`video_cache_by_file: {file_id: cache_name}` alongside `video_by_file`, and
add every created cache name to the cleanup list so teardown deletes them
(`delete_pass2_cache` works for any cache resource name).

`p1.VideoHandle` gains an optional `cache_name: Optional[str] = None`.

### 3.3 Persist cache handles for the (optional) deferred path
If Pass 2 runs inline (section 6, recommended) the caches never need to be
persisted. If you keep Pass 2 deferred, persist `{file_id: cache_name}` via a
new `store.persist_video_caches(run_id, mapping)` / `load_video_caches`
(mirror `persist_vcut_cache`/`load_vcut_cache`) and handle TTL expiry in Pass
2 (re-upload + re-cache, or inline). **Recommendation: run Pass 2 inline and
skip persistence entirely.**

---

## 4. Question planner (content-aware, text-only, cheap)

**New file:** `backend/app/services/vcut/qplan.py`. **One** text-only Gemini
call per run (no video, no frames -> nearly free). Revives + replaces
`pass2._question_ids_by_file`.

### 4.1 Inputs (all cheap, no media)
- Every moment flag's `summary` (from the plan), grouped by file, in time
  order.
- **Local transcript context** around each moment: the speech words/segments
  in a small window `[t_ms - QPLAN_CTX_MS, t_ms + QPLAN_CTX_MS]` (reuse the L1
  transcript already loaded by the speech channel; pass it in, don't re-load).
- Cheap L1 hints if handy: `screen_text` presence, `look`/framing -- optional,
  additive.

### 4.2 Two-level reasoning (single call, structured output)
1. **Genre / intent** for the whole project from all summaries + transcript:
   one of e.g. `performance | tutorial | vlog | product_demo | interview |
   event | broll | other`. Genre implies a **default relevant dimension
   template** (a constant map in `qplan.py`).
2. **Per-moment question ids**: start from the genre template, then refine per
   moment using its own summary + local transcript. E.g. transcript near a
   moment says "and here's the price" -> add `on_screen_text`; every summary
   is "playing piano" -> drop `on_screen_text/notable_object`, keep framing /
   `motion_quality` / `moment_type`.

### 4.3 Output & vocabulary
- **Closed bank + a few custom probes.** Selection is restricted to
  `questions.BANK` ids (run through `questions.validate_question_ids`), PLUS
  up to `QPLAN_MAX_CUSTOM` (e.g. 2) free-text probes per moment for genuinely
  unusual content. Custom probes are stored as `{key, prompt}` and answered by
  Pass 2 into the specifics dict under their `key`.
- **Expand the bank first.** `questions.BANK` today is monument/event-centric.
  Add edit-decision dimensions so the planner has good ids to pick from:
  `shot_size` (wide|medium|close|extreme_close), `camera_move`
  (static|pan|tilt|push|handheld), `motion_direction` (screen L→R etc., for
  match cuts), `subject_entry_exit`, `headroom_lookroom` (for reframe),
  `usable` (strong|ok|weak take), `energy_emotion`, `hook_potential`,
  `tags` (keyword array for search). Keep `output_hint`s enum-constrained
  where possible so downstream stays deterministic.

### 4.4 Where results go
Write `question_ids: List[str]` and `custom_questions: List[{key,prompt}]`
onto each `resolve.MomentFlag` (section 7), plus a run-level `genre` (persist
alongside the plan, e.g. in the plan dict under a reserved `__meta__` key or a
new `store.persist_qplan_meta`). Planner failure -> every moment gets
`DEFAULT_QUESTION_IDS` (exactly today's behavior; never abort).

### 4.5 Model
Text-only, so it's cheap even on a slightly stronger model. Default to
`settings.vcut_pass1_model` (flash-lite) for simplicity; add
`settings.vcut_qplan_model` as an optional override.

---

## 5. Pass 2 rework -- planned questions on cached video

**File:** `backend/app/services/vcut/pass2.py` (`run_enrich`).

### 5.1 Call structure (minimal cost)
**One Pass 2 call per file**, referencing that file's **video cache**, listing
that file's moments each with its own planned `question_ids` (+ custom probes)
and its `t_ms`/window as the anchor. This is the current
`_cuts_listing` shape, but per **moment** and driven by the **planner's**
ids instead of `DEFAULT_QUESTION_IDS`, and against **cached video** instead of
the frame cache / hero still.

- Reference the cache with `complete_gemini(..., cached_content=cache_name)`
  and pass only the task text as `blocks` (the cache carries the video +
  `NEUTRAL_SYSTEM`).
- Concurrency + fail-open **per file** (mirror `pass1.run_pass1`'s pool): one
  file's enrich failure logs and leaves that file's moments summary-only,
  never aborts the run.

> Deeper per-cluster calls (cluster moments by question set, one call each) are
> a **future** option (section 12) if one-call-per-file attention proves too
> shallow. The cache makes multi-call cheap when we want it. Start with
> one-call-per-file to keep cost minimal.

### 5.2 Answer schema
Replace the fixed `_AnswerOut` with a schema keyed by moment (use the moment's
`t_ms` or a per-file moment index as the id, since flags have no stable id --
section 7 adds one) carrying every **bank field** the planner might have
selected, plus a `custom: Dict[str,str]` for probe answers. Keep the
"answer only the requested fields, leave the rest default" contract
(`scene_specifics_from_answers` already trims to the selected ids -- keep that
trimming, driven by each moment's own `question_ids`).

### 5.3 Fallbacks (no cache / expired)
- No cache name for a file (short clip, creation failed) -> send the video
  **inline** in the Pass 2 call (`video_file_block(uri, fps=...)`,
  `media_resolution=...`), same content, just uncached. Needs the file's
  `file_uri` -> thread it through (or keep the upload alive until Pass 2).
- No video at all (frames-mode run) -> keep today's frame-cache / hero-still
  fallback path unchanged.

### 5.4 Write specifics onto the PLAN, not cut rows
`run_enrich` no longer calls `update_cut_scene_specifics(cut_id, ...)`.
Instead it writes each moment's answered specifics onto the matching
`MomentFlag.specifics` (match by file + moment index / `t_ms`), then:
1. re-persist the plan (`store.persist_seam_and_plan`), and
2. re-resolve at `DEFAULT_ENERGY` and rewrite `cut_records` so the shown cuts
   carry composed specifics (section 7.3).

---

## 6. Orchestration / flow (run Pass 2 inline while the cache is warm)

**File:** `backend/app/services/vcut/orchestrate.py`.

### 6.1 Recommended: fold enrich into `run_vcut_ingest` (video mode)
Caches have a TTL (`vcut_cache_ttl_s`, default 900s). Running Pass 2 as a
deferred background task risks the cache expiring between passes. Since ingest
already runs synchronously here, run the planner + Pass 2 **inline at the end
of `run_vcut_ingest`**, before `_cleanup_video_inputs` tears the caches down:

```
... Pass 1 -> resolve -> insert video cut_records -> speech channel ...
qplan.plan_questions(plan, transcript, ...)          # writes question_ids onto flags
store.persist_seam_and_plan(run, seam, plan)         # persist planner output
pass2.run_enrich_inline(project, run, plan, seam,
                        video_cache_by_file, video_by_file)  # writes specifics onto flags
store.persist_seam_and_plan(run, seam, plan)         # persist specifics
resolved = resolve_cuts(plan, seam, DEFAULT_ENERGY)  # recompose with specifics
rewrite video cut_records                             # now carry composed specifics
# finally: _cleanup_video_inputs(cleanup)  (deletes caches + uploads)
```
Keep the existing `vcut_enrich` deferred task as a thin wrapper for
frames-mode / back-compat (it loads the frame cache, no video). Video-mode
enrich happens inline.

Trade-off: cuts appear a little later (after enrich) instead of enrich-after-
show. Acceptable -- correctness + cache reuse + no TTL race win, and the
synchronous reingest already tolerates it. (If you want cuts on screen ASAP,
you can still insert the un-enriched cut_records first, then run enrich inline
and rewrite -- the frontend already polls.)

### 6.2 Teardown
Add cache names to the `cleanup` list so `_cleanup_video_inputs` deletes them
(extend it to `delete_pass2_cache(cache_name)` per file in addition to
`delete_uploaded_file`).

---

## 7. Specifics on moment flags + composition in `resolve.py`

**File:** `backend/app/services/vcut/resolve.py`.

### 7.1 Extend `MomentFlag`
```python
@dataclass
class MomentFlag:
    t_ms: int
    shape: str = DEFAULT_TAG
    summary: str = ""
    question_ids: List[str] = field(default_factory=list)      # planner
    custom_questions: List[Dict[str, str]] = field(default_factory=list)
    specifics: Dict[str, Any] = field(default_factory=dict)     # Pass 2
```
Update `FilePlan.to_dict`/`from_dict` to round-trip the new fields (keep the
legacy `from_dict` branch working -- old runs simply have empty
question_ids/specifics). This is the stable id problem's fix too: within a
file, a flag is identified by its **index in `flags`** (Pass 1 order) -- pass
that index to Pass 2 as the moment id.

### 7.2 Compose specifics onto cuts
A resolved cut may absorb several flags (loose energy). Add composed specifics
to `ResolvedCut`:
```python
@dataclass
class ResolvedCut:
    ...
    specifics: Dict[str, Any]        # composed from the group's flags
```
In `_resolve_file`, alongside `_joined_summary`, compose the group's flags'
specifics: for a single-flag cut it's that flag's specifics; for a merged
multi-flag cut, either (a) the representative peak's specifics + a `moments`
list of the others, or (b) a merged dict. Recommend: `{...representative
flag's fields, "moments": [each flag's {t_ms, summary, ...key fields}]}` so a
loose cut shows a mini shot-list. Keep it JSON-serializable for
`scene_specifics`.

### 7.3 `build_cut_records` writes composed specifics
**File:** `backend/app/services/vcut/store.py`. `build_cut_records` currently
leaves `scene_specifics` at the column default. Have it set
`scene_specifics = resolved_cut.specifics`. Then `insert_cut_records` writes it
directly -- no post-insert `update_cut_scene_specifics` needed for video cuts.

### 7.4 Retire the preservation band-aid
Once specifics are composed from the plan on every resolve, the energy
re-resolve path no longer needs to preserve/re-map them:
- In `routers/projects.py`, `set_cuts_energy` / `energy_levels` /
  `_resolve_energy_preserving` can drop the `_snapshot_video_specifics` +
  hero-containment re-map. `resolve_cuts` -> `build_cut_records` now yields
  specifics-bearing rows at any energy directly.
- Delete `_snapshot_video_specifics` and simplify `_resolve_energy_preserving`
  to just resolve+build+delete+insert (as it was pre-band-aid). Keep the
  energy_levels one-time snapshot behavior only if any non-plan field still
  needs it (it shouldn't).

> This is the clean win the whole rework buys: **specifics can never be wiped
> by the dial again, because they're derived, not stored-and-fragile.**

---

## 8. Data shapes & persistence summary

- `MomentFlag`: + `question_ids`, `custom_questions`, `specifics` (section 7.1).
- Plan dict (`MomentPlan.to_dict`): each flag round-trips the new fields;
  add run-level `genre` (planner) via a reserved meta key or a small side
  table -- keep `from_dict` back-compat for pre-existing runs.
- `ResolvedCut`: + `specifics` (composed).
- `cut_records.scene_specifics`: now written by `build_cut_records` at
  resolve time for video cuts (speech cuts keep their own path).
- `VideoHandle`: + `cache_name`.

---

## 9. Config / params

Add to `app/config.py` (Settings) / `vcut/params.py` as appropriate:
- `vcut_qplan_model` (default = `vcut_pass1_model`).
- `vcut_pass2_model` -- keep, but for cache-sharing default it to
  `vcut_pass1_model`; a different value means "no cache, own model" (section
  12). Document the coupling.
- `QPLAN_CTX_MS` (transcript window around each moment, e.g. 4000).
- `QPLAN_MAX_CUSTOM` (custom probes per moment, e.g. 2).
- Reuse existing: `vcut_cache_ttl_s`, `vcut_video_fps`,
  `vcut_video_media_resolution`, `SUBCLIP_MIN_MS`, `DEFAULT_ENERGY`.

---

## 10. Cost model (per file, video mode)

| item | when | rate |
|---|---|---|
| upload sub-clip | prep | Files API (free/cheap) |
| **create video cache** | prep | **video input once**, flash-lite rate |
| Pass 1 call | per file | cached-input read + small output |
| planner call | once/run | text-only, tiny |
| Pass 2 call | per file | cached-input read + rich output |
| teardown | end | free |

The one expensive thing -- processing the video -- is paid **once** (cache
creation). Pass 1 and Pass 2 both read from cache. Net marginal cost of adding
rich Pass 2 ≈ **its output tokens** + one cheap cached read. This is the whole
point: "stress Pass 2 hard" stays affordable.

---

## 11. Testing

- **`test_vcut_resolve.py`**: extend for `MomentFlag.specifics` round-trip and
  `ResolvedCut.specifics` composition (single-flag cut = that flag's
  specifics; merged multi-flag cut = representative + `moments` list). Confirm
  composition is energy-invariant in the sense that a moment's specifics
  always land on whatever cut contains it, at energy 0 and 1.
- **`test_vcut_store.py`**: `build_cut_records` now populates
  `scene_specifics`; `FilePlan.to_dict/from_dict` round-trips new fields;
  legacy `from_dict` still works.
- **New `test_vcut_qplan.py`** (pure/mocked): genre template application,
  per-moment id refinement from transcript context, `validate_question_ids`
  closure, custom-probe cap, planner-failure -> `DEFAULT_QUESTION_IDS`.
- **New/updated `test_vcut_pass2.py`**: per-file call builds the right
  moment→question listing; answers trim to each moment's own ids; fail-open
  per file; no-cache -> inline-video path; specifics written onto plan flags.
- **`test_projects_router.py`**: after retiring the band-aid, energy
  re-resolve at 0/0.5/1 yields specifics-bearing rows directly (no
  snapshot/re-map); a healthy project keeps full specifics coverage across
  `energy_levels` and single-energy POSTs.
- **Live smoke** (one small multi-file project, video mode, foreground): cache
  created per file, Pass 1 + Pass 2 both cached (check usage has
  `cache_read_input_tokens`), specifics populated and tailored (piano project
  drops text/graphics fields), energy dial recomposes without wiping.
- **Cost check**: log `create/read` cache token counts to confirm video input
  is paid once.

---

## 12. Explicitly out of scope (future knobs)

- **Stronger Pass 2 model (flash/pro).** Would need its **own** cache (model-
  bound) -> re-pays the video at that model's rate. Add only if flash-lite
  specifics prove insufficient; gate behind `vcut_pass2_model != pass1_model`
  and create a second per-file cache with that model.
- **Per-cluster deep Pass 2 calls.** Cluster moments by question set, one call
  each against the shared cache for deeper attention. The cache already makes
  multi-call cheap; defer until one-call-per-file quality is judged.
- **Sub-beats within a moment** (timestamped micro-beats). Not needed:
  moments are already the atoms energy fragments on, so one specific per
  moment == per-cut at max energy.
- **Reframe geometry from `headroom_lookroom`/`shot_size`.** These fields make
  the later aspect-ratio/reframe work possible but that work is separate.

---

## 13. Rollout order (for the implementer)

1. `resolve.MomentFlag`/`ResolvedCut` + `FilePlan` round-trip + composition
   (section 7.1-7.2); `build_cut_records` writes specifics (7.3). Tests green.
2. Retire the preservation band-aid in `routers/projects.py` (7.4). Router
   tests green.
3. `ingest_gemini.create_video_cache` + `VideoHandle.cache_name` +
   `_prepare_video_inputs` builds/returns/cleans up caches (section 3).
4. Pass 1 reads the per-file video cache instead of inline (section 3.2 /
   `pass1._run_pass1_for_file` video branch: pass `cached_content=cache_name`,
   drop inline `video_file_block` when a cache exists; keep inline as the
   no-cache fallback).
5. `qplan.py` planner + bank expansion (section 4). Tests green.
6. `pass2.run_enrich` rework to planned questions on cached video, writing
   specifics onto plan flags (section 5); inline wiring in `run_vcut_ingest`
   (section 6).
7. Live smoke on one project; verify cached reads, tailored specifics, dial
   recomposition. Then a full re-ingest (video mode, no fallbacks).

Each step is independently testable; steps 1-2 already deliver the "dial never
wipes specifics" fix even before the video/questions work lands.
