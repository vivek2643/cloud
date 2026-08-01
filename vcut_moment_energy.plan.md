# vcut_moment_energy.plan.md — Flag-based, energy-emergent video cuts

## 0. Goal & non-goals

**Goal.** Make the vcut video pipeline produce **long, loose cuts by default** and make the **energy dial actually work**, by moving to a *flag-based, energy-emergent* model:

- **Pass 1 (flash-lite) plants "moment flags" only** — a keep-frame timestamp, a `build`/`settle`/`both` shape, and a one-line summary. It never draws cut boundaries, never groups, never filters, never selects "good" ones, never selects question ids.
- **Cut geometry is a pure deterministic function of a single global `energy`** over those flags + the seam curve `S(t)`: each flag grows a span (direction from its shape, reach from `1 − energy`), edges snap to clean seams, a *strong* seam walls off different content, and **overlapping spans fuse into one loose cut**. Energy 0 → few long cuts holding many moments; energy up → the moments shrink, overlaps break, and the long cut *falls apart* into more, tighter cuts.
- **The energy dial re-resolves live** (it already can, server-side) so the editor sees loose↔tight in real time.

**Non-goals (explicitly deferred).**
- **No selection / no dropping.** Nothing is ever ranked or discarded in this phase. Every flag survives; energy only changes *extent* and (as a side-effect of extent) *cut count*. "Best moment" selection is a later phase. ("Best *take*" for speech retakes already exists and is untouched.)
- **No shot-point detection.** Deliberately out of scope; `S(t)` (which already excludes shot points) is the only boundary signal.
- **No pace/reframe/speed-ramp.** Video cuts stay flat-pace, as today.

**Framing insight for the implementer:** most of the machinery already exists in `backend/app/services/vcut/resolve.py` (energy→keep-width `_window_width_ms`, tag asymmetry `_tag_window`/`TAG_SPLIT`, emergent merge `_clamp_windows`/`_merge_windows`, seam-snap `_snap_group_edges`). Today it runs **per VLM loose-cut**, clamped to that loose-cut's `span`. This plan **flattens it to a file-wide pool of flags** and adds **strong-seam walls**. Keep the reused helpers; change the input types, the per-file iteration, and the neighbor-clamp.

---

## 1. Invariants / principles

1. **The VLM only plants flags.** It supplies *semantics* (where the payoff is, its shape, what it is). It never supplies geometry (boundaries, counts, merges).
2. **Geometry is deterministic and reversible.** `cuts = f(flags, S(t), energy)`. Same inputs + same energy ⇒ same cuts. Re-resolving at a new energy needs no model call and no I/O beyond the persisted artifacts.
3. **Energy only shrinks.** Raising energy only ever shrinks a flag's reach; it never invents footage or merges across a strong seam. Cut count is a *side effect* of shrinking (overlaps breaking), never a selection.
4. **Nothing is dropped.** No flag is discarded for being "weak." (The existing dead-air floor `_drop_dead_air` only removes peak-less candidates, which by construction don't occur; keep it as a defensive invariant.)
5. **`S(t)` decides the frame, never the cut.** Seams snap edges and wall off content; they never *create* a cut on their own.
6. **vcut isolation stays intact** (per the existing plans' principle 7): only `app.services.l3.post.CutRecord/PaceEnvelope` and `ingest_store` are imported from L3.

---

## 2. Pass 1 → flag-only

**File:** `backend/app/services/vcut/pass1.py` (and its input dataclasses in `backend/app/services/vcut/resolve.py`).

### 2.1 New prompt (replaces `_task_text()`, lines 47–93)

Replace the entire `_task_text()` body with the flag-only prompt. It drops: the speech warning, the "exactly one entry in clips" line, `meaning`, `loose_cuts`, nested `peaks`, and `question_ids`/the whole question bank.

```
You are watching sampled frames from raw video footage, in time order, so an
editor can later turn it into finished videos.

Your only job: mark every distinct MOMENT worth landing on — a single thing
happening that an editor might want to cut to. For each moment, give:
  - t_ms: the timestamp of the HEART of the moment — the single frame the cut
    would land on (the payoff).
  - shape: how the interest sits around that frame:
      build  — it builds UP TO this frame (the run-up matters; land on this)
      settle — this frame IS the peak, then it eases off (the after matters)
      both   — interesting roughly equally before and after
  - summary: one short sentence naming what the moment is.

Mark as MANY moments as the footage holds — do NOT group them, filter them, or
judge which are good; that happens later. If an action repeats (several reps of
the same thing), mark EACH occurrence as its own moment. Do not describe
boundaries, camera work, counts, or per-frame detail — only each moment's frame,
its shape, and its one-line summary.
```

`NEUTRAL_SYSTEM` (lines 41–44) is unchanged. The frame transport (`build_frame_blocks`, inline base64 JPEGs, cached-content path) is unchanged.

### 2.2 New response schema (replaces `Pass1Schema` and its sub-models, lines 96–130)

Flat moments per file. Drop `_LooseCutOut`, `_ClipOut.meaning`, `_ClipOut.question_ids`, the `_QuestionId = Literal[questions.ALL_IDS]` enum, and `_check_nonempty`'s clip wording.

```python
class _MomentOut(BaseModel):
    t_ms: int
    shape: Literal["build", "settle", "both"] = "both"
    summary: str = ""

class _FileOut(BaseModel):
    file_id: str
    moments: List[_MomentOut] = []

class Pass1Schema(BaseModel):
    files: List[_FileOut] = []
```

`_check_nonempty` → require at least one file with ≥1 moment. Keep the `shape` `Literal` (the live-tested reason the tag comes back correct is documented at lines 106–113 — the same rationale applies).

### 2.3 `run_pass1` (lines 196–242)

- Build the plan from `parsed.files[*].moments[*]` instead of `parsed.clips[*].loose_cuts[*].peaks[*]`.
- Keep the **defensive clamp** idea: a flag whose `t_ms` doesn't fall in any non-speech span for its file snaps to the nearest non-speech span (reuse `_find_enclosing_span`, lines 133–150; adapt `_clamp_loose_cut` → `_clamp_moment` that clamps a single `t_ms` rather than a span).
- Remove `questions.validate_question_ids(...)` (line 238) and the `questions` import (line 31).
- Return the new plan type (§2.4).

### 2.4 Input dataclasses (in `resolve.py`, lines 32–93)

Replace `Peak`/`LooseCut`/`ClipLoosePlan`/`LooseCutPlan` with a flat moment model. **Keep the class names `LooseCutPlan`/`ClipLoosePlan`? No — rename** for clarity, but preserve a back-compat `from_dict` (see §7.3):

```python
@dataclass
class MomentFlag:
    t_ms: int
    shape: str = "both"      # build | settle | both
    summary: str = ""

@dataclass
class FilePlan:
    file_id: str
    flags: List[MomentFlag] = field(default_factory=list)
    def to_dict(self) -> dict: ...        # {"flags": [{"t_ms","shape","summary"}, ...]}
    @staticmethod
    def from_dict(file_id, data) -> "FilePlan": ...   # handles BOTH new & legacy shapes (§7.3)

@dataclass
class MomentPlan:
    files: List[FilePlan] = field(default_factory=list)
    def to_dict(self) -> dict:  # {file_id: {"flags":[...]}}
    @staticmethod
    def from_dict(data) -> "MomentPlan": ...
```

`pass1.py` imports these instead of `ClipLoosePlan/LooseCut/LooseCutPlan/Peak`.

> Note: `meaning` (per-clip) and `question_ids` are gone. The **summary now lives per flag**. Pass 2's contract changes accordingly (§9).

### 2.5 Pass 1 runs PER FILE (bounded output — the real fix, not a token-cap bump)

**Problem this fixes.** The flag-only prompt asks for "as MANY moments as the footage holds," so a **single call over all files** emits output that scales with *total footage*. On a real 44-file project it blew past a 48k-token budget and truncated (twice) — and truncation silently drops real cuts, which violates §1.4 ("nothing is dropped"). Raising `max_tokens` only moves the ceiling; it is a band-aid, not a fix.

**Why per-file is correct (loses nothing).** Pass 1 is now *flag-only with no cross-file reasoning* — grouping, selection, and questions are all deferred (§1.1, §9). A file's flags do **not** depend on any other file. So batching every file into one call buys nothing but the ceiling.

**Change.** `run_pass1` fans out to **one `complete_gemini` call per file**, each emitting only that file's moments, then merges the results into a single `MomentPlan`.

- **Per-call schema.** Each call returns just that file's moments — collapse `Pass1Schema` for the call to `class Pass1Schema(BaseModel): moments: List[_MomentOut] = []` (the `file_id` is known from the call, not echoed by the model). `_FileOut`/`files` disappear from the wire; the merge sets `FilePlan.file_id` itself. `_check_nonempty` → "at least one moment" (per call), and a file legitimately yielding zero moments is fine (contributes an empty `FilePlan`, not an error).
- **Output is now bounded by one file**, so the truncation ceiling is gone permanently — 44 files, 400 files, or one 2-hour clip all complete. Keep `max_tokens` at a sane per-file value (the old `48000` is now vastly more than one file needs).
- **Concurrency.** The calls are independent → run them concurrently with a bounded pool (e.g. 4–8 in flight) so 44 files don't serialize. Merge in deterministic `file_rows` order regardless of completion order.
- **Per-file failure is isolated.** A single file's Pass 1 error retries that file only; on persistent failure that file contributes an empty `FilePlan` (logged) — never aborts or truncates the whole project. Strictly better than today's all-or-nothing.

**Frames / caching (§3, `vcut_pass2_rich` shared cache).** Bounding output alone already fixes truncation, but bound the **input** too so each call reads only its own file's frames:
- **Recommended:** a **per-file `CachedContent`** (that file's frame blocks from `build_frame_blocks` for a single file), reused by that file's Pass 1 call *and* its Pass 2 enrich. Input per call = one file's frames (cache-discounted), not the whole project.
- **Acceptable interim:** keep the single shared cache but scope each call's prompt/schema to one file. Output is bounded (fixes truncation) but each call re-reads the whole cache → input cost multiplies by file count. Note this trade-off; prefer per-file caches.
- **Uncached fallback** (`build_frame_blocks`) is already per-file-structured — just pass a single file's rows/frames per call instead of all.

**Downstream is unchanged.** The merged `MomentPlan` is identical in shape to today's, so §3–§7 (resolver, store, persistence) need no change from this. Only `run_pass1`'s internals, the per-call schema, and the cost note (§12.5) change.

---

## 3. Seam function role — edge-snap + strong-seam walls

**Seam data** is consumed as the per-file dict `seam[file_id] = {hop_ms, S, action_energy, frame_diff}` (built by `orchestrate._build_seam_cache`, persisted via `store.persist_seam_and_plan`). No change to how it's built; shot points already excluded.

Two roles in the resolver:

1. **Edge snap (already implemented).** `_argmax_s_ms` / `_snap_group_edges` (resolve.py lines 236–277) snap the outer edges of a merged cut to the local `argmax S(t)` within `SNAP_MS`. Keep as-is.
2. **Strong-seam wall (NEW).** Between two adjacent flags, if the maximum `S(t)` in the interval between their refined peaks exceeds a threshold, that position is a **wall**: neither flag may grow past it, so they can never fuse even at energy 0. This is what stops genuinely different content from merging into one blob.
   - Add `STRONG_SEAM_FRAC` (params, §4) — a seam is "strong" if its `S` ≥ `STRONG_SEAM_FRAC × max(S over the file)` (or a percentile; implementer's choice, keep it in params).
   - Helper `_strong_seam_between(peak_i, peak_j, hop_ms, S) -> Optional[int]`: returns the ms position of the strongest seam in `(peak_i, peak_j)` if it clears the threshold, else `None`.

---

## 4. The resolver — grow flags into cuts (file-wide)

**File:** `backend/app/services/vcut/resolve.py`. This is the core change. Reuse helpers; change the iteration + neighbor clamp.

### 4.1 What changes vs today

- **Today:** `resolve_cuts` loops `for clip → for loose_cut → _resolve_loose_cut`, and `_clamp_windows` clamps each peak's window to the **loose-cut `span`** and to the **midpoint between neighbor peaks**. Merge only happens *within* one loose-cut.
- **New:** per **file**, pool **all** flags. Grow each; clamp to the **file extent** and to the **strong-seam wall (if any) else the neighbor midpoint**; merge across the whole file.

### 4.2 Algorithm (new `resolve_cuts` + `_resolve_file`)

For each file with valid seam (`S`, `hop_ms>0`):
1. **Refine** each flag's `t_ms` onto real content: `_refine_peak_ms` (lines 136–155) — keep verbatim.
2. **Sort** flags ascending by refined `t_ms`. Keep parallel `shapes` and `summaries`.
3. **Width** from energy: `width = _window_width_ms(energy)` (lines 163–165) — keep. (`energy 0 → REACH_MAX`, `energy 1 → REACH_MIN`.)
4. **Grow** each flag: `_tag_window(peak, shape, width)` (lines 168–170) — keep (`TAG_SPLIT` gives build/settle/both asymmetry).
5. **Clamp** (rewrite `_clamp_windows`): for flag `i`, lower/upper bound is:
   - file extent `[0, duration_ms]` (or `[min valid ms, max valid ms]`),
   - **left:** `max(extent_lo, wall(i-1,i) if strong else midpoint(i-1,i))`,
   - **right:** `min(extent_hi, wall(i,i+1) if strong else midpoint(i,i+1))`.
   The midpoint fallback preserves today's emergent-merge behavior where there's no wall; the wall hard-stops fusion across content boundaries.
6. **Merge** overlapping/touching clamped windows: `_merge_windows` (lines 194–211) — keep. Each connected run = one loose cut.
7. **Representative peak** per run: `_representative_peak` (lines 214–229) — keep (`hero_ts_ms`). **Summary:** concatenate the merged flags' summaries (or take the representative flag's summary; see §4.4).
8. **Seam-snap** outer edges: `_snap_group_edges` (lines 259–277) — keep.
9. **Floors:** `_drop_dead_air` / `_widen_to_min_cut` / `_enforce_min_gap` (lines 305–356) — keep.

`_Candidate.span_lo/span_hi` (used by `_widen_to_min_cut`) become the file extent (or the wall-bounded sub-range) rather than the loose-cut span.

### 4.3 Emergent behavior (state in a docstring + prove in tests)

- **Energy 0:** `width = REACH_MAX` → adjacent flags with no strong seam between them fuse → **few long loose cuts**, each carrying several hero moments.
- **Energy ↑:** `width` shrinks → overlaps break at the midpoints → the long cut **falls apart** into more, tighter cuts. No re-combination logic — it's the same sweep at a smaller reach.
- **Strong seam:** a wall between two flags keeps them in separate cuts at *every* energy.

### 4.4 `ResolvedCut` (lines 100–108)

Rename `meaning` → `summary`. A merged cut's `summary` = the joined summaries of its flags (short), or the representative flag's summary — pick one and note it; the joined form is more informative for the loose (multi-moment) cuts. Optionally add `hero_ts_list: List[int]` carrying every merged flag's refined peak (for future UI hero-scrubbing); not required for v1 since flags are persisted in the plan.

### 4.5 params.py changes (`backend/app/services/vcut/params.py`)

- `WIDE_MS = 2500` → rename intent to **`REACH_MAX_MS`** and **raise it** (e.g. `4000`–`6000`) so energy-0 fuses into genuinely *long* loose cuts. Tunable — call out that this is *the* looseness knob.
- `TIGHT_MS = 600` → **`REACH_MIN_MS`** (energy-1 floor), keep ~600.
- `DEFAULT_ENERGY = 0.5` → **`0.0`** (§5).
- Add `STRONG_SEAM_FRAC = 0.6` (or similar) for §3's wall threshold.
- `TAG_SPLIT`, `PEAK_SNAP_MS`, `SNAP_MS`, `S_DEAD_FRAC`, `MIN_CUT_MS`, `MIN_CUT_GAP_MS` — unchanged.

### 4.6 Worked example (put in the resolver docstring + tests)

Flags at 2000, 5000, 9000 ms (one continuous scene, no strong seam between them) and 30000 ms (a strong seam at ~25000). `REACH_MAX_MS = 4000`, `energy = 0`, `shape=both`:

- 2000/5000/9000 grow to ≈[0,4000],[3000,7000],[7000,11000] → overlap → **one cut ≈ 0–11000**, snapped to clean edges, 3 heroes inside.
- 30000 grows to ≈[28000,32000], walled off by the seam at 25000 → **separate cut ≈ 26000–33000**.
- At `energy = 0.7` the 2000/5000/9000 windows shrink below their midpoints (2500/3500... wait: midpoints 3500 & 7000) → they stop overlapping → **three cuts**. Same three moments, no re-combine.

---

## 5. Default energy = 0 (loose)

**File:** `backend/app/services/vcut/orchestrate.py`, the ingest call (currently line 164):

```python
resolved = rv.resolve_cuts(plan, seam_cache, energy=DEFAULT_ENERGY)
```

With `DEFAULT_ENERGY = 0.0` (§4.5) this now emits long loose cuts by default. No other change here for geometry.

---

## 6. Make the energy dial actually work

### 6.1 Root cause (verified)

- The **server re-resolve endpoint already works**: `POST /api/projects/{id}/cuts/energy` (`backend/app/routers/projects.py` lines 113–150) loads `seam_cache`+`loose_plan`, calls `resolve_cuts(plan, seam, energy=body.energy)`, rebuilds records, and replaces only `kind='video'` rows (`delete_video_cuts_for_run`). It needs no model call/IO.
- The **frontend never calls it.** `setCutsEnergy` exists (`frontend/src/lib/api.ts` ~lines 371–377) but has **no call site**. The on-screen dial in `frontend/src/components/cuts-view.tsx` is pure client-side view-math (`tightenedSpan`/`playSegments`/`resolveCluster`) over the stored `pace` envelope — and since `store._pace_for` writes `pace.min_ms == natural_ms` with empty `salience`, that math resolves to "no change." Hence inert.

### 6.2 Fix — wire the dial to the server re-resolve

**File:** `frontend/src/components/cuts-view.tsx`.

- Replace the client-side `tightenedSpan`/`resolveCluster` view-math path for **video** cuts with a call to the real re-resolver: on dial change, **debounce** (~300–400 ms) → `await setCutsEnergy(projectId, energy)` → refetch `GET /cuts` → re-render. Persist on release.
- Default slider position = **0** (loose), matching §5.
- Keep optimistic UX if desired (show a spinner/skeleton while the round-trip runs); the endpoint is cheap (no model/IO).
- Speech cuts keep any existing `remove_spans` behavior — the endpoint only touches `kind='video'`.

### 6.3 Fix — the endpoint must not re-enrich on every drag

`set_cuts_energy` currently calls `defer_vcut_enrich(project_id, ingest_run_id)` (projects.py line 150) after every re-resolve. On a live system with a worker, dragging the dial would enqueue a Pass-2 (Gemini) job per change — wasteful, and it re-derives `scene_specifics` for geometry that will change again on the next drag.

**Change:** decouple enrich from the energy re-resolve. Options (pick one, note it):
- (a) Drop the `defer_vcut_enrich` call from `set_cuts_energy` entirely — geometry-only; enrich stays whatever it was from ingest. (Simplest, recommended for v1.)
- (b) Debounce/coalesce enrich to fire once after the user settles (e.g. only on an explicit "apply", not on every slider tick).

### 6.4 Persisted artifacts already suffice

`persist_seam_and_plan` (store.py lines 121–129) stores `seam_cache` + `loose_plan` (now the flags plan, §2.4). The endpoint re-resolves from those, so **all hero flags survive** for re-splitting at any energy. No extra persistence needed for the dial to work.

---

## 7. Store / data-model / migration

**File:** `backend/app/services/vcut/store.py`.

### 7.1 `build_cut_records` (lines 77–105)

- `cut.meaning` → `cut.summary` (§4.4). `label = _short_label(cut.summary)`, `summary = cut.summary`.
- `_pace_for` flat envelope (lines 71–74) can stay — the client no longer does pace math (§6.2). (Optional cleanup: leave as-is.)
- `hero_ts_ms = cut.peak_ms` (representative). Optional: also stash the merged flags' hero timestamps into the `salience` jsonb for future UI; **not required** for v1.

### 7.2 `persist_seam_and_plan` (lines 121–129)

Unchanged signature; it now receives `MomentPlan.to_dict()` (flags shape) as `loose_plan`. Column `ingest_runs.loose_plan` (jsonb, migration 052) is reused as-is — **no migration** for the plan shape change.

### 7.3 Back-compat for `from_dict`

Old runs (including the ~21 just re-ingested on the *previous* vcut model) have `loose_plan = {file_id: {meaning, question_ids, loose_cuts:[{span_ms,peaks:[...]}]}}`. The energy endpoint (§6) runs `MomentPlan.from_dict` on whatever is persisted, so `FilePlan.from_dict` **must detect both shapes**:
- If `data` has `"flags"` → new shape.
- Elif `data` has `"loose_cuts"` → legacy: flatten every `loose_cut.peaks[*]` into `MomentFlag(t_ms, shape=tag, summary=meaning)`. This lets the dial keep working on old runs until they're re-ingested.

### 7.4 Migration

Only needed for speech observability (§8). If choosing a new column there, add `backend/.../054_*.sql`; otherwise none.

---

## 8. Speech channel — make the silent fallback loud

**File:** `backend/app/services/vcut/orchestrate.py`, `_run_speech_channel_with_fallback` (lines 114–128).

Today it catches **all** exceptions from `run_speech_channel` and silently falls back to `store.copy_prior_speech_cuts` — so a broken speech pipeline looks identical to a working one (old cuts get copied). Keep the fail-open behavior; make it **observable**:

- Return/record a **status**: `speech_source ∈ {"pipeline", "copy_prior"}` plus the exception summary on fallback.
- Persist per run: add `ingest_runs.speech_channel_status` (jsonb or text; migration 054) **or** stuff `{source, error}` into an existing meta/jsonb on the run. Surface it in `GET /cuts` so the frontend/logs can show "speech: pipeline" vs "speech: fallback (reason)".
- Log at `warning`/`error` (already `logger.exception`) — additionally increment a counter or print in the re-ingest script so batch runs report how many projects fell back.

This is self-contained and independent of the geometry work; it directly answers "is the new speech pipeline actually running, or silently degrading?"

---

## 9. (Phase 2) Questions move out of Pass 1, formed from summaries

Because Pass 1 no longer emits `question_ids`, Pass 2's input contract changes.

**Files:** `backend/app/services/vcut/pass2.py` (`run_enrich`, `scene_specifics_from_answers`) and wherever it reads `loose_plan[...].question_ids`.

- **Today:** Pass 1 selects a closed-bank `question_ids` per clip; Pass 2 answers them.
- **New:** Pass 2 derives what to answer **from each cut's summary** (per moment). Two options — flag as an open decision, don't over-build:
  - **(a) Closed bank, selected in Pass 2 (recommended):** keep `questions.py`'s bank; a cheap step maps each summary → relevant bank ids, then answers them. Structured/filterable, consistent.
  - **(b) Dynamic from summary:** generate free-form specifics per summary. More specific, less consistent/parseable.
- Mark this section **Phase 2**: the geometry + dial + speech-observability work (§2–8) ships first; `scene_specifics` can temporarily fall back to summary-only until this lands. Ensure removing `question_ids` from the plan doesn't crash Pass 2 in the interim (guard the read).

---

## 10. File-by-file change list

| File | Change |
|---|---|
| `backend/app/services/vcut/pass1.py` | New `_task_text()` (§2.1); per-call `Pass1Schema` collapses to `moments: List[_MomentOut]` (§2.2, §2.5); **`run_pass1` fans out one call PER FILE, concurrent + merged, per-file failure isolated** (§2.5); builds `MomentPlan`, drop `questions` import + `validate_question_ids` (§2.3); `_clamp_moment` from `_clamp_loose_cut`. |
| `backend/app/services/vcut/resolve.py` | New `MomentFlag`/`FilePlan`/`MomentPlan` dataclasses w/ back-compat `from_dict` (§2.4, §7.3); rewrite `resolve_cuts`→ per-file flag pool (§4.2); rewrite `_clamp_windows` to use strong-seam walls + midpoints (§3, §4.2 step 5); add `_strong_seam_between`; `ResolvedCut.meaning`→`summary` (§4.4). Keep `_refine_peak_ms`, `_tag_window`, `_merge_windows`, `_representative_peak`, `_snap_group_edges`, floors. |
| `backend/app/services/vcut/params.py` | `WIDE_MS`→`REACH_MAX_MS` (raise, e.g. 4000–6000); `TIGHT_MS`→`REACH_MIN_MS`; `DEFAULT_ENERGY=0.0`; add `STRONG_SEAM_FRAC` (§4.5, §5). |
| `backend/app/services/vcut/orchestrate.py` | `energy=DEFAULT_ENERGY` now 0 (§5); build per-file frame cache(s) for the per-file Pass 1 fan-out (§2.5) instead of one shared all-files cache; `_run_speech_channel_with_fallback` records/returns speech status (§8). |
| `backend/app/services/vcut/store.py` | `build_cut_records` uses `cut.summary` (§7.1); `persist_seam_and_plan` receives flags plan (§7.2, no code change). |
| `backend/app/routers/projects.py` | `set_cuts_energy`: drop/decouple `defer_vcut_enrich` (§6.3); surface `speech_channel_status` in `GET /cuts` (§8). |
| `frontend/src/components/cuts-view.tsx` | Wire dial → `setCutsEnergy` (debounced) + refetch; remove client `tightenedSpan` view-math for video; default energy 0 (§6.2). |
| `frontend/src/lib/api.ts` | `setCutsEnergy` already exists — ensure it posts `{energy}` and returns the refreshed cuts (§6.2). |
| `backend/app/services/vcut/pass2.py` | Phase 2: derive questions from summaries; guard against missing `question_ids` in the interim (§9). |
| `backend/scripts/test_vcut_resolve.py` | New tests (§11). |
| `backend/.../054_*.sql` | Only if adding `ingest_runs.speech_channel_status` column (§8). |

---

## 11. Testing

Follow the existing pure-unit style under `backend/scripts/test_*.py` (e.g. `test_vcut_resolve.py`, `test_speech_delivery.py`): synthetic `S(t)`/flags, no I/O, printed `ok`/`FAIL`.

New/updated `test_vcut_resolve.py` cases:
1. **Fuse at energy 0:** flags 2000/5000/9000 with flat weak `S` between → one cut spanning all three; hero = representative.
2. **Fall apart as energy rises:** same flags at energy 0.7 → three cuts; assert boundaries hug each flag; assert **monotonicity** (cut count non-decreasing, each cut length non-increasing as energy 0→1).
3. **Strong-seam wall:** inject a high-`S` spike between two flags → they stay in separate cuts even at energy 0.
4. **Shape asymmetry:** `build` keeps run-up (edge extends backward), `settle` keeps the landing (forward) — assert via `TAG_SPLIT`.
5. **Flag-only input / back-compat `from_dict`:** new `{"flags":[...]}` and legacy `{"loose_cuts":[...]}` both resolve; legacy peaks flatten to flags.
6. **Floors:** `MIN_CUT_MS`, `MIN_CUT_GAP_MS`, dead-air invariant still hold.
7. **Pyflakes clean** on all touched backend modules.

Manual/live: re-ingest one project, confirm loose long cuts at energy 0; drag the dial and confirm cuts tighten and split live (network call to `/cuts/energy`).

---

## 12. Rollout

1. **Order:** ship §2–§8 (geometry + dial + speech-observability) first; §9 (questions) is a follow-up. §8 is independent and can land anytime.
2. **Back-compat:** `from_dict` handles both plan shapes (§7.3), so the dial works on existing runs immediately; but existing runs still carry *old* geometry until re-ingested.
3. **Re-ingest:** after implementation, re-run the new pipeline across projects (the synchronous, no-workers script pattern already used: `backend/scripts/_reingest_all_vcut.py`). The ~21 projects re-ingested on the *previous* vcut model must be re-ingested to get flag-based loose cuts. Watch the new **speech_channel_status** to see real-pipeline vs fallback rates.
4. **Tunables to sweep on real footage:** `REACH_MAX_MS` (looseness of energy-0 cuts) and `STRONG_SEAM_FRAC` (how eagerly different content walls off). Start `REACH_MAX_MS≈5000`, `STRONG_SEAM_FRAC≈0.6`; adjust by eye.
5. **Cost:** Pass 1 is now **one flash-lite call per file** (§2.5) rather than one per project — same total frames analyzed, but bounded output per call (no truncation) and concurrent. With per-file caches the input stays cache-discounted; the interim shared-cache path costs more input (re-reads the cache per file) so prefer per-file caches. The dial re-resolve is still free (no model/IO). Ensure §6.3 so the dial doesn't trigger paid Pass-2 enrich.
