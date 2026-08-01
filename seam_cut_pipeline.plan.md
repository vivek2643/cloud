# Seam-Driven VLM Cut Pipeline — Implementation Plan

**Status:** FINAL PLAN — ready to hand to an implementing chat.
**Depends on:** `seam_function.plan.md` (DONE — `backend/app/services/seam/` ships `build_seam_signals` + `compute_seam_curve`).
**Prime directive:** a *separate* pipeline that produces `cut_records` for the **existing** frontend cuts channel. The current L3 ingest (`app.services.l3.ingest`) stays untouched and keeps working; this is a parallel path we can A/B and later swap in.

---

## 0. One-paragraph summary

Speech is already separated out upstream (kept exactly as-is). For the **non-speech** footage of each clip we run **one cached VLM pass** that emits *loose* cut spans, each with 1+ **interest peaks** tagged `build` / `settle` / `both`, plus a **rough meaning** per clip. A deterministic engine then uses the **seam curve `S(t)`** and a **single global energy slider** to turn each loose span into final cut boundaries: energy only ever **shrinks** (never expands), a span with one peak just tightens toward it, a span with several peaks **divides** into one cut per peak as energy rises, and every cut edge is **snapped to the cleanest nearby frame via `S(t)`**. Cuts are shown immediately after Pass 1. A background **Pass 2** (same cached frames) answers meaning-derived questions and writes scene specifics onto the same `cut_records`. Everything lands in the existing `cut_records` table under a fresh `ingest_run`, so the frontend renders it with zero display changes.

> **Out of scope (deferred, ideate later):** reframe / aspect-conversion / shot-scale matching. That is an export/delivery-layer concern, uses already-persisted signals, and touches no boundaries — it will be planned separately and is intentionally excluded here.

---

## 1. Principles / non-negotiables (from the ideation)

1. **Speech is a separate channel.** This pipeline never sees speech spans. It operates only on the non-speech complement of each file. The old speech-cut logic is reused verbatim.
2. **Seam decides the *frame*, not the *cut*.** `S(t)` is a quality field only. It never decides whether or where a cut exists — it only refines an edge to the cleanest nearby frame. (This is the explicit non-goal baked into `curve.py`.)
3. **Energy is unidirectional — shrink only.** Cuts are authored *loose* by the VLM. The slider only ever trims/divides. It never expands a cut and never merges across VLM spans.
4. **Weighting, never veto.** No hard boolean gate decides a cut. `S(t)` is a soft attractor consumed by `argmax`; peaks + tags + energy are geometry. No branch anywhere on footage type ("aerial vs push-up vs podcast").
5. **Do the expensive work once.** VLM frames are cached (Gemini explicit cache). Pass 1 + Pass 2 run off the same cache. The energy slider re-derives instantly from persisted artifacts — **no VLM, no proxy download, no motion recompute** on a slider drag.
6. **Cost first.** Default VLM model = **`gemini-3.1-flash-lite`** for both passes (single config knob so Pass 2 can be upgraded later). Sampling = "medium" over non-speech spans only.
7. **Separate pipeline, shared sink.** New code lives under `backend/app/services/vcut/`. It reuses exactly two things from the old world: the `CutRecord` dataclass and `ingest_store.insert_cut_records` (i.e. the `cut_records` / `ingest_runs` tables). Nothing else is shared or modified.

---

## 2. What stays untouched

- `app.services.l3.ingest` and everything it calls (pass1/pass2/post/v4_segment/scene_taxonomy). Do **not** edit.
- All of L1. No new L1 signals, no schema changes to `motion_dynamics`/`audio_features`.
- `app.services.seam/*` — consume it, don't change it.
- Frontend cut **rendering** (`cuts-view.tsx`) — it already reads `GET /api/projects/{id}/cuts`. The only frontend additions are (a) pointing ingest at the new task and (b) the energy slider calling a new re-derive endpoint (see §9).
- Speech cutting — reuse the existing speech-cut output as the speech channel; this pipeline only fills the non-speech gaps.

---

## 3. Module layout (all new)

```
backend/app/services/vcut/
  __init__.py
  params.py         # all tunable constants (energy curve, snap radii, floors, model id, sampling)
  spans.py          # non-speech span extraction per file (invert transcript speech spans)
  sampling.py       # medium frame sampling over non-speech spans; Gemini cache handle
  pass1.py          # ONE VLM call -> LooseCutPlan (spans + peaks + tags + rough meaning)
  resolve.py        # THE ALGORITHM: seam + energy -> final cut boundaries  (pure, unit-tested)
  pass2.py          # meaning-derived questions + scene specifics (background enrich)
  store.py          # persist per-file S(t) + per-run LooseCutPlan; build+insert CutRecords
  orchestrate.py    # procrastinate tasks: vcut_ingest (foreground path) + vcut_enrich (bg)
backend/scripts/vcut/
  run_vcut.py       # CLI: run the whole pipeline for a project/folder (dev + smoke)
backend/scripts/
  test_vcut_resolve.py   # unit tests for resolve.py (the algorithm) — calibration cases in §7.6
  test_vcut_spans.py     # non-speech span inversion edge cases
```

`resolve.py` is the crown jewel and must be a **pure function** (no I/O) so it is fully unit-testable and cheap enough to run on every slider drag.

---

## 4. Data the pipeline persists (so the slider is instant)

Reuse `ingest_runs` + `cut_records`. Add **two** JSON artifacts, both keyed by the new run:

- **Per-file seam curve** — `S(t)`, `hop_ms`, and the raw component tracks needed for peak refinement (`action_energy`, `frame_diff`). Store as a JSON blob. Cheapest option: a new column `ingest_runs.seam_cache jsonb` holding `{file_id: {hop_ms, S, action_energy, frame_diff}}`. (A dedicated `seam_curves` table is fine too, but a run-scoped blob avoids cross-run staleness and matches the "a run is re-run in full, never patched" convention in `ingest_store`.)
- **Per-run LooseCutPlan** — the VLM Pass-1 output verbatim: `{file_id: [{span_ms:[L,R], peaks:[{t_ms, tag, strength}], meaning}]}`. Store in `ingest_runs.pass1_output` (that column already exists and is exactly this role in the old pipeline) or a sibling column `ingest_runs.loose_plan jsonb`.

**Migration:** one small additive migration `NNN_vcut_artifacts.sql` adding `seam_cache jsonb` and (optionally) `loose_plan jsonb` to `ingest_runs`. Additive only — the old pipeline ignores them.

With both persisted, the energy endpoint (§9) re-runs `resolve.py` from these blobs alone: **no model call, no R2, no motion recompute.**

---

## 5. Stage 1 — non-speech spans (`spans.py`)

For each file in the project:
1. Load transcript speech segments (same `transcripts.segments` shape `signals.py` already reads: `start_ms`/`end_ms`).
2. Compute the **complement** over `[0, duration_ms]`: the gaps between speech = non-speech spans.
3. Drop spans shorter than `MIN_NONSPEECH_SPAN_MS` (e.g. 500ms) — nothing worth cutting there.
4. Files with **no transcript** → the whole clip is one non-speech span (correct for silent B-roll / drone footage).

Output: `{file_id: [(start_ms, end_ms), ...]}`. These bound everything downstream — the VLM only ever sees non-speech frames, and `resolve.py` never emits a cut outside them.

---

## 6. Stage 2 — VLM Pass 1 (`sampling.py` + `pass1.py`)

**Sampling (`sampling.py`):**
- Extract frames at "medium" density over the non-speech spans only (reuse `app.services.l3.frames` for on-demand extraction from the R2 proxy — same helper the old pass2 uses).
- Upload once and create a **Gemini explicit cache handle** (mirror `pass2.py`'s `get_pass2_cache_handle`). Both Pass 1 and Pass 2 reuse this handle. Cache TTL short (single ingest run's lifetime).

**One VLM call (`pass1.py`), model `gemini-3.1-flash-lite`.** Ask for a *closed, generic* schema — nothing phenomenon-specific:

```json
{
  "clips": [{
    "file_id": "…",
    "meaning": "one plain sentence: what this footage is",
    "loose_cuts": [{
      "span_ms": [L, R],                 // a coherent stretch worth keeping, loosely bounded
      "peaks": [                          // 1+ moments worth keeping inside the span
        { "t_ms": 8300, "tag": "build" }  // tag ∈ {build, settle, both}
      ]
    }]
  }]
}
```

Rules given to the model:
- `span_ms` must fall inside a provided non-speech span (we clamp defensively anyway).
- `tag` semantics (closed menu, generic):
  - `build` — the interesting part **builds up to** the peak (payoff is *at* the peak; keep the run-up, land on the climax).
  - `settle` — the peak then **settles** (keep from the peak into the landing; trim the run-up).
  - `both` — symmetric around the peak.
- `meaning` is one sentence, used later to *frame questions* — not shown to the user.
- **No bounding boxes, no counts, no per-frame description.** Boxes are spotty and not needed by this pipeline (reframe/aspect is out of scope — see §0).

Output → `LooseCutPlan`, persisted (§4). **Immediately after this, cuts are resolved at default energy and shown** (§7 + §8). Pass 2 runs in the background.

---

## 7. Stage 3 — THE ALGORITHM (`resolve.py`) — "use seam well"

Pure function:

```python
def resolve_cuts(
    plan: LooseCutPlan,        # per-file loose spans + peaks + tags
    seam: dict,                # per-file {hop_ms, S, action_energy, frame_diff}
    energy: float,             # single global slider, 0..1  (default 0.5)
) -> list[ResolvedCut]:        # (file_id, in_ms, out_ms, peak_ms, tag, meaning)
```

Nothing here calls a model, touches R2, or branches on footage type. It is pure geometry over `peaks`, per-peak keep-windows, and `S(t)`.

### 7.1 Step 0 — refine peaks onto real content
VLM timestamps are approximate. Snap each `peak.t_ms` to the nearest **local maximum of `action_energy`** (fallback `frame_diff`) within `±PEAK_SNAP_MS` (~250ms). Keep the VLM `tag`. This grounds "moment worth keeping" on an actual motion event and stabilizes everything downstream.

### 7.2 Step 1 — energy → keep-width (the shrink)
One monotone map, shared by **all** peaks (uniform shrink):

```
W(e) = lerp(WIDE_MS, TIGHT_MS, e)      # e=0 → WIDE_MS (~2500ms), e=1 → TIGHT_MS (~600ms)
```

### 7.3 Step 2 — per-peak window with tag asymmetry
Split `W(e)` into (before, after) around the refined peak by tag:

```
both:   before = 0.5·W,  after = 0.5·W
build:  before = 0.7·W,  after = 0.3·W     # keep run-up, land on/just past climax
settle: before = 0.3·W,  after = 0.7·W     # cut into peak, keep the settle/landing
window_i = [p_i - before, p_i + after]
```

Clamp every window to its own loose `span_ms` **and** to the midpoints to its neighbor peaks (a window can never cross past the halfway point to the next peak). This is what makes division emergent and prevents overlap-through.

### 7.4 Step 3 — division (emergent, not special-cased)
Within a span, **merge overlapping windows, keep disjoint ones separate**:
- Low energy → `W` large → windows overlap → merge into **one** loose cut ≈ the whole span.
- High energy → `W` small → windows separate → span **divides** into one tight cut per peak.
- **Single peak → exactly one window, ever → never divides**, just shrinks toward the peak. (Matches: *"If there is only one peak then it need not [divide]."*)
- Merging only happens **inside** a span. Two different loose spans (different meanings) **never** merge — this is what keeps "camera setup" and "the push-ups" as distinct cuts.

### 7.5 Step 4 — seam-snap the edges (this is "use seam well")
Each merged window has provisional `[a, b]`. Snap each edge to the cleanest nearby frame:

```
a* = argmax_{t ∈ [a - SNAP_MS, a + SNAP_MS]}  S(t)
b* = argmax_{t ∈ [b - SNAP_MS, b + SNAP_MS]}  S(t)
```

Constraints on the search:
- Never snap the IN edge **past** the earliest peak in the window; never snap the OUT edge **before** the latest peak. The peaks must stay inside.
- Keep both edges inside the loose `span_ms` and inside the file's non-speech span.
- Enforce `a* < b*`.

Because `S(t)`'s gates already suppress blur (`g_sharp`), mid-gesture (`g_gest`) and camera-wobble frames, and its attractors reward stillness + beats/onsets, `argmax(S)` lands on a **sharp, settled, ideally beat/onset-aligned** frame automatically. That is the entire point: **VLM + energy pick the region, seam picks the exact frame.** No blur/bad-start heuristics needed — they fall out of `S`.

### 7.6 Step 5 — junk / dead-air floor (energy-independent)
Regardless of energy:
- Drop any resolved window whose **median `S`** over its extent is below `S_DEAD_FRAC · (clip median S)` **and** which contains no peak (pure dead/blur air).
- Enforce `MIN_CUT_MS` (~500ms): a snapped window narrower than the floor is widened symmetrically and re-clamped.
- Enforce `MIN_CUT_GAP_MS` between consecutive emitted cuts within a file.

This is the "usability" guarantee, separate from the energy dial: bad frames never survive as an edge or as a whole cut.

### 7.7 Worked example — the push-up clip (the user's own test case)

Clip: walk in + set camera (static hold, ~0–4s) → 8 push-ups (~4–20s, rhythmic tops) → walk back + switch off (~20–24s).

Pass 1 loose plan:
```
[ {span:[0,4000],    peaks:[{3500, settle}],            meaning:"setting up the camera"},
  {span:[4000,20000],peaks:[8 push-up tops, build],     meaning:"doing push-ups"},
  {span:[20000,24000],peaks:[{20500, build}],           meaning:"walking over, switching off"} ]
```

- **Low energy:** each span → one loose cut (`W` large ⇒ push-up windows all merge). → **3 cuts**: setup, the push-up block, shutdown — kept **separate** because they're separate spans. (Exactly the "keep both as separate cuts" the user demanded.)
- **High energy:** push-up span **divides** into ~8 tight cuts, each cut on the top with `build` asymmetry (keep the down-up, land on the top). Setup/shutdown shrink toward their single peaks. → a punchy short-form montage.
- Every edge is seam-snapped, so no rep starts on a blurry down-phase frame.

### 7.8 Diagram

```
loose span:      L├─────────────●p1──────────────●p2────────────────┤R
                                 (build)          (both)

energy LOW   →   [======= one merged cut, edges snapped to S peaks =======]
                 a*↑ (max S near L)                             b*↑ (max S near R)

energy HIGH  →   [== cut A ==]                 [==== cut B ====]
                 keep run-up→p1,land            symmetric around p2
                 a*,b* = argmax S in ±SNAP      a*,b* = argmax S in ±SNAP

S(t):        ▁▂▅█▂▁▁▂▃▂▁▁▁▁▂▅█▇▂▁▁▁▂▃▂▁▁▁▂▅█▆▂▁   ← edges snap onto the █ (clean) frames
```

---

## 8. Stage 4 — write cut_records / wire to the existing cuts channel (`store.py`)

For the default energy, turn each `ResolvedCut` into a `CutRecord` (`app.services.l3.post.CutRecord`) and insert via the **existing** `ingest_store`:

- `create_ingest_run(project_id, pass1_model="vcut:gemini-3.1-flash-lite", pass2_model=...)`.
- Build minimal `CutRecord`s: `kind="video"`, `channel="shown"` (non-speech visual channel), `file_id`, `src_in_ms`, `src_out_ms`, `label` = short phrase from `meaning`, `summary` = `meaning`, `hero_ts_ms` = refined peak, `on_camera=None`, `junk=False`, empty `framing`/`look`/`caption_zones`, a default `PaceEnvelope`. `scene_specifics` is filled by Pass 2.
- `delete_cut_records_for_run` then `insert_cut_records` (the "re-run in full, never patch" convention).
- Persist `seam_cache` + `loose_plan` on the run (§4).
- `set_status(run, "ready")`.

The frontend's `GET /api/projects/{id}/cuts` → `load_cuts` returns the **latest** `ingest_run` for the project, so these cuts appear with **no frontend rendering change**. The only wiring choice: make the "ingest" action for a project enqueue `vcut_ingest` instead of the old `l3_cuts_ingest` (a single dispatch switch — see §11 for how to keep both live behind a flag).

---

## 9. Energy slider — interactive re-derive (new endpoint)

Add `POST /api/projects/{id}/cuts/energy  { energy: float }`:
1. Load latest run's `seam_cache` + `loose_plan`.
2. Call `resolve.resolve_cuts(plan, seam, energy)` — pure, ~milliseconds.
3. `delete_cut_records_for_run` + `insert_cut_records` with the re-resolved cuts (preserving Pass-2 enrichment where the peak identity is unchanged — simplest v1: re-insert and let Pass 2 re-attach on next enrich; acceptable because Pass 2 is cheap off the cache).
4. Return the new cuts (same shape as `GET …/cuts`).

Frontend: the single energy slider calls this on release (debounced) and re-renders from the response. No model calls, so it's instant. This is the whole reason `S(t)` and `loose_plan` are persisted.

---

## 10. Stage 5 — Pass 2 enrichment (`pass2.py`), background

Runs as `vcut_enrich` **after** cuts are shown. Same Gemini cache handle, model `gemini-3.1-flash-lite`.

1. **Questions from meaning.** For each clip's `meaning`, generate a few targeted questions (cheap text step) — e.g. "who/what is the subject," "is there a distinct climax frame," "any on-screen text." Ask them against the cached frames in one batched call. Write answers to `cut_records.scene_specifics` (existing column) via `update_cut_scene_specifics` (existing helper).
2. Accumulate token usage with `accumulate_pass2_usage`; `set_timings`.

Pass 2 never changes cut *boundaries* — it only enriches records. If it fails, cuts still stand (fail-open, per-cut isolation like the old `l3_scene_enrich`).

---

## 11. Orchestration (`orchestrate.py`)

Two procrastinate tasks on the existing `l2` queue:

- `vcut_ingest(project_id)` — spans → sample+cache → Pass 1 → persist artifacts → resolve at default energy → write cut_records → `set_status("ready")` → enqueue `vcut_enrich`.
- `vcut_enrich(run_id)` — Pass 2 meaning-derived questions / scene specifics, best-effort per cut.

**Keeping old + new both live (recommended):** add `settings.cuts_pipeline: str = "v3"` (env-driven). `projects.kick_ingest` dispatches to `l3_cuts_ingest` when `"v3"` and `vcut_ingest` when `"vcut"`. This lets you flip per-environment (e.g. `vcut` on local-dev, `v3` on prod) and A/B without deleting anything. Auto-kick-on-L1-complete uses the same dispatch.

---

## 12. Frontend touchpoints (minimal)

- **None for rendering** — `cuts-view.tsx` reads the same `cut_records` shape.
- Energy slider → `POST …/cuts/energy` (§9), debounced, re-render from response.

---

## 13. Validation

1. **`cutviz` overlay (reuse existing).** The debug visualizer already overlays `S(t)` (from the seam plan). Add an overlay of resolved cut boundaries at a chosen energy so you can *see* edges snap onto `S` peaks and watch a span divide as you drag energy. No new tool — extend the existing `cutviz` router additively.
2. **`test_vcut_resolve.py`** — pure unit tests for `resolve.py`:
   - single peak never divides; only shrinks with energy.
   - k peaks divide into k cuts at high energy; merge into 1 at low energy.
   - `build`/`settle`/`both` asymmetry produces the expected before/after split.
   - edges snap to the local `S` maximum within `SNAP_MS`; never snap past a peak.
   - dead-air/min-cut/min-gap floors fire correctly.
   - two separate loose spans never merge, at any energy.
   - the §7.7 push-up scenario end-to-end at low and high energy.
3. **`test_vcut_spans.py`** — non-speech inversion: no transcript → whole clip; speech at edges; tiny gaps dropped.
4. **Smoke:** `scripts/vcut/run_vcut.py` on folder `8a5217b7-f491-4517-b4a9-27a60d3ff192` (the drone/factory set) and on a spoken clip; eyeball in `cutviz`.

---

## 14. Sequencing (build order)

1. `params.py` + migration `NNN_vcut_artifacts.sql` (additive columns).
2. `spans.py` + `test_vcut_spans.py`.
3. **`resolve.py` + `test_vcut_resolve.py`** — build and lock the algorithm *first*, against synthetic seam + plans. This is the highest-risk, highest-value piece; get it right in isolation before any I/O.
4. `store.py` (CutRecord build + reuse `ingest_store`) + the `seam_cache`/`loose_plan` persistence.
5. `sampling.py` + `pass1.py` (Gemini cache + one call).
6. `orchestrate.py` `vcut_ingest` + dispatch flag in `projects.kick_ingest`.
7. Energy endpoint (§9).
8. `pass2.py` + `vcut_enrich`.
9. `cutviz` cut-overlay + smoke on real folders.

Ship after step 7 is a fully working cuts pipeline with an interactive energy slider; steps 8–9 add scene specifics and validation polish.

---

## 15. Open decisions / knobs to tune with the harness

- `WIDE_MS` / `TIGHT_MS` (energy range), `SNAP_MS`, `PEAK_SNAP_MS`, `S_DEAD_FRAC`, `MIN_CUT_MS`, `MIN_CUT_GAP_MS`, `build`/`settle` split ratios — all in `params.py`, all retunable without touching `resolve.py`.
- Default energy `e0` (start at 0.5).
- Whether the energy slider re-insert should preserve Pass-2 enrichment (v1: re-run enrich; later: diff peaks and keep unchanged ones).
- Pass-2 model: default flash-lite per cost directive; single config knob to upgrade later.

## 16. Honest risks

- **VLM peak timing is coarse.** Mitigated by Step 0 (snap peaks to `action_energy` maxima) — but if the VLM misses a whole span, no cut appears there. Acceptable for v1; the loose-span framing is forgiving.
- **`tag` reliability.** `build`/`settle` is a soft asymmetry, not a veto — a wrong tag shifts the keep-window a little, never breaks the cut. Low blast radius by design.
- **Cache TTL vs. slider lifetime.** The energy slider deliberately needs **no** cache (it re-derives from persisted `seam_cache` + `loose_plan`), so cache expiry only affects re-running Pass 1/2, not interactivity.
- **`S(t)` compute cost on long clips.** See seam quick-check note — vectorize `_gaussian_kernel_sum` before this runs on 30-min clips in the hot path.
