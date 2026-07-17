# V4 Cluster-Tree Cuts — multi-event moments resolved across the energy ladder

## 0. One-line intent

A video cut is no longer a single fixed span. It's a **cluster** — one continuous
moment — that carries a **tree** of the salient events inside it. The **energy
ladder resolves that tree**: broad = the whole moment as one cut; punchy = each
event as its own tight piece; in between = partial breakdowns. One label per
cluster, shared down. Everything content-derived, nothing hardcoded beyond outer
bounds.

## 1. Scope

**In scope — solve ALL cut levels:**
- Segmentation extracts every salient event (point *and* span kinds, coexisting).
- Group events into clusters; build each cluster's merge tree.
- The pace ladder resolves a cluster into the right piece-set + trims at every
  energy level (broad → punchy), reusing `keep_spans`/`remove_spans`.
- One Pass 2 per cluster (frames sampled across it); broadcast the label to all
  levels.
- Frontend energy dial renders any level (fused ↔ broken).

**Out of scope (handle later, do NOT touch now):**
- Brain / `converse.py` / `tools.py` / `observe.py` communication of the tree or
  its sub-pieces. We only **store** the structure so it's ready; we do not surface
  it to the brain yet.

**Guardrails:**
- Behind the existing `settings.cuts_segmenter == "v4"` flag. V3 untouched.
- **Backward compatible by construction:** a moment with a single event is a
  **cluster of one** — a trivial one-node tree that resolves to exactly today's
  single V4 cut at every level. So single-event footage (talking-head, most
  b-roll) behaves identically to current V4; only genuinely multi-event moments
  gain the tree.
- Generic. No domain/sport-specific logic anywhere.

## 2. Core model (recap of the agreed design)

- **Cluster** = one continuous moment = the cut at the **lowest/broadest** energy.
  Events close enough to fuse belong to one cluster; a big dead gap starts a new
  cluster.
- **The energy ladder is ONE operation: trim each event's window toward its peak.**
  "Breaking" is **emergent** — when two events' windows stop touching, the pieces
  separate. So *how many pieces* and *how tight* both fall out of **window width
  (energy) × event spacing (content)**. There is no separate hardcoded "break
  threshold."
- **Tree = the merge history** of that operation: sweep energy broad→punchy and
  events split in order of the gap between them (a dendrogram). Root = whole
  cluster; leaves = per-event tight windows.
- **Labeling:** one Pass 2 per cluster, fed a bounded frame set sampled across it;
  the label/junk is **broadcast to all levels**. Timing/salience stay precise per
  level (code-derived); only the semantic label is shared ("averaged" over the
  moment). Accepted tradeoff: a huge diverse cluster gets a generic label but
  exact per-piece timing — fine for now.

## 3. Data representation

**The `cut_record` (video) becomes the CLUSTER.** One record per moment, holding
all levels inside it. Stable identity (the cluster never changes as the dial
moves).

Extend the V4 cut payload (`VideoCut` → `v4_meta_by_ref` → `cut_record`):

- `src_in_ms/src_out_ms` = the **broadest** span (whole cluster).
- `salience` becomes **multi-peak**:
  ```
  salience = {
    "events": [
      {"peak_ms", "score", "kind": "point"|"span"|"none",
       "onset_ms", "settle_ms",        # decay-walk bounds (the tight window)
       "span_ms": [in,out]|None},       # for camera-move (span) events
      ...
    ],
    "primary": <index of strongest event>,   # back-compat single-anchor
    "density": <0..1>,
  }
  ```
  A single-event cluster has `events` of length 1 — downstream single-anchor code
  reads `events[primary]` and behaves exactly as today.
- `pace` envelope carries the **per-level resolution** (see §5): for each energy
  level, the set of kept pieces (`keep_spans`) and removed valleys
  (`remove_spans`), plus the level's total duration.
- Store the **tree/children map** (ordered event indices + merge order) in the
  cut payload so later brain-comm work can expose sub-pieces. Not read by anything
  now — just persisted.

## 4. Segmentation changes — `v4_segment.py`

### 4.1 Events of all kinds coexist (remove the `if not out` gating)
`_candidates_for_span` currently tries transitions, THEN peaks, THEN camera
(`if not out`), THEN fallback — first non-empty wins, so a span with both a peak
and a pan keeps only one kind. Change it to **collect every event from every kind**
into one list:
- transition seams (as point events),
- novelty peaks (`_prominent_peaks`) as point events,
- **all** camera-move cores as span events (see 4.2),
- fallback representative window only if the span produced **no** events at all.

Each event carries its tight window from `_point_edges` (point) or the move core
(span), plus `onset_ms`/`settle_ms` from the decay-walk (`_decay_bound`).

### 4.2 Camera: emit ALL sustained moves, not just the longest (the pan-loss bug)
`_camera_move_core` returns `max(runs, key=length)` — a second good pan is
silently dropped. Replace with `_camera_move_cores(...) -> List[(s,e)]`: return
**every** run that clears `CAMERA_MOVE_MIN_MS`. Each becomes its own span event.

### 4.3 Cluster detection + tree (replaces flat `_consolidate`)
Today `_consolidate` merges near candidates into one flat cut keeping a single
anchor. Replace with a two-step build **per working span**:

1. **Cluster grouping:** sort events by time; a **new cluster** starts when the
   gap to the previous event exceeds the cluster-separation bound (§7 — derived
   from the span's own gap distribution, clamped). Events within a cluster are
   near enough to fuse at the broadest window.
2. **Tree per cluster:** the merge order is the events sorted by inter-event gap
   (ascending) — a dendrogram. This is not materialized as a heavy structure; it's
   implicit in the event list + gaps, and consumed by the per-level resolver (§5).

Output: **one `VideoCut` per cluster** (not per candidate), carrying all its
events. A working span with one dead gap yields two clusters → two `VideoCut`s.

### 4.4 `_finalize_cuts` stays (geometry only)
Keep the disjoint/clamp + same-span sub-floor merge from the current fix, but it
now operates on **cluster** cuts. The min-duration merge only welds same-working-
span clusters (already enforced via `span_key`). A cluster spanning multiple
events is never below the floor, so the merge only affects degenerate single-event
clusters — unchanged behavior there.

## 5. The ladder: per-level resolution (the heart — "solve all levels")

This lives where the ladder is synthesized today: `cutrecord_map._video_rung`
(per-level windows for the frontend dial) + `post.compute_pace_envelope`
(min/natural/max + `remove_spans`). Generalize both from "one contiguous window
collapsing toward one peak" to "resolve a cluster tree at a level."

**Resolver `resolve_cluster(events, level) -> (pieces, removed)`:**
1. Compute a **window half-width** `w(level)` per event: broad → wide (fills toward
   neighbors), punchy → tight (each event's `onset..settle` core). `w` slides
   monotonically with level, bounded by `RUN_UP_FLOOR`/`FOLLOW_THROUGH_FLOOR`
   (min) and `MAX_PAD` / the event's own decay bounds (max).
2. Expand each event to `[peak−w_in, peak+w_out]`, keeping the **asymmetric shape**
   per event (build-to-impact vs reveal) exactly as `_before_rung`/`_after_rung`/
   `_span_rung` do today — applied **per event**, not per cut.
3. **Merge touching windows:** adjacent expanded windows that overlap/touch fuse
   into one **piece**. Maximal runs of touching windows = the pieces at this level.
   - Broad: windows wide → all touch → **1 piece = whole cluster**.
   - Punchy: windows tight → few touch → **N pieces = per-event hits**.
4. `pieces` = the kept spans; `removed` = the valleys between pieces (the dropped
   connective tissue).

**Envelope:** `keep_spans = pieces`, `remove_spans = removed`, and the level's
duration = Σ piece lengths. `min_ms`/`natural_ms`/`max_ms` derive from the punchy/
mid/broad resolutions respectively. `_chosen_remove_spans` generalizes to pick the
level's `removed` set.

**Result:** dragging the dial from broad→punchy makes one flowing cut progressively
break into tight hits, with valleys removed — exactly the intended behavior, and a
single-event cluster degenerates to today's single window at every level.

## 6. `post.py` wiring
- `assemble_cut_records`: a cluster's span = its broadest resolution (already the
  V4 `src_in/out`). Build the multi-peak `salience` and the per-level `pace` via
  §5. `_validate_no_overlap` still holds (clusters are disjoint by construction;
  pieces are internal to one record).
- Continuity keys off cluster boundaries (already V4 behavior).

## 7. Content-derived knobs — `v4_segment_params.py`
Replace fixed thresholds with **ratios of the clip's own statistics**, keeping the
current constants only as **outer clamps**:

| Knob | Was | Derive from | Clamp |
|---|---|---|---|
| cluster separation | `MIN_CUT_GAP_MS` (fixed) | e.g. a multiple of the span's median inter-event gap | `[MIN_CUT_GAP_MS, MAX_*]` |
| window width `w(level)` | implicit single window | slides broad→punchy; per-event `onset..settle` at punchy | `RUN_UP/FOLLOW floors … MAX_PAD` |
| min piece duration | `MIN_CUT_DURATION_MS` | event envelope width + readability floor | `≥ readability floor` |
| run-up / follow-through | decay-walk (keep) | already content-derived | floors/`MAX_PAD` |

Every derived value must have a sane fallback for degenerate clips (one event, no
variance) → collapses to today's constants.

## 8. Labeling — `ingest.py`, `image_plan.py`, `pass2.py`
- **One Pass 2 per cluster.** Since a `VideoCut` is now one cluster, the existing
  "one video group → one Pass 2 label" already gives one call per cluster. Keep
  the V4 "label the whole cut, no split" rule from `v4_cuts_as_primitive`.
- **Frame sampling (`image_plan`):** sample frames at **each event peak** in the
  cluster (not just the primary), capped at `N` (evenly sample if more events than
  `N`). So the model sees every sub-moment even though it labels the cluster once.
  Bounds cost flat.
- **Broadcast:** the cluster's label/junk lives on the one `cut_record` and thus
  applies to all levels/pieces automatically (pieces are internal). No per-piece
  labels.

## 9. Frontend (cut-level only — NO brain comm) — `cuts-v3-view.tsx`, resolve/preview
- The energy dial must render a cut that resolves to **multiple pieces** at higher
  energy: honor `keep_spans`/`remove_spans` for video (play kept pieces, jump the
  valleys — hard cuts, which the compositor already supports).
- Verify the preview/`resolve-timeline` path plays video `keep_spans` (speech
  already does). Flag if `resolve`/compositor needs a small change to honor video
  keep_spans on playback/export.
- Do **not** add any UI to expose/select individual sub-pieces — that's brain/UI
  work for later. The dial just shows the correct resolution per level.

## 10. Tests
- `test_v4_segment.py`:
  - two pans + disturbance → **two** span events (not one) — the pan-loss regression.
  - peak + pan in one span → **both** events kept (coexistence).
  - tight cluster of N peaks → one cluster with N events; big gap → two clusters.
  - single event → cluster of one (degenerate == today).
- new `resolve_cluster` tests: broad → 1 piece; punchy → N pieces; monotonic piece
  count as energy rises; pieces disjoint; valleys = complement; asymmetric shape
  preserved per event.
- `test_post.py` / `test_cutrecord_map.py`: envelope `keep_spans`/`remove_spans`
  correct per level; durations monotonic; single-event cluster == current output
  (regression).
- `test_image_plan.py`: frames sampled at each event peak, capped at `N`.
- Keep all V3 tests green.

## 11. Rollout
1. Land behind V4 flag; single-event regression must be byte-identical to current
   V4.
2. Unit tests green.
3. Re-ingest a multi-event project (e.g. the sports/action ones) and eyeball: broad
   = whole rally, punchy = crisp per-hit pieces, both pans present.
4. Re-ingest all projects; confirm no overlap/label regressions.

## 12. Explicitly deferred
- Surfacing the tree / sub-pieces to the brain (`converse`/`tools`/`observe`).
- Any per-sub-piece semantic labeling.
- Selection-by-target-length (dropped — not helpful now).
