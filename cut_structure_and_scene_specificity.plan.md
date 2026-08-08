# Cut structure + scene specificity — implementation plan

Two independent-but-related upgrades, scoped to be implemented **on `local-dev`**,
tested locally, and **not auto-merged to `main`** until verified (the brain / VO
work on this branch is still under local test).

- **Part 1 (cuts):** fix the "camera-start still" leaking into video cuts by
  re-sequencing V4 into a **structure-first** segmenter (camera+blur define
  where it's *clean* to cut; action+rms define *what* to keep; snap content
  edges to structural seams). Deterministic, no ML, no new data.
- **Part 3 (vision specificity):** make per-cut scene descriptions *specific*
  ("pheras — couple circling the fire", "CNC lathe turning a steel shaft")
  instead of generic ("a machine in a factory"), via a **two-pass vision**
  design with a cheap **middle text layer** that turns generic summaries into
  sharp, footage-derived questions. Runs **after cuts are shown**, in the
  background, cached with a short TTL.

Both were locked in with the user during ideation. This doc is written so
another chat can pick it up cold.

---

## Guiding constraints (locked with user)

- **Do not touch Pass 1** (`ingest_pass1_model = claude-sonnet-5`). It's the
  expensive stage; leave it alone for now.
- **Keep it low cost overall.** Prefer approximations over big ML. The user
  explicitly rejected the change-point-detection / learned-boundary approach
  for now as too big/risky — Part 1 is the "clever approximation" instead.
- **Part 3 middle layer:** use a *capable* Gemini model (not necessarily
  flash-lite) — quality matters here. **One call, no re-ask loop.**
- **Part 3 outputs:** do **not** aggressively shrink Pass A / Pass B / middle
  outputs yet. Optimize for quality first; output-size tuning is a later pass.
- **Part 3 timing:** the enrichment runs **after** cuts are generated and shown
  to the user — it must never block cut display.
- **Caching:** never hold a provider cache idle. Short TTL as a safety net +
  **delete on completion**. Frames are always re-extractable from the R2 proxy
  (`frames.py`), so a cache miss is never a correctness problem.

---

# PART 1 — Structure-first video cut segmentation

### The problem
Video cuts sometimes begin on a static "camera-start still" (the setup frame
before the camera/subject moves), and camera-movement handling feels off. This
is a **cuts-level** issue, not a VLM one: the VLM never chooses *where* a cut is
(`v4_segment` owns location; pass 2 only labels + picks `shape`).

### Why it happens today (grounded in `backend/app/services/l3/v4_segment.py`)
Cuts already fuse camera + action + rms into one novelty curve, but the static
head leaks in via three spots:
1. **Fallback lands on the still.** When a shot's opening is static and the pan
   is too gentle to register as a move, the span produces no events and
   `_representative_window` picks the *steadiest, sharpest* instant — literally
   the setup still (`_cost = blur - stability`, minimized).
2. **Run-up padding reaches into dead air.** A point event pads
   `RUN_UP_FLOOR_MS` before its peak, pulling the in-point back into the static
   hold.
3. **Slow pans fall below `CAMERA_MOVE_MAGNITUDE_MIN`**, so they never become a
   span event and drop to case 1.

### The design: three ordered steps
Restructure the *order of reasoning* inside `segment_video` (this is a refactor
of existing primitives, not a rewrite — all signals already exist on
`motion_dynamics`):

**Step 1 — Structural pass (camera + blur ONLY): where it's clean to cut.**
Ignore action/rms. Classify each motion hop in a working span into a **camera
regime** from thresholds on already-computed signals:
- `camera_stability` / relative-jerk → `steady` vs `transient` (whip/bump)
- `camera_coherence` + magnitude (`|dx|+|dy|+|zoom|`) → `static-hold` vs
  `coherent-move` vs `shake` (incoherent)
- `blur` → `sharp` vs `blurred/degenerate`

Regime **boundaries** + premium seams (whips, blur spikes, move onset/offset,
existing `transition_points`) are the candidate clean cut points (the "A–b,
b–e, e–k" divisions). These are places a cut looks intentional.

**Step 2 — Content pass (action + rms + beat): what to keep / emphasize.**
Keep today's novelty machinery (`_novelty_curve`, `_prominent_peaks`,
`action_points`, musical `onsets_ms`) — this decides salient moments and their
natural extent (onset/settle). Unchanged in spirit.

**Step 3 — Reconcile: snap content edges to structural seams; protect content.**
- For a content span `S` and a nearby camera seam `b`, compute
  `min(left_frac, right_frac)` of `S` around `b`:
  - `min < SNAP_FRAC` (≈25%) **and** sliver `< SNAP_MS_FLOOR` → **snap** S's
    edge to `b` (clean trim, only a sliver lost).
  - else → `b` is *inside* content → **do not cut at `b`** (keep `S` whole).
- Snap target set is `{camera seams} ∪ {beats}` (beats now reliable after the
  L1 musicality fix) — snap to the nearest clean seam.
- **Protect content-bearing moves:** a `coherent-move` that overlaps
  action/rms energy is *content*, not a free seam — never split/trim it as if
  it were pure structure. Only content-*less* moves/holds are structural.
- **Drop content-less structural segments:** a `static-hold` (or dead) segment
  with no action/rms/beat carries nothing → produce no cut there. **This is
  what removes the camera-start still — it falls out of the decomposition, no
  special-case dead-edge trim needed.**

**Also:** change `_representative_window` to bias toward **energy** (peak of
action+camera+rms), not stillness, and to **return no event on a fully-dead
span** (max energy across all channels below a floor) instead of fabricating a
still.

### Why this is safe re: "no breathing room"
Breathing room is a *pacing/duration* concern (`post.compute_pace_envelope`) —
quiet *around* a real moment. Step 3 only removes quiet at the *extremes* of a
raw shot where nothing has happened yet / anymore (setup + tail-off). Different
quiet, different mechanism; they compose.

### Concrete changes
- `backend/app/services/l3/v4_segment.py`
  - New: `_camera_regimes(motion, span, hop_ms) -> List[(regime, start_ms, end_ms)]`
    and `_structural_seams(...) -> List[int]` (regime boundaries + whip/blur/
    move-edge seams + existing `transition_points`).
  - New: `_snap_edge_to_seam(edge_ms, span, seams, ...)` implementing the
    25%+ms-floor rule.
  - New: `_move_is_content(move_core, motion, span)` (overlap with action/rms).
  - Rework `_events_for_span` / the window-building in `segment_video` to:
    (a) build structural seams first, (b) run content events, (c) reconcile via
    snap + content-protection, (d) drop content-less structural segments.
  - Rework `_representative_window` bias + dead-span suppression.
- `backend/app/services/l3/v4_segment_params.py`
  - Add: regime thresholds (`REGIME_STABILITY_TRANSIENT_MAX`,
    `REGIME_COHERENCE_MOVE_MIN`, `REGIME_MAGNITUDE_MOVE_MIN`,
    `REGIME_BLUR_MAX`, `DEAD_ENERGY_FLOOR`), and snap thresholds
    (`SNAP_FRAC ≈ 0.25`, `SNAP_MS_FLOOR`). Reuse existing
    `CAMERA_MOVE_*` where sensible.

### Calibration & testing (no ML — validate like the music fix)
- Extend `backend/scripts/test_v4_segment.py` (pure synthetic arrays) with:
  - a static-head-then-move shot → cut excludes the static head;
  - a whip between two holds → cut snaps to the whip;
  - a coherent move overlapping action → move preserved (not split);
  - an action crossing a seam at 40/60 → not split; at 10/90 → snapped.
- **Ground the thresholds on the real offending clip**: pull the cut's
  `salience` + the span's `camera_*`/`blur`/`action_energy`/`rms` from L1
  (`build_l1_snapshot`) and set `REGIME_*` / `DEAD_ENERGY_FLOOR` from real
  numbers, not guesses. Eyeball before/after cut spans.

### Explicitly out of scope for Part 1 (future, if heuristic ceilings out)
- Multivariate change-point detection (PELT/kernel/BOCPD) replacing the
  novelty curve.
- A single global DP objective replacing the rule cascade.
- Bootstrapped VLM boundary labels + a boundary F1/IoU metric + parameter
  fitting; learning from user edit deltas.
These were discussed and deferred. Note them so the flywheel option isn't lost.

---

# PART 3 — Intent-conditioned scene specificity (two-pass vision)

### The problem
Pass 2 runs at ingest, before any intent, so summaries are generic ("a man in a
kitchen", "a machine in a factory"). For good editing the brain needs *specific*
scene identity ("pheras", "the annealing furnace stage"). Specifics can't be
derived from generics — but the *right question* can ("a machine" → "what is
being manufactured, and what operation?"). So: **generic pass → understand →
targeted pass.**

### Current code (grounded)
- Pass 2 = one vision call per batch, `pass2.run_pass2_batch`, model
  `ingest_pass2_model = gemini-3.1-flash-lite`, provider `gemini`.
- Frames extracted fresh from the R2 proxy per run (`frames.py`), transient.
- A **per-run Gemini `CachedContent`** is already created + deleted in
  `ingest._run_ingest` (`ig.create_pass2_cache` / `ig.delete_pass2_cache`) and
  scoped over batches — **reuse this exact machinery** for Pass B.
- Cut records persisted via `ingest_store.insert_cut_records`; the brain reads
  them in `converse`/`observe`.

### Design: Pass A → middle text layer → Pass B

**Pass A — generic (this is today's pass 2), one tweak.**
Keep it. **Tweak the prompt** (`pass2.system_prompt`) so every cut always
names **observable specifics it can see** — objects, on-screen text, part
numbers, the concrete action — even when it can't interpret them. These are the
"hooks" the middle layer hangs questions on (crucial for silent/industrial
footage). Do not shrink output.

**Middle text layer — one cheap-ish, capable-Gemini, TEXT-only call.**
New module `backend/app/services/l3/scene_taxonomy.py`:
- Input: all persisted cut records' `label`/`summary`/`channel`/`people`/
  `screen_text` + the transcript. **Dedupe/cluster** near-identical summaries
  first (a factory reel of 200 "machine on a line" cuts → ~10 distinct types)
  so the call stays small.
- Output (structured/pydantic):
  ```
  {
    "domain": "Indian wedding (Gujarati)" | "CNC machine shop" | "unknown/mixed",
    "confidence": "high|med|low",
    "evidence": [...],
    "taxonomy": [{"id","def"}]        # ONLY where a real closed set exists
                                       # (wedding rituals, sports plays); always
                                       # include "other"/"unsure". Empty otherwise.
    "clusters": [ {"cut_refs":[...], "questions":[ ... ]} ],  # targeted, hook-derived
    "needs_pass_b": ["video_group[3]", "speech_cut[12]", ...] # ambiguous cuts only
  }
  ```
- **Mechanism = question generation, not taxonomy classification.** Closed-set
  taxonomy is a *special case* used only when a genuinely known answer set
  exists; the general case is sharp, footage-derived open questions
  ("what part/product and what operation?"). This is the key correction from
  ideation: you can't enumerate a correct taxonomy from generic summaries, but
  you *can* write the right question.
- One call, capable Gemini model, **no re-ask** (low stakes). New config
  `ingest_scene_text_model` (a good Gemini model, e.g. `gemini-2.5-pro` /
  current best) — add to `client._STAGE_MODEL_ATTR` as a new stage, or call
  `ingest_gemini` directly.
- Also `needs_pass_b` **triages** which cuts get Pass B — this is a **cost
  reducer**: skip Pass B on cuts already specific enough.
- Degenerate case: `domain = unknown/mixed` → fall back to a generic moment
  rubric (establishing/action/reaction/detail/dialogue/transition) rather than
  forcing a wrong taxonomy.

**Pass B — targeted vision, cached, small output.**
New module `backend/app/services/l3/scene_specificity.py` (or extend `pass2`):
- For each `needs_pass_b` cut, re-extract its planned frames from the proxy
  (`image_plan` → `frames.extract_for_planned_frames`) and send with its
  cluster's targeted questions + the project gist.
- Output per cut (kept small but **not artificially shrunk**): the closed-set
  label where applicable (or `other`), plus a one-line specific
  ("pheras — couple circling the fire"). May carry a couple of specifics if the
  questions ask for them.
- Model: `ingest_pass_b_model` (gemini flash/flash-lite; thinking low). Provider
  gemini.

**Caching (reuse the Pass 2 pattern):**
- Build a per-run **short-TTL** Gemini `CachedContent` holding the stable
  prefix (system + gist + taxonomy) — mirror `ig.create_pass2_cache` /
  `pass2_cache_scope` / `delete_pass2_cache`.
- **Delete the cache on completion** (finally block); keep a short TTL (~30 min)
  only as a crash safety net. This makes the "months of idle" cost impossible.
- If Pass B fires **multiple questions per cut/cluster**, the cache earns its
  keep (frames sent once, queried many). Single-question is ~a wash on flash —
  fine either way.

### Orchestration & timing (must not block cut display)
- Add a deferred task `l3_scene_enrich(project_id, ingest_run_id)` on the
  `ingest` (or a new `enrich`) queue.
- Fire it from `ingest._run_ingest` **right after**
  `store.set_status(ingest_run_id, "ready")` — cuts are already visible; the
  enrich job runs in the background. (Alternative, cheaper variant: run it
  in-process *after* setting `ready` but *before* the `finally` cleanup, reusing
  the still-warm `images_b64` + cache — note but default to the decoupled job
  for resilience: enrichment failure must never fail the ingest run.)
- The enrich job: load cut records + transcript → middle text layer → Pass B →
  write specifics onto cut records + persist project gist/taxonomy → done. Wrap
  so a failure is logged and non-fatal (cuts remain fully usable without
  specifics).

### Schema & persistence
- Migration `backend/migrations/0NN_scene_specifics.sql`:
  - `cut_records`: add `scene_specifics JSONB` (per-cut: `{moment_id, specific,
    extras...}`), nullable.
  - Store project-level gist/taxonomy on `ingest_runs` (JSONB column
    `scene_taxonomy`) or a small `project_scene_model` table.
- `ingest_store`: add setters (`set_scene_taxonomy`, `update_cut_scene_specifics`).

### Brain wiring (the payoff)
- `converse._context_block` / `observe.build_context`: surface each cut's
  `scene_specifics.specific` (falling back to `summary`) and the project
  `domain`/`taxonomy`, so edit decisions use the specifics. This is the whole
  point — do not skip it.

### Cost & latency notes (for reviewers)
- Middle layer: deduped input + one capable-Gemini call ≈ cents/project; it
  *reduces* net cost by triaging Pass B volume.
- Pass B: flash + small output + short-TTL cache (deleted on completion);
  frames re-extractable from proxy so no image store.
- Everything runs after `ready` → zero user-facing latency on cut display; the
  only "wait" is one-time, background, per cut, and cached forever.

### Testing
- `scripts/test_scene_taxonomy.py`: mock the LLM; assert dedupe/clustering,
  `needs_pass_b` triage, closed-set-only-where-applicable, unknown→generic
  fallback.
- `scripts/test_scene_specificity.py`: mock Pass B; assert per-cut specifics
  written, cache created+deleted, non-fatal on failure.
- Extend `scripts/test_ingest.py` orchestration mocks to include the enrich
  hook without spending money.
- Real-run validation on the wedding + factory clips: confirm specifics land
  and read correctly to the brain.

---

# PART 4 — Brain self-model / provenance awareness (REQUIRED, not optional)

Both parts change *what the brain sees and how it was produced*. If the brain's
prompt isn't updated, it will misread the new inputs. So this part is a
first-class deliverable, not a footnote: **the brain must be completely aware of
how cuts are formed and where its scene information comes from.**

### Principle: describe provenance + reliability, never dictate trust
Add a dedicated **"How your inputs are built"** section to the brain's
system/context prompt (`backend/app/services/l3/converse.py` — the system prompt
and/or a block in `_context_block`; keep it always-on like the VO block).
Describe *how each signal is produced and its known limits*, so the brain can
calibrate its own confidence. **Do not** tell it "trust X over Y" — state the
mechanism and honest reliability, and let it reason.

### What the self-model block must convey
- **Cut location is deterministic and signal-derived (Part 1).** Explain: video
  cut boundaries are chosen by code from camera-motion + blur (where a cut is
  visually clean — whips, blur, move edges) and action + rms/beat (what's
  salient); a static, silent, motionless head/tail is dropped; the VLM never
  sets *where* a cut is — it only *labels* it and picks `shape`. So a cut's
  span is a best-effort structural guess from signals, not ground truth about
  meaning.
- **Speech cuts vs video cuts** have different provenance (Pass 1 grouping vs
  the V4 segmenter) — say so.
- **Scene specifics are a two-pass, best-effort derivation (Part 3).** Explain:
  a generic vision pass first describes what's visible; a text step infers the
  project domain and writes targeted questions; a second vision pass answers
  them. The resulting `scene_specifics` (and `domain`/`taxonomy`) are
  **model inferences, possibly approximate** — `other`/`unsure`/`unknown` are
  honest, not failures. The brain should treat a confident specific as strong
  evidence and an `unsure` as weak, without being told a fixed rule.
- **Music/beats** are L1-derived (`is_musical`, `bpm`, `onsets_ms`); beat-locked
  cuts only exist for musical tracks — say when they're trustworthy.
- **Identity** (`speaker_person`, `visible_persons`, voices) is code-derived
  from L1 face tracks + diarization, clustered — approximate, `None` when
  unbound.
- **New per-cut fields.** Document the new `scene_specifics.specific` /
  `moment_id` and the project `domain`/`taxonomy` so the brain knows they exist
  and what they mean, and uses the specific in preference to the generic
  `summary` when present (surfaced via Part 3's brain wiring).

### Maintenance rule (write this into the plan's outcome)
Whenever cut-formation logic (Part 1) or the scene-specificity pipeline
(Part 3) changes, **this self-model block must be updated in the same PR.** Add
a short comment at the block in `converse.py` pointing back to this plan so it
doesn't drift out of sync with reality.

### Testing
- `scripts/test_converse.py` (or the relevant context test): assert the
  self-model block renders, includes the cut-provenance + scene-specifics
  language, and that a cut with `scene_specifics` surfaces the specific over the
  generic summary.

---

## Suggested build order
1. **Part 1** first (self-contained, deterministic, unblocks better cuts):
   regimes → seams → snap/reconcile → representative-window fix → tests +
   real-clip calibration.
2. **Part 3** next: Pass A hook tweak → schema/migration → middle text layer →
   Pass B + short-TTL cache → enrich orchestration after `ready` → brain wiring
   → tests + real-run validation.
3. **Part 4** alongside/after: update the brain self-model prompt to reflect the
   new cut provenance (Part 1) and scene specifics (Part 3). Ship the relevant
   half in the *same PR* as each part so the brain never describes stale
   behavior.

## Locked decisions
- Pass 1 unchanged. Part 1 is heuristic (structure-first), not ML. Part 3 is
  two-pass with a middle question-generation layer on a capable Gemini model,
  outputs kept full, enrichment after cuts shown, short-TTL cache deleted on
  completion. Part 4 (brain self-model/provenance prompt) is REQUIRED and must
  stay in sync with Parts 1 & 3.

## Open questions (decide during impl)
- Part 1: exact `REGIME_*` / `DEAD_ENERGY_FLOOR` / `SNAP_MS_FLOOR` values
  (set from the real clip).
- Part 3: auto-detect domain vs one-line user brief vs both (auto + one-line
  correction). Default: **auto-detect**, with the gist surfaced so a brief can
  override later.
- Part 3: how aggressively to cluster before questioning (cheap+consistent vs
  risk of lumping distinct moments).
