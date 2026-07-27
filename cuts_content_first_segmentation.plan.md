# Cuts: Content-First Segmentation (fix the inputs, keep the cluster machine)

## Motivation

The V4 segmenter's cluster/energy machine is sound, but it is fed by a
boundary/event layer that is **camera-scale-dependent and content-narrow**.
On low-motion / static-camera footage (drone-aerial factory, locked cameras,
constant-motion machinery) the content layer goes silent and the result
degenerates to the old behavior (static "camera-start" lead-ins, boundaries
that don't align to real content changes).

A ~180-variant coverage test of this plan scored **≈158/180 clips producing
usable cuts** (104 solid, 54 partial, ~22 fail). The fails — and most of the
hard partials — are static/flat footage whose "correct" cut is purely
aesthetic/semantic (the known ML/taste ceiling), and are **deliberately left
to the brain + Part-3 scene-specificity**, not to signal design.

**One principle, applied at two layers:** *nothing enters a cut unless the
content earns it.* Content decides WHERE a cut belongs; camera + audio only
refine HOW and the exact frame. We change only the inputs to the existing
cluster/energy machine; we do not touch the machine or the brain's dial.

---

## Design discipline (how we avoid over-engineering)

The deterministic layer owns the **generic ~80%: find defensible boundaries
from physical signals, and stop there.** The semantic/aesthetic tail (the ~22
fails + hard partials — "which ritual", empty vista, somber hold) is the
**brain's + Part-3's job**. The over-engineering trap is making the signal
layer approximate *meaning*; that yields something both more complex *and*
still not generic (taste doesn't reduce to signals).

**Acceptance test for every change — general principle or special case?**
Clip-relative, scale-free, reuses an existing signal, no footage-type branch =
keep. Footage-type-specific, or imposing structure signals didn't find =
reject. Anything borderline is **harness-gated** (below).

**ML-scaffold framing.** Cuts are ultimately an ML problem. This model's job
is to be *clean scaffolding*: work well now, and define a tight feature space +
generate labeled data for the eventual learned model. A small, principled
signal→boundary model is good scaffolding; a heuristic pile is bad scaffolding.
Parsimony now buys a better ML transition later — so we bias toward the
smallest model that clears the harness.

---

## Non-goals / keep unchanged

- The cluster / energy-dial model (`_cluster_events`, `resolve_cluster`,
  broad→punchy). It resolves *granularity*; our fixes just feed it better
  events.
- Start-broad-then-only-shrink + asymmetric padding (settle > build). We only
  make the *floors* conditional (Part 4); the shape is unchanged.
- "Content places, camera refines" (camera seams only SNAP, never invent a
  cut). Already the Part 1 design intent — we make it *effective* on
  low-motion footage, not reverse it.
- The brain's granularity controls (`tighten`/`level`/`density`/pace). They
  cover granularity mistakes, NOT missed-boundary mistakes — which is why the
  boundary/event layer must be fixed at the segmenter.
- Pass 1 model, speech path (`dialogue_segments`), Part 3 scene-specificity.

---

# CORE CHANGES (ship these)

## Part 1 — Clip-relative camera-move gate

**Problem.** `_camera_regime_at` gates a "coherent-move" on an ABSOLUTE
`|dx|+|dy|+|zoom| >= REGIME_MAGNITUDE_MOVE_MIN (0.03)`. Aerial/drone flow is
~50x below that, so the regime is permanently "static-hold": no move
onset/offset seams are produced near content. This is the mode-B root cause.

**Change.** Make the move-magnitude decision clip-relative, matching how
`action_energy`/`camera_motion`/`blur` are already normalized. A hop reads as a
deliberate move when its combined magnitude is high **relative to this clip's
own magnitude distribution** (robust percentile of the per-hop
`|dx|+|dy|+|zoom|` series), AND coherence clears `REGIME_COHERENCE_MOVE_MIN`.

**Guardrails.**
- Engage the relative gate only when the clip has genuine motion spread (a
  minimum absolute floor far below 0.03, plus a spread test), so a truly
  locked/still clip does not have sensor noise promoted to "a move."
- Keep coherence as the quality gate so shake stays "shake."

**Files.** `_camera_regime_at`, `_camera_move_cores`, `v4_segment_params.py`.
**Risk.** Low-med (over-firing on still handheld; mitigated by spread test).

## Part 2 — Turn on `composition_points` as a content boundary

**Problem.** `scene_cuts.py` already computes `composition_points` (soft
within-shot content change via HS-histogram drift) but the segmenter uses only
`shot_points`. Static-camera content change (a machine moving to a new
operation, a new subject entering) is invisible to camera+action but IS visible
to composition drift, and is currently discarded.

**Change.** Feed `composition_points` into the content/event layer as a
first-class boundary source alongside action runs/lulls and beats — so a
scene/content change produces a boundary even when action is flat and the
camera is still.

**Notes / caveats.**
- Noisier (`COMPOSITION_DRIFT_FLOOR=0.15`) and coarser cadence (scene hop
  200ms vs motion 100ms). It must pass the same prominence/periodicity
  discipline as other events (a periodic blink shouldn't spam boundaries).
- Only thresholded points persist (HS-drift curve is NOT stored). Lowering the
  drift floor later requires re-running the `scene_detect` L1 stage.

**Files.** `_events_for_span`, `_structural_seams`/`_working_spans`; consumes
`scene.composition_points`. **Risk.** Med (noise; gated by prominence).

## Part 3 — Widen the content layer: action peaks -> peaks + runs/lulls

**Problem.** Content anchors are isolated `action_points` (maxima > p75).
Constant-motion footage has no isolated peaks, so the content layer produces
nothing to anchor on.

**Change.** Also derive boundaries from action-energy **structure**: a
contiguous run of above-baseline action is one moment; a drop toward the clip's
own baseline (a **lull**) is a candidate edge.

**Definition of "a lull" (the fuzzy part).** Baseline clip-relative (reuse
`_series_lohi`/`_norm_in_clip`); a lull = normalized action drops below a
fraction of the run's own sustained level for a minimum duration. Start
conservative (fewer, cleaner edges) and **calibrate against the harness, not by
eye.**

**Files.** `_events_for_span`, `_novelty_curve`, `_true_runs`.
**Risk.** Med (over-segmentation; harness-calibrated; cluster+dial absorb
residual granularity error).

## Part 4 — Energy-gated padding floors

**Problem.** `_broad_window_for_event` applies `RUN_UP_FLOOR_MS (300)` and
`FOLLOW_THROUGH_FLOOR_MS (500)` UNCONDITIONALLY. When an event sits right after
a static intro, the run-up floor reaches backward into the dead establishing
frame — the "camera-start still" at the padding level. (`_span_is_dead` misses
it because whole-span action is "high"; only the head sub-region is dead.)

**Change.** Make the floors **conditional on energy**: padding may only extend
across time carrying energy (action or rms above `DEAD_ENERGY_FLOOR` relative
to the clip). In a dead sub-region the floor does not apply — the edge trims
inward to where energy begins. Keep the settle>build asymmetry where energy
exists.

**Files.** `_broad_window_for_event`, `_decay_bound` clamps. **Risk.** Low.

## Part 5 — Protect gate includes content continuity

**Problem.** A camera transient (whip/bump) INSIDE a continuous action must not
become a cut (it's content). `_move_is_content` already excludes transients
inside content-bearing coherent-moves via action/rms.

**Change.** Extend the protect test so **composition continuity** also
protects: if action is sustained OR composition drift is low across a candidate
seam, forbid a camera-triggered cut there. Only a genuine content change or a
real lull may open a boundary inside an ongoing moment.

**Files.** `_structural_seams` (`_inside_content_move` ->
`_inside_continuous_content`). **Risk.** Low (extends existing mechanism).

## Part 6 — Period-aware representative cut (lever A)

**Problem.** Periodic/repetitive footage (reps, turntable, waves, waterfall,
conveyor, DJ, packaging, 3D-print, tech-rotate) is periodicity-discounted to a
generic representative window — a whole bucket of ~13 partials.

**Change.** Reuse the autocorrelation we ALREADY compute for
`_periodicity_score` to also extract the **period**, and cut **one clean
cycle** (a single representative repetition) instead of an arbitrary window.

**Why it's core, not gated.** It reuses an existing signal, adds no new
mechanism family, is scale-free, and converts a clean bucket (~10 partials ->
solid) with low regression surface.

**Files.** `_periodicity_score` (return period), `_representative_window`.
**Risk.** Low (mis-locked period -> a slightly-off cycle, no worse than
today's generic window).

---

# HARNESS-GATED EXPERIMENTS (add only if the harness proves gain without regression; willing to DROP)

These are borderline on the general-vs-special-case test. They are NOT part of
the core ship. Implement each behind the harness: keep it only if it converts
partials AND does not regress the 104 solids. If it doesn't clear that bar, it
is over-engineering by definition — drop it and leave those clips to the brain.

## Experiment B — Cumulative-drift (CUSUM-style) boundaries

For gradual-change footage (timelapses, slow reveals, strolls, window views,
establishing holds, real-estate rooms, whiteboard) with no spike to anchor:
accumulate composition-drift + slow camera displacement since the last cut and
open a boundary when the *accumulated* change crosses a threshold.
Upside ~9-10 partials. **Risk: med — can over-segment static/talky footage.**

## Experiment C — Paced subdivision fallback

For continuous no-lull action (cycling/hiking POV, long rally, DJ set):
subdivide a long no-lull span at a target cadence, snapping to micro-lulls
(weakest-action instants) or beats. Upside ~4 partials.
**Risk: high — imposes structure signals didn't find (the epicycle smell).**
This is the first candidate to drop.

---

## Part 7 — Cross-regime eval harness (BUILD FIRST)

**Why.** Today cuts are judged by eyeballing one clip — how the drone
regression shipped. The harness is also the **over-engineering detector**: any
change (esp. B/C) that doesn't move the number without regressing solids is
dropped.

**Change.**
- 6-10 clips spanning regimes: handheld, gimbal, locked-drone/aerial, static
  tripod, talking-head, constant-motion machinery, screen-recording,
  music/beat, periodic (for lever A). Reuse real projects incl. factory
  `873d4e35`.
- Hand-mark good boundaries (or a rubric: no dead lead-in; boundary aligns to a
  real content change; no cut through a coherent action).
- Metric: boundary precision/recall vs marks (+/- tolerance) + a dead-lead-in
  count. Run before/after each Part.
- `segment_video` is a pure function of loaded signals (no DB/model), so the
  harness runs offline on persisted L1 (motion 10fps + coarse rms persisted;
  scene-drift curve is not — see Part 2 caveat).

**Files.** New `scripts/eval_cuts.py` + a small fixtures manifest.

---

## Sequencing

1. **Part 7 (harness) FIRST** — baseline number across regimes.
2. **Part 4 (energy-gated padding)** — lowest risk, kills static lead-in.
3. **Part 1 (clip-relative camera gate)** — restores seams on low-motion; watch
   handheld regression.
4. **Part 2 (composition_points on)** — biggest coverage gain; watch noise.
5. **Part 3 (runs/lulls)** — continuous-motion coverage; watch
   over-segmentation.
6. **Part 6 (period-aware cut, lever A)** — converts periodic bucket.
7. **Part 5 (protect gate + composition continuity)** — final safety on
   shake-mid-action.
8. **Experiments B then C** — only if the harness shows regression-free gains;
   C is the first to drop if marginal.
9. Re-run the ~180 coverage assessment against measured harness numbers; update
   this plan's score with real results.

## Expected outcome

- Industrial/process + drone-nature move FAIL -> ✓/~.
- Core (Parts 1-6) converts the periodic bucket and removes static lead-in;
  ~104 solid holds, no regression is the bar.
- If B (and maybe C) clear the harness, ~20-25 more partials -> solid.
- Residual fails (~22) stay the static/flat, semantically-defined class —
  deferred to the brain + Part 3, not a segmentation flaw.

## Brain awareness (follow-through)

Update the brain's provenance/self-model block (as in the Part 1/3/4 work) to
state that cut boundaries now come from content (action runs/lulls +
scene/composition change; periodic content = one representative cycle) with
camera/audio as refiners, and that the energy dial subdivides but cannot
recover a missed boundary — so it should trust cut *locations* as
content-derived.
