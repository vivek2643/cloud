# V4 Clusters — frame revert + brain read/act (linear editing)

## 0. Intent & relationship to the in-flight plan

The cluster-tree segmentation plan (`v4_cluster_tree_cuts.plan.md`) is **already
being implemented** — this plan does **not** touch segmentation, the ladder, or the
per-level resolution. It's a thin follow-up with three jobs:

- **A. Frame budget:** keep **2 frames per cluster** (undo the per-event frame
  sampling in that plan's §8). No VLM budget increase.
- **B. Read:** give the brain generic awareness of a cluster — its pieces, their
  salience/position/duration, and the cluster's broad↔punchy range.
- **C. Act:** a small handle so the brain can place a cluster **at a level** (whole
  ↔ broken) or place **one specific piece** of it.

Target, agreed: **read (B) + a small place-at-level/piece handle (C) = enough for
linear editing.** Everything stays generic and behind the V4 flag.

## 1. Scope

**In:** frame revert; Beat Index cluster/piece presentation; per-cut deep view
(`read_state`) piece breakdown; `place` level/piece handle + resolver; one optional
generic guidance line; tests.

**Out (explicit non-goals — see §6):** per-event frames, per-event semantic
captions (parked option 3), exploding a cluster into separate re-orderable timeline
items, selection-by-target-length.

## 2. Part A — Frame budget: 2 frames per cluster (undo §8)

**Coordinate with the in-flight plan:** its §8 ("sample a frame at each event peak,
capped at N") should **not** be built. If already built, revert it. End state:
`image_plan.build_image_plan` treats a **cluster as a single unit**, framed exactly
like a single cut today.

Concretely, in `image_plan.py` the video-unit branch for a V4 cut keeps the
**current 1–2 frame scheme**, unchanged:
- **short/runt cluster →** one mandatory frame (sharpest instant).
- **long enough cluster →** two frames: the sharpest instant in the **first half**
  and in the **second half** of the cluster span (the existing `_early_late_ms`
  path) — i.e. "start and end" of the moment, which reads the first vs last beat of
  a multi-event cluster naturally.
- keep the existing **point-shape straddle** (peak ± offset) for a single-peak
  cluster where it already applies.

Do **not** loop over `salience.events` to emit a frame per event. The 40-frames-
per-**clip** budget and the mandatory/extras tiers are unchanged.

**Why this is safe:** the per-cluster label is generic and one-per-cluster anyway;
2 frames caption a moment fine. The per-piece info the brain reads in Part B is
**100% code-derived** (`salience.events`) and needs **zero** vision. So no budget
increase, no loss of what the brain actually uses.

## 2.1 Part A.1 — (Optional) advisory "flow" line from Pass 2 (text-only, gated on need)

A candidate, **not required**. Since Pass 2 already sees the 2 frames (start + end)
and already infers `shape`, we can ask the **same single call** for a brief
natural-language **flow/progression** line — "pushes in from wide to close," "winds
up then strikes late," "subject enters, then reacts." **Text output only — zero
extra frames**, so it does not touch the image budget Part A protects.

- **What it adds:** complements the structural `shape` field with prose the brain
  can reason on (what *kind* of motion carries the moment), and gives a **cluster-
  level arc for free** — a cheap partial stand-in for the parked per-event captions
  (option 3), at no image cost.
- **How:** one small optional field on Pass 2's output schema + a brief prompt line;
  grounded by the 2 frames + `shape` + code-derived salience. Keep it short and
  **hedged** ("appears to…").
- **Honest caveat & rule:** with only 2 frames the middle is **inferred, not
  observed** — reliable for simple single-shot moments (the majority), a soft guess
  for busy multi-event clusters. So it is **advisory brain context only, never a
  gate**, and never drives a hard decision on its own.
- **Gating:** build this **only if** testing shows the brain's flow/piece choices
  look weak with `shape` + salience alone. Otherwise leave it out. Mirrors how
  option 3 is parked.

## 3. Part B — Read: cluster awareness (generic)

All presentation must be **generic and structural** — position, salience, duration,
kind. **No domain words, no special-casing.** A **single-event cluster renders
exactly as today** (one line); the breakdown only appears when a cluster has >1
event, so the common case is untouched.

### 3.1 Beat Index — `footage_map.py` (`_moment_line` / entry builder)
For a cluster cut, the entry stays one primary line (its shared summary + transcript
if any), plus, **only when it has >1 piece**, a compact generic sub-list read off
`salience.events` + the per-level resolution:

- **cluster range:** the broad duration ↔ punchy piece-count/duration, e.g.
  "whole ~7.2s, or up to N tight pieces (~1.6s each)". Derived, not hardcoded.
- **per piece (ordered):** index/position in the moment, relative **salience**
  (strongest/…), **duration**, and **kind** (impact / camera-move / etc. from
  `kind`). Enough for the brain to pick "the strongest piece" or "the last beat"
  positionally.

Keep it terse (a few tokens per piece). This is read-only context; it adds nothing
for single-event footage.

### 3.2 Per-cut deep view — `observe.py` `read_state`
When the brain inspects a specific cluster, `read_state` mirrors the same generic
breakdown (pieces with position/salience/duration/kind + the level→resolution) so a
deep look agrees with the Beat Index. No new sense; extend the existing per-cut
report.

### 3.3 (Optional) one generic guidance line — `guidance_doc.md`
A single generic sentence, no specifics: *"A moment may hold several beats — take it
whole or take its key beat(s), guided by your length budget and the video's
purpose."* Fits the existing "select for purpose" section. Skip if it risks
over-steering.

## 4. Part C — Act: place at level / place a piece

The **place-whole-at-level** path largely already exists: placing a cluster and
setting its energy `level` resolves it to 1..N pieces via the in-flight plan's
`keep_spans`/`remove_spans` — broad = one flowing cut, punchy = the tight pieces as
an internal montage. Verify `place(ref, level)` + `trim`/`retime` operate correctly
on a cluster cut (they act on spans, so they should).

**New: place one specific piece.** Add a small, generic selector so the brain can
place just a piece (e.g. the climax) instead of the whole moment:

- **Addressing:** reuse the stable piece ids the in-flight plan stores in the tree
  (`salience.events` order / children map). A piece ref = the cluster ref plus a
  piece index (e.g. `ref="<cluster>", piece=k`), or an encoded ref
  `"<cluster>#k"` — pick whichever fits the existing `ref` grammar. **Requires the
  in-flight plan to give pieces stable ids — confirm this (see §8).**
- **Resolver:** extend the map resolver (`_MapIndex.resolve`) so `(cluster, piece)`
  resolves to that piece's span (from the punchy-level resolution of that event).
  A placed piece is then an ordinary cut (single span) — `trim`/`retime`/review all
  work on it unchanged.
- **Tool schema (`tools.py`):** add an optional `piece` field to `place` (integer,
  "index of a sub-beat within a multi-beat cluster; omit to place the whole
  moment"). Generic wording only.

That's the whole act surface: **level for whole-vs-broken, `piece` for one beat.**

## 5. What stays the same
- Segmentation, cluster tree, ladder, per-level resolution — owned by the in-flight
  plan; untouched here.
- Speech path — untouched.
- `trim`, `retime`, `diagnose`, `validate`, `review`, done-gate, length math —
  unchanged; they operate on placed spans.
- Single-event footage — identical to today in framing, reading, and acting.

## 6. Explicit non-goals (deferred / dropped)
- **Per-event frames / per-event semantic captions** (parked option 3). Revisit
  only if the brain is caught making bad piece choices — and even then it's opt-in
  and only cheap if frames were being sent, which they won't be.
- **Exploding a cluster into separate, re-orderable timeline items.** Placing at
  punchy = an internal montage within one cut (stable identity). For linear editing
  this plays identically to N sequential cuts, so it's out. Only needed if the user
  wants to reorder/intersperse individual beats — future.
- **Selection-by-target-length** (dropped earlier — not helpful now).

## 7. Tests
- `image_plan`: a multi-event cluster yields **≤2 frames** (regression against the
  per-event increase); single/runt cluster yields 1; existing point-straddle
  preserved. Total per-clip budget unchanged.
- `footage_map`: single-event cluster → one line (byte-identical to today);
  multi-event cluster → primary line + terse generic piece list; no domain strings.
- `observe.read_state`: piece breakdown matches the Beat Index for a multi-event
  cluster; empty for single-event.
- `act`/resolver: `place(ref, level)` on a cluster resolves whole↔pieces;
  `place(ref, piece=k)` resolves to piece k's span; `trim`/`retime` then work on it;
  bad/out-of-range `piece` is a clean no-op/error.
- All V3 and in-flight-plan tests stay green.

## 8. Anything else? — honest assessment

For **linear editing**, I believe **A + B + C is complete**. The brain reads the
whole Beat Index up front (so "forward awareness" is already covered), sees each
moment's pieces/durations/salience, and can place a moment whole, broken, or as a
single beat. `trim`/`retime`/done-gate/length-math handle propagation. Nothing else
is structurally required.

**One thing to confirm, not add:** the in-flight plan must store **stable piece
ids** in the cluster tree (§3 of that plan says it stores the children map) — Part C
addressing depends on it. If those ids aren't stable/exposed, add that to the
in-flight plan rather than here.

**Watch-items (not work, just eyes on):**
- After placing a cluster, if the brain later changes its `level`, `keep_spans`
  change (more/fewer pieces) — make sure `review`/audit still read cleanly (they
  key off spans, so they should).
- Keep the piece list terse — piece count is small because well-separated moments
  are already separate clusters, so this never bloats context.

## 9. Rollout
1. Land Part A first (frame revert) — pure budget safety.
2. Part B (read) — additive context, zero behavior change for single-event footage.
3. Part C (act) — the `piece` handle + resolver.
4. Re-ingest a multi-event project; verify 2 frames/cluster, correct piece list, and
   that placing whole / a single piece both work.
5. **Commit and push** the changes once implemented and tests are green (a clear,
   descriptive commit message; push to the working branch).
