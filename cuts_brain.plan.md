# Cuts → Brain — phased plan (smaller prompt, more freedom)

Goal: make the cuts the brain reads **leaner and less opinionated**, while
keeping *total awareness* on tap. The resident prompt shrinks and stops
pre-deciding narrative; the deep detail (every cut facet + raw VLM events /
transcript) becomes **retrievable on demand**, not resident.

North star (locked principles):
1. **Limited resident prompt.** One short line per cut. Detail lives in Tier-1.
2. **Maximum freedom.** Emit *what was captured* + *how cuts connect*; never
   *what a cut is FOR* (no narrative role baked in). The brain decides at
   placement, from content + relations + energy.
3. **Combine, never collapse.** A moment is ONE unit by default, but its members
   are always reachable (the zoom ladder). We never delete the parts.
4. **No duplication.** If the transcript already says it, don't repeat it (the
   graphic-gist rule).

## What we already have (no new perception needed)
- Cuts carry `primitives` (P6) and graphic `summary` (P5).
- Clusters carry a zoom ladder: Broad = whole run, Sharp = peak member (TREE v6/7).
- Tier-1 retrieval exists (`footage_map.moment_detail` / orchestrator
  `inspect_moment`) — today it returns the moment + clip context only.
- L2 perception holds the raw `events`, `content_units`, `cutaways`, and the
  transcript — none of it currently reachable from a moment.

---

## The Tier-0 line: before → after

Today (`footage_map._moment_line`):
```
m03 speech+reaction S1 *answer* .82 [1:12-1:18] "well, the thing is…" · nrg:broad|balanced|tight · rel:responds_to>m02
```
After:
```
m03 person+speech S1 .82 [1:12-1:18] "well, the thing is…" · nrg:broad|balanced|tight · rel:responds_to>m02
```
Changes: **+primitive** (the honest atom), **−role** (`*answer*` gone), summary
shown only when additive (see P3). Net: shorter, less biased.

---

## Phases

### P1 — Drop `role` from the brain (kill the narrative bias)
- `footage_map._moment_line`: remove the `*role*` tag from the resident line.
- `footage_map.build_clip_tree` / `moment_detail`: stop carrying `role` into the
  brain-facing moment (leave the stored `HeroCut.role` field for back-compat,
  just don't surface it to the arranger).
- Frontend `hero-cuts-view.tsx`: remove the role badge + `ROLE_LABEL`.
- Rationale recap: `answer` = the `answers` relation; `listener` = a
  `reaction/person` cut; `establishing` = `primitive=place` at the open;
  `hook/cta/climax` = pure intent the brain should own. All redundant or biasing.
- Bump `TREE_VERSION`. (No hero-cut recompute — purely the tree/prompt layer.)
- Non-goal: removing the VLM's role emission from L2 (defer; harmless once unused).

### P2 — Add `primitive` to the Tier-0 line
- `footage_map._affordance_tag` → `_capture_tag`: lead with the cut's
  `primitives` (e.g. `person+speech`, `graphic`), so the brain sees *what was
  captured*. Keep it to the distinct primitives (cheap: one or two words).
- Decide affordance vs primitive in the line: **primitive wins** as the visible
  atom; affordance stays in the struct for any use that wants the editorial lens.
- Same `TREE_VERSION` bump as P1 (one version step covers P1+P2).

### P3 — Minimal, conditional graphic `summary`
- Keep the gist a **short tag** (≤ ~6 words). Enforce a hard cap when surfacing
  it (truncate in `_moment_line`); the full text stays in the struct + Tier-1.
- **Conditional surfacing**: in `build_clip_tree`, mark a graphic/insert moment's
  summary as "redundant" when a `speech` cut overlaps its span (the voiceover
  already narrates the screen — the screen-recording case). Only surface the
  summary in the resident line when **no concurrent speech** covers it.
- Carry a boolean (e.g. `summary_covered_by_speech`) on the moment so the line
  renderer can decide; the brain can still pull the full gist via Tier-1.
- Same `TREE_VERSION` step.

### P4 — Combined moment as ONE unit (UI + brain)
- **Brain**: a cluster is already emitted in the `MOMENTS` section with its
  whole-run→peak ladder. Make it the *default candidate* the brain places (take
  the run loose, or the peak tight) — members reachable via Tier-1, not as N
  competing siblings in the resident text. (Tighten `_cluster_line`; keep member
  lines but signal they roll up under the cluster.)
- **UI** (`hero-cuts-view.tsx`): render a moment as **one combined preview** that
  plays the whole run (chained member spans, like the existing breath-jump
  preview), with an expand affordance to reveal member cards. Replaces today's
  flat grid-of-members bundle. Combined by default, decomposable on click.
- Invariant: never drop members — expansion + ladder always restore them.
- **Energy exposes members (the release valve).** A fused member is never
  hidden: it re-emerges as energy rises, via TWO ladders that already exist —
  (1) the cluster zoom ladder narrows whole-run → peak member
  (`k = round(n·(1−band/4))`), so a typical small moment shows its **peak alone
  by Tight**; (2) the fuse-gap shrinks (P6) so loose moments dissolve. And the
  cluster line carries ALL its rungs at the anchor band, so the brain can grab
  "peak alone" without moving the global slider. This is what resolves the
  moment-vs-granularity tension — don't add a separate un-fuse control.
- Bump `TREE_VERSION` if the cluster line shape changes; frontend-only otherwise.

### P6 — Re-center the fuse↔atomize ladder onto the 2-4 working range
- Observation: energy **1 (Broad) and 5 (Sharp) are extremes**; the genuinely
  used range is **2-4 (Calm / Balanced / Tight)**. But today full atomize
  (`fuse_gap_ms = 0`) is parked at the rarely-touched Sharp end, so a member
  only cleanly stands alone at the extreme.
- Re-center `_FUSE_GAP_MS` so the **entire fuse→atomize transition lives in 2-4**,
  with 1/5 as saturated anchors: e.g. `(1500, 1000, 600, 250, 0)` →
  `(1500, 800, 350, 0, 0)` (atomize arrives at **Tight**; Sharp adds only the
  breath-removal punch). Values to **validate, not lock**.
- The brain's default read = the **Balanced pivot (band 3)**: moments exist, peak
  is one rung away.
- Bump `PARAMS_VERSION` (this changes grouping) → hero caches recompute lazily.
- Validate on the reel-trail (wants moments) + demo/screen clips: confirm small
  moments expose their peak member by Tight, big bundles still hold at Calm.

### P5 — Tier-1: make every cut + VLM event detail retrievable
- Enrich `footage_map.moment_detail` (and the orchestrator `inspect_moment`
  tool) to return, for the moment's span:
  - full cut facets (already have: primitives, summary, people, framing,
    quality, variants, relations),
  - **the raw VLM `events` overlapping the span** (the physical-business
    timeline),
  - **the transcript window** over `[in_ms, out_ms]` (verbatim words),
  - the underlying `content_units` / `cutaways` for the span.
- Load lazily from L2 perception + transcripts by `file_id` (best-effort; absent
  artifacts → fewer fields). This keeps the resident prompt tiny while giving the
  brain "open the file and read" depth when it chooses to inspect.
- No `TREE_VERSION` change (retrieval path, not the cached tree).

---

## Versioning & recompute
- P1–P4 (tree/prompt shape) → **one `TREE_VERSION` bump**; trees rebuild lazily.
- P6 changes grouping → **one `PARAMS_VERSION` bump**; hero caches recompute, and
  trees rebuild off them.
- P5 is a live retrieval path — no cache version touched.

## Risk / tension to watch
- **Moment-as-unit vs granularity**: collapsing members in the resident text
  could hide a member the brain wanted alone. Mitigation: the ladder (loose→peak)
  + Tier-1 member retrieval keep every part reachable. Validate on the reel-trail
  and demo clips before locking.
- **Conditional summary**: speech-overlap is a heuristic; if it suppresses a
  gist the brain needed, the full text is still one `inspect_moment` away.

## Out of scope (explicitly deferred)
- Music / SFX primitives (separate audio pipeline).
- Removing role emission from L2 perception.
- Rich arc/density UI views (kept "later" per P7).
