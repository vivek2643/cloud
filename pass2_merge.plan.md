# Fold Pass 2a into Pass 2b — one unified Pass 2

## Goal

Collapse the two Pass-2 LLM passes into **one per-cut vision pass**. Today:

- **Pass 2a** (`pass2a.py`) — identity + take resolution. Carries the expensive
  **take-group co-location sharding** so the model can compare take/outlook
  members side by side.
- **Pass 2b** (`pass2b.py`) — visual judgment (framing/look/captions/people).
  Pure chunking, no cross-cut dependency.

Both send the **same frames** (2b re-sends the pixels 2a already saw), so images
are transmitted **twice**. Merging them → one call per batch, images sent
**once**, no co-location machinery.

## Why this is safe now (the honest premise)

Pass 2a was built to make three calls that needed cross-cut pixels: **winner
selection, outlook vs take, cross-file identity**. All three have since moved
out from under it:

- **Take grouping** → Pass 1 `take_candidates` (semantic token overlap).
- **Winner** → deterministic `post._enforce_take_winner` (highest
  `total_quality` in a same-setting cluster) — "replaces pass 2's own winner call
  entirely".
- **Outlook roles** → deterministic `pass2.apply_outlook_roles` (from Pass 1
  `outlook:` groups).
- **on_camera** → deterministic `identity/apply.py` (motion↔voice binding); 2a's
  guess is only a *fallback* when binding abstains.
- **Identity / appearance fingerprints** → **Pass 2b's `PersonLook.appearance`**
  (NOT 2a) + L1 diarization.

The one thing still riding on 2a's co-location is that the **model assigns
`take_group_id`/`take_role` for genuine (non-outlook) retakes** — see
`merge_identity_and_visual` copying `identity.take_group_id/take_role`. That is
the last reason co-location exists. Move it to code and the co-location
requirement disappears, unblocking the fold.

Everything else 2a emits (`label`, `summary`, `channel`, `on_camera`,
`natural_sound`, `speaker`) is **per-cut** and belongs in the per-cut pass.

## Principle

**Behavior-preserving except for one deliberate structural change:** model
take-grouping → deterministic code. Every downstream consumer
(`identity/apply.py`, `post.assemble_cut_records`, `footage_map`, `observe`)
reads `Pass2Cut` / the cut-record shape, which stays **byte-compatible**. So the
change is contained to `pass2a.py` / `pass2b.py` / `pass2.py` + the `ingest.py`
wiring; nothing downstream changes.

---

## Phase 0 — Decisions (defaults chosen; flagged for the executor)

- **D1 — Take grouping owner.** Move it fully to **code** (recommended;
  required for the fold to drop co-location). Stamp `take_group_id`/`take_role`
  onto cuts from Pass 1 `take_candidates` deterministically, exactly as
  `apply_outlook_roles` already does for outlook groups.
- **D2 — Drop `take_group_id`/`take_role` from the model schema** (follows D1).
  The model never emits them again; code owns them end to end.
- **D3 — Visual "same-setting" confirmation.** Dropping model take-grouping also
  drops 2a's pixel check that a same-line group is *visually* the same setting.
  **Recommended: drop it** (Pass 1 semantic grouping is the authority; the winner
  is picked by quality anyway, and current footage has ~0 non-outlook retakes).
  *Optional safety*: keep a single boolean `same_setting` the model may set per
  cut, used only to split a take cluster when the shots clearly differ. Note as a
  known trade-off, not a blocker.
- **D4 — Module layout.** Recommended end state: **one `pass2.py`** that owns the
  merged model, the runner, batching, locator backfill, and the deterministic
  take/outlook/junk stampers; **delete `pass2a.py` and `pass2b.py`** (move their
  still-used schema pieces — `Framing`, `Look`, `Appearance`, `PersonLook`,
  `TasteFences` — into `pass2.py`). Lighter-touch alternative: keep the runner in
  `pass2b.py` (rename its docstring to "Pass 2"), delete only `pass2a.py`. Pick
  one; the plan below assumes the single-`pass2.py` target.

---

## Phase 1 — Deterministic take-group stamping (prerequisite)

Generalize `pass2.apply_outlook_roles` → **`apply_take_groups(pass2, pass1)`**:

- Iterate ALL Pass 1 `take_candidates`. For each member cut `(file_id,
  word_span)`:
  - group_id prefixed `outlook:` → `take_role="outlook"` (unchanged behavior),
  - otherwise → `take_role="take"`, same shared `take_group_id`.
- `post._enforce_take_winner` is unchanged: it reads `take_role in
  ("winner","take")` to form the same-setting cluster and crowns the highest
  `total_quality` as `winner`.
- Verify the outlook path is byte-identical to today; the retake path is now
  code-driven instead of model-driven.

Result: `take_group_id`/`take_role` no longer depend on the model seeing members
together → **co-location is no longer needed.**

---

## Phase 2 — Unified Pass 2 schema

Define the single per-cut model the LLM emits (expand `VisualJudgment` into a
`CutJudgment`, or emit `Pass2Cut` directly). It carries **the union of 2a's
model-owned fields + 2b's**, minus the code-owned ones:

**Model emits (per cut):**
- `source_ref` (join back to Pass 1), `kind`
- `label`, `summary`
- `speaker` (keep current behavior), `on_camera`
- `channel` (said/done/shown), `natural_sound`
- `framing` (subject_box, crops, `rotation_deg`, `shot_size`)
- `look` (with **nested** `white_reference` — keep the pass2b prompt fix)
- `people` (`PersonLook` incl. structured `appearance` — unchanged; the identity
  map depends on this)
- `caption_zones`, `taste_fences`, `readability_ms`

**Code owns (NOT model-emitted):**
- `word_span` / `atom_ids` — backfilled from `source_ref` (`backfill_locators`).
- `file_id` — authoritative from Pass 1 (keep 2a's file_id reconciliation guard).
- `take_group_id` / `take_role` — **removed from schema**; set by
  `apply_take_groups` (Phase 1).
- `junk` / `junk_reason` — set by `apply_junk_suspects` from Pass 1 (as today);
  keep the model's optional junk flag only if it currently adds value, else drop.

Addressing: the model references cuts by `source_ref` (+ an emitted ordinal when
it splits a video group). The cross-pass `cut_index` that 2b used to index into
`IdentityOutput.cuts` is **gone** — there's no second pass to align to.

Video-group **splitting is preserved**: the single pass may still emit >1 cut per
`video_group[...]` source_ref (all sharing the group's planned frames);
`backfill_locators` assigns each sub-cut its `atom_ids`.

---

## Phase 3 — Single runner + batching

- **`build_pass2_batches(pass1_output, planned_frames)`** — chunk Pass 1 units
  (speech cuts, video groups, take members) into batches by simple size, with the
  ONE constraint that a given `source_ref`'s planned frames stay within a single
  batch (so a video-group split sees all its frames). **No take co-location.**
- **`run_pass2_batch(pass1_output, batch_units, planned_frames, images_b64)`** —
  one LLM call returning `CutJudgment`s for the batch. Reuse from the old passes:
  - `render_pass1_output` scoped to the batch's refs (from 2a),
  - `_images_for_cut` / image-block builder (from 2b),
  - `backfill_locators`, invented-ref filtering, file_id reconciliation (from 2a).
- **Re-tune batch size.** Each cut's output is now larger (identity + visual in
  one object), so set `MAX_CUTS_PER_PASS2_BATCH` conservatively (start around the
  old identity-shard size, not the larger visual-batch size) and keep the
  existing one-re-ask-on-schema-violation loop. Batches run in parallel
  (`MAX_PARALLEL_*`).
- **Images sent once** (was twice) — the concrete cost win, on top of halving the
  call count.

---

## Phase 4 — Wire into `ingest.py`

Replace the two blocks (2a shards + 2b batches + merge, ~lines 172–218) with:

```
store.set_status(ingest_run_id, "pass2")
batches = pass2.build_pass2_batches(pass1_output, planned_frames)
cuts = []  # parallel run_pass2_batch, extend
pass2_output = pass2.Pass2Output(cuts=cuts)
pass2_output = pass2.apply_junk_suspects(pass2_output, pass1_output)
pass2_output = pass2.apply_take_groups(pass2_output, pass1_output)   # Phase 1
```

- `pass2_output, identity_map = identity_apply.run(...)` — **unchanged**.
- `post.assemble_cut_records(...)` — **unchanged**.
- Keep the DB status label `"pass2"` (no migration).
- Delete `merge_identity_and_visual` (the single pass emits the merged shape
  directly), `build_identity_shards`, `run_identity_shard`, `build_visual_batches`,
  `run_visual_batch`, `IdentityOutput`/`IdentityCut`.

---

## Phase 5 — Prompt

Merge the two system prompts into one lean Pass-2 brief:
- From 2a: enumerate/confirm cuts by `source_ref`; may split a video group;
  `channel` (said/done/shown); `label`/`summary`; `on_camera` + per-person
  `speaking`; `natural_sound`.
- From 2b: `framing`/`shot_size`; `look` with **nested `white_reference`**;
  `people` with structured `appearance`; `caption_zones`/`taste_fences`.
- **Remove** all take/winner/outlook instructions (code owns those now).
- Keep it categorical/descriptive (house rule): the model describes and
  classifies; it never emits numbers it can't ground, IDs, or take verdicts.

---

## Phase 6 — Tests + verification

- **Unit:** fold `test_pass2a.py` into `test_pass2.py`; update `test_ingest.py`
  fakes from `run_identity_shard`/`run_visual_batch` → the single
  `run_pass2_batch`. `test_image_plan.py` is unaffected.
- **`apply_take_groups`:** outlook path byte-identical to `apply_outlook_roles`;
  a synthetic non-outlook take group gets `take`/`winner` via
  `_enforce_take_winner`.
- **Real re-ingest of podcast trail 3** (`57b689b3-…`), diff `cut_records`
  against the current baseline run `df99a8c5-…`:
  - same cut count and kinds,
  - `on_camera` distribution unchanged (identity map is upstream-agnostic — was
    58/42 overall, 28/33 clean 1-on/1-off),
  - `framing`/`look`/`channel`/`people` all populated,
  - outlook roles + winners unchanged,
  - confirm frames are pulled/sent **once** (log the image count).
- **Downstream smoke:** `footage_map` renders, brain `observe`/`read_state`
  works, a small edit runs.

---

## What this unlocks (out of scope here, noted)

- **~2× image-token cut** immediately (images sent once, one call instead of two).
- A **single vision-pass integration point** — the natural place to later swap in
  a cheaper model (e.g. Gemini Flash) and gated multi-frame, without touching the
  cutting logic. Do those as separate tasks.

## Risks

- **Lost visual same-setting filter** on non-outlook retakes (D3) — low impact on
  current footage; revisit with the optional `same_setting` field if a real regression
  shows up.
- **Batch-size tuning** — bigger per-cut output; start small and watch for schema
  re-asks.
- **Video-group split parity** — verify the single pass still splits groups and
  `backfill_locators` assigns sub-cut `atom_ids` correctly (covered by the
  re-ingest diff).

## Files to touch

| File | Change |
|---|---|
| `l3/pass2.py` | Home of merged `Pass2Cut` model, single runner, `build_pass2_batches`, `backfill_locators`, `apply_take_groups` (generalized from `apply_outlook_roles`), `apply_junk_suspects`; move `Framing`/`Look`/`Appearance`/`PersonLook`/`TasteFences` in; delete `merge_identity_and_visual` |
| `l3/pass2a.py` | **Delete** (helpers re-homed into `pass2.py`) |
| `l3/pass2b.py` | **Delete** (runner + schema re-homed into `pass2.py`), or keep as the renamed single-pass runner per D4 |
| `l3/ingest.py` | Replace the 2a/2b/merge blocks with the single-pass flow; swap `apply_outlook_roles` → `apply_take_groups` |
| `l3/pass2_params.py` | Rename/retune batch/parallelism constants for one pass |
| `scripts/test_pass2a.py`, `scripts/test_ingest.py`, `scripts/test_pass2*.py` | Update to the single runner |

## Non-goals

- No change to Pass 1, the identity map, quality scores, `post`, `footage_map`, or
  the brain.
- No model swap and no multi-frame here (separate follow-ups the single pass sets up).
- No new take-grouping *logic* — just moving the existing Pass-1 grouping onto cuts deterministically.
