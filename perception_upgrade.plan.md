# Perception Upgrade + Pass 1 Cut Tightening — implementation plan

Status: ready to implement (from another chat). Branch base: `gemini-pass2`
(already carries the validated Flash-Lite Pass 2 fix).

This plan bundles five independent changes. They can land in one branch but
are ordered so the **proven** switch ships first and the new perception work
is separable/reversible.

Guiding principles (unchanged, enforce throughout):
- **Code owns numbers/structure; the LLM owns categories/text.** No model-
  emitted scores, thresholds, or millisecond values.
- **Deterministic keep, semantic cull.** Never silently drop content.
- **Flash-Lite guardrail (hard-won):** every NEW model field is
  additive, OPTIONAL, and prompt-nudged — NEVER made `required`/non-nullable in
  the Gemini sanitized schema. Requiring a field or raising `thinking` above
  `low` triggers unbounded dynamic-thinking spirals on hard b-roll batches
  (`think_tok` 30k+, `finish=MAX_TOKENS`, zero JSON). Keep `ingest_pass2_thinking
  = "low"`.

---

## Part A — Make Flash-Lite the default and push to `main`

Current branch config state (`backend/app/config.py`) — VERIFIED:
- `ingest_pass2_model = "gemini-3.1-flash-lite"` — already set.
- `ingest_pass2_thinking = "low"` — already set (MUST stay low; see guardrail).
- `ingest_pass2_provider = "anthropic"` — **STILL NOT FLIPPED.** The implementer
  MUST change this to `"gemini"`. This single line is what actually activates
  the switch; without it everything else is dormant and Pass 2 stays on Claude.
- Pass 2 fix set already on branch: optional `subject_box`, soft
  `_GEMINI_REINFORCE`, `video_cut` kind alias, `_diag()` logging (`pass2.py`,
  `ingest_gemini.py`).
- Also on branch (intended, keep): `converse.py` Lever-2 "GUESS FROM CONTEXT"
  prompt block + the `split_edit` A/V-sync guidance line.

A/B verified across podcast + drone/b-roll reel + montage reel: 6/6 ingests OK,
coverage at Sonnet parity (`subject_box` 100%, `people` 94–99%, `shot_size`
95–100%), Pass 2 cost 28–47× cheaper, no `IngestFailure`.

Steps:
1. Set `ingest_pass2_provider = "gemini"` in `backend/app/config.py` (the one
   line that turns the switch on).
2. Commit any remaining branch changes.
3. Merge `gemini-pass2` → `main` (no rebase needed unless conflicts) and push.
4. **Re-ingest every project on `main`** so all runs use Flash-Lite Pass 2.
   (Existing older runs stay valid — all new fields below are optional with
   pre-migration fallbacks — but a uniform re-ingest is cleanest.)

> Decision point for the implementer: ship Part A as its own merge FIRST, then
> do Parts B–E as a second change set, and re-ingest once at the end. This keeps
> the proven switch independent of the new perception work.

---

## Part B — Two frames per cut (early + late)

Today Pass 2 sees ONE still per cut (`image_plan` picks the single sharpest
instant). A second frame later in the span lets the model perceive **change
over time** — the pipeline's biggest perceptual blind spot. Plumbing already
supports multiple frames per `ref` (drift/anchor extras share a ref; the batch
loop already gathers all a ref's frames — `ingest.py`
`batch_frames = [f for f in planned_frames if f.ref in ref_set]`).

Scope note: this improves how cuts are **described**, not where they land
(Pass 1 is text-only). Cuts stay in the same places.

### B1 — `image_plan.py`: emit an early + late frame per unit
File: `backend/app/services/l3/image_plan.py`

- Add a `phase: str = "only"` field to `PlannedFrame` ("only" | "early" |
  "late"); include it in `to_dict`.
- For every MANDATORY unit (speech cut, take member, unanchored video group),
  produce up to TWO frames instead of one:
  - `early` = `_sharpest_ms(blur, hop_ms, s, mid, default=s+quarter)` over the
    first ~half of the span.
  - `late`  = `_sharpest_ms(blur, hop_ms, mid, e, default=e-quarter)` over the
    second ~half.
  - where `mid = s + (e - s) // 2`.
- For ANCHORED video groups: first frame = first anchor (as today, `phase=early`),
  second frame = LAST anchor if there is more than one anchor and it is far
  enough from the first; else the calm+sharp late-half instant. Remaining
  anchors stay budgeted extras (unchanged).
- **Runt guard (deterministic, code-owned):** send only ONE frame (`phase=only`)
  when EITHER:
  - the span is short by the clip's own distribution — compare `(e - s)` to a
    data-driven floor derived from that clip's unit-span distribution (e.g.
    below the clip's own median unit length AND below ~2× `hop_ms` worth of
    frames apart); OR
  - the two candidate timestamps are within one `hop_ms` of each other (the two
    frames would be near-duplicates that teach nothing).
  No hardcoded absolute ms — derive the floor from the clip's own spans/hop.
- **Budget:** the 2nd frame is a NEW extras tier, `REASON_SECOND_MOMENT`,
  prioritized ABOVE `extra_anchors`/`drift` but BELOW the 1st mandatory frame,
  so under budget pressure a clip gracefully falls back to 1 frame per cut
  rather than dropping whole units. Raise `FRAME_BUDGET_PER_CLIP` 24 → ~40 to
  accommodate the doubling on dense clips.

### B2 — `pass2.py`: label the pair + tell the model
File: `backend/app/services/l3/pass2.py`

- `build_pass2_batch_blocks`: sort by `(file_id, ref, ts_ms)` so a ref's two
  frames are adjacent, and label with phase:
  `IMG n = clip X, T.s, ref R (early)` / `... (late)` (fall back to no suffix
  when `phase=="only"`).
- `_SYSTEM` prompt: add a short paragraph — "You may see up to TWO frames for one
  cut (labelled early/late): they are the SAME cut at two moments. Read them
  together and describe what CHANGES between them. NEVER emit two cut records for
  one source_ref just because it has two frames." This reinforces the existing
  "one cut per source_ref" rule (already validated by `_source_refs_exist` /
  `backfill_locators`).
- `MAX_CUTS_PER_PASS2_BATCH` (`pass2_params.py`): halve it (image bytes per
  batch roughly double). Cheap on Flash-Lite; keeps request size safe.

### B3 — verify frame extraction
`frames.extract_for_planned_frames` keys by `(file_id, ts_ms)` — two frames per
ref with distinct `ts_ms` extract independently; no change needed. Confirm no
dedup collapses same-ref frames.

---

## Part C — New per-cut perception fields (all optional, prompt-nudged)

### C1 — `summary` becomes the "what's happening" narrative (no new column)
`summary` already exists end-to-end. Do NOT add an `action_summary` column —
repurpose `summary`:
- `_SYSTEM`: require `summary` to describe **what is happening / what changes
  across the frames** (the action, the beat, the on-screen event), not a static
  description. `_GEMINI_REINFORCE` already forces non-empty label+summary; add
  one line that summary must be concrete about the action/change.
- No schema/DB/frontend change (field already flows).

### C2 — `shot_quality` (rides the `framing` jsonb — NO migration)
Technical stability a single still can't judge; the 2nd frame makes it
answerable.
- `pass2.py` `Framing`: add `shot_quality: str = "unsure"` with a closed
  vocabulary constant `SHOT_QUALITY = ("stable","shaky","whip","soft_focus",
  "racking_focus","exposure_shift","unsure")`. It persists inside the existing
  `framing` jsonb column — **no migration, no read-path change.**
- `ingest_gemini.gemini_schema`: inject `framing_props["shot_quality"]["enum"] =
  list(SHOT_QUALITY)` next to the existing `shot_size` enum injection. Keep it
  OPTIONAL (do NOT add to `required`).
- `_SYSTEM`: one line defining `shot_quality` and its categories.
- Frontend `api.ts` `Framing`: add `shot_quality?: ...` (optional).
- Optional downstream (additive): let `post._total_quality`'s visual term
  penalize `shaky`/`soft_focus`/`racking_focus`. Keep as a deterministic map;
  gate behind "field present" so old runs are unaffected.

### C3 — `screen_text` (new column, migration required)
Reads on-screen text/graphics (slides, lower-thirds, screen shares) and whether
it changed A→B. Unlocks tutorials/explainers/screen-recordings/news.
- `pass2.py`: add `screen_text: str = ""` to `CutJudgment` AND `Pass2Cut`; copy
  it in `to_pass2_cuts`.
- `post.py`: add `screen_text: str = ""` to `CutRecord`, its `to_dict`, and the
  constructor call in `assemble_cut_records`.
- `ingest_store.py`: add `screen_text` to the INSERT column list + values.
- `cuts_v3_read.py`: add `screen_text` to the SELECT list.
- Migration `backend/migrations/038_pass2_perception.sql`:
  `alter table cut_records add column if not exists screen_text text not null
  default '';`
- `ingest_gemini.gemini_schema`: `screen_text` is a plain nullable string —
  needs no enum; ensure the sanitizer leaves it as an optional string (it will).
  Do NOT make it required.
- `_SYSTEM`: one line — "screen_text: any legible on-screen text/graphics
  (title, lower-third, slide, UI); note if it changes between the frames; empty
  if none."
- Frontend `api.ts` `CutRecord`: add `screen_text?: string`.
- `footage_map.py`: include `screen_text` in the `_moment` dict (near `summary`,
  ~line 282) so the brain sees it.

---

## Part D — Deterministic salience / peak-moment (F8), code-owned

A single deterministic "strongest instant" per cut, fused from signals L1
already computed — useful for emphasis, thumbnail choice, punch-in timing, and
for the brain to know where a cut peaks. This is NOT the LLM's job (it's a
number) and is DISTINCT from `hero_ts_ms` (which is the best *still* for
display; salience is the strongest *event* moment).

File: `backend/app/services/l3/post.py` (`assemble_cut_records` already receives
`audio_by_file` with `rms_db`/`hop_ms` and `motion` with `action_energy`/
`anchors`; `onsets_ms`/`silence_intervals` are in `audio_features`).

### D1 — compute the curve
- New `_salience(...)`: over the cut span, build a per-hop curve fusing
  (a) normalized loudness (`rms_db`, normalized in-clip like `speech_quality`),
  (b) `action_energy`, and (c) an onset/anchor proximity bump. Pick
  `peak_ms` (absolute) = argmax, and `score` = normalized peak height 0..1.
  Fully deterministic; reuse existing helpers (`_norm_in_clip`,
  `_span_slice`, `_mean_in_span`).
- Guard: if no signals available, `peak_ms = hero_ts_ms`, `score = 0.0`.

### D2 — persist + surface
- `CutRecord`: add `salience: Dict[str, Any] = field(default_factory=dict)`
  ({peak_ms, score}); add to `to_dict`.
- Migration 038 (same file as C3): `alter table cut_records add column if not
  exists salience jsonb not null default '{}'::jsonb;`
- `ingest_store.py`: INSERT `json.dumps(r.salience)`.
- `cuts_v3_read.py`: add `salience` to SELECT.
- Frontend `api.ts` `CutRecord`: add `salience?: { peak_ms: number; score: number }`.
- `footage_map.py` `_moment`: include `salience` so the brain can read a cut's
  peak.

### D3 — other deterministic signals worth surfacing (low-cost, optional)
We ALREADY compute but don't always surface these — the plan notes them; include
if cheap:
- `_energy_grade(mean_action_energy)` (calm/active/etc.) → could ride in
  `salience` or `pace` — already derived, purely additive.
- `camera` (already a column + in `_moment`) — confirm it stays surfaced.
No new heavy analysis (no MIR / no segmentation model) — out of scope.

---

## Part E — Tighten Pass 1 speech-cut logic (stop tiny / wrong splits)

Symptom observed: some speech cuts are very small / fragmented, or split where
they shouldn't be (a single delivered thought broken across cuts). Fix in two
layers, keeping the split cleanly: **LLM owns "is this one thought"; code owns
"is this a runt AND safe to absorb".**

### E1 — Prompt tightening (semantic), `pass1.py` `_SYSTEM`
Sharpen the `speech_cuts` instruction:
- Define a speech cut as **a complete, deliverable thought** — a full sentence/
  clause or a coherent multi-sentence beat — that an editor could place on its
  own. NOT a fragment, connector, or trailing tail.
- Explicit anti-patterns: do NOT emit a 1–3 word runt (a stray "yeah", "so",
  "and then", a trailing "…right?") as its own cut — absorb it into the adjacent
  thought it belongs to. Do NOT break mid-thought on a dramatic pause (this rule
  exists; strengthen it).
- Keep the existing `beat_id` mechanism (for a thought genuinely split across
  cuts) and the junk/false-start guidance.

### E2 — Deterministic runt guard (code disposes)
New step after Pass 1 + `_bridge_beats` + `_fold_silent_speech_cuts`
(`pass1.py`). Detect a "runt" speech cut — deterministically short by the clip's
OWN distribution (median word-count/duration of that clip's speech cuts; NO
hardcoded ms) — and ABSORB it into an adjacent cut ONLY when ALL hold (mirrors
the beat-bridge safety, "never wrong-merge"):
- same `file_id` and same speaker,
- word spans are contiguous (`prev.word_span[1] + 1 == runt.word_span[0]` or the
  symmetric next case),
- the seam between them is weldable (`seam.classify_seam` — no shot change /
  transition / speaker change, gap not longer than what it bridges),
- neither the runt nor the neighbor is a take/outlook member,
- the runt is not flagged junk.
Otherwise leave it split (prefer split on any doubt). Log how many runts were
absorbed.

Rationale: whether two ranges are one thought is semantic (E1); whether a cut is
a runt and whether merging is *safe* is quantitative + seam-based (E2). This is
the same "propose/dispose" split already used for `beat_id`.

---

## Part F — Verification (do before calling done)

Re-ingest podcast (`57b689b3-…`), drone/b-roll reel (`a294f9da-…`), montage reel
(`72d87ca9-…`). Check:
1. **No `IngestFailure`**, no schema fallbacks, `thinking` stays `low`.
2. **Pass 1 tightening:** fewer tiny speech cuts vs the pre-change baseline;
   spot-check that no content was lost (runts absorbed, not deleted); take/
   outlook groups unchanged.
3. **Two frames:** confirm image plan emits early/late where spans allow and
   falls back to one frame on short spans; batch sizes sane.
4. **New fields populated:** `summary` reads as an action/change narrative;
   `shot_quality` present (in `framing` jsonb); `screen_text` populated where
   text is on screen (tutorials/graphics), empty otherwise; `salience.peak_ms/
   score` present and sane (inside the span).
5. **Coverage parity holds** (`subject_box` ~100%, `people` 90%+), `on_camera`
   distribution unchanged.
6. **Downstream smoke:** `footage_map`/`observe` render the new fields; brain
   `PIC:` still resolves; frontend `tsc` passes with the new optional types;
   captions/color views unaffected.
7. **Side-by-side spot check** of ~15 matched cuts to confirm the richer
   summaries + new fields read correctly.

Tests: extend `image_plan` tests (two-frame + runt fallback), a `_salience`
unit test, a Pass 1 runt-guard unit test (absorb vs leave-split cases), and a
`gemini_schema` test asserting `shot_quality` gets its enum and `screen_text`
stays an optional string (neither becomes `required`).

---

## Migration summary
One new file `backend/migrations/038_pass2_perception.sql`, idempotent/additive:
- `cut_records.screen_text text not null default ''`
- `cut_records.salience jsonb not null default '{}'::jsonb`
(`shot_quality` needs no migration — it lives in the existing `framing` jsonb.)

## Field ownership recap
| Field | Owner | Storage | Migration |
|---|---|---|---|
| `summary` (repurposed to action/change) | LLM | existing column | no |
| `shot_quality` | LLM (enum) | `framing` jsonb | no |
| `screen_text` | LLM (free text) | new column | yes (038) |
| `salience {peak_ms, score}` | **code** | new column | yes (038) |
| two-frame plan / runt guard | **code** | n/a | no |
