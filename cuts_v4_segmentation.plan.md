# Cuts V4 — Signal-Driven "Extract the Usable, Discard the Scrap" Segmentation

## 0. What this is and why

Today, non-speech (video) cuts are created by: `lattice.build_atoms` (carves atom
boundaries) → `pass1.run_pass1` (LLM groups atoms into `VideoTentativeGroup`s + flags
junk) → `post.assemble_cut_records` (spans, salience, pace, quality). The LLM owns
*grouping*; boundaries are code-derived from a narrow vocabulary (shot cuts, wipe/
degenerate transitions, an Otsu energy-regime split). Rich signals we already compute
(`action_points`, `rms_db`, `camera_*`, composition drift) mostly **annotate** cuts —
they don't **create** them. Result: uniform clips get kept whole, mid-action openings
get absorbed, energy barely moves duration, and good single-sentence-equivalent moments
get lost.

**V4 replaces the video-segmentation half with a deterministic, signal-driven
extractor** built on one principle:

> **A raw clip is mostly scrap. Find the small usable part(s) and discard the rest.**
> Default is to trim hard to the usable core — never "keep the whole clip."

Speech is untouched (Pass 1 still owns speech grouping + junk). V4 only changes how the
**non-speech remainder** becomes cuts, plus how the **energy ladder** trims every cut.

**Two hard rules threaded throughout:**
- **Salience = contrast/novelty** (how much a moment stands out from its local
  surroundings), measured on whichever channels are present (motion and/or audio),
  discounting periodic/constant energy. NOT absolute level. NOT requiring audio+motion
  consensus — consensus only *raises confidence*.
- **VLM decides the *shape* (semantic), signals decide the *point* (timing).** The VLM
  only ever sees ~2 frames, so it can name "what kind of moment" but never *when*.
  Location is always deterministic.

V4 ships behind a **feature flag**, emits the **exact same `cut_record` contract** as
V3 (so `image_plan`, `pass2`, `post` storage, and the entire frontend need no changes),
and runs side-by-side with V3 until proven, then V3 is retired.

---

## 1. Architecture — where V4 sits

Current `ingest.run_ingest` order (`backend/app/services/l3/ingest.py`):

```
L1 signals (motion_by_file, audio_by_file, scene, silences)
  → lattice (atoms)              # per file
  → pass1.run_pass1              # LLM: speech cuts + video groups + junk
  → enforce_lattice_partition
  → image_plan.build_image_plan  # frames per unit
  → pass2.run_pass2_batch        # VLM labels each cut
  → identity resolution
  → post.assemble_cut_records    # spans, salience, pace, quality, continuity
  → store cut_records → frontend
```

V4 inserts a **deterministic video segmenter after Pass 1 (which keeps speech + junk)
and before `image_plan`**, and it **supersedes Pass 1's video groups**:

```
pass1.run_pass1                  # KEEP speech_cuts + junk_suspects; IGNORE video groups when V4 on
  → v4_segment.segment_video(...)   # NEW: non-speech remainder → video cut spans + salience
  → image_plan.build_image_plan     # (small change: frame-for-shape)
  → pass2.run_pass2_batch           # (+ "shape" field)
  → identity resolution
  → post.assemble_cut_records       # (branch: V4 salience + shape-aware ladder)
```

**Flag:** add `settings.cuts_segmenter` (`"v3" | "v4"`, default `"v3"`). Branch in
`run_ingest`. When `"v4"`:
- Pass 1's `VideoTentativeGroup`s are dropped before `enforce_lattice_partition`
  (speech partition still runs on speech cuts only).
- `v4_segment.segment_video` produces the video cuts.
- Downstream ladder + salience use the V4 branch.

The flag lives on the **ingest run**, so the frontend can show either by pointing at a
V3 or a V4 `ingest_run_id` — same window, no UI change.

---

## 2. The output contract V4 MUST satisfy

V4's segmenter produces, per video cut, exactly what `post.assemble_cut_records`
currently derives per video group, so the rest of the pipeline is untouched. Concretely
each V4 video cut must yield a `pass2.CutJudgment`-compatible unit and a
`post.CutRecord`. The fields that matter for V4:

- `file_id`, `src_in_ms`, `src_out_ms`, `kind="video"`, `atom_ids` (the atoms the cut
  covers — keep populating for continuity/image_plan compatibility), `channel`
  (`"done"|"shown"` decided by pass2 as today).
- `hero_ts_ms` (best still — unchanged, `post.pick_hero_ts_ms`).
- `salience` = `{peak_ms, score}` **plus new** `{kind: "point"|"span"|"none", span_ms:
  [in,out] | null}` (see §4).
- `pace` = `PaceEnvelope` (min_ms/natural_ms/max_ms/levels/remove_spans/natural_sound) —
  min_ms becomes shape/content-aware (see §6).
- `shape` (new, from pass2 §5) carried onto the record for the ladder.

**No new DB columns are required** if `shape` and the extra salience keys are folded
into the existing `salience` JSONB blob and/or `pace` blob. (Simplest: put `shape` inside
`salience` as `salience.shape`, and `salience.kind`/`salience.span_ms` alongside
`peak_ms`/`score`. Then `cut_records.salience` already persists them — no migration.)

---

## 3. Component A — the V4 video segmenter

New module: `backend/app/services/l3/v4_segment.py`.

**Signature (pure, deterministic, no model call):**

```
def segment_video(
    *,
    file_id: str,
    duration_ms: int,
    speech_spans: List[Tuple[int,int]],        # from pass1 speech cuts (to subtract)
    motion: Dict[str, Any],                    # motion_by_file[file_id]
    audio: Dict[str, Any],                     # audio_by_file[file_id]
    scene: Dict[str, Any],                     # scene_by_file[file_id] (shot/composition/transition)
    lattice: Lattice,                          # for atom_ids mapping + clip edges
) -> List[VideoCut]                            # VideoCut = {src_in_ms, src_out_ms, atom_ids, salience}
```

### Step 0 — Define the working units (single-shot, non-speech)
- Split the file's timeline at shot boundaries + transitions (`scene.shot_points`,
  `motion.transition_points`) into single-shot segments. *(These are the mechanical
  pre-split — the working "clip" is one shot. We do NOT treat them as editorial cut
  choices; they just bound the unit.)*
- Subtract `speech_spans` from each shot segment. What remains = the **non-speech
  working spans**. Everything below runs **per working span**.

### Step 1 — Start from "nothing kept"
The default answer for a working span is **not** "keep it all." We will earn the usable
part(s) from evidence, and everything else is scrap (dropped, or handed to pass2 as a
low-priority candidate that it may junk).

### Step 2 — Build the contrast/novelty curve (the trust foundation)
Per working span, at the motion `hop_ms` grid, compute a **novelty curve** — how much
each instant stands out from its *local* neighborhood — NOT absolute level:

- **Motion novelty:** `action_energy` minus a rolling local baseline (e.g. rolling
  median over a window ≈ 1–2 s), clamped ≥ 0. A sustained-high stretch (blinking light,
  waves, timelapse) has ≈0 novelty; a burst out of calm spikes.
- **Audio novelty:** same treatment on `rms_db` (rolling-baseline-subtracted). Present
  for all clips (prosody envelope), unlike `onsets_ms` which only exists for musical
  clips.
- **Discrete event bumps:** `action_points` (`motion.action_points`) and, when
  `audio.is_musical`, `onsets_ms` — a bump at each, but only counted if it also shows
  novelty (a lone periodic blip doesn't survive).
- **Periodicity discount:** if the span's signal is highly self-similar/periodic
  (autocorrelation peak, or evenly-spaced repeated `action_points`), scale novelty down
  — a blink is all "change" but no *event*.

`novelty[i] = w_m·motion_novelty + w_a·audio_novelty + event_bumps`, then
periodicity-discounted. Confidence rises when motion and audio agree at the same instant,
but **either alone can produce a point** (no hard consensus requirement).

Reuse the existing clip-relative normalization (`post._series_lohi` / `_norm_in_clip`)
so there are no absolute constants.

### Step 3 — Find the usable anchor(s), in preference order
Per working span, pick anchors in this order (stop when you have a confident one; allow
more than one only if clearly separated and each strong):

1. **Transition point inside the span** (`transition_points`) — premium natural seam.
2. **A novelty peak** (motion and/or audio) that clears a prominence threshold relative
   to the span's own curve — this is the real "event." Audio+motion agreement → higher
   confidence, but not required.
3. **A camera-move payload** — a stretch of sustained `camera_motion` with high
   `camera_coherence` (deliberate move, not shake). The anchor here is a **span** (the
   move's dynamic core), not a point (see §4 "span").
4. **Fallback (no contrast anywhere):** the **representative window** — the steadiest
   (`camera_stability` high, `blur` low), most-central slice. Salience kind = `"none"`.

### Step 4 — Choose edges so both ends make sense
For each anchor, carve a **tight** cut; both edges must be sensible:

- **Point anchor (event):** `in = peak − run_up`, `out = peak + follow_through`, where
  `run_up`/`follow_through` are content-scaled and asymmetric (see §6). The natural
  (energy=0) span still includes a comfortable build + settle.
- **Span anchor (camera move):** `in = move start` (or where it steadies),
  `out = move settle` (`camera_dx/dy/zoom` returns toward 0). Never end mid-move.
- **No anchor (representative):** a modest window centered on the steadiest instant.

Snap edges to clean instants using the **camera-quality gate**: never place an edge
mid-smooth-move; prefer a whip/bump/blur (`camera_stability` low or `blur` high) or a
settle. Never open/close on a blur frame unless intended.

### Step 5 — Consolidate (stop micro-cuts AND stop over-keeping)
- **Minimum length:** if two anchors' cuts are closer than a floor, keep the stronger,
  merge the sliver. Floor is content-aware (see §6), not a hardcoded ms.
- **Merge overlaps:** enforce zero-overlap (post already validates via
  `_validate_no_overlap`).
- **Scrap is dropped:** anything outside the chosen cuts is not tiled — gaps are legal
  (`post` already allows coverage gaps).
- If nothing survived → one **representative-window** cut (never the whole span by
  default).

### Step 6 — Emit
Each surviving cut → `{src_in_ms, src_out_ms, atom_ids (covered), salience:{peak_ms,
score, kind, span_ms}}`. Map spans back to `atom_ids` via the lattice (atoms are still
carved; V4 just chooses spans over them) so `image_plan`/continuity keep working.

Hand these to `image_plan` + `pass2` in place of Pass 1's video groups.

---

## 4. Component B — salience as contrast/novelty (shape-agnostic point/span/none)

Replace/extend `post._salience` (currently absolute action_energy + loudness + anchor
bump, argmax). V4 wants:

- `kind`: `"point"` (event), `"span"` (camera move), `"none"` (uniform).
- `peak_ms`: for `"point"`, the novelty argmax; for `"span"`, the move-core center; for
  `"none"`, `hero_ts_ms`.
- `span_ms`: for `"span"`, the `[in,out]` of the move core; else `null`.
- `score`: peak prominence relative to the cut's own novelty range (0..1).

Implementation: the V4 segmenter already computes the novelty curve and the anchor type,
so it should **emit salience directly** (don't recompute in post). In the V4 branch,
`assemble_cut_records` reads salience from the segmenter's `VideoCut` instead of calling
`_salience`. Keep `_salience` for the V3 branch.

**Key change from today:** salience is contrast-based (novelty over local baseline), so a
blinking light / timelapse yields `kind="none"` (correct), and a burst-out-of-calm yields
a strong `"point"`.

---

## 5. Component C — the VLM shape field (semantic only, no timing)

Add `shape` to the pass-2 output.

- `pass2.CutJudgment` (`backend/app/services/l3/pass2.py` ~line 195): add
  ```python
  shape: str = "center"   # one of: "before" | "after" | "both" | "center" | "none"
  ```
  Semantics (from ~2 frames + label, NO timestamp):
  - `"before"` — everything **before** a moment matters; end **on** it (build-to-impact).
  - `"after"` — everything **after** a moment matters; keep the tail (reveal/payoff).
  - `"both"` — the moment and both sides matter; sit around it.
  - `"center"` — safe default; no strong asymmetry.
  - `"none"` — nothing to trim to (screen recording / uniform).
- `pass2.Pass2Cut` (~line 274): carry `shape` through.
- Prompt (`pass2.gemini_system_prompt` / the cut-judgment instructions): add a short,
  generic line asking the model to classify the moment's shape from the frames — framed
  as "which side of the key moment carries the value," explicitly telling it NOT to
  guess timestamps.
- `post.assemble_cut_records`: stamp `shape` into `salience.shape` (or a dedicated field)
  on the `CutRecord`.

**Arbitration (VLM vs signals):** the VLM `shape` is a *coarse prior*. If it says
`"none"` but the segmenter found a strong, high-confidence `"point"` (audio-confirmed),
the ladder still treats it as a point (a real event is real). Rule: **VLM proposes the
shape; a high-confidence signal can override `"none"`→point.** Default when VLM missing
or low-info: `"center"`.

---

## 6. Component D — the shape-aware asymmetric energy ladder

Change the **video rung** synthesis (`cutrecord_map._video_rung`,
`backend/app/services/l3/cutrecord_map.py` ~line 99) and `post.compute_pace_envelope`'s
`min_ms`. Branch behind the V4 flag (V3 keeps `hero_ts_ms`-centered symmetric shrink).

Today `_video_rung` shrinks a window symmetrically around `hero_ts_ms` toward
`pace.min_ms`. V4 instead collapses toward the **salience anchor**, asymmetrically, per
**shape**:

Let `natural = out − in`, `target = round(natural − energy·(natural − min_ms))`
(same energy→duration mapping). Then place the `target`-length window as:

- **`shape="before"` (point):** anchor the **out edge** near `peak + follow_through_floor`
  and let the **in edge** absorb the shrink. As energy↑, the window ends on the impact
  with a shrinking lead-in. (Out collapses slow — it's already near peak; in collapses
  fast.)
- **`shape="after"` (point):** anchor the **in edge** near `peak − lead_floor`; the **out
  edge** keeps the tail/settle and the head absorbs the shrink.
- **`shape="both"` / `"center"`:** symmetric around `peak_ms` (this is ≈ today's
  behavior but anchored on the salience peak, not `hero_ts_ms`).
- **`salience.kind="span"` (camera move):** trim the **head** (drop the slow ramp-in),
  keep the **settle** (`span_ms[1]`); as energy↑, `in` moves toward `span_ms[0]` (the
  dynamic core), `out` stays at the settle. Never end mid-move.
- **`salience.kind="none"`:** at energy=0 keep the whole cut (long-form/"done"); as
  energy↑ collapse to the **representative window** around `hero_ts_ms`. (User-approved:
  a screen recording is available whole under long-form and tightened under punchy.)

`follow_through_floor` / `lead_floor` are small (so a punchy cut never *clips* the impact
— always land a hair after the peak, respecting that `peak_ms` is coarse at ~100ms audio
hops).

**`min_ms` becomes content/shape-aware** (`post.compute_pace_envelope`): instead of
`max(readability_ms, _anchor_span_ms(anchors))`, scale the floor by the cut's **event
density / novelty** — a sparse/monotonous clip collapses hard at high energy (small
`min_ms`); a dense clip holds more so real events aren't clipped. Density = the
segmenter's novelty stats (count/prominence of novel instants) — pass it through on the
`VideoCut`. This directly answers "monotonous 15s → short at high energy."

Frontend parity: `cuts-v3-view.tsx`'s `tightenedSpan` mirrors `_video_rung`. If the
frontend must preview V4 ladders identically, port the same shape-aware math there (read
`salience.kind`/`shape` off the cut). If the frontend only ever renders the
backend-synthesized `ladder` array, no change needed — confirm which during impl.

---

## 7. Component E — frame extraction for shape (image_plan)

`image_plan.build_image_plan` already gives every unit ≥1 frame and, for long-enough
units, an `"early"` + `"late"` pair (`REASON_SECOND_MOMENT`). This is *usually enough*
for the VLM to judge shape (start vs end look).

Change needed: for a V4 video cut with a **`"point"` salience**, bias the two frames to
**straddle the peak** — one shortly before `peak_ms`, one shortly after — so the VLM can
tell before/after/both. Add a `REASON_SHAPE_STRADDLE` (or reuse early/late by seeding the
"late" candidate at/after `peak_ms`). Keep the `FRAME_BUDGET_PER_CLIP=40` and the runt
guard. This is a targeted tweak to which two instants get picked, not new budget.

No change for `"span"`/`"none"` cuts beyond today's early/late.

---

## 8. Orchestration wiring (`ingest.run_ingest`)

1. Add `settings.cuts_segmenter` (default `"v3"`).
2. After `pass1_output` is built and speech partition enforced, when `v4`:
   - Strip `VideoTentativeGroup`s from `pass1_output` (or skip feeding them to
     `image_plan`/`pass2`).
   - Build `speech_spans` per file from `pass1_output.speech_cuts`.
   - Call `v4_segment.segment_video(...)` per file → video cuts.
   - Convert V4 video cuts into the same intermediate the image_plan/pass2 path expects
     (synthesize `source_ref="video_group[i]"`, `atom_ids`, span).
3. `image_plan` + `pass2` run unchanged (pass2 now also returns `shape`).
4. `post.assemble_cut_records`: in the V4 branch, use segmenter-supplied salience +
   shape-aware ladder + content-aware `min_ms`.
5. Store — identical `cut_records` shape.

Speech path, identity resolution, take/outlook grouping, continuity, audio coupling — all
unchanged.

---

## 9. Frontend

- No schema change (salience/shape ride in existing JSONB). The current cuts-v3 view
  renders V4 runs as-is (it reads `cut_records`).
- If ladder preview is computed client-side (`cuts-v3-view.tsx tightenedSpan`), port the
  shape-aware math (§6) behind the same notion, reading `salience.kind`/`shape`.
- "Remove V3" is a later cleanup once V4 is validated: delete the V3 branch in
  `run_ingest`, `_salience`, the symmetric `_video_rung`, and the flag.

---

## 10. Testing

- **`v4_segment` unit tests** (new `backend/scripts/test_v4_segment.py`): synthetic
  motion/audio arrays for each table case —
  - burst-out-of-calm → one tight `"point"` cut ending after peak;
  - blinking/periodic high energy → `kind="none"`, representative window (NOT split);
  - smooth pan → `"span"` cut, edges at move start/settle;
  - uniform/static → single representative cut, not whole-span;
  - two separated bursts → two cuts; two near bursts → consolidated to one.
- **Salience tests:** contrast-based peak beats absolute-level peak on a
  ramp-then-constant-high fixture.
- **Ladder tests** (`cutrecord_map`): before/after/both/span/none produce the expected
  asymmetric windows; punchy never clips the peak; monotonous clip collapses smaller than
  a dense one at the same energy.
- **Pass2 schema test:** `shape` round-trips; default `"center"`; arbitration overrides
  `"none"`→point on high-confidence signal.
- **Golden run:** ingest a known project under `v3` and `v4`, diff `cut_records`; confirm
  speech cuts identical, video cuts tighter and event-anchored.

---

## 11. Rollout

1. Land V4 behind flag (`v3` default). No behavior change for anyone.
2. Ingest test projects with `v4`, compare in the same frontend window.
3. Iterate on thresholds (novelty prominence, min-length floor, follow-through floors)
   using the table cases as the acceptance set.
4. Flip default to `v4`.
5. Remove V3 branches + flag.

---

## 12. Open questions / risks (call out during impl)

- **Periodicity detection** (rejecting blinks/waves) is the trickiest signal-processing
  bit; start simple (rolling-baseline novelty + evenly-spaced-`action_points` discount)
  and harden with the test fixtures.
- **VLM shape reliability from 2 frames** is a coarse prior — the `"center"` default and
  the arbitration rule keep a wrong shape from wrecking a cut (it only tilts asymmetry).
- **Camera-move core detection** (`"span"`) depends on `camera_coherence`/`stability`
  being trustworthy on real footage — validate on drone/pan clips.
- **Frontend ladder parity** — confirm whether previews use the backend `ladder` array or
  recompute; only port math if the latter.
- **`atom_ids` mapping** — V4 chooses spans; ensure covered-atom mapping stays correct so
  continuity/image_plan don't break.
```
