# Cuts v3 — LLM-grouped cuts over a deterministic lattice

Supersedes the *detection* half of `cuts_v2.plan.md` / `cuts_v2_boundaries.plan.md`.
Keeps (and builds on) the v2 base layer: the non-overlapping boundary lattice,
anchors, and all L1 signals. Everything here is ADDITIVE and versioned — new
modules, new tables, new endpoints — the current surfaces keep working until v3
replaces them.

---

## North star

Editors pick from cuts. They never touch raw footage. For that to hold:

1. **Boundaries are deterministic.** The LLM never emits a millisecond. Speech
   cuts are spans of WORDS (word timings are ours); video cuts are merges of
   ATOMS (atom edges are ours). Precision is guaranteed by construction.
2. **Meaning is the LLM's.** What belongs together, what is a take vs an
   outlook, what is junk, what a cut is about, how to frame it — one ingest
   pass decides all of it, once, and it is stored forever.
3. **Compute once, read forever.** No LLM call ever happens at browse time,
   dial-drag time, or arrange time. Ingest produces the complete per-cut
   record; every surface reads the DB.
4. **No fallbacks.** There is no parallel rescue path, no "degraded mode".
   Strict schema validation + one structured re-ask on violation; a clip that
   still fails shows as `failed` for re-run. The pipeline is made to work.

Division of labor, one line: **signals find physics, the LLM judges meaning,
code enforces invariants, the brain (arranger) does taste at assembly time.**

---

## Pipeline at a glance

```
L1 (existing + additions)
  └─ transcripts (word-timed, diarized), motion, scenes, audio, NEW: transition_points
Lattice
  └─ speech: WORD lattice (turns/pauses = hints, never constraints)
  └─ video:  ATOMS (shot cuts, camera move/settle, disturbance, transitions)
Ingest (one logical pass, two model calls, cached prefix)
  └─ PASS 1  text-only, ALL clips at once:
       speech grouping (word spans) · candidate take groups (cross-clip) ·
       tentative video groups · junk suspects · project + per-clip summaries
  └─ image selection (deterministic, uses pass-1 output)
  └─ PASS 2  vision, cached prefix + images (sharded only by context budget):
       video grouping confirmed · takes vs outlooks · per-cut: summary,
       framing (3 crops + rotation), look, caption zones, on-camera speaker,
       taste caps, pace envelope
Post-compute (deterministic)
  └─ pace_levels arrays · hero frame JPEGs · assembled cut records → DB
Product surfaces (read-only over DB)
  └─ /cuts API · Cuts view (takes stacked, labels, summaries) · dial (view math)
```

---

## 0) Storage & versioning (first, everything lands here)

New tables (additive; migration `024_cuts_v3.sql`):

- **`ingest_runs`** — one row per project ingest: status
  (`pending|pass1|images|pass2|post|ready|failed`), model ids, token/cost
  accounting, project_summary, error detail on failure.
- **`cut_records`** — one row per final cut:
  `file_id, src_in_ms, src_out_ms, kind (speech|video), word_span, atom_ids,
  label, summary, speaker, on_camera, take_group_id, take_role
  (take|outlook|winner), junk (bool + reason), framing jsonb
  {subject_box, crop_16x9, crop_9x16, crop_1x1, rotation_deg},
  look jsonb {graded, palette, exposure}, caption_zones jsonb,
  pace jsonb {min_ms, natural_ms, max_ms, energy_grade, levels[5],
  natural_sound}, hero_ts_ms, hero_key (stored JPEG), transition_in/out,
  ingest_run_id`.
- **Hero frames**: JPEG per cut extracted at `hero_ts_ms` (region of interest
  ~768px), uploaded to R2 under `heroes/{file_id}/{cut_id}.jpg`. These are the
  thumbnails AND the raw material for any future question — future fields are
  a backfill over stored JPEGs, never a re-watch of video.

Invariants enforced in code at write time: full coverage per file, zero
overlap, speech cuts end only at word edges (snapped into inter-word silence),
video cuts end only at atom edges.

---

## 1) L1 signals — changes

Existing signals stay as-is (motion_dynamics, scene_cuts, audio_features,
transcripts). Additions:

**1a. `transition_points`** (extend `motion_dynamics`; new columns, no new
table): premium natural cut instants.
- *Occlusion wipe*: large near-field blob sweeps the frame — fraction of grid
  with high flow magnitude > `WIPE_AREA_FRAC`, coherence collapse, recovery
  within ~600ms. Classic pass-by transition editors hunt for.
- *Degeneracy*: edge-density/detail-entropy collapse (frame becomes one
  texture — over-zoom, lens blocked). Marks "must cut by here".
- Stored as `[{ts_ms, kind: wipe|degenerate, strength}]`. Cheap: computed in
  the existing flow pass. Params in `motion_params.py`.

**1b. Frame-accurate word snapping.** Verify word timings + silence intervals
are dense enough to place a boundary in the gap between any two words
(they are — Whisper words + `silence_intervals`). No new signal; a helper
`snap_word_edge(file_id, word_idx) -> ms` in the lattice module.

**Not needed for v3**: pitch/f0 (speech granularity is now the LLM's), the
per-clip done/shown classifiers. They stay for other uses but v3 has no
dependency on them.

---

## 2) Lattice — atoms change meaning

`base_cuts.py` evolves into the **lattice builder** (`lattice.py`, new module;
base_cuts stays untouched until cutover):

- **Speech side**: no atoms. The lattice is the word list itself. Diarization
  turns, long pauses, speech edges are computed exactly as today but exported
  as PROMPT HINTS (`"long pause after word 141 (1.8s)"`), not boundaries.
- **Video side**: atoms exactly as base_cuts builds them today — shot cuts,
  camera move/settle, disturbance edges — PLUS `transition_points` as atom
  edges. Atoms only exist over non-speech spans (unchanged rule: never cut
  under speech; a video atom overlapping speech attaches to the speech cut).
- **Atom table** builder: the compact numbered text block for prompts —
  `ATOM 7 [12300–15800] move→settle act=0.7 cam=pan coh=0.9 anchors@13100`
  plus per-atom stats the model reads as text (motion enters the prompt as
  NUMBERS, pixels as stills).

Over-split stays safe (LLM merges); under-split is the only fatal error, so
atom params keep their current over-segmenting bias.

---

## 3) Model layer — Sonnet with caching, model-swappable

New `backend/app/services/llm/` (v3 does not touch the existing brain client):

- **`client.py`** — thin provider-agnostic wrapper: `complete(stage, system,
  blocks, schema) -> dict`. Stages resolve model ids from config:
  `INGEST_PASS1_MODEL=claude-sonnet-*`, `INGEST_PASS2_MODEL=claude-sonnet-*`.
  Swapping models = env change; prompts are model-agnostic; the Anthropic
  specifics (cache_control blocks, image blocks) live only in this file.
- **Caching discipline**: the shared prefix `[system + all transcripts + atom
  tables + pass-1 output]` is marked with `cache_control` breakpoints. Pass 2
  shards run back-to-back within the TTL (5 min), sequenced by the
  orchestrator — cache write once, read at 10% for every shard.
- **Structured output**: every call gets a JSON schema; responses are
  validated with pydantic. On violation: ONE re-ask containing the validation
  errors. A second violation fails the ingest run loudly (`failed` + reason).
  That is enforcement, not fallback.

---

## 4) Ingest PASS 1 — text, all clips at once

One call per project (shard only if transcripts alone exceed ~150k tokens —
that is ~10h of speech, effectively never).

**Input**: full word-timed diarized transcripts of every clip + atom tables +
speech hints + file metadata (names, durations, created_at).

**Output schema** (persisted raw on `ingest_runs` for pass 2 + audit):
- `speech_cuts`: `[{file_id, word_span: [a,b], label, speaker_ids}]` — the
  final speech grouping. THE LLM's grouping is final; no dial, no override.
- `take_candidates`: `[{group_id, members: [{file_id, word_span}]}]` — same/
  near-same transcript spans across ALL clips (cross-clip takes solved here).
- `video_tentative_groups`: `[{file_id, atom_ids[]}]` — signal-homogeneous
  atom runs the model tentatively reads as one moment (bounded by shot cuts +
  composition drift, so visually homogeneous by construction).
- `junk_suspects`, `project_summary`, `clip_summaries`.

**Image plan falls out of pass 1** (deterministic, `image_plan.py`):
- one frame per `speech_cut` (sharpest, lowest-blur, inside the span) — the
  speech image economy;
- +1 frame after any composition-drift point inside a speech cut;
- one frame per `video_tentative_group` at its anchor (impact/audio onset) if
  any, else sharpest near motion valley; +1 per additional anchor;
- every candidate take member always keeps its own frame (eye-to-eye
  comparison is the point);
- budget ~24 frames/clip at ~768px; priority: take members > speech cuts >
  anchored video groups > unanchored > drift extras.

---

## 5) Ingest PASS 2 — vision, cached prefix + images

Sharded ONLY by context budget (≤ ~120k tokens of images per call, whole clips
per shard). Take-group members are co-located in the same shard so takes vs
outlooks is judged by direct visual comparison, not description-matching.

**Input**: cached prefix (everything pass 1 saw + produced) + numbered images
(`IMG 12 = clip 3, 41.2s, speech_cut 7`).

**Output schema — the complete per-cut record** (everything, one call, never
ask twice about the same pixels):
- video grouping confirmed: final `atom_ids[]` per video cut (may split a
  tentative group back at atom edges — never below atoms);
- `take_groups` resolved: `takes` (same words, same setting) vs `outlooks`
  (same words, different setting), winner marked, losers kept but stacked;
  a take boundary is always a HARD split;
- per cut: `label`, `summary` (best guess from image + transcript — guessing
  is fine and marked as such), `on_camera` (does the visible person match the
  diarized speaker), `junk` (+reason), `framing` {subject_box, crop_16x9,
  crop_9x16, crop_1x1, rotation_deg (orientation fix + horizon tilt)},
  `look` {graded|log-flat, palette, exposure flags}, `caption_zones`
  (normalized boxes clear on hero AND drift frame), taste fences
  {`max_tasteful_speed`, `min_tasteful_speed`} and `readability_ms` (how long
  the frame needs to be read), `natural_sound` flag.

---

## 6) Post-compute — deterministic assembly (`post.py`)

- **Pace envelope**: `min_ms` = max(readability_ms, anchor span + pad, move
  completion); `natural_ms` = span; `max_ms` = LLM boredom estimate bounded by
  flatline detection (action_energy static ⇒ it won't get better).
- **`pace_levels[5]`**: fixed product-wide target visual velocities L1..L5;
  per cut `mult[k] = target[k] / intrinsic_velocity` (median flow over span),
  clamped by taste fences and source-fps slow-mo limits, then **saturated**
  (unreachable levels repeat the nearest reachable multiplier — arrays stay
  monotonic; a repeated value IS the "maxed out" signal; never a fake 1.0).
  Speech cuts: `[1,1,1,1,1]`. `energy_grade` from action stats.
- **Hero frames**: extract + upload JPEG per cut; `hero_ts_ms` = anchor >
  subject-sharp > midpoint.
- **Framing motion**: store L1 subject-centroid track reference per cut so
  crops FOLLOW the subject (LLM box anchors semantics at hero frame; centroid
  propagates it).
- Assemble `cut_records`, enforce invariants, mark run `ready`.

Orchestrated by `ingest.py`: `pass1 → image_plan → extract frames → pass2
(shards, back-to-back for cache) → post`. Triggered when a project's clips are
all L1-ready; re-runnable per project (idempotent by `ingest_run_id`).

---

## 7) API

- `POST /api/projects/{id}/ingest` — kick/re-kick the v3 ingest.
- `GET  /api/projects/{id}/cuts-v3` — all cut_records + project summary +
  ingest status. Pure DB read. Zero model calls.
- Existing `/cuts` (v2 base) stays untouched until cutover.

---

## 8) Frontend — Cuts view over cut_records

Keeps the v2 shell (rows per clip, horizontal filmstrip, weld-on-adjacent-
selection, drag to timeline, framing dropdowns, mute). Changes:

- Tiles show `label` + `summary` line + speaker/on-camera chip; boundary-
  reason debug chips remain behind a dev toggle.
- **Take stacking**: winner shown in the row; siblings collapse behind a
  "3 takes" badge that fans out on click. Outlooks show side by side (they are
  different content).
- Junk cuts render collapsed into a thin strip (present, honest, unobtrusive —
  content is never hidden, per the no-filter principle).
- Project summary header above the rows.
- Crop preview: Landscape/Portrait/Square dropdown applies the stored crop
  (+rotation) to tile playback — the editor sees what the export would frame.

---

## 9) Dial — final, reduced role (and better for it)

- **Speech granularity: GONE from the dial.** The LLM's grouping is the
  grouping. Never re-split by a slider.
- **Video granularity (top end only)**: at the highest band the view splits
  grouped video cuts back to their atoms (IDs are stored — pure view math).
- **Tightness (all bands)**: interpolates video cuts from full span toward the
  anchored core (existing anchor-aware inset — payoffs can never be trimmed
  out); speech cuts trim edge dead-air only at the top band, never words.
- Instant, deterministic, zero model calls, nothing ever filtered — the dial
  changes VIEW, never truth.
- Pace levels are NOT the dial: they are an arranger/timeline instrument
  ("play this sequence at level 4"), stored per cut, used at assembly.

---

## 10) Cost (2h raw, Sonnet-class, prices per M: $3 in / $15 out / cache read 10%)

| Piece | Tokens | $ |
|---|---|---|
| Pass 1 (all transcripts + atoms) | ~70k in / ~15k out | ~0.45 |
| Prefix cache re-reads (pass-2 shards) | ~200k @10% | ~0.06 |
| Pass 2 images (~250 after speech economy) | ~130k in | ~0.40 |
| Pass 2 structured output | ~70k out | ~1.05 |
| **Total per 2h project** | | **~$2.00 (range 2–4)** |

Once per ingest; browsing/dial/arranging cost $0 afterward. Cost scales with
image count only; resolution (768→512px) is the emergency lever. Token + $
actuals recorded on `ingest_runs`.

---

## 11) Build order (each step verifiable alone)

| Step | What | Verify |
|---|---|---|
| A | Migration 024 + `llm/client.py` + model config | schema applies; dummy schema round-trip |
| B | L1 `transition_points` | synthetic wipe/degeneracy fixtures + real clips |
| C | `lattice.py` (word lattice + atom table + snapping) | unit tests: coverage, word-edge snapping |
| D | Pass 1 + `image_plan.py` | run on real project; inspect groupings/takes/plan dump |
| E | Frame extraction + Pass 2 + `post.py` | full ingest on one project; validate invariants + records |
| F | API + frontend cuts-v3 view | browse real project; takes stack; crops preview |
| G | Dial view-math (tightness + atom split) | drag = instant, content never disappears |
| H | Cutover: default Cuts view reads v3; v2 path retired | side-by-side eyeball on 3 projects |

Non-negotiables carried from v2: never cut mid-word; anchors are inviolable
(no tightening ever drops an impact); full coverage / zero overlap enforced in
code at every write; no fallback paths anywhere — validation, one re-ask,
loud failure.
