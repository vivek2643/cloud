# Cuts v2 — deterministic non-overlapping partition + per-video row

Goal: replace the overlapping, energy-laddered, channel-tabbed hero-cuts feed
with a **deterministic, non-overlapping partition** of each video into
tag-bearing cuts, presented as **one video = one horizontal row**. No VLM. No
image pass. No junk removal yet. Detection quality is the whole bet — there are
no in-surface split/merge tools — so the segmenter must be solidly good.

Everything here is built as **NEW, additive, versioned** modules/endpoints/
components running IN PARALLEL with the existing pipeline. The old path
(`atoms` + `combine` + `hero_cuts` + channel tabs + energy ladder + take
greying + `/hero-cuts`) stays fully functional behind a flag until v2 is
validated, then is deleted in one removal pass (Phase R).

---

## North star (locked principles)

1. **One unified partition pass owns the timeline.** Never independent per-channel
   passes merged after the fact — that is what creates overlap today. One sorted
   set of boundaries → disjoint intervals → one cut per interval. Overlap is
   impossible by construction.
2. **Simultaneity = tags, not parallel cuts.** A cut carries a SET of tags
   (`said` / `done` / `shown`, up to all three). Talking-while-gesturing is ONE
   cut `[said, done]`, not a said cut overlapping a done cut.
3. **Priority, deterministic.** `said (1.0) > done (0.6) > shown (0.3)`. Higher
   priority claims a contested span; the lower one is demoted to a tag (overlap
   ≥ 60% of the shorter span) or trimmed to its free remainder. Reproducible:
   same L1 signals → same partition, no model, no randomness.
4. **Protect the words, cut at the breaths.** "Never cut said" = word-level veto
   (a boundary never lands mid-word); clean pauses / sentence-ends still segment
   a long turn. This reuses the fused-seam cost field.
5. **Detection over a dial.** Boundaries are DETECTED, not chosen by a
   granularity slider. The only knob is **tightness** (see Phase B3).
6. **Over-split, never under-split.** Over-split self-heals via auto-weld;
   under-split is unrecoverable (no split tool). When ambiguous, cut — but only
   at clean, self-contained sub-units, never fragments.

---

## What we reuse (no new perception needed)

- `l1/motion_dynamics.py` — action energy (subject motion after camera removal),
  camera motion/coherence/stability, sharpness/blur, `action_points` (impacts),
  action/camera cut-cost grids.
- `l1/fused_seams.py` — `compute_fused_field` / `snap_bounds` / protected spans:
  the priority-as-cost substrate we snap boundaries onto.
- `l1/dialogue_segments.py` + diarization (words carry `speaker`) — speech turns
  and speaker-change boundaries.
- `l3/thought_segments.py` — clean linguistic boundaries (repurposed as a
  BOUNDARY map, no longer a granularity zoom).
- `l1/audio_features.py` — silence intervals / RMS for pauses.

## The one genuinely NEW signal

- **Scene / shot detection** — currently absent (L2 assumed "one continuous
  take"). Needed so real multi-shot footage partitions honestly and so `shown`
  boundaries land on composition changes.

---

## Phases

### B1 — L1 scene/shot detection (new stage, additive)
- New module `backend/app/services/l1/scene_cuts.py`:
  - ffmpeg scene score (`select='gt(scene,x)'`) + frame-to-frame histogram drift
    over the proxy → **hard shot-cut boundaries**.
  - within a held shot, **composition-change points** (histogram drift above a
    softer threshold) → candidate `shown` sub-boundaries.
  - Best-effort, CPU-only, bounded by the L1 duration cap (mirror
    `motion_dynamics` failure semantics: a decode failure returns empty, never
    fails L1).
- New migration: `scene_cuts` table (`file_id`, `hop_ms`, `shot_points` jsonb,
  `composition_points` jsonb, `schema_version`). Do NOT overload an existing
  table — additive, easy to drop.
- New idempotent stage `scene_detect` added to a **new STAGES tuple** in
  `l1/pipeline.py` (keep the old tuple constant; introduce `STAGES_V2`), wired
  into the motion track (it shares the proxy decode). Register the frame in the
  L1 snapshot/audit additively.

### B2 — the unified priority partition (new module, the core)
- New `backend/app/services/l3/partition.py` — pure `(clip artifacts) -> List[Cut]`,
  no DB/model call, trivially testable (mirror `atoms.py` purity).
- New dataclass `Cut` (parallel to `HeroCut`, do NOT edit `HeroCut`):
  ```
  Cut:
    file_id: str
    src_in_ms: int
    src_out_ms: int          # contiguous; INVARIANT: no two cuts overlap in a file
    tags: List[str]          # subset of {said, done, shown}, >=1, priority-ordered
    primary: str             # the highest-priority tag (drives tightness + label)
    label: str               # transcript text (said) | type placeholder (video)
    speaker: Optional[str]
    peak_ms: int             # representative frame instant (for the thumbnail)
    keep_spans: Optional[List[(in,out)]]  # set by tightness later; None = contiguous
    # deferred/empty for now: people, framing, subject, summary, quality
  ```
- Algorithm (deterministic priority-claim):
  1. **Candidate spans per channel** (energy-independent):
     - `said`  = speech turns from `dialogue_segments` + diarization, split at
       speaker-change and clean pauses; NEVER split mid-word.
     - `done`  = action beats from `motion_dynamics` (energy rise→peak→fall,
       `action_points`), bounded by calm.
     - `shown` = held, stable, in-focus stretches, bounded by scene / composition
       change (B1) + low action energy.
  2. **Fused cost field** (`fused_seams.compute_fused_field`): protection =
     max over channels of (priority weight × in-span), attractors = impacts /
     beats / pauses. Said word-spans are hard protected spans.
  3. **Claim in priority order** onto ONE timeline:
     ```
     for ch in [said, done, shown]:
       for span in ch.candidates:
         free = span minus already-claimed time
         empty     -> demote to a TAG on the covering cut
         ~full     -> new cut; snap edges via the field
         partial   -> trim to free; usable -> cut, else -> tag
     ```
     Overlap→tag threshold: absorb as a tag when overlap ≥ 60% of the shorter
     span; else keep the trimmed free remainder as its own cut.
  4. **Merge** adjacent same-primary cuts only when truly continuous; speaker
     change / scene cut always break. Bias toward over-split at clean sub-units.
  5. **Snap** every boundary to the nearest low-cost seam, never crossing a
     said word.
- New `CUTS_VERSION = 1` constant gating the v2 precompute cache.
- Assert the non-overlap invariant in code + tests (`scripts/test_partition.py`).

### B3 — tightness only (reuse energy, drop granularity)
- New `cut_tightness` helper (in `partition.py` or a sibling): applies ONLY the
  tightness axis of `energy_to_params` — it does NOT re-scope the unit.
  - `said` primary → breath / dead-air excision into `keep_spans` (reuse the
    existing `_breath_keep_spans` logic; dead-air floor at every level).
  - `done` / `shown` primary → peak inset toward `peak_ms` (reuse the
    proportional-handle logic).
- v1 ships with a fixed default tightness (Balanced); the slider is optional and
  can be wired later. Granularity params (`speech_unit`, `fuse_gap_ms`) are
  ignored by v2 — boundaries are fixed by detection.

### B4 — new API + precompute (parallel endpoint)
- New router handlers `GET /api/files/{id}/cuts` and `POST /api/files/cuts`
  (parallel to `/hero-cuts`), returning `{ cuts, tightness, ready }` grouped by
  file, each file's cuts sorted by `src_in_ms`. Do NOT touch the `/hero-cuts`
  handlers.
- New precompute mirroring `hero_store` (or compute-on-read first, precompute
  later) keyed by `CUTS_VERSION`.
- **Peak thumbnails without new infra (v1):** the frontend `<video>` seeks to
  `peak_ms` for the still (same mechanism the current hover-preview uses). A
  server-side thumbnail/sprite endpoint is a later optimization, not v1.

---

## Frontend

### F1 — new Cuts view (parallel component)
- New `frontend/src/components/cuts-view.tsx` + `Cut` type + `getCuts()` in
  `lib/api.ts`. Leave `hero-cuts-view.tsx` / `getHeroCutsFeed` untouched; a
  feature flag (or a settings toggle) selects which view renders where the Cuts
  tab lives today.
- **One video = one horizontal row** (`overflow-x-auto`); multiple videos stack
  vertically. Row header: filename + duration + select-all/clear.
- **Uniform-width cards** in `src_in_ms` order (chosen: uniform, easiest to
  scan). Card: peak thumbnail · duration · **tag badge(s)** (multiple, colored,
  reuse `CHANNEL_STYLE`) · label snippet · speaker chip (said) · hover preview ·
  selected state.
- **Dropdown filter** at top (All / Said / Done / Shown), default **All**;
  multi-tag **"includes"** semantics — a `[said, done]` cut appears under both
  Said and Done. Replaces the tab row.
- Cards flush; contiguous filmstrip of the whole video (no junk removal yet, so
  no gaps — pause spans render as low-priority cards or a thin spacer; decide in
  build, default: contiguous).

### F2 — selection + auto-join
- Click to select cuts. **Adjacent selected cards visually weld** (remove the
  divider between them) to show they will play seamlessly.
- Feed the selection into the existing edit/timeline flow (`EditButton`); on the
  timeline, adjacent source-contiguous picks join via `keep_spans`. Keep this
  minimal for v1 (visual weld + selection list); deep timeline welding can reuse
  the existing compile-time weld.
- No split/merge tools in the cuts surface (by design).

---

## Detection robustness (make it solidly good)

Because there is no in-surface rescue, invest here:
- **Centralize thresholds** in a new `l1/cut_grid_params`-style module for v2
  (scene score, histogram drift, action peak/floor, pause boundary, overlap→tag
  fraction, min sub-unit length). One place to tune.
- **Over-split + weld** is the safety net: tune toward more, cleaner cuts.
- **Word-level said protection** via the fused field guarantees no mid-word cut.
- **Validation harness** (de-risk before building the full surface): new
  `scripts/viz_cuts.py` that renders the partition over a real clip (boundaries,
  tags, snapped seams) so an editor can eyeball whether cuts land where they
  would. Extend the existing `scripts/viz_video_signal.py` approach.
- **Eyeball metric**: on a handful of real clips, compare boundaries to where an
  editor would cut; iterate thresholds. This is the go/no-go for the whole bet.

---

## Versioning & removal

- Additive everywhere: `scene_cuts.py`, `partition.py`, `CUTS_VERSION`,
  `scene_cuts` table, `STAGES_V2`, `/api/files/cuts`, `cuts-view.tsx`,
  `getCuts()`. Nothing existing is edited in place beyond registering the new
  L1 stage.
- Flag selects v1 (hero-cuts) vs v2 (cuts) at the view + endpoint level so both
  run in parallel during validation.

### Phase R — removal (only after v2 is validated on real footage)
- Delete: `l3/atoms.py` VLM path + `l3/combine.py` + the speech/video overlap
  builders in `l3/hero_cuts.py`, `/hero-cuts` handlers, `hero-cuts-view.tsx`,
  channel tabs, the energy LADDER (keep only tightness), take-stacking in the UI
  and its greyed-loser rendering.
- Retire the granularity concept from `l3/energy.py` (keep tightness).

---

## Out of scope (explicitly deferred)

- **Image pass / perception replacement** (peak frame → Claude): the whole
  still-annotation call, on/off-speaker reconciliation, colour/reframe/caption
  outputs. `done`/`shown` classification is PROVISIONAL (motion/scene only)
  until this lands.
- **Junk removal**: dropping pause / filler / false-start / off-camera spans.
  For now the row is a contiguous filmstrip; junk stays as low-priority cuts.
- **Best-take / take comparison / stacking.** v2 shows all cuts; no dedup.
- **VLM removal wiring** (disabling the L2 task) — that rides with the image
  pass, not this plan.

## Risk / tension to watch

- **Segmentation quality is the whole bet.** Under-split is unrecoverable in the
  surface; mitigations are over-split+weld and the eventual NLE handoff. Validate
  on real clips (viz harness) BEFORE building the full frontend.
- **Video cuts are provisional** without the image pass — strong for `said`,
  rough for `done`/`shown`. Acceptable for this slice; improves later.
- **Scene-cut mid-speech** hard-splits a turn into two `[said,…]` cuts; correct
  but fragmenting — auto-weld recovers it.
