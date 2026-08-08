# Restore reframe geometry in the new vcut pipeline

## Problem

The new `vcut` pipeline persists **no per-cut framing geometry** — `vcut/store.py:102`
writes `framing={}`. The old cuts-v3 Pass-2 produced `subject_box` +
`crop_16x9/9x16/1x1` + `rotation_deg` (an image judgment), and the frontend was
built against exactly those fields:

```398:402:frontend/src/components/cuts-view.tsx
function cropForAspect(cut: CutRecord, aspect: Aspect): [number, number, number, number] | null | undefined {
  if (aspect === "portrait") return cut.framing?.crop_9x16;
  if (aspect === "square") return cut.framing?.crop_1x1;
  return cut.framing?.crop_16x9;
}
```

With `framing` empty, the cuts-browser aspect toggle falls back to a blind
centered CSS `cover` crop (`cuts-view.tsx:1440`) and never uprights sideways
footage (`rotation_deg` unused). This is a **quality regression vs the old
pipeline**, and it's the only thing blocking a production push of the new cuts.

The fix restores that geometry **through the Pass-2 VLM call the new pipeline
already makes** (`vcut/pass2.py` sends cached video to flash-lite per file) — so
the marginal cost is ~zero. Once `framing` is populated, the frontend works
unchanged and reframe is at **parity** with the old pipeline (no degradation).

## Target shape (already defined — match it exactly)

The frontend `Framing` interface (`frontend/src/lib/api.ts:156-166`) and the old
`l3/pass2.Framing` (`backend/app/services/l3/pass2.py:93-104`) agree. Each new
video `CutRecord.framing` must carry:

```
subject_box:  [x, y, w, h] | null    # normalized, source (uprighted) frame
crop_16x9:    [x, y, w, h] | null    # normalized target-aspect crop rects
crop_9x16:    [x, y, w, h] | null
crop_1x1:     [x, y, w, h] | null
rotation_deg: number                 # correction-to-upright (0/±90/180)
shot_size:    string (optional; already answered in specifics)
```

## Design principles

1. **The VLM emits the ANCHOR, code solves the crops.** Pass-2 returns
   `subject_box`; a deterministic solver computes `crop_16x9/9x16/1x1`. Matches
   the locked framing philosophy ("the model never picks a crop rectangle by
   hand" — `grade/steer.py:5`, `framing_transforms.plan.md`).
2. **Rotation-to-upright is metadata, not a model call.** Read the source
   rotation side-data at prep/ingest and bake `rotation_deg`. Deterministic and
   the most visible fix (sideways footage delivered sideways).
3. **The anchor lives on the moment flag → energy-invariant.** Store
   `subject_box` on `MomentFlag` and let it compose per-cut on the SAME rails as
   `specifics` (`resolve._composed_specifics`), so the energy dial can't wipe it
   and a merged loose cut still gets a sensible representative anchor.
4. **Zero frontend change.** Write `framing` in the exact shape `cuts-view.tsx`
   already reads.
5. **Near-zero cost.** `subject_box` is one extra output field on a Pass-2 call
   that already happens against cached video.

## Implementation

### 1. Pass 2: emit `subject_box` per moment (`vcut/pass2.py`)

- Add `subject_box: Optional[Tuple[float, float, float, float]] = None` to
  `_MomentAnswerOut` (`pass2.py:60-91`).
- **Always request it**, independent of the per-moment question-id selection —
  reframe applies to every video cut, so `subject_box` is NOT a bank question.
  Mention it in `_task_text`'s preamble (`pass2.py:122`), e.g. "For every moment
  also return `subject_box` — the normalized [x,y,w,h] of the main subject in
  frame (or null if none)."
- Capture it onto the flag regardless of `question_ids`. In
  `_write_answers_onto_flags` (`pass2.py:149`) / `_specifics_from_answer`
  (`pass2.py:134`), attach the box to the flag. Prefer a **dedicated
  `MomentFlag.subject_box` field** (Step 3) over stuffing it into `specifics`,
  so geometry stays separate from descriptive scene_specifics. (If you'd rather
  minimize plumbing, you may instead put it under `specifics["subject_box"]` and
  pop it in Step 4 — but the dedicated field is cleaner.)
- Frames-mode Pass-2 (`run_enrich`, per-moment hero-stills) should emit it too;
  a still is enough to locate a subject. Fail-open: no box → `null`.

### 2. Source dims + rotation from metadata (per file)

The crop solver needs the source **aspect** (w/h) and the **rotation**. Both are
deterministic container facts.

- At the prep stage that already runs ffmpeg on each file
  (`orchestrate._prepare_video_inputs` / `cut_non_speech_subclip`), or from the
  existing L1 file probe, read `width`, `height`, and the rotation side-data
  (ffprobe `stream_tags=rotate` or the display-matrix rotation).
- Thread `(src_w, src_h, rotation_deg)` per `file_id` down to
  `store.build_cut_records` (alongside the `seam` dict, which is already keyed by
  file). `rotation_deg` = the correction to bring the frame upright.
- If metadata is unavailable, default `rotation_deg=0` and treat dims as the
  proxy's — never fail the ingest for missing rotation.

### 3. Crop solver (new small pure function)

Add `solve_crops(subject_box, src_w, src_h) -> {crop_16x9, crop_9x16, crop_1x1}`
(e.g. in a new `vcut/reframe.py`, or beside `l3/framing.py` if isolation allows;
`vcut` has an isolation contract — keep it self-contained under `vcut/`).

- Work in the **uprighted** frame (apply rotation's w/h swap first: if
  `rotation_deg` is ±90, swap src_w/src_h before solving).
- For each target aspect `A`: compute the largest axis-aligned rect of aspect `A`
  that fits inside the source, then position it to keep `subject_box`'s center in
  view, clamped to the source edges. Output normalized `[x,y,w,h]` in source
  space.
- `subject_box is None` → return a **centered** crop of each aspect (still a real
  improvement over today: centered + uprighted, matching old-pipeline behavior
  when the model gave no box).
- Also thread `MomentFlag.subject_box` through `resolve._composed_specifics` (or
  a parallel compose) so a **single-flag** cut passes its box through and a
  **merged** cut carries the **representative flag's** box (the resolver already
  picks a representative via `_representative_peak`).

### 4. `store.build_cut_records`: write `framing` (`vcut/store.py:77-106`)

- Replace `framing={}` (line 102) with a computed `Framing`:
  - `rotation_deg` from Step 2 (per `file_id`).
  - `subject_box` = the resolved cut's representative box (from Step 3 compose).
  - `crop_16x9/9x16/1x1` from `solve_crops(subject_box, src_w, src_h)`.
  - Optionally surface `shot_size` here too (it's already in `cut.specifics`).
- If you routed the box through `specifics`, **strip `subject_box` out of the
  scene_specifics** written by `insert_video_cuts` (`store.py:127-129`) so
  scene_specifics stays purely descriptive (the brain map ignores unknown keys,
  but keep it clean).
- Because `build_cut_records` is called by BOTH the initial ingest and the
  energy re-resolve (`insert_video_cuts`, shared by `run_vcut_ingest` and
  `routers/projects.py`), framing is **re-derived at every resolve** from the
  flag's box → automatically energy-invariant, same guarantee as specifics.

### 5. Frontend — no change

Verify only: `cuts-view.tsx` `cropForAspect`/`cropStyle` (lines 381-402) and the
`Framing` interface (`api.ts:156-166`) already consume `crop_*` + `rotation_deg`.
Nothing to edit.

### 6. Backfill existing projects

Populated only on ingest, so existing cuts stay `framing={}` until re-run. New
ingests get it automatically. To clear the regression in production for existing
projects, **re-ingest (or re-run Pass-2 enrich) once** — folds into the same
batch you already run. Flag this as a required step, not just a code change.

## Cost

Negligible. No new call — `subject_box` rides the existing per-file Pass-2 video
call; rotation/dims are a local ffprobe read. One extra output field per moment.

## Testing

- **`solve_crops`** unit tests: subject centered; subject near each edge (crop
  clamps, subject stays visible); landscape/portrait/square targets; ±90
  rotation swaps dims; `subject_box=None` → centered crop.
- **Pass-2 schema**: `_MomentAnswerOut` parses `subject_box`; it's captured onto
  the flag even when the moment selected no bank questions.
- **Compose**: single-flag cut passes its box through; merged cut carries the
  representative flag's box (energy 0 and 1 give the same box for a given
  moment).
- **`build_cut_records`**: writes a fully-populated `Framing` (all three crops +
  rotation); scene_specifics contains no `subject_box`.
- **Energy-invariance**: framing survives a dial re-resolve unchanged.
- All fixtures in-memory — zero real spend (mock the model, hand `solve_crops`
  synthetic boxes/dims).

## Rollout order

1. `solve_crops` + tests (pure, no deps).
2. Metadata rotation/dims read + threading to `store`.
3. `MomentFlag.subject_box` + Pass-2 emit + compose.
4. `store.build_cut_records` writes `framing`.
5. Full suite + pyflakes clean.
6. Live smoke on ONE project (real Pass-2), eyeball `framing.crop_*`/`rotation_deg`
   populated and the cuts-browser toggle uprighting + subject-centering. Only
   then re-ingest/backfill all projects.

## Out of scope (follow-ups, not needed for parity)

- **Moving-subject tracking** (a crop path over the cut, not one static box). Old
  pipeline was static per-cut; a static box is parity. The edit/render path
  already bakes motion from `motion_dynamics` centroids (`framing.annotate_
  document`) for the timeline preview — untouched here.
- **Speech-cut framing.** This plan targets `kind='video'` cuts (the reframe
  demo). Speech cuts (`vcut/speech`, frame analysis in `frames.py`) could emit
  `subject_box` the same way as a later extension.
- Subject-aware crops richer than a single box (headroom/lookroom-driven
  recompose) — a separate, model-heavier piece.
