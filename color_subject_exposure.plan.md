# Subject-Aware Exposure Evening (v1 color-grade pipeline)

> Implementation plan. Self-contained: an implementer with **no other context**
> should be able to build this from this file alone. All file:line refs were
> verified against the real code on the date this was written (see "Line-ref
> drift" at the end for the deltas found vs. the briefing). **You are wiring the
> FRONT of an existing chain** — most of the plumbing (`subject_box → subject_luma
> → solve_leveling`) already exists and is dormant only because nothing ever
> populates a per-shot `subject_box`. Do not rebuild it.

---

## 1. Goal & non-goals

**Goal.** Make the *subject* (the speaker / on-camera person, or the VLM-chosen
subject) **correctly and consistently exposed across shots** in the v1 grade
pipeline. Today Leveling evens shots on *whole-frame* mid-gray; a face in a dim
corner of a bright frame (or vice-versa) reads as inconsistent brightness even
after leveling. This feature makes the exposure signal track the *subject's own
luma* and makes subjects **converge to a common per-scene/sequence target**.

**Non-goals.**
- Do **not** touch the matching/grouping math already shipped (`balance.py`,
  `reference.py`, `match.py`, `scene_group.py`, `scene_meta.py`). We only *read*
  their outputs and *add* a subject signal alongside them.
- Do **not** change the composite guardrails' *intent* (`resolver.py`'s
  `COMPOSITE_SLOPE_MAX`, mid-gray + shadow-probe offset floors). They stay
  authoritative over the final CDL.
- No per-frame / intra-shot relight, no spatial masking of the grade (the
  compositor applies one CDL per shot — a subject box only informs the *scalar*
  exposure target, it does not spatially isolate the correction). This matches
  `leveling.py`'s documented non-goals.
- **Legacy pipeline stays byte-identical.** This is v1-only and gated behind a
  new flag that defaults **off**.
- **Graceful fallback is mandatory:** any shot where a subject box can't be
  resolved must behave *exactly* as today (whole-frame exposure). Never worse.

---

## 2. How this differs from competitors

Frame-agnostic color tools even shots on whole-frame statistics — they have no
idea *where the person is*. We reuse the edit's **own analysis** (ASD face
tracks + the VLM's per-cut `framing.subject_box`) to expose and even the
**subject** specifically, so a speaker's face reads at a consistent brightness
across a cut sequence even when the surrounding frame brightness differs.

---

## 3. What already exists (wire the front of this chain)

The full `subject_box → subject_luma → convergence` chain is present and dormant:

| Stage | Location | What it already does |
|---|---|---|
| Accepts a box, measures subject luma | `backend/app/services/l3/grade/measure_span.py` `_measure_subject_luma` (L90–107) and `measure_span(..., subject_box=...)` (L110–177) | Given a **normalized `(x, y, w, h)`** box, measures mean luma (Rec.709 weights, 0..1) inside it **on the hero frame specifically** (L168–171). Returns `stats["subject_luma"]`. Cache is keyed `(file_id,in_ms,out_ms)` only; a cache hit lacking `subject_luma` while a box was requested is treated as a partial miss and recomputed (L139). |
| Passes the box + feeds the luma | `backend/app/services/l3/grade/job.py` | `subject_box = s.item.get("subject_box") if settings.grade_semantic else None` (**L393**), passed into `measure_span` (**L394–395**). `stats.get("subject_luma")` → `ShotLevelInput.subject_luma` (**L520–522**). `subject_box` is already in the `input_hash` payload (**L217**). |
| Consumes the luma | `backend/app/services/l3/grade/leveling.py` `ShotLevelInput.subject_luma` (**L87**), `_usable_subject_luma` (L133–145, gated by `SILHOUETTE_RATIO=3.0` L62), `_exposure_value` (L148–150), `solve_exposure_leveling` (L153–173, cap `EXPOSURE_CAP_STOPS=0.5` L51) | When a usable `subject_luma` exists, exposure leveling targets **it** instead of whole-frame `mid_gray`, moving it toward a smooth low-pass target (or an explicit `target_mid_gray`), bounded to ±0.5 stops. |
| Documents the gap | `backend/app/services/l3/grade/resolver.py` Step 1.7 note (**L269–283**), `subject_box = item.get("subject_box")` (**L278**) | Explicitly says the box is "the normalized `(x,y,w,h)` box a caller that's already done the segment→`cut_records.framing.subject_box` mapping can pass through — no caller does that mapping yet." |

**The single missing link:** nothing populates a per-shot `subject_box`. The
timeline seg / `place_video` op (`s.item`) never carries one (`framing.py` bakes
`transform.focus = {cx,cy}`, *not* a `subject_box`). So `s.item.get("subject_box")`
is `None` in practice, `measure_span` never measures `subject_luma`, and Leveling
always falls back to whole-frame `mid_gray`.

---

## 4. Data-flow findings (verified against DB + code)

### 4.1 Where subject boxes live — TWO real, populated sources

**(A) `cut_records.framing.subject_box` — the recommended primary source.**
- **Format:** already **normalized `[x, y, w, h]`** in 0..1 (`pass2.py::Framing.subject_box`, L94: `# normalized x,y,w,h`). Verified sample values: `[0.2, 0.32, 0.6, 0.55]`, `[0.28, 0.38, 0.68, 0.55]`, `[0.05, 0.2, 0.85, 0.75]`.
- **Population (verified live):** `8684 / 8701` cut_records have a valid array `subject_box` (**99.8%**). This is the VLM's per-cut judgment of "the subject" (person *or* product), so it also covers non-face b-roll.
- **Time resolution:** one box per cut (not per-frame). Good enough for a scalar exposure target.
- **Join:** the **exact** `scene_meta.py` max-overlap pattern already used by the pipeline — join a grade shot `(file_id, in_ms..out_ms)` to its covering `cut_record` by greatest span overlap (`scene_meta.py::lookup_shot_cut_meta` L29–71; the join loop L54–60). `cuts_v3_read.rows_for_run` already SELECTs `framing` and `hero_ts_ms` (`cuts_v3_read.py` L62–68), so no query change is needed there.
- **This is normalized already**, so it maps directly onto `measure_span`'s decoded frame with **no proxy-dimension problem** (see 4.2).

**(B) `face_tracks` (ASD) — the higher-precision, speaker-aware refinement.**
- **Table:** `face_tracks` (one row per `file_id`), `tracks` jsonb (migration `041_asd_identity.sql`). Schema comment: *"per-file face tracks (embedding + sampled boxes + ASD-speaking intervals) … proxy pixel space."*
- **Shape** (`active_speaker.py` `FaceTrack`/`FaceFrame` L70–110): each track = `{track_id, embedding, frames: [{t_ms, box: [x,y,w,h]}], speaking: [{start_ms,end_ms,score}], best_crop_ms}`.
- **Coordinate space:** `box` is **pixel** `(x, y, w, h)` in **proxy pixel space** (`FaceFrame.box` comment L73), sampled at `TRACK_SAMPLE_FPS = 5.0` fps (L54) — i.e. a box every ~200 ms. `t_ms` is source/proxy-relative ms (same axis as a shot's `in_ms/out_ms/hero_ts_ms`).
- **Population (verified live):** 4 files present, all non-empty; e.g. one track with **3506** frames, **109** speaking intervals, sample `frame = {"box": [912,415,165,202], "t_ms": 0}`, `speaking = {"start_ms":2178,"end_ms":3564,"score":0.47}`.
- **`speaking` is the differentiator:** it tells us *which* track is the active speaker at a timestamp — so we can expose the **speaker's** face, not just any detected face.
- **Catch — normalization:** boxes are in **proxy pixel space**, not normalized, and the proxy's pixel dimensions are **not stored**. `files.width/height` is the *original* resolution (verified `3840×2160`); the proxy is a 1080p rescale, so you cannot divide by `files.width/height`. You must obtain the **proxy's native (W,H)** (see 4.2 + Phase 3).

### 4.2 Why normalized boxes are the safe currency

`measure_span` measures on a frame decoded by `color_stats._decode_rgb_frame_at`
(`color_stats.py` L125–139), which force-scales to `COLOR_STATS_W × COLOR_STATS_H
= 320×180` (`-vf scale=320:180`, L55–56) — this does **not** preserve aspect
ratio. `_measure_subject_luma` applies the box as *fractions* of `frame.shape`
(L100–102). A **normalized** box therefore maps to the correct relative region
regardless of source aspect (per-axis fractions are scale-invariant). So:
- Source (A) `framing.subject_box` is normalized → plug in directly.
- Source (B) ASD pixel boxes → must be divided by the **proxy's** native pixel
  dimensions (the proxy ASD ran on == the proxy `measure_span` downloads via
  `r2_proxy_key`, so the two agree once normalized).

### 4.3 Verdict / recommendation

Both sources are real and populated. **Recommendation: a tiered per-shot
`subject_box` resolver**, built in leverage order:

1. **Phase 1 (primary, lowest risk, "get a real box in first"):** join
   `cut_records.framing.subject_box` via the existing `scene_meta` max-overlap
   pattern. Normalized already → zero decode cost, no proxy-dim problem, 99.8%
   coverage. This alone lights up the entire dormant chain.
2. **Phase 3 (precision refinement):** when a `face_tracks` row exists, prefer
   the **active-speaker** track's box near `hero_ts_ms` (normalized by probed
   proxy dims). More accurate for talking-head content; falls back to (A).
3. **Fallback:** no box resolves → `subject_box = None` → today's whole-frame
   behavior, byte-identical.

---

## 5. Design

### 5.1 Obtain a per-shot `subject_box`

Add a small resolver `subject_box_for_shots(...)` returning `Dict[shot_key ->
[x,y,w,h] normalized]`. Put it next to the existing join code:
`backend/app/services/l3/grade/scene_meta.py` (it already owns the shot→cut_record
join and its `cuts_v3_read` import).

- **Phase 1 path:** extend `ShotCutMeta` (L13–22) with `subject_box:
  Optional[List[float]] = None` and read `best.get("framing", {}).get("subject_box")`
  inside `lookup_shot_cut_meta` (L62–70), validating it's a length-4 array of
  finite floats in a sane range (clamp to 0..1; reject if `w<=0` or `h<=0`). No
  new query — `framing` is already selected.
- **Phase 3 path:** a sibling helper `subject_box_asd(file_id, in_ms, out_ms,
  hero_ts_ms) -> Optional[[x,y,w,h]]` in a new module
  `grade/subject_box_asd.py` (keeps ASD/insightface-shaped concerns out of the
  pure-lookup `scene_meta.py`).

### 5.2 Attach the box so `run_grade_job` / `measure_span` see it

Mirror the `scene_meta` grade-time attach — **do NOT add a document schema field**
and do not persist onto `s.item`. Resolve boxes into a `Dict[key -> box]` inside
`run_grade_job` and pass the per-shot box into `measure_span`.

**Ordering fix (important):** today `measure_span` runs (job.py **L392–398**)
*before* the `cut_meta` join (**L414–437**). You must resolve the subject box
**before** the measure loop. The existing `lookup_shot_cut_meta` only needs
`(key, file_id, in_ms, out_ms)` (available pre-measure); only its `span_rgb`
consumer runs post-measure. So hoist the subject-box lookup above the measure
loop (you may hoist `lookup_shot_cut_meta` itself and reuse its result for both
the subject box *and* the later scene grouping — the `rgb_mean` is added
separately after measurement, so this is safe).

Then, in the measure loop (currently L393–395):
```python
box = subject_boxes.get(s.key) if settings.grade_semantic and settings.grade_subject_exposure else None
stats = measure_span(s.file_id, s.in_ms, s.out_ms, hero_ts_ms=s.hero_ts_ms, subject_box=box)
```
(Keep the existing `s.item.get("subject_box")` as an override if present, for
forward-compat, but in practice it's always `None`.)

### 5.3 Make `solve_leveling` actually CONVERGE subject luma (not just clamp)

**What happens today.** `solve_exposure_leveling` (leveling.py L153–173):
- `value = _exposure_value(s)` = usable `subject_luma` else `mid_gray`.
- For **ungrouped** shots: target = the smooth low-pass of the `values` array
  → subjects *do* converge toward a local smooth target, but the array **mixes**
  subject lumas (subject shots) with whole-frame mids (subjectless shots), so the
  target is muddied.
- For **grouped** shots (job.py L541–546 sets `target_mid_gray = ref.mid_gray`):
  the explicit target is the group's **whole-frame** mid-gray reference. So a
  grouped subject shot pushes its *subject_luma* toward a *whole-frame* target —
  a **mismatch**. This is the core correctness gap.

**The change — add a subject-specific target:**
1. In `leveling.py`, add `target_subject_luma: Optional[float] = None` to
   `ShotLevelInput` (alongside `target_mid_gray`, L88).
2. In `solve_exposure_leveling`, when a shot has a **usable** subject
   (`_usable_subject_luma` is not None) **and** `target_subject_luma` is set, use
   `target_subject_luma` as that shot's target (instead of `target_mid_gray` /
   the smooth target). Otherwise behavior is unchanged. Concretely, in the target
   list comprehension (L163–166), select `target_subject_luma` first when the
   shot's exposure value came from a usable subject luma, then `target_mid_gray`,
   then the smooth target.
3. Compute the group's **subject** reference: the **median** `subject_luma`
   (working-space) over the group's members that have a usable subject. Add this
   as `GroupReference.subject_luma: Optional[float]` in `reference.py`
   (`compute_group_reference`, L30–48) — but note its inputs there are per-stat
   dicts; the simplest wiring is to compute the median subject-luma **inline in
   job.py** where the group index and each member's `subject_luma` are already in
   scope (leveling block L514–555), and pass it as `target_subject_luma`. Keep
   `reference.py` untouched if you prefer minimal blast radius (recommended).
4. For **ungrouped** shots, optionally make the smooth-target array subject-clean:
   compute the low-pass over subject values only for shots that have a usable
   subject. Minimal version: leave as-is (still converges, just muddier). Note
   the choice; default to the minimal version to reduce risk.

**Result:** subjects within a scene group converge to a common subject
brightness; ungrouped subject shots track a smooth subject target across the
sequence. All still bounded by `EXPOSURE_CAP_STOPS`.

### 5.4 Optional: bias the Balance/Match reference toward subject luma

Out of scope for the default build (keep matching math untouched per non-goals).
If pursued later: `balance.py`/`reference.py` could weight the group reference's
`mid_gray` toward member subject lumas so *matching* also aligns on the subject.
Leave behind the `grade_subject_exposure` flag; do **not** enable by default.
Recorded as a follow-up, not part of this plan's acceptance.

### 5.5 Interaction with the composite guardrails

A genuinely dark subject needs positive slope (lift). Two bounds apply, in order:
- **`EXPOSURE_CAP_STOPS = 0.5`** (leveling): a single pass lifts a subject by at
  most ~1.41×. A very dark subject is therefore only *partially* lifted toward
  the group target in one grade — **this is intended** (never over-correct;
  never-worse). Do not raise this to "fully fix" dark subjects.
- **`COMPOSITE_SLOPE_MAX = 2.0`** + the mid-gray/shadow-probe **offset floors**
  (`resolver.py::_clamp_composite_v1`, L108–135) still clamp the *final* composed
  CDL. Because the subject lift is a slope (`Grade(slope=(g,g,g))`,
  leveling.py L172), it composes multiplicatively with Correct/Match/Balance;
  the composite ceiling prevents runaway contrast and the shadow-probe floor
  prevents crushing a display-~0.15 shadow to black while lifting the subject.
- **Guidance for the implementer:** do **not** add a subject-specific slope cap
  that exceeds `EXPOSURE_CAP_STOPS`, and do **not** relax the composite floors.
  If a dark subject hits the exposure cap before reaching the group target,
  accept the partial lift — it's still a monotonic improvement and stays inside
  the guardrails. Optionally introduce `SUBJECT_EXPOSURE_CAP_STOPS` (default =
  `EXPOSURE_CAP_STOPS` = 0.5, i.e. no behavior change) as a *future* tuning knob,
  but ship it equal to the frame cap.

---

## 6. Phased implementation

All new behavior is gated behind a **new flag `grade_subject_exposure` (default
off)** **AND** the existing `grade_semantic` (the subject signal is a semantic
one). When either is off, output is byte-identical to today.

Order is by leverage — Phase 1 alone makes subjects measured + evened; Phase 3
is precision on top.

### Phase 0 — config + hash + flag (no behavior yet)
- **File:** `backend/app/config.py`. **Add** `grade_subject_exposure: bool = False`
  near the other grade flags (L146–168).
- **File:** `backend/app/services/l3/grade/job.py`. **Add**
  `"grade_subject_exposure": settings.grade_subject_exposure` to the `flags` dict
  in `compute_input_hash` (**L225–231**). **Bump** `INPUT_HASH_SCHEMA_VERSION`
  **5 → 6** (**L71**) with a one-line comment noting subject-aware exposure.
- **Acceptance for this phase:** with the flag off, `compute_input_hash` differs
  only by the schema bump (all grades re-computed once) and the grade math is
  otherwise unchanged.

### Phase 1 — populate `subject_box` from `framing.subject_box` (primary)
- **File:** `backend/app/services/l3/grade/scene_meta.py`.
  - Add `subject_box: Optional[List[float]] = None` to `ShotCutMeta` (L13–22).
  - In `lookup_shot_cut_meta` (L62–70), set `subject_box` from
    `best.get("framing")` → `.get("subject_box")`, validated (length-4, finite,
    `w>0`, `h>0`, clamp components to 0..1). Fail-open: invalid → `None`.
- **File:** `backend/app/services/l3/grade/job.py` (`run_grade_job`).
  - **Hoist** the `lookup_shot_cut_meta` call (currently L416–421, inside the
    `grade_semantic` block after measurement) to **before** the measure loop
    (before L392), guarded by
    `settings.grade_semantic and settings.grade_subject_exposure` (still also run
    it for `grade_scene_join` as today — combine the guards; only *one* lookup).
    Build `subject_boxes: Dict[str, List[float]] = {k: m.subject_box for k,m in
    cut_meta.items() if m.subject_box}`.
  - In the measure loop (L393–395), resolve the box as in §5.2 and pass it to
    `measure_span`.
  - Ensure the later scene-grouping block reuses the same `cut_meta` instead of
    re-querying (it can — the `span_rgb` add is independent).
- **Result:** shots with a covering cut_record now measure `subject_luma`; the
  existing Leveling chain (still with today's target math) starts evening on the
  subject for ungrouped shots immediately.

### Phase 2 — subject convergence in Leveling
- **File:** `backend/app/services/l3/grade/leveling.py`.
  - Add `target_subject_luma: Optional[float] = None` to `ShotLevelInput` (L88).
  - In `solve_exposure_leveling` (L153–173), route a usable-subject shot's target
    to `target_subject_luma` when set (see §5.3 step 2). Keep everything else
    identical.
- **File:** `backend/app/services/l3/grade/job.py` (leveling block L514–555).
  - For a grouped shot with a usable subject, compute the group's **median
    working-space `subject_luma`** over members that have one, and pass it as
    `target_subject_luma`. (Members' `subject_luma` values are already computed
    at L520–522; collect per group index.) When the group has <2 subject members,
    leave `target_subject_luma=None` (falls back to today's `target_mid_gray`).
- **Result:** subjects in a scene group converge to a common subject brightness.

### Phase 3 — ASD active-speaker box (precision refinement, optional but recommended)
- **New file:** `backend/app/services/l3/grade/subject_box_asd.py`.
  - `load_face_tracks(file_id) -> List[dict]` (read `face_tracks.tracks`, best-effort/fail-open → `[]`).
  - `proxy_dims(file_id) -> Optional[(w,h)]`: obtain the **proxy's native**
    pixel size. Preferred: a one-shot `ffprobe` on the downloaded proxy (reuse
    `measure_span`'s temp path, or accept the small extra probe — negligible vs.
    the decode already done). Cache per file_id. (Do **not** use
    `files.width/height` — that's the original, not the proxy.)
  - `subject_box_asd(file_id, in_ms, out_ms, hero_ts_ms) -> Optional[[x,y,w,h]]`:
    1. Candidate tracks = tracks with ≥1 `frames[].t_ms` inside `[in_ms,out_ms]`.
    2. Prefer the track whose `speaking` intervals cover / are nearest
       `hero_ts_ms` (else the track with the most in-span speaking ms; tie-break
       by largest `box` area at `best_crop_ms`).
    3. Take that track's `frames[].box` at the frame nearest `hero_ts_ms` (else
       the median box over the in-span frames).
    4. Normalize by `proxy_dims` → `[x/W, y/H, w/W, h/H]`, clamp to 0..1.
       Return `None` on any failure.
- **File:** `job.py`. In the box resolver (§5.2), try ASD first, fall back to the
  Phase-1 `framing.subject_box`, then `None`.
- **Result:** talking-head shots expose on the *speaker's* face specifically.

Each phase is independently shippable; Phases 1–2 give the full feature, Phase 3
improves fidelity.

---

## 7. New / changed constants & config flag table

| Symbol | File | Default | Rationale |
|---|---|---|---|
| `grade_subject_exposure` | `app/config.py` | `False` | Master flag for this feature; off until validated. Respected only when `grade_semantic` is also on. |
| `INPUT_HASH_SCHEMA_VERSION` | `grade/job.py` | `6` (was `5`) | Grade math can change → invalidate all cached grades once. |
| `ShotCutMeta.subject_box` | `grade/scene_meta.py` | `None` | Carries the joined normalized `[x,y,w,h]`. |
| `ShotLevelInput.target_subject_luma` | `grade/leveling.py` | `None` | Explicit per-group subject target; `None` = today's behavior. |
| `SUBJECT_EXPOSURE_CAP_STOPS` *(optional)* | `grade/leveling.py` | `0.5` (== `EXPOSURE_CAP_STOPS`) | Future tuning knob; ship equal to the frame cap so it's a no-op initially. |
| `GroupReference.subject_luma` *(optional)* | `grade/reference.py` | `None` | Only if you choose to compute the subject reference in `reference.py` instead of inline in `job.py`. Prefer inline (smaller blast radius). |

No existing constant's value changes. No document-schema field is added.

---

## 8. Switchability / rollback

- `grade_subject_exposure` defaults **off**. With it off (or `grade_semantic`
  off), `subject_boxes` is never built, no box reaches `measure_span`,
  `subject_luma` is never measured, and `target_subject_luma` stays `None` — so
  Leveling, Balance, Match, and the resolver are **byte-identical** to today
  (verify via `grade_hash` equality in tests).
- The only unconditional change is `INPUT_HASH_SCHEMA_VERSION 5→6`, which forces
  a one-time re-grade producing the *same* grades when the flag is off.
- **Clean revert point:** all new code is additive and flag-gated. Reverting =
  set the flag off (runtime) or revert the small diffs in `config.py`,
  `job.py`, `scene_meta.py`, `leveling.py` (+ delete `subject_box_asd.py`).

---

## 9. Testing (`backend/scripts/test_grade.py`)

Plain script, no DB/ffmpeg/R2 (mirror the file's existing `mock.patch` +
scripted-fake convention). Run: `cd backend && .venv/bin/python scripts/test_grade.py`.
Add these tests (import helpers already imported at the top of the file:
`_measure_subject_luma`, `ShotLevelInput`, `solve_exposure_leveling`,
`solve_leveling`, `lookup_shot_cut_meta`, `resolve_clip_grade`):

1. **`test_subject_box_measures_subject_luma`** — build a synthetic RGB frame
   (numpy `uint8 H×W×3`) with a bright rectangle in a known normalized region on
   a dark background; assert `_measure_subject_luma(frame, box)` returns the
   bright region's mean, and `None` for a degenerate box (`w=0`, or fully
   out-of-frame).
2. **`test_two_subjects_converge_after_leveling`** — two `ShotLevelInput`s with
   equal whole-frame `mid_gray` but `subject_luma` far apart (e.g. 0.25 vs 0.55),
   same group, `target_subject_luma` = their working-space median. Assert the
   post-`solve_exposure_leveling` gains move both subject lumas toward the target
   (spread after < spread before), within `EXPOSURE_CAP_STOPS`.
3. **`test_no_subject_box_identical_to_today`** — run `solve_leveling` on inputs
   with `subject_luma=None` and `target_subject_luma=None`; assert output equals
   the pre-change behavior (snapshot the grades). Also: `resolve_clip_grade` with
   no `subject_box` produces the same `grade_hash` as before.
4. **`test_join_finds_nothing_no_crash`** — `mock.patch` `cuts_v3_read` so
   `lookup_shot_cut_meta` sees no covering run / no overlap; assert it returns
   entries with `subject_box=None` and never raises (fail-open).
5. **`test_silhouette_subject_falls_back`** — `subject_luma` 3× off the shot's
   `mid_gray`; assert `_usable_subject_luma` returns `None` and the shot levels on
   whole-frame `mid_gray` (existing `SILHOUETTE_RATIO` gate still governs).
6. **`test_legacy_unchanged`** — a `resolve_clip_grade(pipeline="legacy")` path is
   byte-identical (already covered patterns exist; assert no regression).

For Phase 3, add a pure-numeric test for `subject_box_asd`'s normalization + speaker
selection using a scripted `tracks` fixture (no DB): given a `speaking` interval
covering `hero_ts_ms`, it picks that track's nearest-frame box and normalizes by a
given `(W,H)`.

---

## 10. Ops runbook

1. Land Phases 0–2 (and 3 if building it).
2. `INPUT_HASH_SCHEMA_VERSION` is bumped **5 → 6** (already in Phase 0).
3. Turn the flag **on**: set `GRADE_SUBJECT_EXPOSURE=true` in the worker env
   (and ensure `GRADE_SEMANTIC=true`, already the default per `config.py` L153).
4. **Restart the grade worker** so it picks up the new env + code.
5. Re-grade every project's current thread:
   `cd backend && PYTHONPATH=. .venv/bin/python scripts/_grade_all_projects.py`
   (it forces `GRADE_PIPELINE=v1`, `GRADE_EVEN_LIGHTING=true`, `GRADE_SEMANTIC=true`;
   **also export `GRADE_SUBJECT_EXPOSURE=true` before running it**, or add it to
   that script's `os.environ` block at the top).
6. **Verify:** the summary prints `state=done`, `rows>0`, `baked>0` per project.
   Spot-check a people-reel thread's `resolved_grades` and confirm cross-shot
   subject-luma spread dropped (see acceptance).

---

## 11. Acceptance criteria (quantified)

- **Cross-shot subject evenness.** On a reel with people, measure the standard
  deviation of per-shot *subject* luma (display-space) across the shots that have
  a usable subject, **before vs. after** enabling the flag. Target: **spread
  drops ≥ 40%** (or absolute post-leveling stdev **≤ 0.05**), while whole-frame
  contrast between distinct scenes is preserved (don't flatten real day/night
  cuts — the smooth-target arc survives per `LEVELING_WINDOW`).
- **Dark subject lifted, guardrails intact.** A deliberately dark-subject shot in
  a group is lifted **toward the group's median subject luma** (monotonic
  improvement), bounded by `EXPOSURE_CAP_STOPS`; the final composed CDL keeps
  `black_point ≥ ~0` (no shadow crush past the shadow-probe floor) and
  `white_point ≤ ~1` (no blown highlights), i.e. `_clamp_composite_v1` still
  holds. A very dark subject may be only partially lifted in one pass — accepted.
- **Frame-only reels unchanged.** For a reel where no `subject_box` resolves (or
  with the flag off), every shot's `grade_hash` is **identical** to the pre-change
  v1 output (byte-identical fallback).

---

## 12. Line-ref drift found vs. the briefing

- ✅ `measure_span(subject_box=...)` and `_measure_subject_luma` exist as
  described; box is **normalized `(x,y,w,h)`**, measured on the **hero frame**
  (measure_span.py L90–177). Cache partial-miss handling at L139.
- ✅ `job.py`: `subject_box` read at **L393**, passed at **L394–395**;
  `subject_luma` → `ShotLevelInput` at **L520–522**; `subject_box` already in the
  `input_hash` payload at **L217**; `INPUT_HASH_SCHEMA_VERSION = 5` at **L71**.
- ✅ `resolver.py` Step 1.7 note at **L269–283** (`subject_box = item.get("subject_box")`
  at L278) — confirms no caller derives the box yet.
- ✅ `leveling.py` consumes `subject_luma` (L87, L133–150); today it converges
  toward a **smooth low-pass** target for ungrouped shots and toward the group's
  **whole-frame** `target_mid_gray` for grouped shots — so it *does* even, but on
  the wrong (whole-frame) target for grouped subject shots. §5.3 fixes this.
- ⚠️ **Key correction to the briefing's premise:** subject boxes are **already
  persisted** — twice. `cut_records.framing.subject_box` (normalized, **99.8%**
  populated, joinable via the exact `scene_meta` pattern) is the recommended
  primary source; `face_tracks` (ASD, per-frame proxy-pixel boxes + speaking
  intervals, populated) is the speaker-aware refinement. So the "must be derived"
  branch is **not** needed — no new detector run at grade time. The only
  derivation cost is Phase 3's proxy-dimension probe for ASD normalization.
- ⚠️ **Ordering caveat:** the `scene_meta` join currently runs **after**
  `measure_span` in `job.py` (L414–437 vs. L392–398). Phase 1 must hoist the
  subject-box lookup **before** the measure loop.
