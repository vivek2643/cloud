# Color QA Harness — Implementation Plan (Phase-1 base-quality parity)

**Status:** plan only. No production code changes here. All new code lives in
`backend/scripts/` as `_diag_`-prefixed, read-only tools plus one reusable
`backend/scripts/qa/` metrics module.

---

## Goal & rationale

We have concluded Phase 1 of color grading is about clearing the **base-quality
bar**: the grade must read as *professional / trustworthy* — rough parity with
tools like ColourLab AI — before we chase the Phase-2 semantic edge. The
`grade_pipeline_standardize.plan.md` work has already collapsed the pipeline to a
single un-flagged v1 path, so the architecture is now stable enough to measure.

**The problem this harness solves: we are flying blind.** There is **no
user-facing quality-feedback signal anywhere in the product** — confirmed:

- No feedback/rating/thumbs endpoint exists in `backend/app/routers/` (grep for
  `feedback|rating|thumbs|complaint|satisfaction|telemetry` → nothing
  quality-related; the only `feedback` hits are a VLM-context helper in
  `l3/tools.py`/`pass2_params.py` and the word inside `scene_group.py`'s
  docstring).
- The only `feedback` reference in the frontend grade view
  (`frontend/src/components/color-grade-view.tsx:375`) is a *UI progress bar*
  ("instant feedback" that a look changed), not a quality signal.

So today the **only** quality signal is a human opening a preview and reporting
"this looks off" ad hoc. The whole grade iteration history proves the cost of
this: the git log shows a filmic contrast S-curve added
(`03f6911 color_tone_contrast.plan.md`) and then hardwired **OFF** in
`grade_pipeline_standardize.plan.md` because it "regressed too-dark looks" — a
full add-then-revert cycle that a measurement harness would have caught before
it shipped.

**This harness is the Phase-1 flywheel.** It renders RAW-vs-GRADED stills across
the *entire project corpus*, computes objective failure metrics per shot,
rolls them into a scoreboard, and lets us (1) iterate toward parity by ranking
and fixing the highest-impact failure classes and (2) guard against regressions
when we change grade math. It replaces "a human eyeballed one preview" with
"the corpus scoreboard moved from X to Y".

Non-goal (out of scope, do NOT build): anything about the LLM/brain
communication layer, Phase-2 semantic grading (identity, narrative arcs), and
any actual grade-math changes beyond **Part B** (the IDT), which is the first
fix the harness is built to drive+prove.

---

## Part A — The harness

Three stages: **enumerate corpus → sample RAW/GRADED stills → measure + score**.

### A.0 — Where it plugs into the existing pipeline (reuse, don't reinvent)

| Need | Reuse |
|---|---|
| Enumerate every project's gradeable thread | `backend/scripts/_grade_all_projects.py` (its `projects` query + `latest_thread`/`latest_document` + `grade_job.ordered_shots(doc)` loop) |
| Ensure fresh grades exist | `grade_job.run_grade_job(tid)` then `grade_job.fetch_latest_grades(thread_id, keys)` |
| Bake a shot's `.cube` | `grade.cache.ensure_cube_file(grade_json, cube_dir)` (same path the render compositor/preview use) |
| Extract a still + apply cube with ffmpeg | `_grade_v1_frames.py` (`extract_still` from `l3/frames.py`, then `ffmpeg -vf lut3d=file=...`) |
| Bake a look's thumbnail cube (look-fidelity metric) | `_diag_look_thumbs.py`'s `thumb_cube_output(spec, ref_rgb01)` (identity CDL + `build_look_grid` → `bake_cube_text` → `_sample_lut_trilinear`) |
| Per-shot color measurement primitives | `l1/color_stats.py::_aggregate` / `_decode_rgb_frame_at`, and `grade.measure_span.measure_span` |
| Subject box + hero frame per shot | `grade.scene_meta.lookup_shot_cut_meta` (gives `subject_box`, `hero_ts_ms`, and scene metadata for grouping) |
| Semantic scene groups (for consistency metric) | `grade.scene_group.group_shots_semantically` + `grade.job`'s exact grouping/fallback logic (`_has_real_groups` → `group_neighbors`) |

**Critical correctness rule for RAW vs GRADED:** the "GRADED" still MUST be the
RAW still put through the shot's baked `.cube` (ffmpeg `lut3d`), NOT a re-render.
This guarantees the harness measures exactly what preview/export produce (same
cube, same math, `color_grading.plan.md` SS4 "Fork A" parity contract). Soft-local
effects (vignette/halation/grain) live *outside* the cube (`lut_bake.py` docstring;
they are a separate ffmpeg pass), so v1 of the harness measures the **cube-graded**
image and treats soft-local as a known blind spot (see A.4 over-processing notes).

### A.1 — Corpus enumeration

`backend/scripts/_diag_qa_corpus.py` (read-only):
- Copy `_grade_all_projects.py`'s project/thread enumeration verbatim.
- For each project, pick its latest gradeable edit thread (first thread whose
  `latest_document` has a non-empty `timeline`/`operations`).
- Emit a manifest `backend/scripts/_out/qa/corpus.json`:
  `[{project_id, project_label, thread_id, shot_count, file_ids}]`.
- Flag: `--regrade` runs `grade_job.run_grade_job(tid)` first (so grades are
  current for the deployed `INPUT_HASH_SCHEMA_VERSION`); default just reads the
  freshest persisted grades via `fetch_latest_grades`.

### A.2 — Sampling (RAW vs GRADED stills)

`backend/scripts/_diag_qa_sample.py` (read-only, ffmpeg-driven):

For each shot in each thread:
1. Resolve the frame(s) to sample:
   - **Hero frame:** the shot's `hero_ts_ms` if present, else the covering
     `cut_record`'s `hero_ts_ms` (from `lookup_shot_cut_meta`), else the span
     midpoint `(in_ms+out_ms)//2` — the same fallback ladder `job.py` uses for
     `measure_span`.
   - **A few keyframes:** additionally sample at ~25%/50%/75% of the span (reuse
     `color_stats._sample_timestamps(span_s, 3)`), so a metric isn't fooled by one
     lucky frame. Keep it to ≤4 frames/shot (matches `measure_span.SPAN_MAX_FRAMES`,
     keeps decode cost bounded).
2. Download the proxy once per `file_id` (reuse `_download_from_r2` +
   `r2_proxy_key`-preferred lookup as `measure_span._fetch_proxy_path` /
   `_grade_v1_frames.py` do), extract each still with `l3.frames.extract_still`.
3. Bake the shot's cube: `ensure_cube_file(fetch_latest_grades(...)[shot_key],
   cube_dir)`; apply it with `ffmpeg -i raw -vf "lut3d=file='cube'" graded`
   (exactly `_grade_v1_frames.apply_cube`). If a shot has no grade row, record
   `graded == raw` (identity) so it still scores (and shows up as "ungraded").
4. Persist RAW+GRADED PNG/JPG pairs under
   `backend/scripts/_out/qa/<project>/<shot_key>_<ts>_{raw,graded}.jpg` and a
   per-shot record (frame paths, ts, subject_box, group_id).

Cost control: cache decoded proxies per run; cap frames/shot; a `--projects`
filter to run a subset while iterating on metrics.

### A.3 — Contact sheets + global index (human eyeball layer)

`backend/scripts/_diag_qa_sheets.py`:
- Per project: an hstacked **RAW | GRADED** row per shot (labeled with shot_key +
  its pass/warn/fail badge), vstacked into one contact sheet PNG (same
  ffmpeg `hstack`/`vstack`/`drawtext` recipe as `_grade_v1_frames.py`).
- Global `backend/scripts/_out/qa/index.html`: thumbnails of every project sheet,
  each annotated with the project's rollup score, sortable by worst-first.
- This is the subjective "feels off" backstop the metrics can't fully capture.

### A.4 — FAILURE TAXONOMY (the core: concrete, computable metrics + bands)

All metrics live in a reusable module `backend/scripts/qa/metrics.py` (pure
functions over decoded RGB frames + optional subject_box + group membership; no
DB, easy to unit-test). Luma is Rec.709 `0.2126R+0.7152G+0.0722B` on 0..1 RGB
(same coefficients as `color_stats._aggregate` and `look_engine.LUMA_*`). Color
metrics use CIE Lab via `cv2.cvtColor(..., COLOR_RGB2LAB)` with OpenCV's 8-bit
packing (L*×100/255, a*/b*−128) — identical convention to
`color_stats._aggregate` and `measure_span._measure_subject_lab`, so numbers are
directly comparable to what the grade solver saw.

Each metric is computed on the **GRADED** frame (and, where noted, also on RAW to
report the RAW→GRADED delta = "did the grade help or hurt?"). Thresholds below are
**starting bands to be calibrated** (see Risks) — they are deliberately explicit
so the first run produces a real scoreboard.

#### 1. Exposure
- `crushed_black_fraction` = mean(luma ≤ 2/255) on the graded frame. Mirrors
  `color_stats.CLIP_LOW_8BIT`. Band: PASS < 0.02, WARN < 0.05, FAIL ≥ 0.05
  (a graded shot crushing >5% to pure black). Also report the RAW→GRADED delta —
  the grade should almost never *increase* crush (this is the exact failure
  `resolver.COMPOSITE_SHADOW_PROBE` was added to fix; the metric guards it).
- `clipped_highlight_fraction` = mean(luma ≥ 253/255) (`CLIP_HIGH_8BIT`). Same
  bands. `tone.from_working`'s shoulder should keep this low; a regression here
  means the shoulder isn't engaging.
- `exposure_band` = median luma out of an acceptable window. Correct targets
  `TARGET_MID_GRAY = 0.42` (display); band PASS 0.30–0.60, WARN 0.22–0.72,
  FAIL outside. Report both mean and median (mean catches overall bias, median
  catches "most of the frame is dark but a window blows the mean").

#### 2. White balance / color cast
- `neutral_axis_deviation`: build a near-neutral mask (pixels with Lab chroma
  `sqrt(a*²+b*²) < 6`), report the **mean residual a*/b*** of that mask. A
  well-balanced image's neutrals sit near (0,0). Band: PASS |ab| < 3, WARN < 6,
  FAIL ≥ 6. If the neutral mask is too small (<1% of pixels) mark **N/A** rather
  than forcing a reading.
- `skin_hue_error` / `skin_chroma_error` — **measured ON the subject box**
  (`cut_records.framing.subject_box`, resolved via `lookup_shot_cut_meta`, 99.8%
  populated per `scene_meta.py`). Sample mean Lab inside the box (reuse
  `measure_span._measure_subject_lab`), decompose against the universal skin
  locus `SKIN_LOCUS_DEG = 50°` (same axis `correct._skin_multiplier` uses):
  - `skin_perp_residual` = |off-locus (green↔magenta tint) component|. This is the
    thing `correct.py` actively corrects, so it's the cleanest signal that WB is
    right on people. Band: PASS < 6, WARN < 12, FAIL ≥ 12.
  - `skin_L` sanity: only score when `SKIN_L_MIN..SKIN_L_MAX` (20..92) and chroma
    ≥ `SKIN_MIN_CHROMA` (3) — same gates as `_skin_multiplier`, so we never score
    a non-skin box. **This is the metric competitors can't compute as precisely**
    — it keys off our known, per-cut subject box, not a guessed center region.

#### 3. Shot-to-shot CONSISTENCY (the primary "does matching work?" metric)
- For each **scene group** (recompute exactly as `job.py`: semantic groups via
  `group_shots_semantically`, fall back to `group_neighbors` when
  `_has_real_groups` is false), over the graded hero frames of its members:
  - `intra_group_luma_std` = stdev of per-shot median luma.
  - `intra_group_chroma_std` = stdev of per-shot mean a* and mean b* (report the
    larger of the two axes).
  - `intra_group_black_std` / `intra_group_white_std` = stdev of black/white
    points across members (ties directly to what Balance/Match/Leveling converge).
  - Bands (per group, ≥2 members): PASS luma_std < 0.03 & chroma_std < 2.5;
    WARN < 0.06 / < 5; FAIL above. Singletons are excluded (nothing to match).
- This is the headline number: if Balance+Match+Leveling actually work, grouped
  members converge. Report **RAW vs GRADED** group std side by side — the grade
  should *reduce* it. If GRADED std ≥ RAW std for a group, matching is a no-op or
  harmful there (this is precisely the failure `_corrected_display_stats`'s
  docstring documents fighting).

#### 4. Over-processing
- `saturation_band`: mean Lab chroma of the graded frame (reuse
  `color_stats` `chroma_mean` math). Band: PASS 12–40, WARN 8–55, FAIL outside
  (dead-flat or garish). Also flag `chroma_increase_ratio = graded/raw`; FAIL if
  > 2.0 (grade doubled saturation).
- `banding_score`: on a small blurred luma ramp region, count adjacent-histogram
  gaps (posterization from an aggressive 33³ LUT / heavy contrast). Concretely:
  histogram the graded luma into 256 bins, compute the fraction of empty bins
  *between* the min and max occupied bin; high empty-fraction in a smooth image
  ⇒ posterization. Band: PASS < 0.15, WARN < 0.30, FAIL ≥ 0.30. (Heuristic; the
  contact sheet is the real judge — keep this as a WARN-only signal at first.)
- `halo/vignette overshoot`: **v1 blind spot** — vignette/halation/grain are a
  separate ffmpeg pass not in the cube (`lut_bake.py`), so the cube-graded still
  doesn't include them. Note explicitly; a follow-up can render the full
  soft-local pass for shots whose look declares `halation/grain` (via
  `resolve_clip_grade`'s `soft_local` descriptor) and measure edge-halo energy
  around the subject box. Not required for the first scoreboard.

#### 5. Look fidelity
- Only for shots whose document `look.mode == "engine"` (has a `look_engine`
  descriptor in the grade row). Bake the look's **own** thumbnail cube (identity
  CDL + `build_look_grid(spec)` via `_diag_look_thumbs.thumb_cube_output`) and
  apply it to the RAW still; compare to the actual GRADED still (which =
  correct+balance+match+leveling+**look**). The metric is the **direction/shape**
  the look adds, not pixel-equality (the graded still also carries correction):
  compute per-channel mean-shift + chroma-shift of (graded − raw) and of
  (look-only − raw), report cosine similarity of those shift vectors. Band: PASS
  cos > 0.8, WARN > 0.5, FAIL ≤ 0.5 (the applied look is pulling a different
  direction than the look intends → a compositing/order bug).

#### 6. Exposure-evenness (LIGHTING — ties `even_lighting`/`subject_exposure` to a number)
- `intra_scene_exposure_spread`: same as consistency metric #3's
  `intra_group_luma_std`, but reported specifically as the **lighting-evenness**
  KPI, and additionally computed on **subject-box luma** where boxes exist
  (`_measure_subject_luma`): `intra_group_subject_luma_std`. This directly
  measures whether `leveling.solve_exposure_leveling` + the
  `target_subject_luma` convergence (`job.py:582-605`) actually evens faces
  across a scene. Band: PASS subject_luma_std < 0.03, WARN < 0.06, FAIL above.
- `subject_convergence_delta` = RAW subject_luma_std − GRADED subject_luma_std per
  group; positive = leveling helped. A negative delta on a real (2+ subject)
  group is a leveling regression.
- Out of scope (state explicitly, do not metricize as failure): **intra-frame**
  lighting unevenness (one side of a face dark) — that's relighting, which
  `leveling.py`'s docstring lists as an explicit non-goal.

### A.5 — Scoring & scoreboard

`backend/scripts/_diag_qa_score.py`:
- **Per-shot:** each metric → {pass, warn, fail}; shot verdict = worst class
  present (any FAIL ⇒ shot FAILs; else any WARN ⇒ WARN; else PASS).
- **Per-project rollup:** counts/percentages of pass/warn/fail shots + the worst
  offending metric classes; plus group-level consistency stats.
- **Global scoreboard** `backend/scripts/_out/qa/scoreboard.json` + a printed
  table (same style as `_grade_all_projects.py`'s SUMMARY): per failure class,
  `count × mean-severity` across the whole corpus, ranked — this is the input to
  the iteration loop's "rank failures" step.
- **Human layer:** the A.3 contact sheets/index, linked from the scoreboard, for
  the subjective residue metrics miss.

---

## Part B — First fix the harness drives: INPUT COLOR TRANSFORM (IDT)

### B.1 — The gap (confirmed)

`tone.to_working(rgb_display, working_space)` (`grade/tone.py:59-70`) does **only**
the inverse sRGB/Rec.709 EOTF — it linearizes *display-encoded* rec709 and nothing
else. Its own docstring (`:13-17`) says this is deliberately "the slot": *"a fuller
ACES input transform (IDT) can replace it later."* So **log/flat/HLG footage is
currently treated as if it were display-encoded rec709.** Everything downstream
inherits this: Correct/Balance/Match/Leveling all solve on `to_working`-projected
scalars (`job.py::_to_working_scalar`, `correct._project`, `match._proj`), and the
bake linearizes with the same `to_working` (`lut_bake.py:117`).

**Impact assessment (be honest about magnitude):**
- For genuinely display-encoded footage (phone video, screen recordings, most
  YouTube-creator content — and the look catalog in `look_engine.py` is explicitly
  "YouTube-centric"), `to_working` is *correct*. For that majority corpus the IDT
  gap is **not** the dominant defect.
- For log/flat footage (prosumer cameras, "flat"/log picture profiles, S-Log/
  C-Log/V-Log/HLG), it's wrong three ways: (a) WB gray-world is computed on
  log-encoded values, so the neutral point is off; (b) the levels/contrast solve
  operates in a pseudo-linear space that isn't scene-linear; (c)
  `from_working`'s highlight shoulder (`_SHOULDER_START = 0.8`) assumes
  scene-linear input, so it engages at the wrong place. Today the *only*
  compensation is `correct.py`'s `is_log_flat` heuristic + `LOG_FLAT_PRE_LIFT`
  (a crude 1-D contrast lift), not a real decode.
- **Therefore the harness must decide whether IDT is a top-3 failure class before
  we invest in it.** The exposure/cast/consistency metrics on the subset of shots
  where `color_stats.is_log_flat == True` are exactly the evidence: if those shots
  cluster in WARN/FAIL, IDT is the fix; if `is_log_flat` shots are rare in the
  real corpus, IDT drops down the ranking and a different class leads.

### B.2 — What profile metadata we actually have

- **`color_stats.is_log_flat`** (`l1/color_stats.py:235-240`): a heuristic (low
  luma std + compressed range + minimal clipping). This is a *detector*, not a
  profile identifier — and its own comment warns it misfires on
  "dim-but-correct" footage (a dim podcast looks statistically like log). It's
  available on every measured file (fetched by `grade.measure.fetch_color_stats`,
  already in the `_COLS` list).
- **No true transfer-function metadata is decoded today.** `color_stats.py`
  decodes proxy frames with ffmpeg to raw `rgb24` and never inspects the
  container's `color_transfer`/`color_primaries`/`color_space` tags. **Step 0 of
  Part B is a read-only spike** (`_diag_qa_profile_probe.py`) that runs
  `ffprobe -show_streams` over the corpus's source files and tabulates
  `color_transfer` (e.g. `arib-std-b67`=HLG, `smpte2084`=PQ, `bt709`, `unknown`)
  and any vendor tags. This tells us whether real metadata is sufficient or we
  must lean on the heuristic.

### B.3 — Where the IDT hooks (buildable, scoped)

Fill the existing slot, don't rearchitect. The transform is chosen per file from
(1) ffprobe transfer tag when trustworthy, else (2) the `is_log_flat` heuristic
as the fallback log-detect:

- **Primary hook — `tone.to_working`:** add an input-transform selector keyed by
  a new `input_transform` argument (default `"rec709"` = today's exact behavior,
  byte-identical). New values decode a known log curve to scene-linear *before*
  the working-space normalization. Start with the two highest-value, safest
  cases and keep everything else identity:
  - `"rec709"` (default): unchanged inverse-sRGB (current code path).
  - `"hlg"` / `"pq"`: standard, well-specified inverse OETFs (only when ffprobe
    reports them — no heuristic guessing for these).
  - `"loglike"`: a **conservative generic log→lin decode** used only when
    `is_log_flat` is true AND no explicit tag exists — deliberately gentle
    (closer to a mild gamma expansion than a camera-exact IDT) so it stays
    *never-worse* even if the guess is wrong. This is the honest fallback: we are
    not claiming per-camera IDTs.
- **Selection lives in `job.py` (measure/correct boundary), not in the solvers.**
  `run_grade_job` already fetches `color_stats` per file and already owns the
  working-space projections. Add the per-file `input_transform` decision there
  and thread it through: (a) the `_to_working_scalar`/`_ws_stats` projections used
  to solve Balance/Match/Leveling, and (b) the `working_space`/new
  `input_transform` field carried on the resolved grade descriptor so
  `lut_bake.bake_cube_text` applies the *same* decode at bake time. This keeps the
  "solve in the space you apply" discipline the whole pipeline already follows.
- **Hash + bake plumbing:** `input_transform` must join the `grade_hash` payload
  (`cdl.grade_hash`) and `bake_cube_text`'s signature (alongside `working_space`),
  and `INPUT_HASH_SCHEMA_VERSION` bumps so every stored grade re-computes (same
  discipline as every prior grade-math change — see `job.py`'s version-history
  comment block). This is the "hash-bump + re-grade" dependency the loop calls out.

### B.4 — How the harness proves it

Run the scoreboard **before** (baseline) and **after** the IDT, filtered to the
`is_log_flat`/log-tagged subset:
- Expect: neutral_axis_deviation and skin_perp_residual **down** on log shots
  (WB now solved on decoded values), exposure_band better centered, and — the
  guardrail — **no regression** on the rec709-tagged majority (whose transform is
  unchanged/identity by construction). If the rec709 subset moves at all, the
  default path wasn't actually byte-identical → bug.

---

## Part C — The ITERATION LOOP (how to actually run it)

This is the repeatable Phase-1 cadence. Each pass is one grade-math change proven
by the scoreboard:

1. **Baseline.** `_diag_qa_corpus.py --regrade` → `_diag_qa_sample.py` →
   `_diag_qa_score.py`. Produces `scoreboard.json` + contact sheets. This is the
   reference point; commit the JSON to `_out/qa/baselines/<date>.json` (locally,
   not to git prod code).
2. **Rank failures by frequency × severity.** Read the global scoreboard's ranked
   failure classes. The top class is the next fix (Part B's IDT is the expected
   first, *if* the log subset ranks high — otherwise whatever leads).
3. **Fix the highest-impact class.** One change, scoped, in the real pipeline
   (`grade/…`), following the existing layer discipline.
4. **Re-run the harness.** Re-grade (the fix changed grade math → bump
   `INPUT_HASH_SCHEMA_VERSION`, re-run `run_grade_job` for all threads — the
   **hash-bump + re-grade dependency**; until re-graded, `fetch_latest_grades`
   serves stale rows, so the harness would score the *old* grade), re-sample,
   re-score.
5. **Compare — two questions, both required:**
   - Did the **targeted class** improve (its count×severity dropped)?
   - Did **anything previously PASS now FAIL**? (the harness is the regression
     guard — this is the check the tone-contrast S-curve regression lacked). A
     fix that improves class A but regresses class B is not accepted until net
     shots-passing rises.
6. **Repeat** until the scoreboard clears the parity bar.

### "PARITY BAR CLEARED" — concrete exit criteria

Phase-1 base-quality parity is declared when, across the whole corpus:
- **≥ 90%** of shots are PASS on **all** classes, and **0 FAIL** shots remain on
  the two never-worse classes: `crushed_black_fraction` and
  `clipped_highlight_fraction` (a professional grade must never crush/blow what
  the raw didn't).
- **Consistency:** ≥ 90% of scene groups PASS `intra_group_luma_std < 0.03` and
  `intra_group_chroma_std < 2.5`, and no group has GRADED std > RAW std (matching
  never makes a group *worse*).
- **Lighting-evenness:** ≥ 90% of multi-subject groups PASS
  `intra_group_subject_luma_std < 0.03`.
- **Skin:** ≥ 90% of subject-box shots PASS `skin_perp_residual < 6`.
- **No net regression** vs. the prior accepted baseline on any class.

(All numeric thresholds are provisional until calibrated against the human-labeled
set in Risks — the *structure* of the bar is the deliverable; the exact numbers
get locked after calibration.)

---

## Risks / open questions

1. **Threshold calibration.** Every band above is a starting guess. Before trusting
   the scoreboard, hand-label ~20–40 shots as "clearly good / clearly bad / borderline"
   (use the contact sheets), then tune each band so the metric verdict agrees with the
   human label on that set. Until calibrated, treat FAIL/WARN as *relative* ranking, not
   absolute truth.
2. **`is_log_flat` false positives.** It misfires on dim-but-correct footage
   (`color_stats.py:229-234` comment). If the IDT keys off it, a dim podcast could get a
   log decode it shouldn't. Mitigation: prefer ffprobe tags; make the `"loglike"` fallback
   deliberately gentle (never-worse); gate it with the levels-slope cap that already exists.
3. **subject_box coverage.** Skin + subject-luma metrics need the box. It's ~99.8% populated
   on cut_records per `scene_meta.py`, but a shot with no covering cut_record (verified real
   condition — the RGB-fallback path exists for exactly this) has no box → those metrics are
   N/A for it, not FAIL. Track box-coverage % as a harness health stat.
4. **Log-profile metadata availability.** Unknown until the `_diag_qa_profile_probe.py`
   ffprobe spike runs — proxies may have stripped/normalized transfer tags (proxies are
   transcoded), in which case we're heuristic-only and IDT confidence drops. Probe the
   *originals* (`files.r2_key`), not just proxies.
5. **Cube-only measurement blind spot.** Soft-local (vignette/halation/grain) isn't in the
   cube, so v1 metrics don't see it. Acceptable for base-quality (correction/consistency are
   cube-side); revisit for over-processing once looks with texture are common.
6. **Proxy resolution.** color_stats/proxies can be as low as 160×90 (`color_stats.py`
   docstring). Fine for luma/cast aggregates; banding/halo metrics need a higher-res still —
   sample those from the best available proxy/original.
7. **Corpus drift / cost.** Full-corpus decode is heavy. Cache proxies per run, cap
   frames/shot, support `--projects` subsetting; the scoreboard is cheap to recompute from
   cached stills.

---

## Ordered implementation checklist

1. `backend/scripts/qa/metrics.py` — pure metric functions (exposure, WB/cast,
   skin-on-box, consistency, over-processing, look-fidelity, exposure-evenness) with
   the A.4 bands as module constants. Unit-testable, no DB.
2. `backend/scripts/_diag_qa_corpus.py` — corpus manifest (reuse
   `_grade_all_projects.py`); `--regrade` flag.
3. `backend/scripts/_diag_qa_sample.py` — RAW/GRADED still sampling (reuse
   `_grade_v1_frames.py` + `ensure_cube_file` + `fetch_latest_grades` +
   `lookup_shot_cut_meta` for boxes/hero_ts + `frames.extract_still`).
4. `backend/scripts/_diag_qa_score.py` — per-shot/project/global scoring →
   `_out/qa/scoreboard.json`.
5. `backend/scripts/_diag_qa_sheets.py` — per-project contact sheets + global
   `index.html`.
6. **Run baseline** on the full corpus; hand-label ~20–40 shots; calibrate bands
   (Risk 1). Lock the parity-bar numbers.
7. `backend/scripts/_diag_qa_profile_probe.py` — ffprobe transfer-tag survey over
   source originals (Part B.2 spike).
8. **Part B IDT** — implement in `tone.to_working` (new `input_transform` arg,
   default byte-identical) + selection in `grade/job.py` + carry through
   `resolver`/`grade_hash`/`bake_cube_text`; bump `INPUT_HASH_SCHEMA_VERSION`.
9. Re-grade corpus; re-run harness; compare baseline vs post-IDT on the log subset
   AND confirm zero regression on the rec709 majority (Part B.4).
10. Enter the Part-C loop: rank → fix → re-grade → re-score → guard-against-regression,
    until the parity bar clears.
