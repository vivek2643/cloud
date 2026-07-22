# color_phase1.plan.md — Phase 1 base color grading: calibrate the QA harness, handle all footage (especially LOG), iterate to a tiered bar

## Goal (one paragraph)

Clear the "looks professional / trustworthy" base-color bar for the current production corpus, **proven by the QA harness scoreboard**, not by eyeballing. The core correctness requirement is that **every footage type the corpus actually contains is correctly understood and graded — especially LOG**. Phase 1 is three moves in strict order: (1) **calibrate** the already-built harness so it stops failing intentional looks, slide decks, and a single rigid exposure target (a metric that fails on-purpose content is worse than no metric); (2) fix the one **real, systematic correctness defect** the calibrated harness will still show — LOG footage is treated as display Rec.709, so fill the long-reserved input-transform (IDT) slot in `tone.to_working`, with a **hard requirement that it is exactly inert (byte-identical) on already-display-Rec.709 footage**; (3) run the **iteration loop** (baseline → rank failures → fix one class → re-run, using the harness as the regression guard) until the tiered parity bar is met. Everything here is grade-math + harness work; nothing about the LLM/brain.

---

## Status of prior work (already built — verify refs, do not rebuild)

- **Harness Part A** (commit `3487915`): pure metric functions in `backend/scripts/qa/metrics.py` (26 unit tests in `backend/scripts/test_qa_metrics.py`) + the read-only diagnostics `backend/scripts/_diag_qa_corpus.py`, `_diag_qa_sample.py`, `_diag_qa_score.py`, `_diag_qa_sheets.py`, `_diag_qa_profile_probe.py`. Output (gitignored) lands in `backend/scripts/_out/qa/` (`corpus.json`, `samples.json`, `scoreboard.json`, per-project contact sheets + `index.html`).
  - **NOTE — path correction**: the source-of-truth prompt refers to `backend/app/services/l3/grade/qa/metrics.py`; the metrics module actually lives at **`backend/scripts/qa/metrics.py`** (with `backend/scripts/qa/__init__.py`). All refs below use the real path. Do not create a second copy under `app/services`.
- **Grade pipeline standardization** (commit `3352623`): one un-flagged v1 path — Measure → Correct → Balance → Match → Leveling → Look → Arc → Soft-local → Bake — in `backend/app/services/l3/grade/resolver.py::resolve_clip_grade`; working space in `backend/app/services/l3/grade/tone.py`. `INPUT_HASH_SCHEMA_VERSION = 11` (`backend/app/services/l3/grade/job.py:98`).

## First-run findings (established — verify, don't re-derive)

- Overall pass **12%**, dominated by **miscalibration, not defects**.
  - `saturation_band` fails 54/83 — driven by a slide-deck project ("demo trail") + intentional mono/near-mono looks; the metric has **no raw baseline** (`saturation_band(graded01, raw01=None)` → cannot diff graded-vs-raw like every other metric).
  - `exposure_band` fails 38/83 against a single rigid center — `EXPOSURE_BAND_PASS = (0.30, 0.60)` centered near `TARGET_MID_GRAY = 0.42`; deliberately dark/bright shots fail.
  - Grade **improves** crushed-blacks on 46/83 shots; only 2 real grade-introduced crush fails (both an intentional high-contrast-mono look).
  - White balance / highlights / banding / chroma-consistency are clean. Matching mostly works (12/17 groups pass luma consistency).
- **LOG subset** (Siri Reel — the only log footage, ~12–17 shots) fails saturation 100% (0 pass) + heavy look-fidelity; its exposure is fine. Real, systematic signal.
- **Root cause for log**: `tone.to_working` (`backend/app/services/l3/grade/tone.py:59-70`) applies only the inverse sRGB/Rec.709 EOTF; its docstring calls itself "the slot" for a future IDT. Log/raw is currently handled by a crude `is_log_flat` heuristic + `LOG_FLAT_PRE_LIFT = 1.06` in `correct.py:55,282-295`. `is_log_flat` is computed in L1 `backend/app/services/l1/color_stats.py:235-240`. `_diag_qa_profile_probe.py` found **0/27 source originals carry a log/HLG/PQ transfer tag**, so detection stays heuristic, not metadata.

---

## The tiered parity bar (the finish line)

Three tiers. Each existing metric in `backend/scripts/qa/metrics.py` maps to exactly one. "Pass" is measured on the **calibrated** bands (Part 1), scored on the hero frame per shot / rolled up per scene group exactly as `_diag_qa_score.py` already does.

### Tier A — Structural / "broken": **100% pass, zero tolerance**
A shot that trips any of these is objectively broken; there is no acceptable count > 0.

| Metric (in `metrics.py`) | Function |
|---|---|
| `crushed_black_fraction` | `exposure_metrics` → `_band_upper(crushed, CRUSHED_BLACK_PASS, CRUSHED_BLACK_WARN)` |
| `clipped_highlight_fraction` | `exposure_metrics` |
| `neutral_axis_deviation` (severe cast) | `neutral_axis_deviation` |
| `exposure_band` **gross** end (deliberately-dark/bright still allowed; only *gross* mis-exposure fails — see 1b) | `exposure_metrics` |

Zero-tolerance applies to the **grade-introduced** case: a Tier-A fail is only a real failure when the grade *caused or worsened* it. Use the raw baseline already carried in `crushed_black_fraction.extra["delta"]` / `clipped_highlight_fraction.extra["delta"]` — a shot whose raw was already crushed and whose grade did not worsen it (`delta <= 0`) is **not** counted against Tier A (that is a source defect, out of scope). This is the "never make it worse" reading of zero-tolerance, and it is exactly why the raw baseline must exist on every structural metric.

### Tier B — Quality: **≥ 95% pass**
| Metric | Function |
|---|---|
| `saturation_band` | `saturation_band` (after 2a: raw baseline + look-aware exemption) |
| `skin_perp_residual` | `skin_perp_residual` |
| `intra_group_luma_std` / `intra_group_chroma_std` / `intra_group_subject_luma_std` (scene consistency) | `group_consistency_metrics`, `group_subject_exposure_metrics` |
| `look_fidelity_cosine` | `look_fidelity_metric` |
| `banding_score` | `banding_score` (WARN-only; never a hard fail — contact sheet is the judge) |

### Tier C — Consistency vs raw: **better than raw on 100% of scenes**
The grade must **never make a scene worse than ungraded**. Every group metric already carries an `improved`/`convergence_delta` field comparing graded-vs-raw:
- `group_consistency_metrics` → `intra_group_luma_std.extra["improved"]`, `intra_group_chroma_std.extra["improved"]`.
- `group_subject_exposure_metrics` → `intra_group_subject_luma_std.extra["convergence_delta"]` (raw_std − graded_std; must be ≥ 0).

Tier C requires `improved == True` (and `convergence_delta >= 0`) for **100% of multi-member groups**. Add these as explicit scoreboard assertions (Part 3).

> Implementation note for the scoreboard: extend `_diag_qa_score.py` to emit a per-tier rollup (`tier_a`, `tier_b`, `tier_c` with pass counts / required threshold / met?) alongside the existing `overall_pass_pct`, so "did we clear the bar" is a single glance, and add the tier to each metric's record. The tier→metric map above is the single source of truth.

---

## Part 1 — Calibrate the harness (DO THIS FIRST, before any grade-math change)

A metric that fails intentional looks or slide decks is worse than none. All four fixes are in `backend/scripts/qa/metrics.py` + `_diag_qa_score.py` + `_diag_qa_sample.py` — **no production grade code changes in Part 1**.

### 1a. Saturation metric — add a raw baseline + make it look-aware

**Problem.** `saturation_band` today (`metrics.py:371-387`) bands the graded frame's absolute mean Lab chroma against `SATURATION_PASS = (12.0, 40.0)` / `SATURATION_WARN = (8.0, 55.0)`. When `raw01 is None` it cannot diff graded-vs-raw. It also has no notion that a look *intends* to be desaturated/mono, so `moody_teal` (`sat=0.82`), `eterna` (`sat=0.88`), `bleach_fade` (`sat=0.80`), and especially `graphic_bw` (`sat=0.05`) — all in `look_engine.py` — fail as "under-saturated" when they are working exactly as designed.

**Fix — two parts:**

1. **Raw baseline (already plumbed, verify wiring).** `_diag_qa_score.py:104` already calls `m.saturation_band(graded01, raw01=raw01)`, and `saturation_band` already computes `extra["raw_chroma_mean"]` + `extra["chroma_increase_ratio"]` and force-fails on `ratio > CHROMA_INCREASE_FAIL_RATIO = 2.0`. Confirm this path is exercised (it is — `raw01` is always loaded in `score_shot`). Change the **verdict basis** so, like every other metric, saturation is judged **relative to raw** rather than only against absolute bands: keep the absolute band as a coarse guard, but the *primary* graded-vs-raw signal is the chroma ratio. Concretely, replace the single absolute-band verdict with: PASS when the chroma ratio sits inside a sane multiplicative window around the look's *intended* multiplier (see next), WARN/FAIL as the ratio strays. Absolute over/under-saturation (chroma way outside `SATURATION_WARN`) stays a backstop.

2. **Look-aware exemption (auto, no manual look list).** The look's own saturation intent is fully specified by its `LookSpec` (`backend/app/services/l3/grade/look_engine.py:42-64`): the global `sat` field plus per-hue `hue_sat` bands. The shot's active look is already available on each sample record as `shot["look_engine"]` (`_diag_qa_sample.py:239`, carried into `score_shot`). Compute the look's **intended global saturation factor** from its `LookSpec.sat` (and, when present, a coverage-weighted contribution from `hue_sat` mults) and use it to set the *expected* graded/raw chroma ratio for this shot. Then:
   - A look declaring `sat < ~0.35` (e.g. `graphic_bw` at 0.05) is an **intentional desaturation** → the low-chroma FAIL is exempt (verdict `na` for the "under-saturated" direction; still guard the *over*-saturated direction and the `> 2.0` ratio backstop).
   - A look declaring `sat` in a mild range → the pass window recenters on `raw_chroma * intended_sat` instead of the fixed `(12, 40)` band.
   - No active look (`look_engine is None`) → judge against raw with the default window (the "correction only, no creative desat" case).

   Implement this as a helper in `metrics.py`, e.g. `def look_saturation_intent(look_engine: Optional[dict]) -> Optional[float]` returning the intended global factor (`None` when no look), and extend `saturation_band(graded01, raw01, *, look_intent: Optional[float] = None)`. `_diag_qa_score.py` passes `look_intent=m.look_saturation_intent(shot.get("look_engine"))`. This keys off the look's **own spec**, so any future look is handled automatically — no hand-maintained "these looks are mono" list.

   Add unit tests to `test_qa_metrics.py`: (i) a mono look exempts a low-chroma graded frame; (ii) an over-saturated grade (`ratio > 2.0`) still FAILs even under a mild look; (iii) `look_intent=None` reproduces today's band behavior on a neutral correction.

### 1b. Exposure metric — content-aware band, not a single rigid target

**Problem.** `exposure_metrics` (`metrics.py:167-200`) bands both mean and median luma against `EXPOSURE_BAND_PASS = (0.30, 0.60)` / `EXPOSURE_BAND_WARN = (0.22, 0.72)`, a single fixed window near `TARGET_MID_GRAY = 0.42`. A deliberately dark (night, moody) or bright (high-key) shot fails even when graded correctly.

**Fix.** Replace the fixed absolute band with a **content-aware** judgment that only fails *gross* mis-exposure (Tier A) and *never* fails legitimately-dark/bright content:
- **Grade-relative core**: judge the graded exposure against the **raw** exposure (raw luma mean/median is available — pass `raw01`, already loaded). The grade should not push a shot far from its own content-appropriate exposure unless it was correcting a genuine under/over. So the primary check becomes: did the grade move mean luma *toward* a plausible mid-range from a genuinely off raw (good), or *away* from a fine raw (bad)?
- **Absolute backstops only at the extremes** (Tier A gross mis-exposure): keep a very wide absolute FAIL band (e.g. mean luma `< ~0.06` = crushed-dark whole frame, or `> ~0.92` = blown whole frame) so a truly broken exposure still trips Tier A, but the normal 0.30–0.60 window becomes a *soft* "typical" reference, not a hard fail.
- Keep `crushed_black_fraction` / `clipped_highlight_fraction` as the precise structural signals (they already carry raw deltas); `exposure_band` becomes the "overall placement sane / not gross" signal.

Concretely: split the current `exposure_band` verdict into (a) a wide gross-mis-exposure FAIL band (Tier A) and (b) a WARN-only "outside typical" band, and add an `extra["raw_mean"]`/`extra["raw_median"]`/`extra["delta_mean"]` set so the calibration in 1d has the raw reference to tune on. The exact numeric bands are **outputs of the 1d hand-labeling**, not guessed here.

### 1c. Synthetic-content exemption (baseline only) — swappable lookup, no classifier

**Problem.** One project — "demo trail", a slide-deck screen recording — is synthetic, not photographic. Scoring it with photographic metrics (saturation, skin, exposure) is meaningless and dominates `saturation_band`'s failures.

**Fix.** Add a small, **swappable** content-source interface with exactly one method, whose only implementation today is a **hand-maintained map**:

```python
# backend/scripts/qa/content_source.py  (new, harness-only)
def content_type_for(project_id: str, project_label: str) -> str:
    """Return "photographic" (default) or "synthetic". Today this reads a
    hand-maintained map keyed by project label/id (the demo-trail slide deck
    is the only "synthetic" entry). This is the SWAPPABLE seam: a later
    VLM-Pass-2 `content_type` field can replace the map body without changing
    this signature or any caller. DO NOT build the VLM path in Phase 1."""
```

- The map is the single hardcoded entry today: the demo-trail project → `"synthetic"`; everything else → `"photographic"`.
- `_diag_qa_score.py` calls `content_type_for(...)` per shot; when `"synthetic"`, the photographic quality metrics (`saturation_band`, `skin_perp_residual`, `exposure_band`) return/record `na` (excluded from Tier scoring) while structural sanity can still be reported for information. Record `content_type` on each shot record in the scoreboard.
- **State clearly in the code + this plan**: **LOG stays signal-derived** (`is_log_flat` from `color_stats`) — it is a *decode* decision, always heuristic. **Synthetic detection is the future VLM-Pass-2 job** — the map is a stopgap that the VLM `content_type` field will replace behind this exact interface. NO classifier is built in Phase 1.

### 1d. Hand-labeling artifact + held-out validation (user does the labeling)

The bands above (saturation window, exposure gross/typical thresholds, skin/consistency PASS/WARN) are calibrated to **human labels**, then validated on a held-out subset so they generalize past these 13 projects (production-grade, not overfit).

**Artifact — a labeling page.** Add a new read-only diagnostic `backend/scripts/_diag_qa_label_sheet.py` that:
- Selects **~20–30 representative shots** spanning: the LOG subset (Siri Reel), a spread of Rec.709 shots (talking-head, indoor/dim, bright/outdoor), and a couple of intentional-look shots (at least one mono/desaturated look and one warm/saturated look). Deliberately include the demo-trail synthetic project as its own labeled row so the exemption is validated.
- For each selected shot, emits a **raw vs graded** still pair (reuse the already-sampled `hero` frame paths from `samples.json`; do not re-extract) plus the current metric readouts, into a single `backend/scripts/_out/qa/label_sheet.html` with a compact form: each row = raw thumbnail | graded thumbnail | metric values | a **good/bad** radio (+ optional "intentional look?" checkbox). The page writes/exports the user's labels to `backend/scripts/_out/qa/labels.json` (a simple `shot_key → {verdict: good|bad, intentional: bool}` map; the page can offer a "copy JSON" button, or the user hand-edits a pre-seeded `labels.json`).
- **Split**: deterministically partition the labeled shots into a **calibration set (~70%)** and a **held-out validation set (~30%)**, seeded by a fixed hash of `shot_key` so the split is reproducible and the validation shots are *never* used to pick thresholds. Record which shots are in which set in `labels.json`.

**Calibration procedure** (a second diagnostic, `backend/scripts/_diag_qa_calibrate.py`, read-only, writes proposed constants):
1. Load `labels.json` + `samples.json` + per-shot metric values (re-run metrics or read `scoreboard.json`).
2. On the **calibration set only**, for each tunable band (saturation window vs `look_intent`/raw ratio; exposure gross + typical thresholds; `SKIN_PERP_*`; `GROUP_*_STD_*`), choose PASS/WARN/FAIL cut points that best separate human good/bad (maximize agreement; prefer the threshold that yields **zero false-PASS on human-bad Tier-A shots** first, then best overall agreement).
3. **Validate on the held-out set**: report agreement between calibrated verdicts and human labels on shots that were *not* tuned. Require held-out agreement ≥ a stated target (e.g. ≥ 90%) before adopting; if it fails, the bands are overfit — widen/simplify and re-check. Print a confusion summary per metric.
4. Emit the proposed constants (the `*_PASS`/`*_WARN` values in `metrics.py`) as a diff-ready block for a human to paste into `metrics.py`. Calibration **proposes**; a person commits the constant change (keeps the metric module the single source of truth, and keeps calibration reproducible/inspectable).

**Deliverable of Part 1**: a **TRUE baseline** scoreboard — the same corpus re-scored with calibrated, look-aware, synthetic-exempt metrics — which is the honest starting point for Parts 2–3. Expect the headline pass rate to jump sharply (most of the 12% was miscalibration), surfacing the small real defect pool.

---

## Part 2 — LOG / input-transform handling (the real correctness fix)

**"Understand the current footage to grade it."** Fill the IDT slot in `tone.to_working` so Correct / Balance / Match / Leveling operate on **properly-decoded** footage instead of a crude 1.06× lift.

### 2.1 Where it hooks

- **Decode lives in `tone.to_working`** (`backend/app/services/l3/grade/tone.py:59-70`) — the documented "slot." Today `to_working(rgb_display, working_space)` returns identity unless `working_space == WORKING_SPACE_V1 ("rec709_v1")`, in which case it applies the inverse sRGB/Rec.709 EOTF.
- **Selection is a per-clip working space.** Introduce a new working-space value, `WORKING_SPACE_LOG_V1` (e.g. `"log_v1"`), added in `tone.py`. `to_working` gains a branch: for `WORKING_SPACE_LOG_V1`, apply a proper **log→scene-linear decode** (a documented, single, well-behaved log curve — a generic log/"flat" inverse that lifts the compressed mid-range into linear, decode + a normalization so the linearized range maps back into 0..1) instead of the sRGB EOTF. `from_working` handles `WORKING_SPACE_LOG_V1` by re-encoding to **display Rec.709** exactly as it does for `rec709_v1` (the output is always display Rec.709; only the *input decode* differs). This keeps the "CDL solved and applied in the working space, baked between to_working/from_working" contract intact (`correct._project`, `lut_bake.bake_cube_text:117-119`).
- **Detection lives in the resolver.** `resolve_clip_grade` (`resolver.py:199-283`) already computes `working_space = item.get("working_space") or WORKING_SPACE_V1` (`:283`) and already receives `color_stats`. Select the log working space there:

  ```
  is_log = bool((color_stats or {}).get("is_log_flat")) or _transfer_tag_is_log(color_stats)
  working_space = item.get("working_space") or (WORKING_SPACE_LOG_V1 if is_log else WORKING_SPACE_V1)
  ```

  `_transfer_tag_is_log` reads any ffprobe `color_transfer` tag when present (`arib-std-b67`=HLG, `smpte2084`=PQ, known log curves) — but per `_diag_qa_profile_probe.py`'s finding (0/27 tagged), this is an *additive* signal; `is_log_flat` remains the workhorse. If a transfer tag is to be persisted, add it to `color_stats` (`backend/app/services/l1/color_stats.py`) as a nullable field and thread it via `measure.fetch_color_stats`; otherwise key purely off `is_log_flat` for Phase 1 and note the tag as an optional refinement.

- **Retire / neutralize the crude lift.** `correct.py`'s `LOG_FLAT_PRE_LIFT = 1.06` block (`:282-295`) was a stand-in for a real decode. Once the log working space decodes properly, the pre-lift must not double-apply. Decide explicitly: gate the pre-lift OFF when `working_space == WORKING_SPACE_LOG_V1` (the decode replaces it), keeping it only as a fallback if a real decode is *not* selected. Document this in `correct.solve_correct_grade` and pass the chosen `working_space` through so the branch is unambiguous.

### 2.2 HARD REQUIREMENT — inert on Rec.709

The input transform must be **exactly identity (byte-identical)** on already-display-Rec.709 footage:
- Non-log footage keeps `working_space == WORKING_SPACE_V1`; `to_working`/`from_working` behavior for `rec709_v1` is **unchanged** (do not touch those code paths). The Rec.709 majority's `grade_hash` payload is unchanged for those clips → same cache key → same baked bytes.
- Add a test asserting `to_working(x, WORKING_SPACE_V1)` and `from_working(y, WORKING_SPACE_V1)` are bit-for-bit identical to the current implementation (golden array), and that a full `resolve_clip_grade` on a non-log `color_stats` produces the **same `grade_hash`** as before this change. This is the regression contract.

### 2.3 Hash + bake threading

- `working_space` is already part of the `grade_hash` payload (`cdl.grade_hash:146,168`) and already flows into `bake_cube_text(..., working_space=...)` (`lut_bake.py:96,117-119`) and the resolver's returned descriptor (`resolver.py:283,352`). So a per-clip `WORKING_SPACE_LOG_V1` **automatically** produces a distinct cache key and a distinct baked cube for log clips, and leaves non-log clips' hashes untouched. No new hash field is required for the working-space selection itself.
- Bump `INPUT_HASH_SCHEMA_VERSION` (`job.py:98`, currently `11` → `12`) because the log clips' grade math changes; add the one-line comment in the version history block (`job.py:80-97`). This forces the one-time re-grade (Part 3 ops step). Non-log clips re-grade to identical bytes (the inert requirement), so only log clips actually change.
- If the ffprobe transfer tag is persisted into `color_stats`, also bump `color_stats.SCHEMA_VERSION` (`backend/app/services/l1/color_stats.py:53`, currently `3`) so measurement rows recompute; skip this if keying purely off the existing `is_log_flat`.

### 2.4 Before/after measurement (the proof)

Using the **calibrated** harness (Part 1) as the judge:
1. Capture the calibrated TRUE baseline scoreboard (pre-Part-2).
2. Implement the log working space; bump the schema; re-grade all threads (`_grade_all_projects.py`).
3. Re-run the full harness (`_diag_qa_corpus.py` → `_diag_qa_sample.py` → `_diag_qa_score.py`), and diff scoreboards.
   - `_diag_qa_score.py` already emits a `log_flat`-vs-`rec709` split (`failure_classes_ranked_log_flat` / `_rec709`). Require: **the LOG subset (Siri Reel) improves** on saturation + look-fidelity, and **zero Rec.709 regression** — every Rec.709 shot that previously PASSed still PASSes (the harness is the regression guard). A convenient check: assert the `rec709` shot verdicts and `grade_hash`es are unchanged for non-log clips.
   - Use `_diag_qa_sample.py --projects "siri"` for a fast, log-only iteration loop before the full re-run.

---

## Part 3 — The iteration loop + true-baseline triage + hash/re-grade dependency

Once the harness is calibrated (Part 1) and log is decoded (Part 2), the residual defect pool is small. Fix it via a disciplined loop, with the harness as the regression guard.

### 3.1 The loop
1. **Baseline**: run the full harness, capture `scoreboard.json` + tier rollup.
2. **Rank** failures by frequency × severity — `_diag_qa_score.py`'s `failure_classes_ranked` (`rank_score = fail*2 + warn`) already does this; use it worst-first.
3. **Fix one class** in the relevant production module (grade math), smallest safe change.
4. **Re-run** the harness (`_diag_qa_corpus.py` → `_diag_qa_sample.py` → `_diag_qa_score.py`; scope with `--projects` first for speed, then full).
5. **Confirm** the targeted class improved **AND nothing previously-passing regressed** (diff the per-shot verdicts / tier rollup; the harness is the guard). If anything regressed, revise or revert.
6. **Repeat** until all three tiers are met (Tier A 100%, Tier B ≥95%, Tier C 100%).

### 3.2 Likely true-baseline defect pool (triage after Part 1 recalibration)
- **Intra-scene consistency** on a few groups — `intra_group_luma_std` / `intra_group_chroma_std` (Balance/Match convergence). Fix in `backend/app/services/l3/grade/balance.py` / `match.py` / `reference.py`; Tier C requires `improved == True` for all groups.
- **Look fidelity** — `look_fidelity_cosine` low on some looks (compositing/order, not taste). Fix in the Look layer / `look_engine.py` grid or the resolver's compose order.
- **Skin residual** on the subject box — `skin_perp_residual` (`correct._skin_multiplier`, `SKIN_LOCUS_DEG`/`SKIN_TINT_STRENGTH`).
Verify each is real (post-calibration) before acting; do not chase WARN-only `banding_score`.

### 3.3 Hash / re-grade dependency (MANDATORY whenever a fix changes grade math)
Any change to grade math (Parts 2–3, not Part-1 harness-only work) **must**:
1. Bump `INPUT_HASH_SCHEMA_VERSION` (`backend/app/services/l3/grade/job.py:98`) by 1 and add a one-line note in the version-history comment block (`:80-97`). If the baked-cube *bytes* change for reasons outside the CDL/working-space payload, also bump `grade_hash`'s `schema_version` (`cdl.grade_hash:149`, currently `3`).
2. Re-grade every thread: `PYTHONPATH=. backend/.venv/bin/python backend/scripts/_grade_all_projects.py` (idempotent; required because the schema bump makes every stored grade stale).
3. Re-run the harness against the freshly-graded corpus. The harness reads each shot's **actual baked `.cube`** via `ensure_cube_file` + ffmpeg `lut3d` (`_diag_qa_sample.py:207,220`) — it scores exactly what preview/export produce, so a stale re-grade would silently invalidate the scoreboard. Order is always: bump → re-grade → re-sample → re-score.

---

## Deferred / Later (explicitly out of Phase 1)
- **Per-clip user-facing "disable/delete grade" toggle** — not built in Phase 1.
- **A real content-type classifier** — Phase 1 uses only the hand-maintained swappable map (1c). No classifier.
- **VLM-Pass-2 `content_type` field** — referenced *only* as the future data source that will replace the 1c map body behind the `content_type_for(...)` interface. Do not build the VLM path now.
- **Multi-frame robustness scoring** — the harness samples ~25/50/75% keyframes (`_diag_qa_sample.py`) but v1 scores the hero frame only (`_diag_qa_score.py`); averaging across frames is a later refinement.
- **Persisting/keying off real transfer-function metadata** at scale — Phase 1 stays heuristic (`is_log_flat`); the ffprobe tag is at most an additive signal (0/27 tagged in this corpus).

---

## Risks / open questions
1. **Threshold calibration is the crux.** The bands in `metrics.py` are admittedly "starting guesses" (module docstring, `:40-41`). If the hand-labeled set is too small or unrepresentative, calibrated bands won't generalize — mitigated by the held-out validation split (1d): adopt bands only if held-out agreement clears the target; otherwise widen/simplify. Document the final bands + their held-out agreement in the commit.
2. **`is_log_flat` false positives.** A dim-but-correct indoor shot (black≈0.15, white≈0.65) is statistically near-indistinguishable from true log (`color_stats.py:229-234` comment; `correct.py:56-64`). Applying a full log decode to such a shot would *over*-lift it. Mitigations: (a) keep the log decode conservative/normalized so a misfire is bounded, never crush/blow; (b) the calibrated harness is the guard — a false-positive log decode that worsens a Rec.709 shot must trip Tier C (worse-than-raw) or a Tier-A/B regression and block adoption; (c) include a couple of dim-but-correct shots in the 1d label set so the risk is measured, not assumed.
3. **Heuristic-only log detection.** 0/27 originals carry a transfer tag, so there is no metadata ground truth for "is this log." Detection is `is_log_flat` + optional tag. Accept this for Phase 1; the inert-on-Rec.709 requirement + Tier C bound the downside of a wrong guess.
4. **`subject_box` coverage.** `skin_perp_residual` / `intra_group_subject_luma_std` are `na` when no subject box resolves (`metrics.py:230-238`, `crop_box`), and Phase-1 callers largely don't thread `subject_box` through (`resolver.py:307` note). So skin/subject-exposure Tier-B coverage may be thin — report *coverage* (how many shots scored vs `na`) alongside pass rate so a high pass% on a tiny denominator isn't mistaken for a cleared bar.
5. **Log decode curve choice.** Which log curve to invert when the source is untagged (generic "flat" inverse vs a specific vendor curve like S-Log/V-Log/C-Log). Phase 1 uses one documented generic log-decode tuned so the Siri Reel subset improves without Rec.709 regression; a per-vendor IDT keyed on a real transfer tag is a later refinement (deferred).

---

## Ordered implementation checklist
1. **Part 1a** — `metrics.py`: add `look_saturation_intent(look_engine)` + extend `saturation_band(..., look_intent=)` (raw baseline verdict + auto mono/desat exemption from `LookSpec.sat`/`hue_sat`); wire `look_intent` in `_diag_qa_score.py`; add tests to `test_qa_metrics.py`.
2. **Part 1b** — `metrics.py`: rework `exposure_metrics`/`exposure_band` to content-aware (wide gross-mis-exposure FAIL + WARN-only typical band + raw deltas); add tests.
3. **Part 1c** — new `backend/scripts/qa/content_source.py` with `content_type_for(...)` (hand-maintained map, demo-trail → synthetic); wire into `_diag_qa_score.py` to `na` photographic metrics on synthetic shots; record `content_type` on scoreboard.
4. **Part 1 tiering** — extend `_diag_qa_score.py` to emit per-tier rollup (A/B/C) using the tier→metric map; add Tier-C `improved`/`convergence_delta` assertions.
5. **Part 1d** — new `_diag_qa_label_sheet.py` (label page → `labels.json`, ~20–30 shots spanning log/Rec.709/looks + demo-trail, seeded 70/30 calibration/held-out split); **user labels good/bad**.
6. **Part 1d calibrate** — new `_diag_qa_calibrate.py` (fit bands on calibration set, validate on held-out, emit diff-ready constants); a human pastes adopted constants into `metrics.py`.
7. **Capture TRUE baseline** — re-run `_diag_qa_corpus.py` → `_diag_qa_sample.py` → `_diag_qa_score.py`; save `scoreboard.json` as the calibrated baseline.
8. **Part 2** — `tone.py`: add `WORKING_SPACE_LOG_V1` + log-decode branch in `to_working` (Rec.709 display re-encode in `from_working`); keep `rec709_v1` byte-identical (golden test).
9. **Part 2** — `resolver.py`: select `WORKING_SPACE_LOG_V1` when `is_log_flat` (+ optional transfer tag); `correct.py`: gate `LOG_FLAT_PRE_LIFT` OFF under the log working space; add inert-Rec.709 hash-unchanged test.
10. **Part 2 hash/regrade** — bump `INPUT_HASH_SCHEMA_VERSION` 11 → 12 (`job.py:98`) + comment; run `_grade_all_projects.py`.
11. **Part 2 proof** — re-run harness; assert Siri Reel (log) improves on saturation + look-fidelity **and** zero Rec.709 regression (verdicts + non-log `grade_hash`es unchanged). Iterate log-only via `--projects "siri"` first.
12. **Part 3 loop** — from the calibrated post-Part-2 scoreboard, rank worst-first, fix one class (consistency → look-fidelity → skin), re-run, confirm target improved + nothing regressed; bump schema + re-grade + re-run on every grade-math change; repeat until Tier A = 100%, Tier B ≥ 95%, Tier C = 100%.
