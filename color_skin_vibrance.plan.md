# Skin-Anchored Tint Correction + Vibrance Normalization (v1 Correct layer)

> Implementation plan. Self-contained: an implementer with **no other context**
> should be able to build this from this file alone. All file:line refs were
> verified against the real code on the date this was written (see "Line-ref
> drift" at the end). **Both features live entirely inside the Correct layer
> (`grade/correct.py`)** — they add a skin-derived tint vote to the existing
> white-balance solve and a bounded global saturation nudge, folded into the
> `Grade` that `solve_correct_grade` already returns. No new stage, no new
> compose site, no schema/table migration.

---

## 1. Goal & non-goals

**Goal.** Make auto-corrected footage look *finished*, not merely *exposed* —
the two things a human notices instantly on people content:
1. **Skin never reads green/magenta.** Correct the off-locus (tint) cast on
   skin that mixed/fluorescent lighting produces, **without** privileging any
   skin tone. This is the fairness-safe half of the skin-WB the Correct layer
   deliberately left out (`correct.py` L10–16).
2. **Flat/desaturated footage gets its color back.** A bounded, never-worse
   global saturation lift toward a target chroma — raising the floor on lifeless
   log/flat clips, leaving already-vivid footage untouched.

**Why this and not more convergence math.** Balance, Match, and Subject Exposure
all converge strongly on synthetic fixtures but move ~9.5% on real reels because
the test footage lacks multi-shot-same-subject/same-scene structure (limited
*opportunity*, not a bug). These two features fire **per shot**, independent of
group structure, so they improve every clip regardless of the reel's shape —
they're the perceived-quality lever the consistency machinery isn't.

**Non-goals.**
- **No skin-tone targeting.** We never push skin toward a reference hue,
  lightness, or saturation. We only remove the component of skin's chroma that
  is perpendicular to the universal skin locus (the green↔magenta tint axis) —
  which is never natural on skin regardless of tone. Legitimate warm/cool
  variation and lightness are preserved by construction.
- **No per-pixel / local saturation (true "vibrance").** The CDL exposes only a
  single global `sat` scalar (`cdl.py` L34, applied L134–137). "Vibrance" here
  is a *bounded global saturation nudge*, skin-protected by a tight cap, not a
  per-pixel skin-masked operation.
- **No face detector.** Skin comes from `color_stats.skin_lab` (center-weighted
  proxy, `color_stats.py` L232–248) and — when a subject box is already resolved
  for `grade_subject_exposure` — from a real face region via a new `subject_lab`
  on `measure_span`. Neither adds a detector run.
- **Legacy pipeline stays byte-identical.** v1-only, gated behind a new flag
  defaulting **off**.
- **Never-worse is mandatory.** No skin sample / low confidence → no skin vote
  (gray-world/white-patch behave exactly as today). Chroma at/above target →
  no saturation change (we only *raise* the floor, never desaturate).

---

## 2. How this differs from what's shipped

`correct.py::_solve_wb` (L93–114) today balances white on **gray-world +
white-patch** (+ an optional verified `white_reference`), and *deliberately*
omits skin (L10–16: fairness risk of a hardcoded skin target). It also never
touches saturation — `solve_correct_grade` returns `sat=1.0` hardcoded (L223).

This plan adds the fairness-safe skin signal that omission was waiting for
(correct only the off-locus tint, never a target), plus the missing saturation
floor — both inside the same solve, both bounded by the existing never-worse
discipline and the resolver's composite guardrails (`resolver.py` L108–135).

---

## 3. What already exists (wire the front of this chain)

| Piece | Location | What it gives us |
|---|---|---|
| Skin sample (proxy) | `l1/color_stats.py` `_aggregate` L232–248 → `ColorStats.skin_lab` L85, `to_dict` L105 | `[L*, a*, b*]` center-weighted region, OpenCV-Lab-derived. Present on both whole-file `color_stats` and per-span `measure_span` (which reuses `_aggregate`, `measure_span.py` L167). |
| Whole-frame cast | `color_stats.py` L190–194 → `lab_ab_cast` L79 | `[a*, b*]` mean over the frame — the reference "where is the whole image, not just skin." |
| WB solve | `correct.py::_solve_wb` L93–114 | Returns `(wb_r, wb_g, wb_b)` multipliers folded into the luma slope (L221). **The skin tint vote folds in here.** |
| Global saturation | `cdl.py::Grade.sat` L34, `apply_cdl` L134–137, `compose` carries it L108 | A `Grade(sat=…)` desaturates/saturates toward Rec.709 luma. **Vibrance sets this.** Under v1 it runs in linear working space (lut_bake: `to_working → apply_cdl → from_working`). |
| Exact sRGB↔Lab | `grade/colorspace.py` `srgb_to_lab` L19–30, `is_neutral` L33–38 | Forward transform + neutrality test. **We add the inverse `lab_to_srgb`** to turn a corrected skin Lab target back into a per-channel RGB multiplier. |
| Subject box (optional) | `job.py` L413–416 (`subject_boxes`), `measure_span` `subject_box` arg L112 + `_measure_subject_luma` L90–107 | When `grade_subject_exposure` is on, a real face box is already resolved per shot. **We add `subject_lab` alongside `subject_luma`** so skin can be metered on the face, not the center proxy. |

**The correct layer already receives per-span stats:** `job.py` L636–641 calls
`resolve_clip_grade(s.item, color_stats=stats, …, pipeline="v1")` where `stats`
is the shot's `measure_span` result (L635) — so `skin_lab`/`chroma_mean` are the
*span's*, not the whole file's, with zero extra plumbing.

---

## 4. Design

### 4.1 New measurement: `chroma_mean` (L1 + span)

`_aggregate` already builds `lab_stacked` (`color_stats.py` L190–194) with a*/b*
per pixel. Add, right after the `a_mean`/`b_mean` block:

```python
a = lab_stacked[..., 1] - 128.0
b = lab_stacked[..., 2] - 128.0
chroma_mean = float(np.sqrt(a * a + b * b).mean())   # mean per-pixel Lab chroma
```

- Add `chroma_mean: float = 0.0` to `ColorStats` (near L86) and to `to_dict`
  (near L106).
- **Bump `color_stats.py` `SCHEMA_VERSION` 2 → 3** (L53) — recomputes cached L1
  rows so the new field lands.
- **Bump `measure_span.py` `SCHEMA_VERSION` 1 → 2** (L32) — same, for `cut_color_stats`.
- **Fail-open:** every consumer treats a missing `chroma_mean` (older row not
  yet re-measured) as "no vibrance data" → identity. Never a crash, never worse.

### 4.2 New measurement: `subject_lab` (span, face-region skin — optional)

In `measure_span.py`, mirror `_measure_subject_luma` (L90–107) with a Lab variant
`_measure_subject_lab(hero_frame, subject_box) -> Optional[[L*,a*,b*]]`
(cv2 `COLOR_RGB2LAB` on the crop, same OpenCV-Lab unpacking as
`color_stats.py` L242–246). In `measure_span` (L168–171), when `subject_box` and
`hero_frame` resolve, also set `stats["subject_lab"]`. This is the **real face**
skin sample; `skin_lab` (center proxy) is the fallback. No cache-key change (the
partial-miss recompute at L139 already covers a box requested but not yet stored).

### 4.3 Skin-anchored tint correction (fairness-safe)

**Signal precedence:** `subject_lab` (face region, when present) → else
`skin_lab` (center proxy) → else no skin vote.

**The locus principle.** Human skin across *all* tones clusters along a line
through the origin in the Lab a*/b* plane at a characteristic warm hue angle
(`+a*` red, `+b*` yellow). What varies legitimately between people and lighting
is the position **along** that line (saturation) and `L*` (lightness). What is
**never** natural on skin is displacement **perpendicular** to it — that
perpendicular axis is the green↔magenta tint a bad/mixed white balance adds.
So we correct only the perpendicular residual and leave along-locus + `L*` alone.
Because we never touch position along the locus or lightness, the correction is
identical in spirit for a very dark and a very light face — no tone is
privileged.

```
SKIN_LOCUS_DEG = 50.0        # skin-locus hue angle in Lab a*/b* (documented, tunable)
SKIN_TINT_STRENGTH = 0.7     # remove this fraction of the perpendicular (tint) residual
SKIN_WB_WEIGHT = 0.5         # skin gets a vote in WB, not a veto (blended with gray-world)
SKIN_L_MIN, SKIN_L_MAX = 20.0, 92.0   # plausible skin lightness; outside -> not skin, skip
SKIN_MIN_CHROMA = 3.0        # near-neutral sample -> not a confident skin read, skip
SKIN_MAX_PERP = 25.0         # residual bigger than this -> not skin (colored object), skip
```

Steps (all pure, in `correct.py`):
1. Read skin `(L*, a*, b*)`. Gate: `SKIN_L_MIN ≤ L* ≤ SKIN_L_MAX` and
   `sqrt(a*²+b*²) ≥ SKIN_MIN_CHROMA`, else return no skin multiplier (identity vote).
2. Decompose `(a*, b*)` about the locus angle `θ = radians(SKIN_LOCUS_DEG)`:
   - along `r∥ = a*·cosθ + b*·sinθ` (preserve)
   - perp `d⊥ = -a*·sinθ + b*·cosθ` (the tint cast, signed)
3. Gate: `abs(d⊥) ≥ SKIN_MAX_PERP` → treat as a colored object, skip (identity).
   Small `abs(d⊥)` → tiny correction, fine.
4. Target skin = measured with the perpendicular reduced by `SKIN_TINT_STRENGTH`:
   `d⊥' = d⊥·(1 - SKIN_TINT_STRENGTH)`; recompose `(a*', b*')` from `(r∥, d⊥')`,
   keep `L*` unchanged.
5. Convert both measured and target skin Lab → sRGB via the new
   `colorspace.lab_to_srgb` (§4.5). Per-channel skin multiplier
   `m_c = target_rgb[c] / max(eps, measured_rgb[c])`, clamped to
   `[1/WB_MULTIPLIER_CLAMP, WB_MULTIPLIER_CLAMP]` (reuse `correct.WB_MULTIPLIER_CLAMP`
   = 1.5, L53).
6. **Blend into WB, don't replace it.** In `_solve_wb`, after computing the
   gray-world/white-patch `wb` (existing L106–113), blend:
   `wb[c] = wb[c] · (1 + (m_c - 1)·SKIN_WB_WEIGHT)`, then the existing final
   clamp (L113) still bounds it. Gray-world stays the temperature workhorse;
   skin only nudges the tint it's blind to.

`white_reference` (when present & verified neutral, L96–104) keeps priority
exactly as today — a real neutral surface still beats the skin vote; the skin
blend applies only on the gray-world/white-patch branch.

### 4.4 Vibrance normalization (bounded global saturation)

```
TARGET_CHROMA = 22.0     # target mean Lab chroma; below -> boost, at/above -> leave
SAT_BOOST_MAX = 1.25     # hard cap: a global sat lift past this over-saturates skin/reds
```

In `solve_correct_grade`, when `skin_vibrance` and `pipeline=="v1"`:
- `chroma = color_stats.get("chroma_mean")`. Missing/None → `sat = 1.0` (fail-open).
- `sat = clamp(TARGET_CHROMA / max(eps, chroma), 1.0, SAT_BOOST_MAX)` — **only ≥ 1.0**
  (never desaturate; already-vivid footage where `chroma ≥ TARGET_CHROMA` gets `sat=1.0`).
- Return `Grade(slope, offset, power, sat=sat)` (replace the hardcoded `sat=1.0`
  at L223 with the computed value).
- **Skin protection** is the tight `SAT_BOOST_MAX` cap plus "boost only when
  genuinely low-chroma." Optional stronger version (note only; default off):
  scale the boost down when `subject_lab`/`skin_lab` chroma is already healthy.
  Ship the minimal version first.

Saturation composes through `compose` (`cdl.py` L108, `sat = lerp_mult`) and the
per-clip override/arc, unchanged. The resolver's composite clamp
(`_clamp_composite_v1`) bounds slope/offset only; `sat` is bounded at source here.

### 4.5 New helper: `colorspace.lab_to_srgb`

Add the exact inverse of `srgb_to_lab` (D65) to `grade/colorspace.py`
(`_gamma` = forward sRGB OETF, `_f_inv` = inverse of `_f` L15–16, Lab→XYZ→linear
→gamma). Pure, no deps. Clamp output to 0..1. Needed only by §4.3 step 5.

### 4.6 Wiring

- **`config.py`** (after the `grade_subject_exposure` block, L176): add
  ```python
  # color_skin_vibrance.plan.md: skin-anchored tint correction (fairness-safe:
  # removes only the off-locus green/magenta cast on skin, never targets a
  # skin tone) + a bounded global saturation floor for flat/desaturated
  # footage. v1-only; the subject_lab (face-region) skin refinement additionally
  # requires grade_semantic (it rides the grade_subject_exposure box). Off =
  # today's WB (gray-world/white-patch) and sat=1.0 exactly.
  grade_skin_vibrance: bool = False
  ```
- **`resolver.py`** `resolve_clip_grade` (L199–209): add `skin_vibrance: bool = False`;
  pass to `solve_correct_grade(color_stats, already_graded=…, skin_vibrance=skin_vibrance, pipeline=pipeline)` (L239).
- **`correct.py`** `solve_correct_grade` (L178–184): add `skin_vibrance: bool = False`.
  Skin vote is used only when `skin_vibrance and pipeline=="v1"`; vibrance sat
  the same. `_solve_wb` gains a `skin_lab`/`subject_lab` arg (pass through from
  `color_stats`).
- **`job.py`**:
  - `run_grade_job` L636–641: pass `skin_vibrance=settings.grade_skin_vibrance`
    into `resolve_clip_grade`.
  - `subject_lab` flows automatically: it's already in `stats` (from
    `measure_span` §4.2) which is passed as `color_stats=stats`. No extra arg.
  - **Bump `INPUT_HASH_SCHEMA_VERSION` 6 → 7** (L76) + one-line comment.
  - Add `"grade_skin_vibrance": settings.grade_skin_vibrance` to the `flags`
    dict in `compute_input_hash` (L230–237).

---

## 5. Switchability / rollback

- `grade_skin_vibrance` defaults **off**. Off (or `legacy`) → `_solve_wb`
  returns the gray-world/white-patch result unchanged and `solve_correct_grade`
  returns `sat=1.0` — the correct layer, and therefore every `grade_hash`, is
  **byte-identical to today** (verify via hash-equality test, §7).
- The only unconditional changes are the three `SCHEMA_VERSION` bumps
  (`color_stats` 2→3, `measure_span` 1→2, `INPUT_HASH` 6→7), which force a
  one-time recompute/re-grade producing the *same* output when the flag is off
  (the new fields are simply unused).
- No SQL migration: `color_stats` and `cut_color_stats` store JSON keyed by a
  `schema_version` column already; bumping the constant is the whole "migration."
- Clean revert: all new code is additive and flag-gated; delete `lab_to_srgb`,
  the `chroma_mean`/`subject_lab` fields, and revert the small `correct.py`/
  `resolver.py`/`job.py`/`config.py` diffs.

---

## 6. Phased implementation

Gated behind **`grade_skin_vibrance` (default off)** AND `pipeline=="v1"`. The
face-region (`subject_lab`) refinement additionally requires `grade_semantic`
(it reuses the `grade_subject_exposure` box); the center-proxy `skin_lab` path
and vibrance work under the flag alone.

### Phase 0 — measurement + flag + hash (no grade change yet)
- `color_stats.py`: add `chroma_mean` (§4.1), bump `SCHEMA_VERSION` 2→3.
- `measure_span.py`: add `_measure_subject_lab` + `subject_lab` (§4.2), bump
  `SCHEMA_VERSION` 1→2. `chroma_mean` flows via `_aggregate().to_dict()`.
- `colorspace.py`: add `lab_to_srgb` (§4.5).
- `config.py`: add `grade_skin_vibrance` (§4.6).
- `job.py`: bump `INPUT_HASH_SCHEMA_VERSION` 6→7 + add the flag to `compute_input_hash`.
- **Acceptance:** with the flag off, grades are byte-identical (only the schema
  bump re-computes, to the same values); `chroma_mean` and `subject_lab` are
  populated on a fresh measure and sane on a known clip.

### Phase 1 — vibrance (simplest, broadest reach)
- `correct.py`: compute `sat` from `chroma_mean` (§4.4); thread `skin_vibrance`
  through `solve_correct_grade` and `resolver.resolve_clip_grade`; `job.py`
  passes the flag.
- **Result:** flat/log footage gains bounded saturation; vivid footage unchanged.

### Phase 2 — skin-anchored tint correction
- `correct.py`: implement the locus decomposition + skin multiplier (§4.3),
  blend into `_solve_wb`. Prefer `subject_lab`, fall back to `skin_lab`.
- **Result:** green/magenta skin casts are pulled onto the skin locus; warmth,
  lightness, and skin-tone identity are preserved.

Each phase is independently shippable; Phase 1 gives the broad "looks finished"
win, Phase 2 the people-content skin win.

---

## 7. Testing (`backend/scripts/test_grade.py`)

Plain script, no DB/ffmpeg/R2 — mirror the file's `mock.patch` + scripted-fake
convention (top-of-file imports L24–38; register each new test in `main()` near
L1765 and keep the final `print("\nall grade tests passed")`). Run:
`cd backend && .venv/bin/python scripts/test_grade.py`.

**Vibrance**
1. `test_vibrance_boosts_low_chroma_bounded` — `color_stats` with `chroma_mean`
   well below `TARGET_CHROMA`; `solve_correct_grade(..., skin_vibrance=True,
   pipeline="v1")` returns `1.0 < sat ≤ SAT_BOOST_MAX`.
2. `test_vibrance_no_desaturation_on_vivid_footage` — `chroma_mean ≥ TARGET_CHROMA`
   → `sat == 1.0` (never worse).
3. `test_vibrance_missing_chroma_is_identity` — no `chroma_mean` key → `sat == 1.0`.
4. `test_vibrance_flag_off_and_legacy_sat_is_one` — flag off (and `pipeline="legacy"`)
   → `sat == 1.0` regardless of chroma.

**Skin tint**
5. `test_skin_tint_corrects_green_cast_bounded` — build a skin Lab on the locus,
   then push it green (perpendicular residual); assert the returned WB multiplier
   moves the *perpendicular* component toward the locus, bounded by `WB_MULTIPLIER_CLAMP`.
6. `test_skin_tint_preserves_warmth_and_tone` — a warm (golden-hour) on-locus skin
   AND a darker on-locus skin both produce a near-identity correction (along-locus
   warmth + `L*` untouched) — encodes "no skin-tone privileging."
7. `test_skin_tint_skips_non_skin` — `L*` outside `[SKIN_L_MIN,SKIN_L_MAX]`, or
   `abs(d⊥) > SKIN_MAX_PERP`, or chroma `< SKIN_MIN_CHROMA` → no skin vote (WB
   equals the gray-world/white-patch-only result).
8. `test_skin_prefers_subject_lab_over_center_proxy` — both present, differing;
   assert the face-region `subject_lab` drives the correction.
9. `test_lab_to_srgb_round_trips` — `lab_to_srgb(srgb_to_lab(rgb)) ≈ rgb` within
   tolerance across a few colors (incl. two skin tones).

**Never-worse / parity**
10. `test_correct_flag_off_byte_identical` — `resolve_clip_grade` with
    `skin_vibrance=False` yields the **same `grade_hash`** as before this plan on
    a fixture (extend the existing legacy-parity pattern, cf. L158–163, L394–441).
11. `test_white_reference_still_wins_over_skin` — a verified-neutral
    `white_reference_rgb` (L96–104) overrides the skin vote.

---

## 8. Ops runbook

1. Land Phases 0–2.
2. Schema bumps are in Phase 0 (`color_stats` 2→3, `measure_span` 1→2,
   `INPUT_HASH` 6→7). **`chroma_mean`/`subject_lab` only appear on
   re-measured files** — the correct layer fail-opens on older rows, so no
   forced re-ingest is required, but color quality is best once L1 re-runs.
3. Turn the flag **on**: `GRADE_SKIN_VIBRANCE=true` in the worker env (ensure
   `GRADE_PIPELINE=v1`; `GRADE_SEMANTIC=true` for the face-region skin path —
   both already default per `config.py` L146/L153).
4. **Restart the grade worker** to pick up env + code.
5. Re-grade every project:
   `cd backend && PYTHONPATH=. .venv/bin/python scripts/_grade_all_projects.py`
   (**export `GRADE_SKIN_VIBRANCE=true` first**, or add it to that script's
   `os.environ` block). Optionally re-run L1 `color_stats` for the projects you
   want the fresh `chroma_mean`/face-skin on.
6. **Verify:** summary prints `state=done`, `rows>0`, `baked>0` per project;
   spot-check a people reel — faces neutralized on the tint axis, warmth intact,
   flat b-roll livelier, vivid shots unchanged.

---

## 9. Acceptance criteria

- **Skin tint, fairness-safe.** On a green/mixed-light interview, the skin
  sample's perpendicular (off-locus) residual drops by ~`SKIN_TINT_STRENGTH`
  while its along-locus warmth and `L*` are preserved; a warm-but-correct skin
  and a dark-but-correct skin are each left ≈unchanged (correction is symmetric
  across tones — no privileging). All bounded by `WB_MULTIPLIER_CLAMP`.
- **Vibrance floor.** A flat/log clip's `sat` lifts toward `TARGET_CHROMA`,
  capped at `SAT_BOOST_MAX`; a vivid clip gets `sat == 1.0` (never desaturated).
- **Flag-off / legacy byte-identical.** Every shot's `grade_hash` is identical
  to the pre-change v1 output when `grade_skin_vibrance` is off, and the
  `legacy` pipeline is untouched.
- **Composite guardrails hold.** The skin-tinted WB slope still passes
  `_clamp_composite_v1` (no over-contrast / shadow crush); `sat` bounded at source.

---

## 10. Line-ref drift found vs. the code (verified this session)

- ✅ `correct._solve_wb` L93–114 folds `(wb_r,wb_g,wb_b)` into the luma slope at
  L221; `white_reference` priority at L96–104. Skin blend inserts after L113's
  clamp input, before the final clamp.
- ✅ `solve_correct_grade` returns `sat=1.0` hardcoded at L223 — the single line
  vibrance replaces.
- ✅ `resolve_clip_grade` calls `solve_correct_grade(color_stats, already_graded=…,
  pipeline=pipeline)` at L239 — no `skin_vibrance` yet; add the kwarg both sides.
- ✅ `job.py` passes the **span** stats as `color_stats=stats` at L637 (so
  `skin_lab`/`chroma_mean`/`subject_lab` are per-span); `INPUT_HASH_SCHEMA_VERSION
  = 6` at L76; flags dict L230–237; `subject_boxes` resolved L413–416.
- ✅ `color_stats._aggregate` builds `lab_stacked` L190–194 (chroma is one line
  off it); `ColorStats.skin_lab` L85 / `to_dict` L105; `SCHEMA_VERSION = 2` L53.
- ✅ `measure_span._measure_subject_luma` L90–107 is the exact template for
  `_measure_subject_lab`; `subject_box`+`hero_frame` set `subject_luma` at
  L168–171; `SCHEMA_VERSION = 1` L32; partial-miss recompute L139.
- ✅ `cdl.Grade.sat` L34, applied L134–137, carried through `compose` L108.
- ✅ `colorspace.srgb_to_lab` L19–30 + `is_neutral` L33–38 — `lab_to_srgb` is the
  missing inverse; D65 white `_D65` L8.
- ⚠️ **No SQL migration needed** — both stat tables are JSON keyed by a
  `schema_version` column; the constant bumps are the migration. Older rows
  lacking the new fields must be handled fail-open (they are, per §4.1/§4.4).
- ⚠️ **CDL `sat` is global**, applied in **linear working space** under v1
  (lut_bake order). "Vibrance" is therefore a bounded global sat nudge, not a
  per-pixel skin-masked op — the tight `SAT_BOOST_MAX` is the skin guardrail.
