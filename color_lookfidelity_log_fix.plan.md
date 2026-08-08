# color_lookfidelity_log_fix.plan.md

Harness-calibration fix for `look_fidelity_cosine` on LOG footage.

## Goal

Make the color-QA metric `look_fidelity_cosine` measure the **LOOK layer's**
contribution to the raw→graded color shift, independent of the
Correct/Balance/Match/decode contribution. Today the metric measures the
**whole** raw→graded shift against a look-only reference; after the log decode
landed (commit `615d4df`, `WORKING_SPACE_LOG_V1` in `tone.py`), log shots need a
large Correct contribution that no longer aligns with the look's direction, so
the cosine drops and produces **false Tier B fails** even though the decode
*visibly improves* the image (confirmed on Siri Reel a002/a003 contact sheets).
This is a harness-calibration gap, not a production defect. The fix is
harness-only: **no grade re-bake, no `INPUT_HASH` bump.**

## Root cause (exact current formula + states)

`look_fidelity_metric` compares two 5-D shift vectors (per-channel RGB mean
shift + Lab a*/b* mean shift, `_shift_vector`, `backend/scripts/qa/metrics.py:534-544`):

```547:561:backend/scripts/qa/metrics.py
def look_fidelity_metric(graded01: np.ndarray, raw01: np.ndarray, look_only01: np.ndarray) -> MetricResult:
    ...
    v_graded = _shift_vector(graded01, raw01)
    v_look = _shift_vector(look_only01, raw01)
    n1, n2 = float(np.linalg.norm(v_graded)), float(np.linalg.norm(v_look))
    if n1 < 1e-6 or n2 < 1e-6:
        return MetricResult("look_fidelity_cosine", None, "na", {"graded_norm": n1, "look_norm": n2})
    cos = float(np.dot(v_graded, v_look) / (n1 * n2))
    return MetricResult(
        "look_fidelity_cosine", cos, _band_lower(cos, LOOK_FIDELITY_COS_PASS, LOOK_FIDELITY_COS_WARN),
    )
```

The two states being diffed (both anchored on **RAW**):

- **actual** `v_graded = shift(graded01, raw01)`.
  `graded01` is the shot's real baked `.cube` applied to the raw still
  (`_diag_qa_score._load_rgb01` on the ffmpeg-`lut3d` output rendered in
  `_diag_qa_sample.apply_cube`). The cube is
  `bake_cube_text(cdl, working_space=ws, creative_lut_grid=look_grid, tone_contrast)`
  (`cache.ensure_cube_file`), i.e. the pipeline is
  `raw → to_working (decode) → apply_cdl (Correct+Balance+Match+Leveling+Arc) → from_working (tone) → look_grid`.
  So `v_graded` bundles **decode + Correct-stack + tone + look**.
- **expected** `v_look = shift(look_only01, raw01)`.
  `look_only01` (`_diag_qa_score._look_only_still:103-118`) bakes
  `bake_cube_text(Grade(), working_space=ws, creative_lut_grid=look_grid, tone_contrast=0.0)`
  — **identity CDL** but the same `working_space` decode + tone + look grid.
  So `v_look` bundles **decode + tone + look** (no Correct stack).

The difference between the two vectors is therefore *exactly the Correct
stack* (Correct+Balance+Match+Leveling+Arc), which appears in `v_graded` but
not `v_look`.

- **rec709** (`WORKING_SPACE_V1`): `to_working` is the inverse sRGB EOTF and
  `from_working` its re-encode, so with identity CDL the decode round-trips to
  near-identity below the highlight shoulder (`tone._SHOULDER_START = 0.8`).
  Well-exposed rec709 needs only a small Correct nudge, so the Correct term is a
  small fraction of `v_graded` → the two vectors stay near-parallel → cosine
  high → PASS. (This is why rec709 shots currently pass.)
- **log** (`WORKING_SPACE_LOG_V1`, `linear = display ** 1.8`): log footage
  legitimately needs a **large** Correct lift/balance. That large Correct term
  is in `v_graded` but absent from `v_look`, dragging `v_graded`'s direction
  away from `v_look` → cosine drops → **false FAIL**. `v_look` *also* carries
  the big log-decode lift (it shares `ws`), which further mis-aligns it from a
  decode-free interpretation of "the look."

Metric→tier wiring that makes this bite: `look_fidelity_cosine ∈ TIER_B_METRICS`
(`_diag_qa_score.py:61-64`), Tier B needs 95% pass (`TIER_B_REQUIRED_PASS_PCT`),
so inflated log fails directly threaten the Tier B bar.

## Proposed new metric definition

Isolate the **look grid's** contribution by removing the shared
decode+tone+Correct terms from *both* sides. Diff the look layer against the
state it actually composes on top of, on both the actual and reference paths:

- **actual look shift** = `shift(graded01, corrected01)`
  where `corrected01` = the real grade **without the look grid**:
  `bake_cube_text(cdl, working_space=ws, creative_lut_grid=None, tone_contrast=tone_contrast)`
  sampled on `raw01`. Because the real cube is
  `look_grid(from_working(apply_cdl(to_working(raw))))` and `corrected01` is the
  same expression minus the final `look_grid`, this diff is **exactly the look
  grid's delta on the corrected image** — decode + Correct-stack + tone all
  cancel out.
- **expected look shift** = `shift(look_only01, decoded01)`
  where `decoded01` = the neutral decode **without the look grid**:
  `bake_cube_text(Grade(), working_space=ws, creative_lut_grid=None, tone_contrast=0.0)`
  sampled on `raw01`. `look_only01` stays the existing identity-CDL+look bake.
  This diff is the look grid's delta on the neutral-decoded base — decode + tone
  cancel out, and there is no Correct term by construction.

`cos = dot(actual, expected) / (|actual| · |expected|)`, bands unchanged.

Both sides now measure a **pure look-grid delta** with decode/tone/Correct
removed. The reference base (`decoded01`) is deliberately the look's *standalone*
base, not `corrected01` — using `corrected01` on both sides would make the two
diffs identical and force cosine ≡ 1 (trivially passing, catching nothing). The
metric still catches its intended failure (a compositing/order/working-space bug
where the look, as actually composed, pulls a *different direction* than the look
standalone).

### Why this is correct for BOTH rec709 and log (not a log-only special case)

- **rec709**: `decoded01 ≈ raw01` (sRGB decode→re-encode round-trips to identity
  below the 0.8 shoulder with identity CDL), so `expected = shift(look_only01,
  decoded01) ≈ shift(look_only01, raw01)` — the *same* reference rec709 uses
  today. And rec709's Correct term was already small, so removing it from
  `actual` barely moves the vector; if anything it makes the cosine *cleaner*.
  Currently-passing rec709 shots stay passing.
- **log**: the large log-decode lift is subtracted from `expected` (via
  `decoded01`) and the large decode+Correct lift is subtracted from `actual`
  (via `corrected01`). Both collapse to the look-grid delta, so the spurious
  Correct/decode-driven mis-alignment is gone. The metric now reflects only
  whether the look composed the way the look intends.

The transform is uniform: both paths subtract a look-free reference rendered in
the shot's own `working_space`. There is no `if is_log` branch anywhere.

## Implementation touch-points

1. `backend/scripts/qa/metrics.py`
   - Change `look_fidelity_metric` signature to
     `look_fidelity_metric(graded01, corrected01, look_only01, decoded01)` and
     compute `v_graded = _shift_vector(graded01, corrected01)` and
     `v_look = _shift_vector(look_only01, decoded01)`. Keep `_shift_vector`
     (`:534-544`), the norm `< 1e-6 → na` guard, and the `_band_lower(cos,
     LOOK_FIDELITY_COS_PASS, LOOK_FIDELITY_COS_WARN)` banding unchanged. Update
     the docstring to state the metric isolates the LOOK contribution measured
     from the corrected (not raw) state.

2. `backend/scripts/_diag_qa_score.py`
   - Generalize `_look_only_still` (`:103-118`) or add two siblings that render
     look-free references in-process via the existing
     `bake_cube_text → parse_cube_text → _sample_lut_trilinear` recipe:
     - `corrected01 = bake(Grade.from_dict(shot["cdl"]), ws, creative_lut_grid=None, tone_contrast=shot["tone_contrast"])` on `raw01`.
     - `decoded01 = bake(Grade(), ws, creative_lut_grid=None, tone_contrast=0.0)` on `raw01`.
     Suggested shared helper `_bake_still(cdl, raw01, ws, grid, tone_contrast)`
     that all three (corrected/decoded/look_only) call, so there is one bake
     path. `Grade` is already imported (`:44`); `LookSpec`/`build_look_grid`
     already imported (`:45`).
   - In `score_shot` (`:143-146`), when `shot.get("look_engine")` is set, render
     `look_only01`, `corrected01`, `decoded01` and call
     `m.look_fidelity_metric(graded01, corrected01, look_only01, decoded01)`.
   - Rendering all three in-process (trilinear) keeps the reference states on one
     consistent sampling path; `graded01` stays the ffmpeg-`lut3d` output
     (preserves export-parity for the other metrics that reuse it). This mirrors
     today's already-accepted mix (`look_only01` in-process vs `graded01`
     ffmpeg): both are trilinear on the same 33³ grid, numerically ~identical.
     (Optional exactness variant: also compute `actual` as
     `shift(_sample_lut_trilinear(look_grid, corrected01), corrected01)` so both
     endpoints are in-process — a pure look-grid delta with zero cross-method
     residual. Not required; note it if band re-cal shows residual noise.)

3. `backend/scripts/_diag_qa_sample.py`
   - The score step needs the composed **`cdl`** and **`tone_contrast`** to
     render `corrected01`; the samples manifest currently carries only
     `look_engine` + `working_space` (`:239-250`, `:214`). Add
     `"cdl": (grade_json or {}).get("cdl")` and
     `"tone_contrast": (grade_json or {}).get("tone_contrast") or 0.0` to each
     shot record (both already present in the persisted grade row — see
     `resolver.resolve_clip_grade`'s returned descriptor,
     `backend/app/services/l3/grade/resolver.py:365-373`). Because the manifest
     schema grows, `samples.json` must be regenerated (see re-scoring note).
   - Note: `look_fidelity_cosine` only runs for **engine-mode** looks (score
     gates on `shot["look_engine"]`), and for engine looks the persisted `cdl`
     is exactly Correct+Balance+Match+Leveling+Arc with **no look** (the look is
     the `creative_lut_grid`, `resolver.py:304-315`, `cache.py:48-58`). So
     `bake(cdl, …, creative_lut_grid=None)` is precisely the corrected-pre-look
     state; no look leaks into `corrected01`.

## Test updates (`backend/scripts/test_qa_metrics.py`)

Existing look-fidelity tests call the 3-arg signature and must move to the 4-arg
one (`:324-347`):

- `test_look_fidelity_identical_shift_is_cosine_one`: pass
  `graded=corrected+look_delta`, `corrected`, `look_only=decoded+same_look_delta`,
  `decoded` such that both diffs are the same vector → cosine ~1, PASS.
- `test_look_fidelity_opposite_shift_is_fail`: actual look delta warms,
  reference look delta cools → negative cosine, FAIL.
- `test_look_fidelity_na_when_no_shift`: `graded==corrected` and
  `look_only==decoded` → both norms ~0 → `na`.

Add new coverage for the actual fix:

- **Correct-invariance (the core fix)**: build `corrected` = `raw` plus a large
  synthetic Correct lift, `graded` = `corrected` plus a small look tint;
  `decoded` = `raw`, `look_only` = `raw` plus the *same* small look tint. Assert
  the new metric is high/PASS (look delta aligned) **and** assert that the OLD
  raw-anchored formula (`shift(graded, raw)` vs `shift(look_only, raw)`) would
  be dragged down by the Correct lift — i.e. document the regression the new
  definition removes.
- **rec709 back-compat**: with `decoded ≈ raw` and a negligible Correct term
  (`corrected ≈ raw`), assert the new cosine ≈ the old raw-anchored cosine for
  the same look (currently-passing shots unaffected).

Wire any new test names into `main()` (`:367-406`).

## Re-scoring + band recalibration

- **Harness-only**: metric *definition* change → **re-run the scoreboard**; no
  production grade change, **no grade re-bake, no `INPUT_HASH` bump**.
- Because the manifest schema grows (`cdl`, `tone_contrast`), re-run both stages:
  `PYTHONPATH=. backend/.venv/bin/python scripts/_diag_qa_sample.py` then
  `… scripts/_diag_qa_score.py`. (Alternatively make the score step tolerant of
  an old manifest by skipping `look_fidelity_cosine` when `cdl` is absent, so a
  stale `samples.json` degrades gracefully rather than crashing.)
- **Bands**: keep `LOOK_FIDELITY_COS_PASS = 0.8`, `LOOK_FIDELITY_COS_WARN = 0.5`
  (`metrics.py:91`) as-is initially. The fix should *raise* log cosines toward
  the rec709 range without moving rec709, so the existing bands should hold. Use
  the re-run scoreboard's `failure_classes_ranked_log_flat` vs
  `_rec709` split as the calibration check; only tighten/loosen if the log/rec709
  distributions still diverge materially after the fix.
