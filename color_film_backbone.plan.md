# Color: a film-emulation / display-rendering backbone as the default look

**Status:** design / feasibility. No pipeline code changed by this document.
**Goal:** make our automatic grade read as *graded by a professional*, not merely
*corrected*, by adding a **film-emulation / display-rendering transform** as the
always-present look backbone — the thing pro/AI auto-color tools rely on for a
cinematic baseline "for free" (filmic contrast + highlight roll-off + tasteful
saturation), applied once and consistently instead of hand-tuned per-constant.

---

## 0. Why (evidence recap, verified against the code)

Our Rec.709 base grade is **correction-only**:

- **Contrast is hardwired off.** Every caller passes `tone_contrast=0.0`; the S-curve
  plumbing exists but is dead. See `resolver.resolve_clip_grade(..., tone_contrast: float = 0.0)`
  (`backend/app/services/l3/grade/resolver.py:215`, docstring `:246-253`), and the job
  passes `tone_contrast=0.0` literally (`backend/app/services/l3/grade/job.py:625`).
  `INPUT_HASH_SCHEMA_VERSION` v11 note: "hardwired the global tone_contrast S-curve
  permanently OFF (0.0)" (`job.py:91-97`).
- **No look is baked into the CDL.** `_solve_look` returns `Grade()` (identity) for
  `mode=="engine"` and for `mode=="lut"`/unset (`resolver.py:199-203`). The parametric
  look only exists if a document explicitly selects one, and there is *no auto-selection*
  (`resolver.py:34-35`, `:181-183`). The default (no `look`) is pure identity into the grid.
- **The tone map that *is* live is highlight-only.** `from_working`'s `_tonemap_shoulder`
  is *exact identity below `_SHOULDER_START = 0.8`* and only compresses above it
  (`tone.py:73`, `:128-138`). Verified numerically: shoulder deltas are `+0.0000` at
  x = 0.05 / 0.18 / 0.42 / 0.5 / 0.8, and only `-0.033 / -0.078 / -0.100` at 0.9 / 0.97 / 1.0.
  So shadows and midtones — where "cinematic" lives — are **untouched** today.
- **Saturation is a floor, not a curve.** Correct only ever *raises* `sat` toward
  `TARGET_CHROMA=22` and never shapes per-hue (`correct.py:88-93`, `:317-325`).
- **Guardrails proven inert (not the throttle).** `COMPOSITE_SLOPE_MAX=2.0`,
  `COMPOSITE_MID_FLOOR=0.02` (`resolver.py:74-91`, `_clamp_composite_v1` `:106-139`) are
  outlier backstops; the correction-only design, not the clamps, is why output looks flat.

Human review: ~84% of graded shots marked "bad," concentrated on Rec.709. The gap is a
**missing creative display-rendering step**, which is exactly the piece pro/AI tools add.

---

## 1. How pro / AI auto-color tools actually get "the look" (research, 2026)

Two distinct families, and the distinction is the crux of our whole design:

### 1a. Scene-referred display transforms (ACES output transform, AgX)
- **ACES RRT + ODT**: a scene-referred → display pipeline. The RRT is an opinionated
  filmic S-curve (highlight shoulder + shadow toe); the ODT maps to the device. "Punchy
  and saturated out of the box," but with a known **hue skew** as colors blow out (yellow/
  magenta shift). Mid-gray convention: scene-linear **0.18 → ~0.10** display-linear on SDR
  (ACESCentral "Where should mid-gray end up," historical LAD reasoning).
- **AgX** (Troy Sobotka; default in Blender): same job, but **desaturates toward white** as
  values increase instead of hue-skewing — preserves gradient/detail. Neutral/"raw" by
  design; needs a "Punchy"/"High Contrast" *look* on top to be pleasing. Openly licensed.
- **Critical property:** both expect **scene-referred linear input with headroom** (values
  running several stops over display white, up to ~16+). Their signature — the highlight
  roll-off and highlight desaturation — *operates in a region above display white*. Feeding
  them **already-SDR display footage** (bounded 0..1) is a documented double-transform:
  "crushed shadows, unnatural highlight roll-off … you must first invert display-referred
  footage back to a linear scene-referred state" (pixls.us, toodee.de). darktable states the
  rule precisely: scene-referred modules use `[0, ∞]`; display-referred modules want `[0,1]`.

### 1b. Display-referred print-film emulation (Kodak 2383 print LUTs, FilmBox print node, Dehancer)
- A **print emulation** maps a *color-corrected Rec.709* image → Rec.709, modeling what a
  digital file looks like forced onto print stock. Signature: **soft highlight roll-off,
  deep shadows, teal-shadow / orange-highlight split-tone, and a gamut limited to the
  Rec.709∩print intersection** (colorgradingcentral, preset guides, lowepost).
- **Applied AFTER primaries.** Every 2383 tutorial: correct exposure + WB + contrast to a
  neutral Rec.709 baseline *first*, then apply the print LUT (often at 60-80% opacity). Apply
  it to raw/Log and you get "oversaturated, crushed blacks, plain ugly" (presetcurator).
- **FilmBox / Dehancer** = finishing layers, not correctors: negative response + print
  response + halation/grain/acutance. They assume an already-balanced grade underneath.
- **Colourlab AI** = the *correction/matching* half (auto primary balance + shot match), then
  you still "shape the final cinematic look" — i.e. exactly our split: Correct does balance,
  a look/print step does cinematic.

### What specifically produces the "cinematic feel" (design targets)
1. **Contrast curve shape** — a gentle filmic **S** through the midtones (toe lifts blacks
   slightly then dips, shoulder rolls highlights), *not* a linear slope. Ours is currently a
   pure slope/offset (`apply_cdl`, `cdl.py:112-139`) plus an identity-below-0.8 shoulder.
2. **Highlight roll-off / shoulder** — compress the top ~1-2 "stops" so speculars don't hard-
   clip; ours does a little of this already (`_tonemap_shoulder`) but starts too high (0.8).
3. **Per-hue saturation behavior** — vibrance that *rolls off* in highlights and near-neutrals,
   with skin protected; teal/orange separation. Ours is a flat global `sat` scalar only.
4. **Gamut containment** — pull the most saturated corners inward slightly (film can't print
   pure Rec.709 primaries), which reads as "expensive."

### Copyable vs. taste
- **Copyable (math, license-permitting):** the tone-curve shape, shoulder, highlight
  desaturation, split-tone direction. All expressible in a 3D `.cube` or our parametric ops.
- **Needs taste / reference:** the exact strength, the skin balance, and *which* film character
  (2383 vs. Portra vs. Eterna). This is tuned on real footage via contact sheets, not derived.

---

## 2. Exactly where the transform slots into OUR pipeline

### 2a. The decisive color-space finding
Our "working space" is **NOT scene-referred**. `to_working` is the inverse sRGB EOTF
(`tone.py:82-97`), producing **bounded 0..1 display-linearized** values with **no highlight
headroom** — display white 1.0 stays 1.0, and mid-gray display 0.5 → linear **0.214**
(verified; matches `resolver.py:76` "≈0.214"). Display 0.42 → linear 0.1473; 0.18 → 0.0272.
There is nothing above 1.0 for a scene-referred shoulder to act on.

**Therefore a scene-referred AgX/ACES ODT is the wrong tool for our inputs** — the exact trap
research and `tone.py`'s own docstring (`:32-40`) already call out: "that calibration assumes
input can run several stops over display white … applied to already-SDR footage it visibly
darkens shadows and midtones." Our footage behaves like the "already Rec.709" case in every
2383 tutorial. **The correct model is a display-referred print-emulation transform (Rec.709 →
Rec.709), applied after primary correction.**

### 2b. The exact seam
There is already a perfectly-placed, display-referred seam in the bake. `bake_cube_text`
(`lut_bake.py:90-137`) does, per grid node:

```118:123:backend/app/services/l3/grade/lut_bake.py
    working = to_working(grid, working_space)
    graded = apply_cdl(working, grade)                 # CDL (SS3 stack: Correct/Match/Arc live in cdl)
    out = from_working(graded, working_space, contrast=tone_contrast)

    if creative_lut_grid is not None:
        lut_grid, _lut_size = creative_lut_grid
        out = _sample_lut_trilinear(lut_grid, out)
```

`out` after `from_working` is **display-encoded Rec.709** — and this is *exactly* the space
`build_look_grid`/creative LUTs already operate in (`look_engine.py:16-19`). So:

> **Slot the film backbone as a display-referred transform applied to `out` immediately
> after `from_working` and BEFORE `_sample_lut_trilinear(creative_lut_grid, ...)`.**

Proposed shape (new stage, *augmenting* the bake — not replacing `from_working`):

```python
out = from_working(graded, working_space, contrast=tone_contrast)
out = apply_film_backbone(out, backbone)        # NEW: display -> display, Rec.709
if creative_lut_grid is not None:
    out = _sample_lut_trilinear(lut_grid, out)  # engine look / uploaded .cube on top
```

Why here and not inside `from_working`:
- `from_working` is the *encode* boundary and is shared with `_corrected_source_stats`
  (`resolver.py:171-177`) and the Balance/Match stat projections (`job.py:118-193`), which
  round-trip through it to keep solver stats comparable. Those projections must stay a clean
  display↔linear round-trip; folding a creative curve into `from_working` would poison them
  (the solver would "see" the look and try to correct against it). Keeping the backbone a
  *separate display→display stage after* `from_working` leaves every stat projection exact.
- It composes at bake time only, so preview (WebGL) and export (ffmpeg `lut3d`) inherit it
  identically — the whole point of the single-`.cube` parity contract (`lut_bake.py:1-7`).

### 2c. Uniform across BOTH content paths — yes, and it's automatic
`from_working` re-encodes **log_v1 and rec709_v1 identically** to display Rec.709; only the
*input decode* in `to_working` differs (`tone.py:155-160`, `:171-177`). So a backbone applied
to `out` (post-`from_working`) is applied **uniformly to both** by construction — a log clip
and a Rec.709 clip both arrive at the backbone as corrected display Rec.709. This is the
correct place: the backbone sees a neutral, corrected Rec.709 image regardless of source,
which is precisely the "clean neutral start" print emulation demands. (The prior working-space
bug was a display-space offset applied to linearized midtones — `correct._project`
`:100-112`; putting the backbone in *display* space, after the encode, sidesteps that class of
bug entirely.)

---

## 3. Coexistence with engine looks + creative LUTs (avoiding the double-contrast regression)

The regression that killed `tone_contrast`: a global S-curve *on top of* engine looks that
already carry their own contrast → double-contrast → "too dark" (`resolver.py:249-253`,
`job.py:91-97`). We must not repeat it. The three options:

- **(a) Backbone = neutral default; engine looks layer on top.** Clean architecture, matches
  pro tools (neutral display render + look on top). BUT if looks aren't retuned, they
  double-apply the baseline (kodak_2383 already bakes `contrast=0.22` + split-tone,
  `look_engine.py:450-455`) → the exact regression.
- **(b) Backbone only when no engine look is set.** Zero double-contrast risk; but shots with
  a look never get the backbone → inconsistent, two-track pipeline.
- **(c) Engine looks re-expressed as deltas relative to the backbone.** Each look drops the
  baseline contrast/rolloff the backbone now supplies and keeps only its distinctive
  character. Correct end-state, but requires retuning all 18 looks (`look_engine.py:376-542`).

### Recommendation: **(a) as the architecture, delivered via (c), staged so (b) is Phase 1.**

Reasoning: the pain (84% bad) is concentrated on the **no-look, correction-only majority** —
and resolver does *no auto look-selection*, so in practice most shots have **no engine look at
all**. That means:

- **Phase 1 (ship first): enable the backbone on the no-look path only.** This is *literally
  option (b)*, but framed as the first slice of (a). It is where all the value is, and it
  carries **zero double-contrast risk** because there is no look to double with. An explicitly-
  selected engine look keeps today's behavior (look bakes its own character; no backbone),
  byte-compatible with the current catalog.
- **Phase 2 (end-state = a via c): make the backbone unconditional and re-express the catalog
  as deltas.** Strip each look's baseline contrast/shoulder (the part the backbone now owns),
  keeping its signature (hue moves, split-tone tint, grain/halation). Then every shot —
  look or no-look — shares one cinematic baseline, and looks are honest deltas on top.

This is decisive: **backbone is the default backbone (a); the mechanism that makes it safe
under looks is re-expressing looks as deltas (c); we sequence it so the first release is the
risk-free no-look slice (b).** Implementation hook: `resolve_clip_grade` already knows whether
a look is active (`sequence_look.get("mode")`, `resolver.py:301-315`) — Phase 1 sets the
backbone descriptor when no engine look/`.cube` is active; Phase 2 sets it always and the
catalog retune lands with it.

---

## 4. Candidate transforms we can actually bake, and the recommendation

All must land as either a display-referred 33³ `.cube` (drop-in for `_sample_lut_trilinear`,
`parse_cube_text`, `lut_bake.py:140-181`) or as parametric ops over the display grid (reusing
`look_engine.py`'s vectorized primitives). Assessment:

| Candidate | Ref. model | License | Fit to our SDR input | Bake path |
|---|---|---|---|---|
| **AgX** (Blender/Sobotka) | scene-referred | Open (CC0/Apache-ish) | **Poor as-is** — needs a synthetic scene-referred domain + headroom we don't have; degrades to a contrast+desat curve on SDR | Would need an sRGB→sRGB refit LUT |
| **ACES 1.x/2.0 ODT** | scene-referred | Permissive (ACES) | **Poor as-is** — same headroom problem, plus documented hue skew | Refit LUT |
| **Kodak 2383 print `.cube`** (Juan Melara / vendor) | display-referred | **Encumbered** (vendor LUTs / paid) — do NOT ship their bytes | N/A (licensing) | drop-in if we had rights |
| **In-house parametric "Filmic Print" backbone** | display-referred, *informed by* 2383 character | **Ours** | **Exact** — designed for corrected Rec.709 in, Rec.709 out | parametric over display grid, baked once to a static `.cube` |
| Dehancer/FilmBox profiles | display-referred | Paid/closed | good but closed | N/A |

### Recommended starting candidate: an **in-house display-referred "Filmic Print" backbone**, modeled on Kodak-2383 character, reusing our existing validated ops.

Why this and not "just grab AgX":
- **Correctness:** it operates in display Rec.709 (where our data actually lives), so it
  cannot trigger the SDR double-transform failure. AgX/ACES cannot be dropped in without first
  building a real scene-referred pipeline with headroom (a much larger, riskier change).
- **License-clean:** we author it; we ship no vendor bytes. We already do exactly this "informed
  authoring, not spectral simulation" for the `kodak_2383` *look* (`look_engine.py:341-359`) —
  and its parameters are already tuned on real footage via `_diag_look_contact_sheet.py`.
- **Reuses proven, tested primitives:** the backbone is a small, fixed spec built from ops that
  already exist and are unit-tested:
  - filmic S-curve via `_contrast_pivot` (`tone.py:141-152`) — verified S-shape (darkens below
    `TONE_PIVOT=0.435`, brightens above);
  - highlight shoulder via a lowered-start `_tonemap_shoulder` (`tone.py:128-138`) — currently
    starts at 0.8, i.e. does almost nothing (verified: identity ≤0.8); a print shoulder wants
    to start ~0.55-0.65 so highlights actually roll;
  - teal-shadow / orange-highlight split via `_apply_split_tone` (`look_engine.py:237-249`);
  - highlight/near-neutral saturation roll-off + gentle vibrance via `_apply_hue_sat` /
    `_apply_global_sat` (`look_engine.py:262-278`);
  - optional gamut nudge (later).

Concretely, the v1 backbone is a single fixed `LookSpec`-like descriptor (NOT user-selectable),
e.g. seeded from the vetted `kodak_2383` character but **milder** (it's a baseline everyone
gets, so ~50-70% strength): a modest positive `contrast`, a low-start shoulder, a small
teal/orange split, a small vibrance with highlight roll-off, **no grain/halation** (those stay
look-scoped). It is baked **once** into a static `film_backbone_v1.cube` at build time (it has
no per-clip parameters), stored in-repo, and composed at `lut_bake` as the display→display
stage from §2b. Because it's static, there is no per-clip bake cost — only the one-time grid
compose.

> **Single best starting candidate: an in-house "Filmic Print v1" display-referred backbone
> (Rec.709→Rec.709), authored from the already-validated Kodak-2383 look character at reduced
> strength, baked once to a static `.cube` and composed after `from_working`.** Keep AgX on the
> roadmap as the target *if/when* we build a true scene-referred working space (out of scope
> here); it is the better long-term engine but not compatible with today's SDR-linear pipeline.

---

## 5. Folding in the exposure fix (Correct must place, backbone must shape — don't let them fight)

Finding: `LEVELS_SLOPE_MAX=1.5` (`correct.py:64`) caps the levels stretch, so a dark Rec.709
shot undershoots `TARGET_MID_GRAY=0.42` (`correct.py:68`). **Verified numerically** on a
plausible dark shot (black 0.03, white 0.75, mid 0.28): the slope needed to hit 0.42 is
**2.374**, the cap allows only **1.5**, so displayed mid lands at **0.338** — an undershoot of
**+0.082** (consistent with the ~0.07 prior finding). See `_solve_levels`/`_solve_levels_v1`
(`correct.py:183-241`), where the cap is applied at `:201`, `:238-239`.

Why this is dangerous *with* a backbone: the film S-curve **darkens everything below its
pivot** (verified: `_contrast_pivot(0.30, g=1.2) = 0.279`, i.e. −0.021; at 0.18, −0.029). If
Correct hands the backbone a mid that's *already 0.08 low*, the backbone's toe pushes it
**further** down → the "too dark" failure, now baked into the default. Correct and the backbone
must be sequenced so **Correct places a correctly-exposed neutral input, then the backbone
shapes it** — never compensating for each other.

### Specification
1. **Retarget Correct to the backbone's *input* pivot, not the final display value.**
   Define the backbone so that its neutral input mid `M_in` maps to the desired *output*
   mid: `film_backbone(M_in) ≈ 0.42`. Because the backbone's S-curve darkens mids, `M_in`
   must be **slightly brighter than 0.42** (e.g. if the toe drops ~0.02-0.03 at mid, aim
   Correct at `M_in ≈ 0.44-0.45`, i.e. right around the existing `TONE_PIVOT=0.435`, which is
   convenient — the pivot is where the curve is identity). Compute `M_in` once by inverting the
   chosen backbone at 0.42; store it as the Correct target under the backbone regime. Correct
   keeps solving in the working (linear) space via `_project` (`correct.py:100-112`), targeting
   `to_working_scalar(M_in)`.
2. **Raise the slope ceiling for the mid-gray retarget so dark shots can actually reach the
   input target.** The `1.5` cap exists to avoid manufacturing contrast on false-positive
   `is_log_flat` misfires (`correct.py:56-64`). But two things changed: (i) real log decode now
   lives in `log_v1` (`tone.py:44-64`, `job.py:98-105`), so the cap no longer has to double as a
   log guard; and (ii) **contrast is now owned by the backbone**, so Correct's job is purely
   *placement*, and a higher slope no longer risks over-contrasting (the backbone, not the
   slope, sets contrast). Recommendation: introduce a distinct, higher **placement** ceiling
   for the mid-gray retarget (`_solve_levels_v1`, `correct.py:209-241`) — e.g. allow up to
   ~2.2-2.5 for the *mid nudge* — while:
   - keeping the never-worse **black anchor** (`target_low`, `correct.py:194`, re-anchored at
     `:240`) so blacks don't lift into noise;
   - keeping `MID_GRAY_EXTRA_CLAMP` (`correct.py:72`) as the guard against a single wild nudge;
   - adding a **noise-aware guard**: cap the lift by how far a dark shot can be pushed before
     revealing sensor noise (use measured `black_point`/shadow spread), since with contrast
     off-loaded to the backbone, *noise*, not *contrast*, is the real reason to limit the lift.
3. **Keep the composite ceiling honest.** `COMPOSITE_SLOPE_MAX=2.0` (`resolver.py:74`,
   `_clamp_composite_v1` `:106-139`) currently clamps the *stacked* CDL below the new placement
   ceiling — raise it in lockstep (or exempt the pure-exposure mid nudge) so the relaxed Correct
   target isn't silently re-clamped away. Verify no shot's *composite* slope exceeds what the
   backbone was tuned to receive.

Net: Correct = **placement to `M_in`** (exposure + WB + skin tint + sat floor, all as today),
backbone = **contrast + shoulder + split + vibrance**. They no longer fight because contrast
moved wholly to the backbone and Correct's target is defined *as the backbone's input*.

---

## 6. Risks & mitigations

| Risk | Mechanism | Mitigation |
|---|---|---|
| **Double-contrast** (the `tone_contrast` regression) | Backbone contrast + engine-look contrast stack | Phase 1 backbone only on the no-look path (no look to double). Phase 2 re-express looks as deltas (§3c). Keep `tone_contrast=0.0` — its role is now the backbone. |
| **Crushed blacks** | Backbone toe + teal split deepen shadows; `crushed_black_fraction` rises | Backbone toe must *lift-then-dip* (film toe), not a pure gamma crush. Keep Correct's never-worse black anchor (`correct.py:194`) and `_clamp_composite_v1`'s shadow-probe floor (`resolver.py:86-91`, `:129-135`). Gate on `crushed_black_fraction` Tier-A (must not worsen vs raw; `_diag_qa_score._tier_a_fail_exempt` `:264-273`). |
| **Highlight clipping vs. roll-off** | A steep shoulder or contrast can clip instead of roll | Lower `_SHOULDER_START` to ~0.55-0.65 so highlights *compress* (verified shoulder is asymptotic to 1.0, never clips, `tone.py:128-138`). Gate on `clipped_highlight_fraction` (should **fall or hold**, `metrics.py:232-235`). |
| **Skin-tone shift** | 2383 pushes highlights orange / shadows teal → skin warms/greens | Keep skin-anchored WB + tint correction in Correct *before* the backbone (`correct.py:74-93`, `_skin_multiplier` `:115-149`). Protect the skin hue band in the backbone's split/hue-sat (don't rotate ~30-50°). Gate on `skin_perp_residual` PASS <6 (`metrics.py:260-283`). |
| **Log vs. Rec.709 interaction** | Backbone applied uniformly post-`from_working`, but log content arrives via a different decode (`LOG_DECODE_GAMMA`, `tone.py:64`) and may present flatter | Backbone is uniform by construction (§2c). Use the harness's built-in **log_flat vs rec709 split** (`_diag_qa_score.py:394-397`, `:443-450`) to confirm both subsets improve; if log clips over-contrast, that's a `to_working` decode issue, not the backbone. |
| **Synthetic / screen-recording content** | Slide decks / screen-recs get a film curve they shouldn't | **Bypass (or gentle-only) the backbone for synthetic content.** The classifier already exists: `content_source.content_type_for` and `_SYNTHETIC_NA_METRICS` (`_diag_qa_score.py:81-83`, `:148-156`). Route the backbone descriptor off for `content_type == "synthetic"` (or a much milder variant). |
| **Over-saturation** | Backbone vibrance + look sat compound | Backbone vibrance small, with highlight/near-neutral roll-off via `_apply_hue_sat` (`look_engine.py:262-271`, whose `_band_weight` already collapses on near-neutral, `:216-226`). Gate on `saturation_band` ratio window `(0.6,1.6)` and the hard over-sat FAIL at ratio>2 (`metrics.py:447-505`). |
| **Banding** | Static 33³ backbone LUT trilinearly sampled twice (backbone then look) | Bake backbone at 33³ (matches `DEFAULT_LUT_SIZE`, `lut_bake.py:27`); it's a smooth monotone curve so gaps are unlikely. Gate on `banding_score` <0.15 (WARN-only, `metrics.py:508-527`); contact sheet (`_diag_qa_sheets.py`) is the real judge. Consider composing backbone∘look into one grid to avoid double resample. |
| **Look-fidelity harness breaks** | `_look_only_still` bakes the look **without** the backbone (`_diag_qa_score.py:103-118`), so `look_fidelity_cosine` would compare graded-with-backbone vs look-without → artificially low cosine | Update `_look_only_still` to include the backbone in the look-only reference once Phase 2 lands (so the comparison is "backbone+look" both sides). Until then, keep looks on the no-backbone path (Phase 1) so the metric stays valid. |
| **Schema / re-grade cost** | Every project's cached grade + every cube must recompute | Bump `INPUT_HASH_SCHEMA_VERSION` 12→13 (`job.py:106`) and add a backbone id to the `grade_hash` payload (`cdl.grade_hash` `:142-175`, alongside `tone_contrast`/`look_engine`) so cubes rebake. This is the standard, expected cost (v2-v12 each did it); the read path's graceful fallback (`fetch_latest_grades`, `job.py:270-291`) keeps preview live during the re-grade. |

---

## 7. Harness-measurable rollout plan

All metrics already exist in `backend/scripts/qa/metrics.py`; the scoreboard + tiered parity
bar in `backend/scripts/_diag_qa_score.py`. Expected direction per property:

| Metric (`metrics.py`) | Tier | What the backbone should do | Gate |
|---|---|---|---|
| `exposure_band` (`:201-237`) | A | Dark Rec.709 mids move **up** into `(0.30,0.60)` (via the §5 exposure fix); PASS-rate up. FAIL only at gross extremes. | Tier A 100% (fails only if grade-introduced) |
| `crushed_black_fraction` (`:228-231`) | A | **Hold or fall** vs raw; must not worsen (toe lifts-then-dips) | must not exceed raw `delta` (`_tier_a_fail_exempt`) |
| `clipped_highlight_fraction` (`:232-235`) | A | **Fall** (shoulder rolls off instead of clipping) | Tier A 100% |
| `neutral_axis_deviation` (`:244-257`) | A | Roughly flat — backbone split-tone is small and hue-weighted, and WB is corrected upstream | <3 PASS |
| `saturation_band` (`:447-505`) | B | Chroma ratio rises modestly, stays in `(0.6,1.6)`; over-sat FAIL (ratio>2) must not trip | Tier B ≥95% |
| `skin_perp_residual` (`:260-283`) | B | **Hold** (<6) — skin band protected, WB upstream | Tier B ≥95% |
| `look_fidelity_cosine` (`:547-561`) | B | Hold >0.8 **after** updating `_look_only_still` to include backbone (Phase 2) | Tier B ≥95% |
| `banding_score` (`:508-527`) | B (WARN-only) | Hold <0.15 | contact sheet judges |
| `intra_group_luma/chroma_std` (`:331-373`) | B + C | Unchanged/held — backbone is per-clip identical, so it can't *increase* cross-shot spread; and Tier C "never worse than raw" holds | Tier B ≥95%, Tier C 100% |

**Tiered parity bar** (source of truth `_diag_qa_score.py:60-71`): Tier A **100%** (structural,
grade-introduced fails only), Tier B **95%**, Tier C **100%**. The backbone must not regress
any tier; the *win* shows up as the **overall pass %** (`:399-408`) and the collapse of the
Rec.709-heavy WARN/FAIL cluster (the 84%-bad population), visible in the log_flat-vs-rec709
split and the per-project rollup (`:443-456`).

### Staged rollout
1. **Author + bake `film_backbone_v1.cube`** (static, in-repo). Unit-test parity: identity
   backbone == today's bytes; `bake_cube_text` with backbone applied to an identity CDL matches
   an offline reference render. Confirm it composes cleanly before/after the creative grid.
2. **One sample project (measure).** Run `_diag_qa_sample.py` → `_diag_qa_score.py` on a
   Rec.709-heavy project **before** (baseline scoreboard) and **after** (backbone on the no-look
   path, exposure fix in). Require: Tier A stays 100% (no new grade-introduced structural
   fails), Tier B ≥95%, and overall pass % **up** with the Rec.709 subset's FAIL rank dropping.
   Eyeball the contact sheet (`_diag_qa_sheets.py`) — the harness is necessary, not sufficient;
   "does it look graded" is a human call.
3. **Tune** backbone strength + `M_in` + the new placement ceiling against that project's
   scoreboard + sheet, iterating on the static `.cube` only (no per-clip cost).
4. **All projects (Phase 1).** Bump `INPUT_HASH_SCHEMA_VERSION` 12→13, add backbone id to
   `grade_hash`, re-grade. Re-run the corpus scoreboard (`_diag_qa_corpus.py`/`_diag_qa_score.py`);
   confirm tiers hold corpus-wide and the "bad" rate drops materially from ~84%.
5. **Phase 2 (end-state).** Make the backbone unconditional, re-express the 18-look catalog as
   deltas on top of it (strip each look's baseline contrast/shoulder), update `_look_only_still`
   to bake the backbone into the look-only reference, and re-run the harness (esp.
   `look_fidelity_cosine`). Bump schema again.

---

## 8. Summary of the recommendation

- **Add a display-referred film-print-emulation backbone (Rec.709 → Rec.709)** as the default
  look, composed at bake time **immediately after `from_working`** and **before** the creative
  LUT grid (`lut_bake.py:118-123`) — a new display→display stage, *not* a change to
  `from_working` (which must stay a clean encode for the solver stat round-trips).
- **Do NOT use a scene-referred AgX/ACES ODT as-is:** our working space is bounded
  display-linearized SDR with no highlight headroom (verified: display 0.5 → linear 0.214), so
  a scene-referred transform double-processes and darkens — the exact trap `tone.py`'s own
  docstring already flags. AgX is the right *long-term* engine only if we later build a true
  scene-referred pipeline.
- **Single best starting candidate:** an **in-house "Filmic Print v1"** backbone, authored from
  the already-validated `kodak_2383` look character (`look_engine.py:444-456`) at reduced
  strength, reusing tested primitives (`_contrast_pivot`, a lowered-start `_tonemap_shoulder`,
  `_apply_split_tone`, `_apply_hue_sat`), baked **once** to a static `.cube`. License-clean,
  correct for our SDR inputs, and cheap (no per-clip bake).
- **Coexistence:** backbone as the neutral default (a), made safe under looks by re-expressing
  looks as deltas (c); ship the risk-free **no-look slice first** (b) since that's the
  84%-bad majority and has no look to double-contrast with.
- **Exposure:** retune Correct to place mid-gray at the backbone's **input** pivot `M_in`
  (≈`TONE_PIVOT` 0.435, s.t. `backbone(M_in)≈0.42`) and raise the mid-gray **placement**
  ceiling above 1.5 (contrast now lives in the backbone, so the cap's original anti-contrast
  reason is gone) with a noise-aware guard — verified the current cap undershoots a dark shot
  by ~0.082.
- **Prove it with the existing harness** (`_diag_qa_score.py` tiered bar: A 100%, B 95%, C
  100%), staged sample-project → all-projects, watching `exposure_band` up, `crushed`/`clipped`
  held or down, `skin_perp_residual`/`saturation_band` in-band, and the Rec.709 FAIL cluster
  collapsing.
