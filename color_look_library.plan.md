# The Look Library (author the catalog — instrument → product)

> Implementation plan. Self-contained: an implementer with **no other context**
> should build this from this file alone. All file:line refs were verified
> against the real code on the date written (see "Line-ref drift" at the end).
> **This is the last major color step.** The engine
> (`color_response_engine.plan.md`) and the film texture
> (`halation_grain.plan.md`) are the *instrument*; this authors the *values* —
> the actual looks users pick. The engine ships 3 validation entries
> (`engine_identity`, `engine_punchy`, `engine_film`); this replaces them with a
> real, YouTube-centric library.

---

## 1. Goal & non-goals

**Goal.** Author a real Look library as `LookSpec` parameter sets in
`look_engine.py::LOOKS`: **6 YouTube/creator looks + ~6 film looks + ~4 ad/
commercial looks**, each tuned on real footage via the contact-sheet loop, and
tag them so the frontend can filter by family. YouTube/creator is the priority
(most users); film is the minority family (carries halation/grain); ad styles
round it out. "As many as we want for now, filter later" — so tag, don't prune.

**Two honest reframes up front** (they shape the whole plan):

1. **"Fit to film-stock data" = informed authoring, NOT spectral simulation.**
   Our `LookSpec` is a *creative* vocabulary (contrast, split-tone, hue-rotate,
   hue-sat, sat, halation, grain), not a physical film model (density curves,
   dye couplers, spectral crosstalk). So a "Kodak 2383" look means *setting our
   knobs to match that stock's known character* (contrasty, teal-orange
   separation, warm highlights, fine grain) — read from published color-science
   descriptions, **not** their `.cube` files (licensing) and **not** a numerical
   spectral fit. This is credible, ownable, and right-sized for a YouTuber
   product. True physical emulation is a much bigger engine with low ROI here —
   explicitly out of scope.

2. **Authoring reveals a real vocabulary gap.** The current `LookSpec` can only
   *add* contrast (`contrast>0`) and *tint* zones — it cannot **lift/fade blacks**
   or **soften** contrast. Those are exactly what "Bright & Airy," "Vintage
   Faded," and high-key ad looks need. So this plan first adds two small,
   bakeable, no-op-at-default knobs (§3.1), then authors the library.

**Non-goals.**
- **No new spatial/tone machinery.** Reuse the engine + halation/grain as-is
  (plus the two tonal knobs in §3.1). No physical film sim.
- **No frontend picker build here.** End-user selection (a gallery that sets
  `document["look"] = {"mode":"engine","look_id":...}`) is the separate Frontend
  Grade UX step. This plan validates via the contact sheet + a directly-set look;
  it notes the picker as the companion that makes the library *usable* (§6).
- **No change to Correct/Match/Balance/Leveling/tone.** Looks sit on top of the
  fully-corrected image; exposure/WB stay the pipeline's job, not a look's.
- **Flag-off / no-look byte-identical.** Everything rides `grade_look_engine`
  (+ `grade_film_texture` for the film family), both default off.

---

## 2. Current state (what we're building on)

| Piece | Location | State |
|---|---|---|
| `LookSpec` vocabulary | `look_engine.py` L41–67 | `contrast, shadow/mid/highlight_tint, hue_rotate[], hue_sat[], sat, halation, grain`. **Add `black_lift` + allow negative `contrast` (§3.1).** |
| `build_look_grid` | `look_engine.py` L253–301 | applies split-tone → hue-rotate → hue-sat → global-sat → contrast (guard `if spec.contrast > 0.0` ~L299). **Extend for the two new knobs.** |
| Catalog | `look_engine.py::LOOKS` L320+ (`EngineLook` L312–317) | 3 validation entries. **Replace with the real library (§3.3).** |
| Gallery listing | `look_engine.py::list_engine_looks` ~L328; `routers/grade.py` `/api/grade/presets` (returns `list_presets()+list_engine_looks()`) | works; **add a `tags`/`family` field per look for frontend filtering.** |
| Contact sheet | `backend/scripts/_diag_look_contact_sheet.py` | renders PROTOTYPE looks (L46–79), not the real catalog. **Extend to render the real `build_look_grid` catalog on real frames (§4).** |
| Film texture | `halation`/`grain` on `LookSpec` (shipped) | film looks set these; requires `grade_film_texture` on. |

---

## 3. Design

### 3.1 Vocabulary extension (two small, bakeable, no-op knobs)

Add to `LookSpec` (after `grain`, L67), both no-op at default:

```python
    # Faded/milky blacks: raise the shadow floor. 0 = off; ~0.03–0.10 tasteful.
    # out = black_lift + (1-black_lift)*out  -> blacks lift to black_lift,
    # whites stay 1 (a classic film "fade"; reduces contrast in the toe).
    black_lift: float = 0.0
```

And **allow `contrast < 0` (soften):** change `build_look_grid`'s guard from
`if spec.contrast > 0.0` to `if spec.contrast != 0.0`, mapping `g = 1.0 +
contrast` and clamping `g >= 0.2` (so `contrast=-0.1 → g=0.9`, a gentle
de-contrast; `_contrast_pivot` is monotonic for any `g>0`, so this never
inverts). Update the field comment to note negative = soften.

**Order in `build_look_grid`:** split-tone → hue-rotate → hue-sat → global-sat →
contrast → **black_lift** (black-lift is a final tonal placement, applied after
contrast so the fade isn't re-crushed). Both stay pointwise/bakeable — no spatial
concern. Extend `to_dict`/`from_dict`/`is_identity` for `black_lift`.

**Tests (grade suite):** identity still exact (default `black_lift=0`,
`contrast=0`); `black_lift` raises min channel value, leaves white at 1;
negative `contrast` reduces midtone slope (softens) and stays monotonic; bake
parity holds with both.

### 3.2 Catalog structure — add a `family` tag

Extend `EngineLook` (L312–317) with `family: str = "creator"` (one of
`"creator"`, `"film"`, `"ad"`), and include it in `list_engine_looks`'s dict
(alongside `mode`). This is the "filter later" hook the frontend uses to group
the gallery. No behavior change — purely metadata.

### 3.3 The first-pass library (starting values — tuned in §4)

These are **informed starting points**, not final. The contact-sheet loop (§4)
sets the final numbers. Values are `LookSpec` fields; omitted fields are no-op.
Film looks carry `halation`/`grain` (need `grade_film_texture`).

**Creator family (priority — most users):**

| look_id | label | Intent | Key params |
|---|---|---|---|
| `clean_natural` | Clean & Natural | Safe true-to-life default | `contrast=0.05, sat=1.03` |
| `bright_airy` | Bright & Airy | Lifted, soft, warm (vlog) | `contrast=-0.10, black_lift=0.05, shadow_tint=(0.02,0.015,0), highlight_tint=(0.03,0.02,-0.01), sat=0.98` |
| `punchy_vibrant` | Punchy Vibrant | Saturated + contrasty (tech/gaming) | `contrast=0.20, sat=1.15, hue_sat=[(30,40,1.2),(150,50,0.9)]` |
| `warm_cozy` | Warm Cozy | Gentle orange warmth (sit-down) | `contrast=0.08, shadow_tint=(0.02,0.005,-0.01), highlight_tint=(0.05,0.02,-0.03), hue_sat=[(30,45,1.15)], sat=1.05` |
| `cool_clean` | Cool Clean | Slightly cool, crisp (reviewer) | `contrast=0.12, shadow_tint=(-0.02,0,0.03), highlight_tint=(-0.01,0,0.02), sat=1.05` |
| `moody_cinematic` | Moody Cinematic | Desat, teal shadows, gentle crush | `contrast=0.18, shadow_tint=(-0.04,0,0.05), highlight_tint=(0.03,0.015,-0.02), hue_sat=[(150,50,0.7)], sat=0.82` |

**Film family (informed by stock character; carry halation+grain):**

| look_id | label | Character emulated | Key params |
|---|---|---|---|
| `kodak_2383` | Kodak 2383 | Print film: contrasty, teal-orange, warm highs | `contrast=0.22, shadow_tint=(-0.04,0,0.05), highlight_tint=(0.05,0.02,-0.03), hue_sat=[(30,40,1.2),(150,50,0.85)], sat=1.05, halation=0.25, grain=0.04` |
| `fuji_eterna` | Fuji Eterna | Soft, low-sat, green-leaning | `contrast=0.04, mid_tint=(-0.01,0.02,-0.01), sat=0.88, halation=0.15, grain=0.05` |
| `vision3_250d` | Vision3 250D | Natural neg, gentle, slight warm | `contrast=0.08, black_lift=0.02, mid_tint=(0.02,0.01,-0.01), sat=0.98, halation=0.12, grain=0.05` |
| `portra_400` | Portra 400 | Warm, skin-flattering, soft, low-con | `contrast=0.0, black_lift=0.03, highlight_tint=(0.04,0.02,-0.02), hue_sat=[(30,45,1.1)], sat=0.95, halation=0.10, grain=0.04` |
| `vintage_faded` | Vintage Faded | Lifted milky blacks, warm, desat | `contrast=-0.05, black_lift=0.08, mid_tint=(0.03,0.015,-0.02), sat=0.80, halation=0.20, grain=0.06` |
| `bw_film` | B&W Film | Near-mono, contrasty, grainy | `contrast=0.20, sat=0.05, halation=0.10, grain=0.06` |

**Ad / commercial family:**

| look_id | label | Intent | Key params |
|---|---|---|---|
| `clean_commercial` | Clean Commercial | Crisp, neutral, punchy (no grain) | `contrast=0.15, sat=1.08` |
| `high_key_beauty` | High-Key Beauty | Bright, airy, soft skin, warm | `contrast=-0.08, black_lift=0.05, highlight_tint=(0.03,0.02,-0.01), hue_sat=[(30,45,1.1)], sat=1.0` |
| `tech_sleek` | Tech Sleek | Cool, high-contrast, restrained sat | `contrast=0.20, shadow_tint=(-0.03,0,0.04), highlight_tint=(-0.01,0,0.02), sat=0.95` |
| `food_vibrant` | Food Vibrant | Warm, saturated, appetizing | `contrast=0.12, highlight_tint=(0.04,0.02,-0.02), hue_sat=[(30,45,1.25),(60,40,1.15)], sat=1.18` |

Example full entry (the pattern for all 16):

```python
EngineLook(
    "punchy_vibrant", "Punchy Vibrant",
    "Saturated, contrasty pop for tech / gaming / product footage.",
    LookSpec(contrast=0.20, sat=1.15, hue_sat=((30.0,40.0,1.2),(150.0,50.0,0.9))),
    family="creator",
),
```

Keep `engine_identity` (parity anchor) in the catalog; drop `engine_punchy`/
`engine_film` once the real library subsumes them (or relabel `engine_film` →
`kodak_2383`).

### 3.4 Fairness + safety (unchanged, re-stated)

- Skin/WB correction stays **upstream** (fairness-safe locus decomposition,
  `color_skin_vibrance`). Looks that "pop skin/orange" do so as a *creative,
  user-selected* hue-band saturation — bounded, and never a per-skin-tone
  privilege. The achromatic guard (grays never shift) still holds for every look.
- Every knob is no-op at default and bounded; `build_look_grid` clamps 0..1.
  Verify each authored look: no channel clip on a real frame, skin stays healthy
  (not plastic/orange), grays stay neutral.

---

## 4. Authoring loop (how the values actually get set)

The values in §3.3 are starting points; final numbers come from eyeballing real
output. Extend `scripts/_diag_look_contact_sheet.py` to render the **real
catalog** (not the prototypes at L46–79):

1. For each `EngineLook` in `LOOKS`, `build_look_grid(look.spec)` → sample the
   corrected frame through it (same `_sample_lut_trilinear` path the bake uses),
   so the contact sheet shows *exactly* what export/preview will produce.
2. Render on **3 real frames**: a talking-head (Siri Reel — skin check), a
   highlight-heavy clip (halation check), and a flat/low-light clip (grain +
   black-lift check).
3. Grid the families into labeled sheets; iterate each look's params until it
   reads as intended without clipping / plastic skin / shifted neutrals.
4. (Film looks) render *with* halation/grain applied (reuse the halation/grain
   apply math or approximate in the sheet) so the texture is tuned in context.

This is taste work — budget iteration, not a one-shot. The sheet makes it fast.

---

## 5. Phased implementation

### Phase 0 — vocabulary extension (gated by nothing; no-op at default)
- `look_engine.py`: add `black_lift`; allow negative `contrast` (§3.1); extend
  `to_dict`/`from_dict`/`is_identity`/`build_look_grid`.
- Tests (§3.1). **Acceptance:** identity still exact; bake parity holds; existing
  3 catalog entries unchanged in output.

### Phase 1 — author the library
- Replace `LOOKS` with the 16 entries (§3.3) + `family` tag on `EngineLook` and
  in `list_engine_looks` (§3.2).
- **Acceptance:** `/api/grade/presets` lists all looks with `mode:"engine"` +
  `family`; each `build_look_grid` is finite, clamped, non-identity.

### Phase 2 — tune on real footage
- Extend the contact sheet (§4); iterate the params on the 3 real frames until
  each look reads right. Commit the tuned values.

### Phase 3 — turn on + validate end-to-end
- `grade_look_engine=true` (+ `grade_film_texture=true` for film looks); set a
  test thread's look to each family representative; real render; confirm
  preview == export and the look reads as intended on real footage.

---

## 6. Companion dependency (not built here)

For end users to actually *use* the library, the frontend grade view must list
engine looks (grouped by `family`) and set `document["look"] = {"mode":"engine",
"look_id":...}` when picked. That's the **Frontend Grade UX** step (roadmap #5).
Until then, looks are validated by setting `document["look"]` directly + the
contact sheet. Flag this so the library doesn't look "shipped but unreachable."

---

## 7. Testing (`backend/scripts/test_grade.py`)

1. `test_black_lift_raises_floor_keeps_white` — `LookSpec(black_lift=0.08)`:
   min channel ≈ 0.08, white ≈ 1.0, monotonic.
2. `test_negative_contrast_softens_monotonic` — `contrast=-0.1` reduces midtone
   slope vs identity, still monotonic non-decreasing, endpoints pinned.
3. `test_identity_still_exact_with_new_fields` — `LookSpec()` (defaults) builds
   the exact identity grid (regression on the parity anchor).
4. `test_catalog_all_looks_valid` — every `LOOKS` entry: `build_look_grid`
   finite + clamped; `list_engine_looks` includes `family` ∈ {creator,film,ad}
   and `mode=="engine"`; no duplicate `look_id`.
5. `test_bake_parity_with_new_knobs` — a spec using `black_lift` + negative
   `contrast` bakes and trilinear-samples within tolerance (preview == export).
6. `test_film_looks_carry_texture` — every `family=="film"` look has
   `halation>0` or `grain>0`; every `creator`/`ad` look (except intentional)
   has `halation==0` (grain optional).

Full suite + pyflakes clean; run `cd backend && .venv/bin/python
scripts/test_grade.py`.

---

## 8. Acceptance criteria

- **Each look reads as its intent** on real frames (contact sheet + a render):
  Bright & Airy is lifted/soft (black-lift working), Punchy pops, Moody is
  crushed/teal/desat, film looks show grain+halation and stock character.
- **Safe on real footage.** No channel clipping, skin stays healthy, neutrals
  stay neutral, no banding — verified per look, not just synthetically.
- **Parity holds.** Preview == export for a representative look per family.
- **Flag-off / no-look byte-identical.** `grade_look_engine` off → no engine
  look resolves → identical to today; identity look stays exact.

---

## 9. Line-ref drift found vs. the code (verify before editing — shifted by prior plans)

- ✅ `look_engine.py`: `LookSpec` L41–67 (`halation`/`grain` L66–67 shipped);
  `to_dict`/`from_dict`/`is_identity` immediately after (~L69–130);
  `build_look_grid` L253–301 (contrast guard ~L299 `if spec.contrast > 0.0` →
  change to `!= 0.0`, clamp `g>=0.2`); `EngineLook` L312–317; `LOOKS` L320+
  (3 entries: `engine_identity`, `engine_punchy`, `engine_film`);
  `list_engine_looks` ~L328 (add `family`).
- ✅ `routers/grade.py`: `/api/grade/presets` returns `list_presets() +
  list_engine_looks()` — `family` flows through automatically once added to the
  dict.
- ✅ `_diag_look_contact_sheet.py`: prototype looks L46–79 — extend to render the
  real catalog via `build_look_grid` + `_sample_lut_trilinear`.
- ✅ `resolver.py` / `cache.py` / `cdl.py` / `job.py`: **no change** — the engine
  + descriptor + hash + bake already handle any `LookSpec`; more catalog entries
  and two more no-op-at-default fields need no new plumbing (the fields flow
  through `to_dict` → `look_engine` payload automatically).
- ⚠️ **No `INPUT_HASH_SCHEMA_VERSION` / `grade_hash` schema bump needed** unless
  Phase 0's math change alters an EXISTING catalog look's output. It doesn't
  (new fields are no-op at default; existing entries don't set them) — but if you
  relabel/retune `engine_*`, that changes `document["look"]` per selection, which
  already re-hashes. Only bump if you change `build_look_grid`'s math for a value
  an existing persisted look already uses.
- ⚠️ **Film looks are inert without `grade_film_texture`** — their color still
  applies (engine flag), but halation/grain need the texture flag too. State this
  in each film look's description so it's not mistaken for a bug.

---

## 10. Roadmap after this

| Next | Stage | Why | Effort |
|---|---|---|---|
| ✅ | color-response engine | the instrument | done |
| ✅ | halation + grain | film-family texture | done |
| — | **This: the look library** | instrument → product; the moat | M (author) + iteration |
| next | **Frontend grade UX** (gallery grouped by `family`, before/after, per-clip pick, progress) | makes the library reachable + the pipeline legible | M–L |
| later | `already_graded` gate; export bundle (`.cdl`/`.cube`); auto narrative arc | completeness / polish | S–L |
