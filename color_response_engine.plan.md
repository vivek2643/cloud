# Parametric Color-Response Engine (the Look instrument)

> Implementation plan. Self-contained: an implementer with **no other context**
> should build this from this file alone. All file:line refs were verified
> against the real code on the date written (see "Line-ref drift" at the end).
> **This builds the instrument, not the looks.** It gives the Look layer a
> vocabulary richer than a CDL slope/offset (per-zone split-tone, hue-vs-hue
> rotation, hue-vs-sat, saturation) baked into a real 3D LUT — so a "look" is a
> small set of numbers, not a hand-crafted `.cube`. Authoring the actual look
> library (YouTube + film + ad styles) is the **next** step, not this one.

---

## 1. Goal & non-goals

**Goal.** Add a **parametric color-response engine** that turns a small
`LookSpec` (a colorist-style parameter set) into a 3D LUT grid, and plug it into
the Look layer as a new `mode == "engine"`. This lets us express the moves a CDL
*can't* — split-toning by tonal zone, per-hue rotation, per-hue saturation — as
genuine per-pixel, per-hue color transforms, baked into the same `.cube` both
preview and export already sample (parity-safe by construction).

Today the Look layer has three modes (`resolver.py::_solve_look` L180–196):
`preset` (a CDL `Grade` delta), `reference` (Reinhard transfer → CDL), and `lut`
(an uploaded `.cube`). CDL presets can only push slope/offset/power/sat — they
approximate teal-orange by "offset biases shadows, slope biases highlights"
(`presets.py` L9–16 admits this), and **cannot** rotate greens toward teal
without also dragging skin, or boost orange saturation without boosting
everything. This engine is the missing fourth mode: **authored looks as
parameters, baked to a LUT.**

**Non-goals.**
- **No look authoring here.** This plan ships the *engine* + wiring + parity +
  1–2 validation looks. The full library (Clean & Natural, Bright & Airy, Punchy
  Vibrant, Warm Cozy, Cool Clean, Moody Cinematic + film + ad styles) is the
  **next** plan (seed from film-stock data + contact-sheet iteration).
- **No spatial / stochastic effects.** Halation and grain are neighbor/random
  ops that *cannot* bake into a color LUT (`lut_bake.py` L12–18 says exactly
  this) — they are a separate later finishing pass. This engine is pointwise
  color only.
- **No new `working_space` string** (same trap as the tone-contrast plan): the
  engine rides the existing `creative_lut_grid` seam, `working_space` stays
  `WORKING_SPACE_V1` / `rec709`.
- **No change to Correct/Match/Balance/Leveling.** The fairness-safe skin/WB
  correction stays upstream; the look sits on top of the fully-corrected image.
- **Legacy / flag-off byte-identical.** `mode == "engine"` is a brand-new mode
  no existing document uses; gated behind `grade_look_engine` (default off). No
  existing preset/reference/lut/no-look path changes one byte.
- **Never clip / never NaN.** Every op is no-op at its default, output clamped
  to 0..1, and hue ops leave achromatic (gray) pixels untouched.

---

## 2. What exists (the seam we plug into)

The engine reuses the **`creative_lut_grid`** seam that uploaded `.cube` looks
already flow through — so preview (WebGL) and export (ffmpeg `lut3d`) inherit it
for free, identically to a hand-authored LUT.

| Piece | Location | Role |
|---|---|---|
| Grid + trilinear sampler | `grade/lut_bake.py::_identity_grid` L30–39, `_sample_lut_trilinear` L42–87 | `_identity_grid(size)` returns a `(size,size,size,3)` float32 grid indexed `[b,g,r]` holding `[r,g,b]` values. **`build_look_grid` produces a grid of exactly this shape.** |
| The bake | `grade/lut_bake.py::bake_cube_text` L90–137 | `to_working(grid) → apply_cdl → from_working(...,contrast) → `**`_sample_lut_trilinear(creative_lut_grid, out)`** L121–123 (in DISPLAY space, after tone). This is where the look grid composes — same slot an uploaded LUT uses. |
| Descriptor → bake | `grade/cache.py::ensure_cube_file` L25–64 | reads the grade dict, resolves `creative_lut_ref` to a grid via `parse_cube_text` L50–54, calls `bake_cube_text` L58–59. **Builds the engine grid here instead when `look_engine` is present.** |
| The Look solve | `grade/resolver.py::_solve_look` L180–196 + `resolve_clip_grade` L199–312 | picks the look mode; sets `creative_lut_ref` L276–278; returns `{cdl, creative_lut_ref, working_space, soft_local, tone_contrast, grade_hash}` L305–312. **Add `mode=="engine"` + a `look_engine` descriptor field.** |
| The hash | `grade/cdl.py::grade_hash` L142–168 | content key every `.cube` is stored under; payload L158–166, `schema_version` default `1` L149. **`look_engine` MUST be in the payload** or the cube won't rebake per-look. |
| Look listing | `routers/grade.py::get_grade_presets` L40–45 → `presets.py::list_presets` L104–108 | the gallery the frontend renders. **Extend to also list engine looks.** |
| Flags + input hash | `config.py` grade flags L176–190; `job.py::compute_input_hash` flags dict L237–247 (`INPUT_HASH_SCHEMA_VERSION = 8` L83); `document["look"]` already hashed L236; `resolve_clip_grade` call L646–655 | gate + cache invalidation. The per-look params live in `document["look"]`, **already in the input hash** — only the flag + schema bump are new. |

**Why this seam is safe.** The creative grid composes at L121–123 *after*
`from_working` (display-encoded, post tone curve) — exactly where a colorist's
creative LUT belongs, and exactly how an uploaded `.cube` already behaves. The
CDL spine, working-space projection, composite guardrails, and tone curve are
all upstream and untouched. `build_look_grid` is pure and deterministic from its
`LookSpec`, so the content hash fully captures it.

---

## 3. Design

### 3.1 `LookSpec` — the parameter vocabulary

A new frozen dataclass in `grade/look_engine.py`. Every field defaults to a
**no-op**, so the empty spec is an *exact* identity grid (parity + "Natural").

```python
@dataclass(frozen=True)
class LookSpec:
    # Extra per-look contrast on top of the global tone curve (reuses
    # tone._contrast_pivot). 0.0 = none.
    contrast: float = 0.0
    # 3-zone split-tone (lift/gamma/gain color balance). Each is an RGB tint
    # added, luma-weighted into that zone; magnitudes are small (~±0.1).
    shadow_tint:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mid_tint:       Tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Hue-vs-hue: rotate a hue band. Each = (center_deg, width_deg, rotate_deg).
    hue_rotate: Tuple[Tuple[float, float, float], ...] = ()
    # Hue-vs-sat: scale saturation in a hue band. Each = (center_deg,
    # width_deg, sat_mult). e.g. (30,40,1.3) pops orange; (140,50,0.7) calms green.
    hue_sat: Tuple[Tuple[float, float, float], ...] = ()
    # Global saturation multiplier (1.0 = unchanged), applied last.
    sat: float = 1.0

    def to_dict(self) -> dict: ...
    @staticmethod
    def from_dict(d) -> "LookSpec": ...        # missing keys -> no-op defaults
    def is_identity(self, eps=1e-9) -> bool: ...  # all fields at no-op
```

`to_dict`/`from_dict` must be **canonical** (sorted, lists not tuples) so the
same look always hashes identically. `is_identity` is what lets an all-default
spec skip the grid entirely (→ byte-identical to no-look).

### 3.2 `build_look_grid` — parameters → 3D LUT grid

```python
def build_look_grid(spec: LookSpec, size: int = 33) -> Tuple["np.ndarray", int]:
    """Evaluate the color-response ops over an identity grid and return
    (grid, size) — the SAME shape parse_cube_text returns, so it drops
    straight into bake_cube_text(creative_lut_grid=...). Pure/deterministic.
    Operates in DISPLAY-encoded RGB (0..1), the space the grid is sampled in."""
    grid = _identity_grid(size)          # from lut_bake; (M,M,M,3) [b,g,r] idx, [r,g,b] vals
    out  = grid
    out  = _apply_split_tone(out, spec)  # zone-weighted RGB tint
    out  = _apply_hue_rotate(out, spec)  # per-band hue rotation (HSV)
    out  = _apply_hue_sat(out, spec)     # per-band saturation (HSV)
    out  = _apply_global_sat(out, spec)  # luma-preserving global sat
    if spec.contrast > 0.0:
        from app.services.l3.grade.tone import _contrast_pivot
        out = _contrast_pivot(out, 1.0 + spec.contrast)
    return np.clip(out, 0.0, 1.0).astype(np.float32), size
```

**Op details (all vectorized numpy, all no-op at default):**

- **`_apply_split_tone`** — luma `y = 0.2126R+0.7152G+0.0722B`. Zone weights:
  `w_shadow = (1-y)**2`, `w_high = y**2`, `w_mid = 1 - w_shadow - w_high`
  (all ≥0, sum to 1). `out = rgb + shadow_tint·w_shadow + mid_tint·w_mid +
  highlight_tint·w_high`. Small magnitudes; this is the teal-orange machinery a
  CDL can only fake.
- **`_apply_hue_rotate` / `_apply_hue_sat`** — convert RGB→HSV (vectorized), for
  each band build a Gaussian weight over hue distance
  `w = exp(-0.5·(Δhue/ (width/2))²)` (hue distance is circular, in degrees), then
  rotate `H += rotate_deg·w` (mod 360) or scale `S *= 1 + (mult-1)·w`. Convert
  back to RGB. **Achromatic guard:** weight every hue op by `S` (or skip pixels
  with `S < 1e-4`) so pure grays/neutrals never shift — critical so the look
  never re-introduces a cast the WB solve removed, and never tints neutrals.
- **`_apply_global_sat`** — `luma + sat·(rgb - luma)` (same math as
  `apply_cdl`'s sat term, L134–137), clamped.

Use a small, dependency-free RGB↔HSV pair inside `look_engine.py` (vectorized;
do **not** loop pixels — a 33³ grid is 35k points, fine as arrays). Order
(split-tone → hue-rotate → hue-sat → global-sat → contrast) is fixed and
documented; it's what the validation looks and contact-sheet tuning assume.

### 3.3 Resolver wiring (new `mode == "engine"`)

In `resolve_clip_grade` (L199–312), add a gate param and resolve the spec:

1. New signature param: `look_engine_enabled: bool = False` (job passes
   `settings.grade_look_engine`). Off → engine mode ignored → identity.
2. After `creative_lut_ref` is computed (L276–278), add:
   ```python
   look_engine = None
   if (look_engine_enabled and sequence_look
           and sequence_look.get("mode") == "engine"):
       spec = _resolve_look_spec(sequence_look)   # catalog by look_id, or inline params
       if spec is not None and not spec.is_identity():
           look_engine = spec.to_dict()
           creative_lut_ref = None    # engine + uploaded LUT are mutually exclusive
   ```
   `_solve_look` returns `Grade()` for `mode=="engine"` (no CDL delta — the whole
   look lives in the grid), so add `if mode == "engine": return Grade()` there.
3. Pass `look_engine=look_engine` into `grade_hash(...)` (L298–304) and add
   `"look_engine": look_engine` to the returned descriptor (L305–312).

`_resolve_look_spec(sequence_look)`: if `sequence_look.get("look_id")` names a
catalog look (§3.6), use it; else build `LookSpec.from_dict(sequence_look.get(
"look_params") or {})` for an inline/preview spec. Returns `None` if neither.

### 3.4 Bake wiring (`ensure_cube_file`, `bake_cube_text`)

`ensure_cube_file` (cache.py L48–59) — build the engine grid when present, in
preference to a fetched LUT (they're mutually exclusive by §3.3):

```python
creative_grid = None
look_engine = grade.get("look_engine")
if look_engine:
    from app.services.l3.grade.look_engine import LookSpec, build_look_grid
    creative_grid = build_look_grid(LookSpec.from_dict(look_engine), size=lut_size)
else:
    lut_ref = grade.get("creative_lut_ref")
    if lut_ref and fetch_creative_lut is not None:
        text = fetch_creative_lut(lut_ref)
        if text:
            creative_grid = parse_cube_text(text)
```

`bake_cube_text` needs **no change** — it already accepts `creative_lut_grid`
and composes it at L121–123. `build_look_grid` returns the exact `(grid, size)`
tuple it expects.

### 3.5 Hash + config + input hash

- **`cdl.py::grade_hash`** (L142–168): add kwarg `look_engine: Optional[Dict]
  = None`; add `"look_engine": look_engine or {}` to the `payload` (L158–166);
  bump the `schema_version` **default `1 → 2`** (L149) + a one-line comment.
  (Anything that changes the baked cube MUST be in the payload.)
- **`config.py`** (after the `grade_tone_contrast` block, L189–190):
  ```python
  # color_response_engine.plan.md: parametric Look engine (mode=="engine") that
  # bakes a LookSpec into the creative LUT grid. Off = engine looks ignored
  # (byte-identical; existing preset/reference/lut/no-look paths never change).
  grade_look_engine: bool = False
  ```
- **`job.py`**: bump `INPUT_HASH_SCHEMA_VERSION` **8 → 9** (L83) + one-line
  comment; add `"grade_look_engine": settings.grade_look_engine` to the `flags`
  dict (L237–247); pass `look_engine_enabled=settings.grade_look_engine` into
  the `resolve_clip_grade` call (L646–655). `document["look"]` (mode/look_id/
  params) is already in the payload (L236), so per-look changes already
  re-hash — only the flag + schema bump are new.

### 3.6 Look catalog (structure only — library authored NEXT)

New `LOOKS: List[EngineLook]` in `grade/look_engine.py` (or a sibling
`looks.py`), where `EngineLook = (look_id, label, description, LookSpec)` —
mirrors `presets.py::Preset` (L26–31). Ship **two** entries only, to prove the
engine end-to-end, NOT to be the product library:

- `engine_identity` — empty `LookSpec()` → exact identity (parity anchor).
- `engine_punchy` — a real look (e.g. `contrast=0.15`, warm `highlight_tint`,
  teal `shadow_tint`, `hue_sat=[(30,40,1.25),(150,50,0.8)]`, `sat=1.08`) — enough
  to eyeball on the Siri frame and confirm per-hue moves land.

Add `list_engine_looks()` (id/label/description, same shape as `list_presets`
L104–108) and surface it from the gallery endpoint (§3.7). The **full library**
(6 YouTube looks + 5–6 film grades + ad styles) is the next plan: seed film
looks from published film-stock characteristic data, hand-author the modern
ones, tune both via the contact sheet.

### 3.7 Gallery endpoint

`routers/grade.py::get_grade_presets` (L40–45) returns `list_presets()`. Add the
engine looks so the frontend can offer them: return
`list_presets() + list_engine_looks()` (tag each entry with its `mode` —
`"preset"` vs `"engine"` — so the frontend sets `look.mode` correctly when the
user picks one). Keep it additive; existing preset entries are unchanged.

---

## 4. Switchability / rollback

- `grade_look_engine` defaults **off** → `look_engine_enabled=False` → `mode==
  "engine"` resolves to no look → `look_engine` never set → `grade_hash`
  identical to today. Existing `preset`/`reference`/`lut`/no-look modes are
  untouched by this plan regardless of the flag.
- Only unconditional change: `INPUT_HASH_SCHEMA_VERSION 8→9` + `grade_hash`
  `schema_version 1→2` — a one-time re-grade producing identical output while
  the flag is off (verify via hash-parity test §6.7–6.8).
- No migration: `look_engine` lives only in the in-memory descriptor + the
  jsonb `resolved_grades.grade_json` (schemaless) + the content hash.
- Clean revert: flag off (runtime), or revert the diffs in `look_engine.py`
  (new file), `resolver.py`, `cache.py`, `cdl.py`, `config.py`, `job.py`,
  `routers/grade.py`.

---

## 5. Phased implementation

### Phase 0 — engine + wiring + hash (wired, gated off)
- New `grade/look_engine.py`: `LookSpec`, RGB↔HSV helpers, the op functions,
  `build_look_grid`, the 2-entry catalog + `list_engine_looks`.
- Wire `resolver.py` (mode + gate + descriptor field), `cache.py` (build grid),
  `cdl.py::grade_hash` (payload + schema bump), `config.py` (flag), `job.py`
  (schema bump + flag in input hash + `look_engine_enabled` arg),
  `routers/grade.py` (list engine looks).
- **Acceptance:** flag off → every `grade_hash` and baked cube byte-identical to
  pre-change; the v1 parity test (`test_lut_bake_v1_parity_direct_vs_baked_cube`,
  test_grade.py ~L110) still green; `build_look_grid(LookSpec())` equals
  `_identity_grid` exactly.

### Phase 1 — turn on + validate the mechanism
- `grade_look_engine=true`; set a test document's `look` to `{"mode":"engine",
  "look_id":"engine_punchy"}`; re-grade; eyeball via the contact-sheet loop
  (`scripts/_diag_look_contact_sheet.py`) on the Siri frame.
- Confirm per-hue moves land: greens calm, orange/skin pops, split-tone reads —
  the moves a CDL preset visibly *can't* make. Confirm neutrals/grays don't
  shift (achromatic guard) and skin isn't plasticky.

### Phase 2 — author the library (SEPARATE, next after halation/grain)
Out of scope here. Seed film looks from film-stock data, hand-author the modern/
creator looks, iterate both on real frames. This plan just makes each new look
"~15 numbers" instead of a hand-built `.cube`.

---

## 6. Testing (`backend/scripts/test_grade.py`)

Mirror the file's conventions (imports ~L24–38; register in `main()`; keep
`print("ok ...")`). Run: `cd backend && .venv/bin/python scripts/test_grade.py`.

1. `test_look_identity_spec_is_identity_grid` — `build_look_grid(LookSpec())`
   equals `_identity_grid(33)` within 1e-6 (empty look = exact no-op).
2. `test_look_grid_deterministic` — same spec → byte-identical grid twice.
3. `test_look_grid_no_clip_no_nan` — for a strong spec, output ∈ [0,1], finite,
   and luma is non-decreasing along a neutral ramp (no inversion/crush).
4. `test_look_split_tone_directional` — `shadow_tint=(0,0,+0.08)` makes a dark
   gray bluer (b up) while a light gray is ~unchanged; `highlight_tint` the
   inverse. Mid-gray shifts by the mid term only.
5. `test_look_hue_rotate_only_targets_band` — a `hue_rotate` on orange rotates an
   orange sample's hue by ~`rotate_deg` while a blue sample is ~unchanged; a
   pure gray is EXACTLY unchanged (achromatic guard).
6. `test_look_hue_sat_only_targets_band` — `hue_sat=(140,50,0.5)` desaturates a
   green sample, leaves an orange sample and a gray unchanged.
7. `test_look_engine_bake_parity` — build a grid from a non-trivial spec, bake it
   via `bake_cube_text(cdl=identity, creative_lut_grid=grid)`, and confirm the
   baked-cube trilinear sample matches directly sampling the built grid within
   tolerance (proves preview == export for engine looks).
8. `test_resolver_engine_off_byte_identical` — `resolve_clip_grade(..., mode=
   "engine", look_engine_enabled=False)` yields the SAME `grade_hash` and no
   `look_engine` field vs. no-look.
9. `test_resolver_engine_on_sets_look_engine_and_changes_hash` — with
   `look_engine_enabled=True` + a non-identity look, the descriptor carries
   `look_engine` and the `grade_hash` differs from flag-off.
10. `test_grade_hash_look_engine_in_payload` — same CDL, `look_engine=None` vs a
    spec dict → different `grade_hash` (cube correctly rebakes per look).

---

## 7. Ops runbook

1. Land Phase 0.
2. `INPUT_HASH_SCHEMA_VERSION` bumped 8→9 + `grade_hash` schema 1→2 (Phase 0).
3. Turn on: `GRADE_LOOK_ENGINE=true` in the worker **and** API env (API must
   match so `compute_input_hash` agrees and the read path serves engine grades —
   same flag/read-path discipline as the tone-contrast + skin-vibrance rollouts).
4. Restart worker + API.
5. Re-grade with an engine look set on a test thread; verify `state=done`,
   `baked>0`; contact-sheet the Siri frame.

---

## 8. Acceptance criteria

- **New expressive moves land.** An engine look visibly rotates one hue band
  (greens→teal) and boosts another's saturation (orange/skin) WITHOUT dragging
  neutrals or other hues — the exact thing the CDL presets can't do.
- **Parity holds.** Preview frame == exported frame within tolerance for an
  engine look (the baked cube is the single source).
- **Flag-off / no-look byte-identical.** Every `grade_hash` identical to
  pre-change when `grade_look_engine` is off or no engine look is selected;
  `legacy` untouched; existing preset/reference/lut modes unchanged.
- **Safe by construction.** Empty `LookSpec` = exact identity grid; all outputs
  clamped to 0..1; grays never shift.

---

## 9. Line-ref drift found vs. the code (verified this session)

- ✅ `lut_bake.py`: `_identity_grid` L30–39 (`[b,g,r]` idx / `[r,g,b]` vals),
  `_sample_lut_trilinear` L42–87, `bake_cube_text` L90–137 composes
  `creative_lut_grid` at L121–123 (AFTER `from_working` L119, display space),
  `parse_cube_text` returns `(grid, size)` L140–181. **`bake_cube_text` needs no
  change** — `build_look_grid` returns exactly this tuple.
- ✅ `cache.py::ensure_cube_file` L25–64: `creative_lut_ref` handling L48–54,
  `working_space`/`tone_contrast` read L56–57, bake call L58–59 — build the
  engine grid here (L48–54 branch).
- ✅ `resolver.py`: `_solve_look` L180–196 (add `mode=="engine"→Grade()`),
  `resolve_clip_grade` L199–312, `creative_lut_ref` set L276–278, `grade_hash`
  call L298–304, return descriptor L305–312 (add `look_engine`), params L199–211
  (add `look_engine_enabled`).
- ✅ `cdl.py::grade_hash` L142–168: payload L158–166, `schema_version` default
  `1` L149 (bump to 2), `tone_contrast` already in payload L165 — add
  `look_engine`.
- ✅ `presets.py`: `Preset` L26–31, `PRESETS` L34–95, `get_preset` L100–101,
  `list_presets` L104–108 — mirror for the engine catalog.
- ✅ `routers/grade.py`: `list_presets` import L29, `/api/grade/presets` L40–45 —
  extend to include engine looks (tag each with `mode`).
- ✅ `config.py`: grade flags block, `grade_tone_contrast`/`_strength` L189–190 —
  add `grade_look_engine` after.
- ✅ `job.py`: `INPUT_HASH_SCHEMA_VERSION = 8` L83 (bump 8→9), flags dict
  L237–247, `document["look"]` in payload L236, `resolve_clip_grade` call
  L646–655 (add `look_engine_enabled`).
- ⚠️ **Do NOT add a new `working_space` string** for the engine — the grid rides
  `creative_lut_grid`; `working_space` stays `WORKING_SPACE_V1`, so
  `correct._project`/`match._proj`/leveling scalars keep projecting.
- ⚠️ **Engine and uploaded-LUT are mutually exclusive** (both fill the single
  `creative_lut_grid` slot) — resolver nulls `creative_lut_ref` when a look
  engine is active (§3.3).
- ⚠️ **Achromatic guard is mandatory** in the hue ops — without weighting hue
  moves by saturation, a near-gray pixel's undefined hue would inject a cast the
  Correct/WB layer just removed.

---

## 10. Remaining pipeline roadmap (after this)

| Next | Stage | Why | Effort |
|---|---|---|---|
| — | **This: color-response engine** | the instrument every look is built from — broadest quality lever after tone | M |
| 2 | **Halation + film grain** (spatial finishing pass, both-sides parity like soft-local; look-scoped intensity) | "feels like film" for the cinematic subset; can't bake into a LUT | M |
| 3 | **Author the look library** (6 YouTube + 5–6 film + ad styles; film from stock data, modern hand-authored, both iterated) | turns the instrument into the product; the actual moat | L (ongoing) |
| 4 | **`already_graded` gate** (segment → `cut_records.look.graded`) | never-worse: already-graded footage is re-corrected today | S–M |
| 5 | **Frontend grade UX** (progress, before/after, per-clip look pick) | make the pipeline legible/controllable | M–L |
