# Cinematic Tone / Contrast Curve (v1 tone map — the flat-base fix)

> Implementation plan. Self-contained: an implementer with **no other context**
> should build this from this file alone. All file:line refs were verified
> against the real code on the date written (see "Line-ref drift" at the end).
> **This is the single biggest universal quality win** and is independent of the
> Look-layer decision — it makes every clip and every future look sit on a
> contrasty, film-like base instead of the flat one that reads as "not graded."

---

## 1. Goal & non-goals

**Goal.** Add a real **filmic contrast S-curve** to the v1 tone map so corrected
footage gets midtone contrast and film-like tonal shape. Today `tone.py::
from_working` only *compresses highlights above 0.8 and is exact identity below
it* — so it never adds contrast, and every grade (and every look on top) reads
flat/washed. This closes that gap, baked into the same `.cube` both preview and
export sample (parity-safe by construction).

**Non-goals.**
- **No new working-space string.** The curve is carried as a *descriptor
  parameter* threaded into `from_working`, NOT a new `working_space` value —
  because `correct.py::_project`, `match.py::_proj`, and the leveling scalars
  all branch on `working_space == WORKING_SPACE_V1` and a new value would
  silently turn their working-space projection back into identity (a real,
  subtle regression). `working_space` stays `WORKING_SPACE_V1`.
- **No Look-layer work.** Looks are deferred; this is the *base tone* under them.
- **No local/spatial or per-hue operations.** A pointwise tone curve only. Grain,
  halation, and per-hue moves are separate later stages.
- **Legacy byte-identical.** Contrast defaults to 0.0 → `from_working` behaves
  exactly as today; `legacy` and any non-v1 space stay pure identity.
- **Never clip / never invert.** The curve is monotonic and endpoint-pinned
  (`0→0`, `1→1`), so it can neither blow highlights nor crush true black beyond
  today, regardless of strength.

---

## 2. What exists (the single parity seam)

The entire grade bakes through **one** function pair, so a change here is
inherited by preview (WebGL) and export (ffmpeg `lut3d`) automatically:

| Piece | Location | Role |
|---|---|---|
| The tone map | `grade/tone.py::from_working` L108–120 | linear working RGB → filmic highlight shoulder (`_tonemap_shoulder` L95–105, identity below `_SHOULDER_START=0.8` L50) → sRGB OETF encode. **The contrast curve is added here.** |
| The bake | `grade/lut_bake.py::bake_cube_text` L90–132 | `to_working(grid) → apply_cdl → from_working(...)` (L112–114). The ONE call site that must pass the contrast through. |
| Descriptor → bake | `grade/cache.py::ensure_cube_file` L25–63 | reads the resolved grade dict, pulls `working_space` (L56), calls `bake_cube_text` (L57–58). **Threads `tone_contrast` in.** |
| The descriptor | `grade/resolver.py::resolve_clip_grade` L285–297 (return dict) | `{cdl, creative_lut_ref, working_space, soft_local, grade_hash}`. **Add `tone_contrast`.** |
| The hash | `grade/cdl.py::grade_hash` L142–164 | content key every baked `.cube` is stored under. **`tone_contrast` MUST be in its payload** or the cube won't rebake when the curve changes. |
| Flags + input hash | `config.py` (grade flags ~L146–176), `job.py::compute_input_hash` flags dict L230–237, `INPUT_HASH_SCHEMA_VERSION = 7` (`job.py` L76) | gate + cache invalidation. |

**Why this seam is safe:** `from_working` is called for the bake only; the
scalar projections in `correct.py`/`match.py` use `to_working_scalar` (inverse
EOTF, no tone curve) and are untouched. The composite guardrails
(`resolver._clamp_composite_v1`) bound the CDL in *linear* space *before*
`from_working`; the contrast curve is a fixed monotonic display-side finish
applied *after*, so the guardrails still hold and the curve never clips.

---

## 3. Design

### 3.1 The curve — pivoted, endpoint-pinned contrast (in display space)

Applied inside `from_working` to the final **display-encoded** output (after the
existing shoulder + sRGB OETF), only when `working_space == WORKING_SPACE_V1`
and `contrast > 0`:

```
TONE_PIVOT = 0.435          # display-space pivot (~linear mid-gray 0.18); contrast rotates about here
```

A **pivoted double-power S** — the standard "contrast around a pivot," monotonic,
`C1`-continuous at the pivot, and pinned at both ends so it can never clip:

```python
def _contrast_pivot(x, g, p=TONE_PIVOT):
    # g = 1.0 -> identity; g > 1.0 -> more contrast (darker below pivot,
    # brighter above). Endpoints fixed: f(0)=0, f(1)=1, f(p)=p. Slope at the
    # pivot is g on both sides (C1). Monotonic increasing for g > 0.
    lo = p * np.power(x / p, g)
    hi = 1.0 - (1.0 - p) * np.power((1.0 - x) / (1.0 - p), g)
    return np.where(x <= p, lo, hi)
```

- Strength maps `g = 1.0 + contrast` (so `contrast=0` → `g=1` → exact identity;
  `contrast≈0.9` → `g≈1.9`, the tasteful default validated on the Siri frame in
  the contact-sheet loop — start there, tune per §6).
- **Toe** (film-black lift) is intentionally OMITTED (default) so true blacks
  stay at 0 (no milky blacks; consistent with the correct/composite floors).
  Expose it later as a look concern, not a base-tone one.
- This is the "slot": a fuller filmic/AgX/ACES output transform can replace
  `_contrast_pivot` later without touching any caller (same "libraries
  deferred" principle as the original `tone.py`).

### 3.2 `from_working` signature

```python
def from_working(rgb_working, working_space, *, contrast: float = 0.0):
    ...  # existing: clip>=0, shoulder, sRGB OETF -> `display`
    if working_space == WORKING_SPACE_V1 and contrast > 0.0:
        display = np.clip(_contrast_pivot(display, 1.0 + contrast), 0.0, 1.0)
    return display
```

`contrast=0.0` (default) reproduces today's bytes exactly. `from_working_scalar`
(L82–92) keeps `contrast=0` — the scalar projection must NOT get the display
contrast (it's used to round-trip stats, not to finish pixels); leave it as-is.

### 3.3 Threading the parameter (no working_space change)

1. **`resolver.py`** `resolve_clip_grade`: add param `tone_contrast: float = 0.0`;
   put `"tone_contrast": tone_contrast` in the returned descriptor (L285–297) and
   pass it into `grade_hash(...)` (§3.4). The caller (`job.py`) supplies the
   value from the flag.
2. **`cdl.py`** `grade_hash`: add kwarg `tone_contrast: float = 0.0` and
   `"tone_contrast": tone_contrast` to the `payload` dict (L155–162). (Anything
   that changes the baked cube MUST be in the payload — this is why the cube
   rebakes when the curve turns on.)
3. **`cache.py`** `ensure_cube_file`: read `tone_contrast = float(grade.get(
   "tone_contrast") or 0.0)` and pass it into `bake_cube_text(...)` (L57–58).
4. **`lut_bake.py`** `bake_cube_text`: add kwarg `tone_contrast: float = 0.0`;
   pass it to `from_working(graded, working_space, contrast=tone_contrast)`
   (L114).
5. **`job.py`** `run_grade_job` (L636–645): pass `tone_contrast=(
   settings.grade_tone_contrast_strength if settings.grade_tone_contrast and
   settings.grade_pipeline == "v1" else 0.0)` into `resolve_clip_grade`.

### 3.4 Config + gate

- **`config.py`** (after the `grade_skin_vibrance` block): add
  ```python
  # color_tone_contrast.plan.md: adds a filmic contrast S-curve to the v1 tone
  # map (tone.from_working), baked into the cube so preview == export. Off =
  # today's shoulder-only tone map (byte-identical). v1-only.
  grade_tone_contrast: bool = False
  grade_tone_contrast_strength: float = 0.9   # g = 1 + strength; tune per contact sheet
  ```
- **`job.py`**: bump `INPUT_HASH_SCHEMA_VERSION` **7 → 8** (L76) + one-line
  comment; add `"grade_tone_contrast": settings.grade_tone_contrast` and
  `"grade_tone_contrast_strength": settings.grade_tone_contrast_strength` to the
  `flags` dict in `compute_input_hash` (L230–237).

---

## 4. Switchability / rollback

- `grade_tone_contrast` defaults **off** → `tone_contrast=0.0` everywhere →
  `from_working` is byte-identical to today; every `grade_hash` unchanged
  (verify via hash-parity test, §7).
- Only unconditional change: `INPUT_HASH_SCHEMA_VERSION 7→8`, forcing a one-time
  re-grade producing identical output while the flag is off.
- No migration (no new stored fields; `tone_contrast` lives only in the
  in-memory descriptor + the content hash).
- Clean revert: flag off (runtime), or revert the small diffs in `tone.py`,
  `lut_bake.py`, `cache.py`, `resolver.py`, `cdl.py`, `job.py`, `config.py`.

---

## 5. Phased implementation

### Phase 0 — curve + flag + hash (wired, gated off)
- `tone.py`: add `_contrast_pivot` + `TONE_PIVOT` + the `contrast` kwarg on
  `from_working`.
- Thread `tone_contrast` through `bake_cube_text`, `ensure_cube_file`,
  `resolve_clip_grade`, `grade_hash` (§3.3).
- `config.py` flags; `job.py` hash bump + flag wiring.
- **Acceptance:** flag off → every `grade_hash` and baked cube byte-identical to
  pre-change; the v1 parity test (`test_lut_bake_v1_parity_direct_vs_baked_cube`,
  test_grade.py L110) still green.

### Phase 1 — turn it on + tune
- Set `grade_tone_contrast=true`; re-grade; eyeball via the contact-sheet loop
  (`scripts/_diag_look_contact_sheet.py`) on Siri + a CANON clip.
- Tune `grade_tone_contrast_strength` (and `TONE_PIVOT` if needed) until the base
  reads graded without crushing shadows or plasticky skin.

---

## 6. Testing (`backend/scripts/test_grade.py`)

Mirror the file's conventions (imports L24–38; register in `main()` ~L1765; keep
`print("ok ...")`). Run: `cd backend && .venv/bin/python scripts/test_grade.py`.

1. `test_tone_contrast_zero_is_exact_identity` — `from_working(x, WS, contrast=0)`
   equals today's `from_working(x, WS)` bit-for-bit across `x∈{0,.18,.5,.8,1}`.
2. `test_tone_contrast_endpoints_pinned` — `contrast=0.9`: `f(0)≈0`, `f(1)≈1`
   (never clips past the range).
3. `test_tone_contrast_monotonic` — output non-decreasing across a dense ramp
   (no inversion at any strength in `{0.3,0.9,1.5}`).
4. `test_tone_contrast_increases_midtone_slope` — with contrast on, values below
   `TONE_PIVOT` move DOWN and above move UP vs. contrast off (real S-shape);
   the pivot itself is fixed.
5. `test_tone_contrast_legacy_and_nonv1_identity` — `from_working(x, "rec709",
   contrast=0.9)` and `"rec709_legacy"` are exact identity (curve is v1-only).
6. `test_bake_parity_with_contrast` — extend the L110 parity test: a direct
   `to_working→apply_cdl→from_working(contrast=k)` matches the baked-cube
   trilinear sample within tolerance (proves preview == export WITH the curve).
7. `test_resolver_flag_off_byte_identical` — `resolve_clip_grade(...,
   tone_contrast=0.0)` yields the same `grade_hash` as before this plan.
8. `test_grade_hash_changes_with_tone_contrast` — same CDL, `tone_contrast`
   0.0 vs 0.9 → different `grade_hash` (cube correctly rebakes).

---

## 7. Ops runbook

1. Land Phase 0.
2. `INPUT_HASH_SCHEMA_VERSION` bumped 7→8 (Phase 0).
3. Turn on: `GRADE_TONE_CONTRAST=true` (and optionally
   `GRADE_TONE_CONTRAST_STRENGTH=0.9`) in the worker **and** API env
   (API must match so `compute_input_hash` agrees and the read-path shows the
   contrasted grade — same flag/read-path discipline as the skin-vibrance
   rollout).
4. Restart worker + API.
5. Re-grade: `cd backend && PYTHONPATH=. GRADE_TONE_CONTRAST=true .venv/bin/
   python scripts/_grade_all_projects.py`.
6. Verify: `state=done`, `baked>0`; contact-sheet a Siri + CANON frame; confirm
   contrast/pop without shadow crush or blown highlights.

---

## 8. Acceptance criteria

- **Adds contrast, stays safe.** With the flag on, midtone contrast visibly
  increases; a display-`0.15` shadow probe stays ≥ its correct/composite floor
  (no new crush) and white stays ≤ 1 (no new clip). Curve monotonic at every
  strength.
- **Parity holds.** Preview frame == exported frame within tolerance WITH the
  curve (the baked cube is the single source).
- **Flag-off byte-identical.** Every `grade_hash` identical to pre-change v1 when
  `grade_tone_contrast` is off; `legacy` untouched.

---

## 9. Line-ref drift found vs. the code (verified this session)

- ✅ `tone.py`: `from_working` L108–120, `_tonemap_shoulder` L95–105,
  `_SHOULDER_START=0.8` L50, `WORKING_SPACE_V1="rec709_v1"` L41,
  `from_working_scalar` L82–92 (keep `contrast=0`).
- ✅ `lut_bake.bake_cube_text` calls `from_working(graded, working_space)` at
  L114 — the one site to pass `tone_contrast`.
- ✅ `cache.ensure_cube_file` L56–58 reads `working_space`, calls
  `bake_cube_text` — thread `tone_contrast` here.
- ✅ `resolver.resolve_clip_grade` returns the descriptor at L285–297 (no
  `tone_contrast` yet); `grade_hash` called at L285.
- ✅ `cdl.grade_hash` payload L155–162 (`schema_version` default 1) — add
  `tone_contrast` to the payload.
- ✅ `job.py`: `INPUT_HASH_SCHEMA_VERSION = 7` (L76, already bumped by the
  skin-vibrance plan), flags dict L230–237, `resolve_clip_grade` call L636–645.
- ⚠️ **Do NOT introduce a new `working_space` string** for the curve:
  `correct._project` (correct.py L88 `if working_space != WORKING_SPACE_V1`),
  `match._proj` (match.py L45), and job.py's leveling scalars all treat any
  non-`WORKING_SPACE_V1` value as identity/legacy — a new space would silently
  disable their projection. Carry the curve as `tone_contrast` instead.
- ⚠️ Deferred-look note: `resolver._corrected_source_stats` projects stats
  through `to_working/from_working` WITHOUT the contrast; when the Look layer is
  revived, decide whether the reference transfer should see the contrasted
  output. Irrelevant while looks are deferred.

---

## 10. Remaining pipeline roadmap (after this)

This plan is step 1 of "complete the pipeline." Recommended order by leverage,
each its own plan when we get to it:

| Next | Stage | Why | Effort |
|---|---|---|---|
| 2 | **Halation + film grain** (finishing pass, both-sides parity like soft-local) | ~half of "feels cinematic," universal, look-independent | M |
| 3 | **`already_graded` gate** (segment → `cut_records.look.graded`) | never-worse hole: already-graded footage is re-corrected today | S–M |
| 4 | **Soft-local depth** (subject pop / graduated sky off `subject_box`) | dimensionality; reuses subject boxes already resolved | M |
| 5 | **Export bundle** (`.cdl`/`.ccc` + `.cube`, XML/EDL round-trip, editable-vs-baked) | pro-workflow completeness | M |
| 6 | **Frontend grade UX** (grading progress bar, before/after, per-clip override) | make the pipeline legible/controllable | M–L |
| 7 | **Auto narrative arc** (EDSO categorical intent → subtle tonal arc) | invisible-by-default polish | M |
| — | **Look layer** (real `.cube` film emulation via OEM license or own engine) | deferred by decision; rides the CDL spine + this tone base | L |
