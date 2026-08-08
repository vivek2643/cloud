# Halation + Film Grain (spatial finishing pass — the "feels like film" layer)

> Implementation plan. Self-contained: an implementer with **no other context**
> should build this from this file alone. All file:line refs were verified
> against the real code on the date written (see "Line-ref drift" at the end).
> **These are spatial/stochastic effects that CANNOT bake into a color LUT** (a
> 3D LUT is a pointwise value→value map with no notion of neighboring pixels or
> randomness — `lut_bake.py` L12–18 says exactly this). They apply as an
> additional deterministic pass on BOTH sides (export ffmpeg + preview WebGL),
> exactly like the existing `soft_local` vignette — same approximate-parity
> trade (`softlocal.py` L17–26), not the byte-exact color-value contract.

---

## 1. Goal & non-goals

**Goal.** Add two **look-scoped** film-texture effects to the grade's spatial
pass:
- **Halation** — a soft red/orange glow bleeding out of bright highlights (light
  scatter in film base). The single biggest "shot on film, not phone" tell.
- **Film grain** — subtle stochastic luminance texture. Adds cohesion, hides
  banding on flat gradients.

Both are **declared by the selected Look** (a film look sets its own grain/
halation intensity; a clean YouTube look sets zero) and applied as a spatial
finishing pass over the already-color-graded picture, on preview AND export.

**Why look-scoped, not global** (the user's explicit decision): grain/halation
belong to the film-look family — a clean modern/creator look wants *zero*. So
they ride the Look, off unless a look asks for them; the base pipeline (and every
no-look / non-film clip) is untouched.

**Non-goals.**
- **Not baked into the LUT.** Impossible by construction (neighbor + random).
  They join the `soft_local` spatial descriptor the vignette already uses.
- **Not byte-exact parity.** Export (ffmpeg) and preview (WebGL) use independent
  implementations that read as the *same* effect at the same strength but are
  not pixel-identical — the exact same bounded trade `softlocal.py` L17–26
  already documents for the vignette. The color LUT's byte-exact contract is
  unaffected (halation/grain run *after* it, on both sides).
- **No new global always-on behavior.** Gated behind `grade_film_texture`
  (default off) AND only non-zero when a look carries texture params → triple-
  safe (flag off, or no engine look, or look with zero texture = untouched).
- **No color-response / tone changes.** This is purely the spatial finish on top
  of the existing color pipeline (color_response_engine + tone_contrast).

---

## 2. What exists (the spatial-pass seam we extend)

The vignette already proves the whole both-sides spatial-pass path end to end —
we extend the same three seams rather than inventing new ones.

| Piece | Location | Role |
|---|---|---|
| Spatial descriptor | `grade/resolver.py` L316–322 builds `soft_local = {"vignette": {...}, "subject_box": [...]}`; carried in the return dict L332–339 | **Add `halation` + `grain` keys** (sourced from the engine look, §3.3). |
| In the hash | `grade/cdl.py::grade_hash` L161–170 — `soft_local` already in the payload L166 | **No signature change**: bigger `soft_local` dict re-hashes automatically. Bump `schema_version` for the one-time reindex. |
| Export apply | `grade/softlocal.py::vignette_ffmpeg_filter` L55–66; `render/compositor.py::_transform_vf` L495–500 (LUT then vignette), threaded via `_produce_segment` L526–537 + overlay path L613–619; read at L225 & L615 | **Add `halation_ffmpeg` + `grain_ffmpeg` builders; append after the vignette.** |
| Preview apply | `frontend/.../preview/lut-gl.ts` — fragment shader vignette L91–98, `VignetteParams` L116–120, `draw()` L129–134/251–288, uniforms L227–228/284–285 | **Add halation (FBO blur pass) + grain (shader hash-noise); thread params like `VignetteParams`.** |
| The look source | `grade/look_engine.py::LookSpec` L41–60, `EngineLook`/`LOOKS` L292–319 | **Add `halation`/`grain` fields to `LookSpec`; `build_look_grid` ignores them (they're spatial, not color).** |
| Flags + input hash | `config.py` grade flags (add after `grade_look_engine` L195); `job.py::compute_input_hash` flags dict L248–252, `INPUT_HASH_SCHEMA_VERSION = 9` L87 | gate + cache invalidation. |
| Render-cache token | `render/compositor.py::_grade_key` L298–308 (already collapses identity incl. `soft_local`) | already correct — a non-empty `soft_local` (halation/grain) already forces a distinct cache key. **Verify, no change expected.** |

**Why this seam is safe.** The vignette already runs after the LUT on both
sides and re-keys the render/segment cache via `soft_local` in `grade_hash` /
`_grade_key`. Halation + grain slot into the identical descriptor and apply
points, so cache invalidation, hashing, and the read path all work unchanged —
only the two apply implementations (ffmpeg + WebGL) are new.

---

## 3. Design

### 3.1 The look-scoped params (on `LookSpec`)

Add to `LookSpec` (`look_engine.py` L41–60), both defaulting to no-op:

```python
    # Spatial finishing (halation_grain.plan.md) — NOT baked into the
    # grid (build_look_grid ignores these); routed into the soft_local
    # spatial descriptor by resolve_clip_grade instead.
    halation: float = 0.0   # 0 = off; ~0.15–0.4 tasteful. Glow strength.
    grain: float = 0.0      # 0 = off; ~0.02–0.08 tasteful. Noise amplitude.
```

- Extend `to_dict`/`from_dict`/`is_identity` to include them (so a look carrying
  only grain is still non-identity and hashes distinctly).
- **`build_look_grid` does NOT read them** — assert this in a test. Color is
  bakeable; texture is not. Keeping both on one `LookSpec` preserves "a look is
  one parameter set" while routing each to the right pass.
- Optional fixed sub-params (keep them **constants** in `softlocal.py`, not spec
  fields, to keep the vocabulary small): halation highlight threshold
  `HALATION_THRESHOLD=0.75`, blur sigma `HALATION_SIGMA≈8` (scaled by frame
  height), tint `(1.0, 0.35, 0.1)` (red-orange); grain is luma-only, size 1px.

### 3.2 `soft_local` descriptor shape (resolver)

`resolve_clip_grade` (resolver.py L316–322) currently only fills `soft_local`
from `vignette_strength`. Extend so an active engine look's texture params also
populate it:

```python
soft_local = None
vignette = solve_vignette(subject_box, strength=float(vignette_strength)) if vignette_strength else None
halation = None
grain = None
if look_engine_enabled and film_texture_enabled and look_engine:
    spec = LookSpec.from_dict(look_engine)          # already resolved above
    if spec.halation > 0.0: halation = {"strength": min(1.0, spec.halation)}
    if spec.grain > 0.0:    grain = {"strength": min(1.0, spec.grain)}
if vignette or halation or grain:
    soft_local = {}
    if vignette: soft_local["vignette"] = vignette
    if halation: soft_local["halation"] = halation
    if grain:    soft_local["grain"] = grain
    if subject_box: soft_local["subject_box"] = list(subject_box)
```

`film_texture_enabled` is a new `resolve_clip_grade` param (job passes
`settings.grade_film_texture`). Off → no halation/grain keys → `soft_local`
byte-identical to today. Reuse the `look_engine` dict already computed at L301–306
(don't re-resolve). Keeping halation/grain **in `soft_local`** (not a new
descriptor field) means `grade_hash` (L166) and `_grade_key` (L300/304) already
account for them — no signature churn.

### 3.3 Export apply (`softlocal.py` + `compositor.py`)

Add two builders in `softlocal.py` mirroring `vignette_ffmpeg_filter` (L55–66):

- **`grain_ffmpeg_filter(grain)`** → ffmpeg `noise` (simplest, temporal):
  ```python
  # amplitude 0..1 -> ffmpeg noise strength ~0..40 (alls), temporal+uniform
  return f"noise=alls={int(strength*40)}:allf=t+u" if strength > 0.001 else None
  ```
- **`halation_ffmpeg_subgraph(halation)`** → returns a filtergraph FRAGMENT that
  isolates highlights, blurs, red-tints, and screen-blends back. Unlike the
  vignette (one clause), this needs a `split`+`blend` subgraph:
  ```
  split=2[hbase][htmp];
  [htmp]lutrgb=... (keep only >THRESHOLD),
        colorchannelmixer=... (red-orange tint),
        gblur=sigma=SIGMA[hglow];
  [hbase][hglow]blend=all_mode=screen:all_opacity=STRENGTH
  ```
  Return `None` for zero strength.

Wire into `_transform_vf` (compositor.py) **after the vignette** (L499–500):
`parts.append(halation_subgraph)` then `parts.append(grain_filter)`. Because
`_transform_vf` builds a comma-joined `-vf` chain, the halation `split/blend`
subgraph (which uses named pads) must be spliced in a way that stays a single
linear graph — the standard `split[a][b];[b]...[g];[a][g]blend` form is valid
inside one `-vf`. Read the params next to the existing vignette read (L225 &
L615): `halation = (grade.get("soft_local") or {}).get("halation")`, same for
grain. Thread both through `_produce_segment` (L526–537) exactly like
`vignette_filter`.

**Order (both sides):** LUT → vignette → **halation → grain**. Halation is a
glow over the graded picture; grain is the final texture on top of everything
(real film grain is in the emulsion, i.e. last).

### 3.4 Preview apply (`lut-gl.ts`)

Two shader additions. **Grain** is cheap (single-pass, in the existing fragment
shader); **halation** needs a blur, which needs an offscreen pass.

- **Grain** — add `uGrainStrength` + `uFrameSeed` uniforms; after the vignette
  (L97), add luma-only hash noise:
  ```glsl
  if (uGrainStrength > 0.001) {
    float n = fract(sin(dot(vUv * uCanvasSize + uFrameSeed, vec2(12.9898, 78.233))) * 43758.5453);
    graded += (n - 0.5) * uGrainStrength;
  }
  ```
  `uFrameSeed` varies per frame (temporal grain, like ffmpeg `allf=t`). Thread
  via `draw()` (L129–134) like `vignette`.
- **Halation** — the current renderer is single-pass (L251–288). Add an
  offscreen framebuffer (FBO) ping-pong: (1) render graded picture to a texture,
  (2) threshold+tint+separable-Gaussian-blur it in a small second program, (3)
  screen-blend the glow back in the final draw. This is the one real structural
  addition on the preview side (an FBO + a blur program) — budget it as the bulk
  of the frontend effort. Gate behind the same param so no-halation looks keep
  the existing single-pass path (zero perf cost when off).

Extend the params interface (rename `VignetteParams` usage or add a sibling):
```ts
export interface SoftLocalParams {
  vignette?: VignetteParams | null;
  halation?: { strength: number } | null;
  grain?: { strength: number } | null;
}
```
and pass `soft_local` from the player through to `draw()`. Where the player reads
the resolved grade's `soft_local.vignette` today, also read `.halation`/`.grain`.

### 3.5 Flag + config + input hash

- **`config.py`** (after `grade_look_engine` L195):
  ```python
  # halation_grain.plan.md: look-scoped spatial film texture (halation
  # glow + film grain), applied as a soft_local pass on both sides (approximate
  # parity, like the vignette). Off = no halation/grain keys in soft_local
  # (byte-identical). Rides engine looks; requires grade_look_engine too.
  grade_film_texture: bool = False
  ```
- **`job.py`**: bump `INPUT_HASH_SCHEMA_VERSION` **9 → 10** (L87) + comment; add
  `"grade_film_texture": settings.grade_film_texture` to the flags dict
  (L248–252); pass `film_texture_enabled=settings.grade_film_texture` into the
  `resolve_clip_grade` call (~L646–661, next to `look_engine_enabled`).
- **`cdl.py::grade_hash`**: bump `schema_version` **2 → 3** (L149) + comment
  (one-time reindex; `soft_local` already in the payload so no field add).

### 3.6 Catalog validation look

Add texture to a catalog entry so it's testable end-to-end. Extend
`engine_punchy` or add `engine_film` in `look_engine.py` LOOKS (L300–319):
`LookSpec(..., halation=0.25, grain=0.04)`. The full film-look library is still
the NEXT plan; this is just a probe.

---

## 4. Switchability / rollback

- `grade_film_texture` defaults **off** → `film_texture_enabled=False` →
  resolver never writes `halation`/`grain` into `soft_local` → `grade_hash` and
  every apply path byte-identical to today. Independent of `grade_look_engine`
  so the color engine and the texture pass roll out separately.
- Triple gate: flag off, OR no engine look selected, OR look with
  `halation==grain==0` → zero texture, existing vignette-only behavior.
- Only unconditional changes: `INPUT_HASH_SCHEMA_VERSION 9→10` + `grade_hash`
  schema `2→3` (one-time reindex, identical output while off).
- No migration (`soft_local` is schemaless jsonb in `resolved_grades`).
- Clean revert: flag off (runtime), or revert the diffs in `look_engine.py`,
  `resolver.py`, `softlocal.py`, `compositor.py`, `cdl.py`, `config.py`,
  `job.py`, and `lut-gl.ts` (+ any player threading).

---

## 5. Phased implementation

### Phase 0 — descriptor + export apply + flag (wired, gated off)
- `look_engine.py`: `halation`/`grain` on `LookSpec` (+ dict/identity); assert
  `build_look_grid` ignores them.
- `resolver.py`: route look texture params into `soft_local`; new
  `film_texture_enabled` param.
- `softlocal.py`: `grain_ffmpeg_filter` + `halation_ffmpeg_subgraph` (+
  constants). `compositor.py`: read + thread + append after vignette.
- `config.py` flag; `job.py` schema bump + flag + arg; `cdl.py` schema bump.
- **Acceptance:** flag off → every `grade_hash`, baked cube, and `-vf` chain
  byte-identical to pre-change; grade tests still green;
  `build_look_grid(LookSpec(halation=.3,grain=.1))` == `build_look_grid` of the
  same spec with texture zeroed (texture never touches the grid).

### Phase 1 — preview apply (WebGL)
- `lut-gl.ts`: grain (single-pass) + halation (FBO blur pass); thread
  `soft_local` through the player's `draw()`.
- **Acceptance:** a film look shows glow around highlights + grain in preview
  that reads like the exported clip (approximate parity); a no-texture look is
  visually + perf identical to today (single-pass path).

### Phase 2 — turn on + tune
- `grade_film_texture=true`; set a film look with texture; re-grade; compare
  preview vs a real export on the Siri frame + a highlight-heavy CANON clip.
  Tune `HALATION_THRESHOLD/SIGMA/tint` and the strength→ffmpeg mappings so the
  two sides read the same and neither is overcooked.

---

## 6. Testing (`backend/scripts/test_grade.py`)

Mirror the file's conventions (register in `main()`, `print("ok ...")`). Run:
`cd backend && .venv/bin/python scripts/test_grade.py`.

1. `test_lookspec_texture_roundtrip` — `halation`/`grain` survive
   `to_dict`/`from_dict`; a texture-only spec is `not is_identity()`.
2. `test_build_look_grid_ignores_texture` — grid for `LookSpec(halation=.4,
   grain=.1, sat=1.2)` is byte-identical to `LookSpec(sat=1.2)` (texture is
   spatial, never baked).
3. `test_resolver_film_texture_off_byte_identical` — with
   `film_texture_enabled=False`, an engine look with texture yields the SAME
   `grade_hash` and `soft_local` as today (no halation/grain keys).
4. `test_resolver_film_texture_on_populates_soft_local` — flag on + a texture
   look → `soft_local` carries `halation`/`grain`; `grade_hash` differs from off.
5. `test_grain_ffmpeg_filter` — zero → `None`; positive → a `noise=...` clause;
   monotonic in strength.
6. `test_halation_subgraph_shape` — zero → `None`; positive → a graph containing
   `split`, `gblur`, and `blend=...screen` (structural, not pixel).
7. `test_grade_hash_changes_with_texture` — same everything, `soft_local` with vs
   without halation/grain → different `grade_hash` (re-bakes/re-renders).
8. `test_grade_key_not_collapsed_with_texture` — `compositor._grade_key` does
   NOT collapse an identity-CDL grade that carries `soft_local.halation` to `""`.

(Preview/WebGL is validated visually in Phase 1–2 — no JS unit harness exists for
the shader, same as the vignette.)

---

## 7. Ops runbook

1. Land Phase 0 (+ Phase 1 preview).
2. `INPUT_HASH_SCHEMA_VERSION 9→10`, `grade_hash` schema `2→3` (Phase 0).
3. Turn on: `GRADE_FILM_TEXTURE=true` (and `GRADE_LOOK_ENGINE=true`) in worker +
   API + the frontend's grade read path (same flag-parity discipline as the
   engine/tone rollouts).
4. Restart worker + API; rebuild frontend.
5. Re-grade a thread with a texture look; verify `state=done`; compare preview
   vs export on a highlight-heavy clip.

---

## 8. Acceptance criteria

- **Halation reads as film.** Bright highlights (windows, specular, skin edges)
  get a soft red/orange glow on export AND preview at the same strength; flat/
  no-highlight footage is ~unchanged (glow is highlight-gated).
- **Grain is present but subtle.** Visible luminance texture that survives to
  export; not blocky, not crushed by codec, monotonic in strength.
- **Approximate parity, honestly bounded.** Preview and export read as the same
  effect (not pixel-identical) — the documented soft-local trade; the color LUT
  stays byte-exact underneath.
- **Off / no-texture byte-identical.** Flag off, or no engine look, or a look
  with zero texture → every `grade_hash`, `-vf` chain, and the preview path are
  identical to today.

---

## 9. Line-ref drift found vs. the code (verified this session)

- ✅ `resolver.py`: engine look resolved L301–306; `soft_local` built L316–322;
  `grade_hash` call passes `soft_local` L324–331; return dict L332–339. Add
  `film_texture_enabled` param + route texture into `soft_local`.
- ✅ `cdl.py::grade_hash`: `soft_local` already in payload L166 (no signature
  change); `schema_version` default `2` L149 — bump 2→3.
- ✅ `softlocal.py`: `solve_vignette` L36–52, `vignette_ffmpeg_filter` L55–66,
  `DEFAULT_STRENGTH` L32, `MAX_ANGLE_RAD` L33 — mirror for grain/halation.
- ✅ `compositor.py`: `vignette_ffmpeg_filter` import L37; read+thread at L225 &
  L615; `_transform_vf` LUT L495–498 then vignette L499–500 (append halation+
  grain after); `_produce_segment` L526–537; `_grade_key` identity collapse
  L298–308 already includes `soft_local` (a non-empty one already re-keys).
- ✅ `look_engine.py`: `LookSpec` L41–60, `to_dict` L62–74, `from_dict` L76–113,
  `is_identity` L115–124, `build_look_grid` L253–281 (must keep ignoring the new
  fields), `LOOKS` L300–319.
- ✅ `config.py`: `grade_look_engine` L195 — add `grade_film_texture` after.
- ✅ `job.py`: `INPUT_HASH_SCHEMA_VERSION = 9` L87 (bump 9→10), flags dict
  L248–252, `resolve_clip_grade` call has `look_engine_enabled` L660 — add
  `film_texture_enabled` alongside.
- ✅ `frontend/.../lut-gl.ts`: vignette in-shader L91–98, `VignetteParams`
  L116–120, `draw()` sig L129–134, uniforms L227–228 + set L284–285, single-pass
  `createLutRenderer` L173–298 (FBO/blur program is a genuine addition here).
- ⚠️ **Halation is a `split`/`blend` subgraph, not a single `-vf` clause** — the
  helper must emit the full `split[a][b];[b]...[g];[a][g]blend=...screen` form,
  which is valid inside one `-vf` but is more than the vignette's append-a-clause
  pattern. Budget accordingly.
- ⚠️ **Preview halation needs an offscreen pass** — the renderer is single-pass
  today; the FBO + separable-blur program is the bulk of the frontend work.
  Grain is cheap (single-pass hash noise).
- ⚠️ **Parity is approximate by design** (ffmpeg `noise`/`gblur` vs WebGL
  hash-noise/FBO-blur) — the same bounded trade `softlocal.py` L17–26 documents
  for the vignette; the color LUT's byte-exact contract is untouched.

---

## 10. Remaining pipeline roadmap (after this)

| Next | Stage | Why | Effort |
|---|---|---|---|
| ✅ | color-response engine | the look instrument (done) | — |
| — | **This: halation + grain** | film-family "feels cinematic"; look-scoped spatial finish | M (backend) + M–L (WebGL) |
| 3 | **Author the look library** (6 YouTube + 5–6 film + ad; film from stock data, modern hand-authored, both iterated) | turns the instrument into the product — the moat | L (ongoing) |
| 4 | **`already_graded` gate** (segment → `cut_records.look.graded`) | never-worse: already-graded footage re-corrected today | S–M |
| 5 | **Frontend grade UX** (progress, before/after, per-clip look pick) | make the pipeline legible/controllable | M–L |
