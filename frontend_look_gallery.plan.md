# Frontend Look Gallery — family grouping + live thumbnails

> Implementation plan. Self-contained: an implementer with **no other context**
> should build this from this file alone. All file:line refs were verified
> against the real code on the date written (see "Line-ref drift" at the end).
> **Read the frontend-design skill first** (`.cursor/skills/frontend-design/
> SKILL.md`): black/white/grey, orange (`--accent`) sparingly, colors via CSS
> tokens (never hex), `cn()` for conditional classes, Tailwind for layout only.
>
> **This makes the 17-look library actually browsable.** The picker already
> *works* (selection, before/after, progress, live engine-look preview all
> exist) — but it renders ~29 looks (12 legacy CDL presets + 17 engine looks) as
> a flat grid of text labels with no grouping and no thumbnails. This plan groups
> engine looks by family and shows a live thumbnail of each, so users pick by
> *seeing* the look, not reading a name.

---

## 1. Goal & non-goals

**Goal.** Turn the flat text-label look grid into a **family-grouped gallery with
live per-look thumbnails**:
- Group engine looks under **Creator / Film / Ad** headers (the `family` tag the
  library already ships).
- Render each look as a **thumbnail** — a bundled reference still put through
  that look's baked cube (reusing the exact WebGL LUT path the preview uses), so
  the swatch shows the real color transform.
- De-clutter: hide the superseded legacy CDL presets from the default picker;
  drop the `engine_identity` parity-anchor from the picker and give a clean
  "None / Original" affordance instead.
- Mark film-family looks (they carry halation/grain, which need
  `grade_film_texture`) with a small badge + honest tooltip.

**Non-goals.**
- **No new grade math / backend pipeline.** Only a listing field is added
  server-side (`look_params`, §3.1); everything else is UI.
- **No per-clip look UI.** Look stays sequence-level (as today). Per-clip
  override is a later step.
- **No removal of `reference` / `.cube` upload / arc / steer / export / before-
  after** — those sections stay exactly as they are; this plan only reworks the
  **Look picker** section (`color-grade-view.tsx` L543–569).
- **Thumbnails show color only.** Halation + grain are spatial (not in the cube),
  so a cube-baked thumbnail can't show them — hence the film badge. This is
  called out to the user, not hidden.

---

## 2. What exists (verified)

| Piece | Location | State |
|---|---|---|
| Look picker gallery | `color-grade-view.tsx` L543–569 | flat `grid grid-cols-3/4` of text-label buttons over `presets` (mixes CDL + engine). **Rework into grouped + thumbnailed.** |
| Selection wiring | `color-grade-view.tsx::selectLook` L388–400 | already handles `mode:"engine"` (sets `look_id`, clears others) and `mode:"preset"`. **Keep as-is; add a "clear/None" path.** |
| Listing fetch | `color-grade-view.tsx` L304–307 (`getGradePresets`), state L178 | works. |
| API type | `api.ts::GradePresetSummary` L1049–1060 | has `mode`, `preset_id?`, `look_id?`, `label`, `description`, `family?`. **Add `look_params?` (§3.1).** |
| Combined endpoint | `routers/grade.py` `/api/grade/presets` → `list_presets()` + `list_engine_looks()` | engine entries carry `family`; CDL entries don't. |
| Engine listing | `look_engine.py::list_engine_looks` L553–564 | returns `{look_id,label,description,mode,family}`. **Add `look_params` (§3.1).** |
| Cube fetch (handles engine) | `grade-cube-client.ts::gradeCubeUrl` L38–52, `prefetchGradeCube` L106–109 | `gradeCubeUrl` ALREADY encodes `look_engine` (L48–50) — a synthetic grade with identity CDL + a look's params fetches that look's cube. **Reuse for thumbnails.** |
| WebGL LUT sampler | `lut-gl.ts::createLutRenderer` L173–298 (`parseCubeText` L142–171, trilinear `TEXTURE_3D`) | draws a video through a cube. **Adapt into a one-shot still→cube thumbnail renderer (§3.3).** |
| Design tokens | `globals.css` (`--accent`, `--border`, `--muted`, `--foreground`, `--accent-soft`, …) | use via inline `style`, never hex. |

---

## 3. Design

### 3.1 Backend: expose each engine look's params (for thumbnails)

The gallery listing must carry each engine look's `LookSpec` so the frontend can
bake its thumbnail cube (identity CDL + that look's `look_engine`).

- **`look_engine.py::list_engine_looks`** (L553–564): add
  `"look_params": look.spec.to_dict()` to each dict; widen the return annotation
  to `List[Dict[str, Any]]`.
- **`api.ts::GradePresetSummary`** (L1049–1060): add
  `look_params?: Record<string, unknown>;` with a comment (only present for
  `mode:"engine"`).
- Legacy CDL presets (`presets.list_presets`) are unchanged — they carry no
  `look_params`/`family`; the frontend hides them from the picker (§3.2).

*No hash/schema/migration change* — this is a read-only listing field.

### 3.2 Family-grouped, decluttered gallery (`color-grade-view.tsx`)

Replace the flat map at L543–569 with:

1. **Filter to engine looks** for the primary gallery
   (`presets.filter(p => p.mode === "engine" && p.look_id !== "engine_identity")`).
   Legacy CDL presets are **hidden by default** (superseded by the engine
   library; showing 29 entries is clutter). Any existing document still on a
   `preset`/CDL look keeps resolving — we just don't surface CDL in the picker.
   *(Product call: if you'd rather keep them, put them in a collapsed "Classic"
   group below — but default = hidden.)*
2. **Group by `family`** in a fixed, priority order: `creator` → `film` → `ad`.
   Render each group under a `SectionLabel` (existing helper L138–144):
   "Creator", "Film", "Ad".
3. **"None / Original" chip** at the top of the picker (before the groups): a
   thumbnail-less chip that clears the look to identity — `applyLook({ ...(look
   ?? {}), mode: undefined, look_id: null, preset_id: null, look_params: null,
   lut_ref: null }, true)` (or `applyLook(undefined, true)` if the store treats
   that as "no look"). Active when `look?.mode` is not `engine` (and no ref/lut).
   This replaces the need to show `engine_identity`.
4. Each look renders as a **`LookCard`** (thumbnail + label + active ring),
   `family:"film"` cards get a small **"grain" badge** (see §3.4).

Grouping helper (pure, testable):
```ts
const FAMILY_ORDER = ["creator", "film", "ad"] as const;
const FAMILY_LABEL = { creator: "Creator", film: "Film", ad: "Ad" };
function groupByFamily(items: GradePresetSummary[]) {
  return FAMILY_ORDER
    .map((fam) => ({ fam, looks: items.filter((p) => (p.family ?? "creator") === fam) }))
    .filter((g) => g.looks.length > 0);
}
```

### 3.3 Live thumbnails — one shared WebGL renderer, cached per look

Add `frontend/src/components/preview/look-thumbnail.ts` — a **singleton** offscreen
renderer (one WebGL context total, NOT one per card) that puts a bundled
reference still through a look's cube and returns a small bitmap/data URL, cached
by `look_id`.

- **Reference still:** add a bundled asset `frontend/public/look-thumb-ref.jpg`
  — a single frame rich in the things looks differentiate: **skin tones, a sky/
  blue, greens, a warm highlight, and a neutral grey**. ~480×270. (Any
  representative, license-clean still; a color-rich portrait-outdoors frame is
  ideal.) It loads once, reused for every thumbnail.
- **Cube per look:** build a synthetic `ResolvedGrade`:
  ```ts
  { cdl: IDENTITY_CDL, working_space: "rec709_v1", creative_lut_ref: null,
    soft_local: null, tone_contrast: 0, look_engine: look.look_params, grade_hash: "" }
  ```
  and fetch via `prefetchGradeCube` (grade-cube-client.ts — already encodes
  `look_engine` into the URL, L48–50, and caches immutably). `IDENTITY_CDL =
  { slope:[1,1,1], offset:[0,0,0], power:[1,1,1], sat:1 }`.
- **Render:** reuse the trilinear `TEXTURE_3D` sampling from `lut-gl.ts` (extract
  the shared shader/sampler, or add a `drawImageThroughLut(image, grid, size) →
  HTMLCanvasElement` sibling in `look-thumbnail.ts`). Draw the still through the
  uploaded cube into a small canvas; cache the result (canvas/`ImageBitmap`/data
  URL) keyed by `look_id`. 17 looks × (1 cached cube fetch + 1 draw) is cheap.
- **`LookThumbnail` React component:** takes `look` + `size`, renders the cached
  bitmap into an `<img>`/`<canvas>`; while the cube/render is pending, show a
  neutral `--border` placeholder; on any failure (no WebGL, cube 404) fall back
  to a flat swatch (graceful — never a broken card).
- **Honest note:** the thumbnail is the LOOK's color grid over a neutral base
  (identity CDL, no per-clip correction, no tone_contrast, no halation/grain) —
  it shows *relative look character*, not the exact final pixel. Good enough to
  choose by; documented in the component header.

### 3.4 `LookCard` (design-system compliant)

A button wrapping the thumbnail:
- `rounded-lg overflow-hidden border`, `borderColor: active ? "var(--accent)" :
  "var(--border)"`; active also gets a faint `--accent-soft` ring/label.
- Label under the thumbnail: `text-[11px] font-medium`, `--foreground` when
  active else `--muted`. `title={description}` for the tooltip.
- Film badge: for `family:"film"`, a tiny corner tag "grain" in `--muted` on a
  `--border` chip; `title` explains "adds film grain + halation (needs film
  texture on)".
- Use `cn()` for conditional classes; **orange only** on the active card's
  border/label (keep ≤ a couple orange elements per view — the active look + the
  before/after toggle already use it, so don't add more).

Grid: `grid grid-cols-2 sm:grid-cols-3 gap-2` per family group (thumbnails need
more width than text chips did).

---

## 4. Switchability / rollback

- Pure additive UI + one read-only listing field. No flags, no hash, no
  migration. If `grade_look_engine` is off server-side, engine looks still LIST
  and thumbnail, but selecting one resolves to identity until the flag is on —
  same as today (the picker already lists them). Consider gating the picker's
  visibility on a lightweight "engine available" signal only if you want to hide
  unusable looks; otherwise leave listed (thumbnails still inform).
- Clean revert: restore the flat map at L543–569; delete `look-thumbnail.ts` +
  the bundled asset; revert the `look_params` additions.

---

## 5. Phased implementation

### Phase 0 — backend listing field
- `list_engine_looks`: add `look_params`; widen annotation. `api.ts`: add
  `look_params?` to `GradePresetSummary`.
- **Acceptance:** `GET /api/grade/presets` returns `look_params` (a
  `LookSpec.to_dict()` shape) on every `mode:"engine"` entry; `tsc` clean.

### Phase 1 — thumbnail renderer
- `look-thumbnail.ts` (singleton WebGL still→cube renderer + cache); bundle
  `public/look-thumb-ref.jpg`; `LookThumbnail` component with pending/fallback.
- **Acceptance:** a `LookThumbnail` for `punchy_vibrant` visibly differs from
  `moody_cinematic` / `bright_airy`; no WebGL context leak (one context reused);
  fallback swatch shows if the cube fetch is blocked.

### Phase 2 — grouped gallery
- Rework the picker section: filter to engine looks (hide CDL + `engine_identity`),
  `groupByFamily`, `SectionLabel` per group, `LookCard` grid, "None/Original"
  chip. Keep `selectLook` + all other sections untouched.
- **Acceptance:** three labeled groups (Creator/Film/Ad) with thumbnails;
  selecting a look sets `{mode:"engine", look_id}` and the live preview updates;
  "None" clears to original; film cards show the badge; before/after still works.

### Phase 3 — polish + validate
- Design-system pass (checklist below); verify on a real thread with
  `grade_look_engine` (+ `grade_film_texture`) on: pick a creator look and a film
  look, confirm preview == the thumbnail's color direction and export matches.

---

## 6. Testing

- **Backend** (`backend/scripts/test_grade.py`): `test_list_engine_looks_has_
  params_and_family` — every engine entry has `look_params` (round-trips via
  `LookSpec.from_dict` to a non-crashing spec) and `family ∈ {creator,film,ad}`;
  `list_presets` entries have neither (regression that CDL stays unchanged).
- **Frontend:** `tsc` clean. Pure `groupByFamily` unit test if a JS test harness
  exists (order creator→film→ad, empty groups dropped). Thumbnail rendering +
  gallery layout are validated visually (no shader unit harness — same as the
  vignette/preview).
- **Design-system checklist** (from the skill): no hardcoded hex; orange only on
  active card + existing toggle; `cn()` for conditional classes; no
  `tailwind.config.js`; spacing over dividers.

---

## 7. Acceptance criteria

- **Browsable by sight.** Looks appear as thumbnails grouped Creator/Film/Ad; a
  user can tell teal-moody from warm-cozy from punchy at a glance.
- **Selection unchanged + live.** Picking a look sets `{mode:"engine",look_id}`,
  the preview repaints via the existing cube path, before/after toggles it.
- **Decluttered.** Legacy CDL presets are not in the picker; `engine_identity` is
  replaced by a clean "None/Original" chip.
- **Honest film cue.** Film looks carry a badge noting grain/halation (+ that it
  needs the texture flag); thumbnails are documented as color-only.
- **Design-system clean.** Passes the skill checklist; one WebGL context for all
  thumbnails; graceful fallback when a cube/WebGL is unavailable.

---

## 8. Line-ref drift found vs. the code (verify before editing)

- ✅ `color-grade-view.tsx`: picker map L543–569 (rework), `selectLook` L388–400
  (keep + add clear), `SectionLabel` L138–144, `getGradePresets` L304–307,
  `presets` state L178, `applyLook` L353–369, `isIdentityGrade` L47–61.
- ✅ `api.ts`: `GradePresetSummary` L1049–1060 (add `look_params?`), `SequenceLook`
  L639–658 (`look_id`/`look_params` already present), `getGradePresets` L1062–1064.
- ✅ `grade-cube-client.ts`: `gradeCubeUrl` L38–52 already encodes `look_engine`
  (L48–50); `prefetchGradeCube` L106–109; `parseCubeText` imported from lut-gl.
- ✅ `lut-gl.ts`: `createLutRenderer` L173–298, `parseCubeText` L142–171 — extract
  the 3D-texture trilinear sampler for the still→cube thumbnail draw.
- ✅ `look_engine.py::list_engine_looks` L553–564 (add `look_params`); `EngineLook`
  L362–372 (`family` L372); `LookSpec.to_dict` present.
- ✅ `presets.py::list_presets` L104–114 (CDL, no `family`/`look_params` — stays).
- ✅ `routers/grade.py`: `/api/grade/presets` returns both lists; `/api/grade/cube`
  already accepts `look_engine` (used by `gradeCubeUrl`).
- ⚠️ **One WebGL context for all thumbnails** — do NOT create a context per card
  (17+ contexts will hit browser limits / leak). Use a singleton offscreen
  renderer that draws each look once and caches the bitmap.
- ⚠️ **Bundled reference still is required** — thumbnails need a real image;
  `public/` currently only has `edso-logo.png`. Add a color-rich reference frame.
- ⚠️ **Thumbnails are color-only** (cube has no halation/grain, no per-clip
  correction, no tone_contrast) — intended; the film badge covers the texture gap.

---

## 9. Roadmap after this

| Next | Stage | Why |
|---|---|---|
| ✅ | color engine · halation/grain · 17-look library | backend complete |
| — | **This: look gallery UX** | makes the library browsable/pickable |
| next | per-clip look override; grading-progress polish; before/after wipe | finer control |
| later | `already_graded` gate; export bundle; auto narrative arc | completeness |
