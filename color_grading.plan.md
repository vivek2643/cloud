# Colour grading — executable build plan (final product)

Executable, re-anchored companion to the strategy doc `docs/color_grading.md`
(philosophy, moat, honest limits — read it first; not repeated here). This file
is what another chat executes. Scope = **the full product, not an MVP** (correct
+ match + 3 look modes + invisible arc + soft-local + NL steering + pro export).

Status: **planning — nothing implemented yet.**

---

## 0. Locked decisions (this session)

1. **No L2/VLM revival.** Perception is read from **Cuts v3 `cut_records`**
   (written by `pass2b`). We do not rebuild `clip_perception`.
2. **`white_reference` is LIVE.** Add it to `pass2b` and actually produce +
   persist it for everything ingested from here on (not dormant). It only lands
   on footage ingested *after* ship — existing cuts need re-ingest.
3. **Three input modes → one target look** (no auto look-selection):
   **(1) authored parametric presets, (2) reference-image drop, (3) `.cube`
   upload.** Modes 1–2 collapse into our parametric CDL spine (steerable +
   arc-able); mode 3 rides on top of the correct+match layer.
4. **Pro round-trip export is in scope** (`.cdl`/`.ccc` + `.cube`, referenced in
   XML/EDL; editable-vs-baked toggle).
5. **Grade location:** measurement is source-level; grade *decision* is
   document-level; a deterministic resolver combines them; EDSO owns only the
   human boundary (steer / ask / explain).
6. **Fork A — parity:** resolve each clip's full grade into **one baked 3D LUT**
   (`.cube`) in a stamped working space, applied **identically** in the WebGL
   preview shader and the ffmpeg `lut3d` export. Same math both sides.
7. **Fork B — soft-only local.** Feathered masks off `framing.subject_box` +
   `color_stats`; **bakes into the per-clip LUT**. No segmentation model, no
   crisp power windows.
8. **Fork C — arc via EDSO categorical.** EDSO tags each segment with an intent
   category; deterministic table → per-beat CDL delta; one user intensity dial
   scales amplitude. LLM never emits color numbers.
9. **Fork D — grade-groups (deterministic).** Group by `color_stats` similarity
   + `look.palette` + `camera`/`framing` + timeline adjacency (`continuity`);
   anchor = highest `total_quality`; optional EDSO "same scene?" categorical as a
   tiebreaker only.

## 0.1 Re-anchoring vs `docs/color_grading.md` (what changed)

| Strategy doc says | Now (execute against this) |
|---|---|
| VLM/L2 fields (`already_graded`, ROI, cues) | Read `cut_records`: `look.graded`, `look.palette`, `look.exposure_flags`, `framing.subject_box`, `framing.shot_size`, `characteristics[]` |
| new `white_reference` VLM field | add to **`pass2b`** `look`, produced live |
| `hero_cuts` / `take_quality` | `cut_records` / `total_quality` (+ `speech_quality`) |
| Director→Editor pipeline | **EDSO** agentic editor (`converse.py`/`tools.py`/`act.py`) |
| `beats[].purpose` drives arc | EDSO **arc-intent categorical** per segment (Fork C) |
| Reframing (§8/§12.7) as new work | **already exists** — `framing.crop_16x9/9x16/1x1` + `rotation_deg` + render transforms; **reuse, do not rebuild**. Grading only *reuses* `subject_box`/`characteristics` for metering + soft-local. |
| local/spatial ambiguous | **soft-only**, bakes to LUT |

---

## 1. Current-state touchpoints (read before coding)

- **L1** — `backend/app/services/l1/pipeline.py` (orchestrator `l1_orchestrate`;
  parallel deep tracks; `processing_jobs` per stage, idempotent). Model stages
  live in `l1/*.py` (`motion_dynamics.py`, `audio_features.py`, …). **Add a new
  deterministic `color_stats` stage here.**
- **pass2b** — `backend/app/services/l3/pass2b.py` produces per-cut `look =
  {graded, palette, exposure_flags}`, `framing`, `characteristics`,
  `caption_zones`, `taste_fences`. **Add `look.white_reference`.**
- **Persistence** — `l3/ingest_store.py` writes `cut_records` (has `look`,
  `framing`, `characteristics`, `total_quality`, `take_group_id`, `continuity`,
  `camera`, `hero_ts_ms`). Migrations in `backend/migrations/`.
- **Edit Document** — `frontend/src/lib/api.ts` (`EditSegment`, `EditOperation`,
  `EditDocument`); versioned via `edit_documents` (append-only). Resolve on
  backend `l3/observe.py:resolve_doc` + `l3/layers.py:resolve`; on frontend
  `lib/resolve-timeline.ts`.
- **Preview** — `components/preview/composite-preview.tsx` +
  `use-program-player.ts` (WebAudio + pooled `<video>`; per-layer transform).
  **This is where the WebGL LUT shader lands.**
- **Render** — `backend/app/services/render/compositor.py`: ffmpeg
  `filter_complex`, `_transform_vf` (rotate/fit/zoom/crop), everything in
  **8-bit `yuv420p`, no color management**. **This is where `lut3d` + working
  space land.**
- **EDSO** — `l3/converse.py` (system prompt, tool loop), `l3/tools.py` (verb
  specs — keep **de-prescribed**: *what it does*, not *when*), `l3/act.py`
  (verbs), `l3/observe.py` (senses), `l3/guidance_doc.md` (editorial advice).
- **UI shell** — `frontend/src/components/project-lenses.tsx` (the
  "Colour grading — Coming soon" placeholder) + `sidebar.tsx` stage rail.

---

## 2. Data model

### 2.1 The grade object (CDL-native spine) — unchanged from strategy §4
```
grade {
  cdl:          { slope[3], offset[3], power[3], sat }   // correct + match + arc; round-trips
  creative_lut: <optional .cube ref>                      // the look; non-CDL nuance + soft-local baked in
  working_space: "rec709" | "logc" | ...                  // stamped so preview == export
}
```

### 2.2 L1 `color_stats` (new, deterministic) — the foundation  [BE][D]
- New stage `l1/color_stats.py` + `processing_jobs` entry + migration for a
  `color_stats` table (per file, keyed by `file_id`). Sample **N frames across
  the clip** (reuse proxy; sample near `hero_ts_ms` + evenly spaced).
- Fields (per clip, aggregated): luma histogram, **black/white points**,
  mid-gray, per-channel RGB mean/median, **Lab a*/b* mean (cast)**, **WB
  estimate** (gray-world + white-patch), **clipping %** (highlight/shadow),
  **log/flat detection** (histogram spread vs known log curves), **skin sample**
  (mean skin Lab from `framing.subject_box`/`characteristics` face region when
  present), dominant palette. **Everything downstream depends on this.**

### 2.3 `pass2b.look` additions  [BE][L=pass2b]
- Add **`white_reference`**: `{ present: bool, region: [x,y,w,h]|null, object:
  str|null }` — VLM proposes a neutral object + region; **deterministic code in
  the correct layer verifies it is actually neutral** before trusting it.
- Reuse (already produced): `graded` (= `already_graded`), `palette`,
  `exposure_flags`; `framing.subject_box`/`shot_size`; `characteristics[]`.
- Persist via `ingest_store.py` → `cut_records.look`. Migration if the JSON
  shape is column-validated.

### 2.4 Edit Document `grade` block  [BE][FE][D]
- **Per-timeline-item** `grade` (on `EditSegment` + `EditOperation`) and a
  **sequence-level `look`** (mode + recipe/ref/LUT + intensity dial). Additive
  fields in `api.ts` + backend schema; versioned through `edit_documents`.
- Document-level because the same source cut can grade differently in context
  (the arc). Source-level measurement stays on `cut_records`/`color_stats`.

---

## 3. The grade stack (fixed order — resolver contract)
```
Measure(color_stats + look) → Correct(semantic-gated, never-worse) →
Match(grade-groups) → Look(1 of 3 modes) → Arc(EDSO intent → CDL delta × dial) →
Soft-local(feathered, bakes to LUT) → NL trims → BAKE per-clip .cube →
render/preview(parity) → export(.cdl/.ccc + .cube)
```
- Implemented as a **deterministic resolver** (new `l3/grade/resolver.py` or a
  module under render) run at `resolve_document`. **No heavy new LLM pass** —
  LLM only for steer/ask/explain (§8).
- **Never-worse guardrail** everywhere: don't lift crushed shadows into noise,
  don't push clipped highlights further; best-effort silently (strategy §11).

---

## 4. Fork A — parity engine (do this early; everything renders through it)  [BE][FE][D]

**Contract: one baked 3D LUT per (clip, grade-hash), applied identically in
preview and export.**
- Backend **bakes a 33³ `.cube`** from the resolved `cdl + creative_lut +
  soft-local + arc-delta` in the stamped `working_space`; cache by
  `grade_hash(clip)`.
- **Export:** `compositor.py` applies it via `lut3d=<cube>` in the video graph
  **after `_transform_vf`, before final `format=yuv420p`**; use a higher-precision
  intermediate (`zscale`/`format=rgb48`/`gbrpf32`) around the LUT so 8-bit
  banding doesn't diverge from preview. Stamp/convert working space explicitly
  (no implicit rec709 assumptions).
- **Preview:** `use-program-player.ts`/`composite-preview.tsx` load the same
  `.cube` (served via a small endpoint) and apply it in a **WebGL 3D-texture LUT
  shader** over the video layer. Same cube = same math.
- Endpoint: `GET` per-clip resolved `.cube` (cached; regenerated on grade
  change — cheap). Frontend fetches on grade/selection change.
- Acceptance: a graded frame in the browser preview matches the exported frame
  within tolerance (eyeball + spot pixel check on a test clip).

---

## 5. Correct layer (global, CDL)  [BE][D]  (strategy §12.3)
- Auto exposure normalize (within captured range), auto WB (**skin-anchored** off
  `subject_box`/`characteristics`, else gray-world/white-patch; **prefer verified
  `white_reference` when present & confirmed neutral**), contrast + black/white
  point, log/flat input transform (from `color_stats`).
- **Semantic gate:** skip already-graded (`look.graded`), protect skin, keep
  golden-hour intent (don't neutralize an intended warm cast).
- Never-worse guardrail (§3).

## 6. Match layer (consistency)  [BE][D]  (Fork D)
- **Grade-groups:** deterministic clustering over `color_stats` similarity +
  `look.palette` + `camera`/`framing` + timeline adjacency (`continuity`).
  Optional EDSO "same scene?" categorical as tiebreaker only.
- Anchor per group = highest `total_quality`. Match members → anchor
  conservatively; **skin-priority** weighting for consistent faces across cuts.
- Master look across groups: unify identity, keep intended day/night differences.

## 7. Look layer — three modes  [BE][FE][D]  (strategy §6; decision 0.3)
1. **Presets:** parametric recipe engine (recipe → CDL + LUT). Author **~10–12**
   canon looks; taste-validate. Live thumbnails on a smart frame (hero frame);
   EDSO may *rank* "suggested for this footage" (orders, **never auto-applies**).
2. **Reference-image drop:** skin-aware Lab/Reinhard transfer fitted into **our
   dials/CDL**, with a **match-strength dial**. Stays steerable + arc-able.
3. **`.cube` upload:** black-box LUT + CDL trims underneath; tag input color
   space. Arc via LUT-mix + CDL trims per beat.

## 8. Arc layer (invisible)  [BE][L+D]  (Fork C)
- **EDSO tags each segment** with an intent category (e.g. `calm | build | peak |
  resolve`) — categorical output only, over the assembled timeline (reuses
  `channel`/`continuity`/`label`/`summary` as evidence).
- Deterministic table maps intent → per-beat CDL delta; the user's **single
  intensity dial** scales amplitude (0 = flat, 1 = full arc). Invisible by
  default.

## 9. Soft-local layer  [BE][D]  (Fork B) — bakes into the per-clip LUT
- Feathered masks from `framing.subject_box` + `color_stats` only:
  attention vignette (subject-anchored), graduated/directional shaping
  (`horizon_y`-aware sky gradient, soft side-lift), gentle subject/depth pop.
- All feathered; **no hard masks**. Bakes into `creative_lut` so the CDL spine
  stays clean and round-trips.

## 10. EDSO integration — the human boundary  [BE][L]  (strategy §9)
- **NL steering:** "warmer", "less teal", "fix the skin", "more cinematic" →
  dial/CDL edits on the document `grade`. LLM emits **intent + dials**, never raw
  numbers (same pattern as `framing.py`).
- **Ask-when-unsure:** relight the stubbed `awaiting_user` path — a few
  high-leverage questions only.
- **Explain the grade** (trust + steerability); **rank/suggest looks** (never
  auto-applies).
- Tool specs in `tools.py` stay **de-prescribed** — describe *what the grade
  verb does*, not *when/how* to use it. Editorial advice (if any) goes only in
  `guidance_doc.md`.

## 11. Render + export + caching  [BE][D]
- Render grading stage in `compositor.py` = per-clip `lut3d` in working space
  (§4). **Proxy == export math.**
- Preview thumbnail cache (per look × frame) for the gallery.
- **Re-grade triggers:** timeline change → regroup/re-arc; **decoupled from the
  cut**. Deterministic + cached, so cheap.
- **Export bundle:** `.cdl`/`.ccc` + `.cube`, referenced in XML/EDL;
  **editable-vs-baked toggle**. (Round-trip working-space stamp is mandatory —
  the #1 CDL round-trip bug.)

## 12. UI  [FE]  (fills the placeholder)  (strategy §12.10)
- **Grade panel** replaces `project-lenses.tsx` "Colour grading — Coming soon".
- Controls: **look picker gallery** (live thumbnails) + **intensity dial** +
  **reference-image / LUT drop** + **NL steering box** + per-clip override.
- **Before/after toggle** on `CompositePreview`.
- Design system: minimalist black/grey, **orange (`--accent`) ≤ ~2 spots**
  (primary action + active look), hairline borders, no shadows. (See
  `.cursor/skills/frontend-design`.)

---

## 13. Build order (final product; ship value as it lands)
1. **`color_stats`** L1 stage + table (§2.2). *Foundation — everything needs it.*
2. **Parity engine** (§4): bake per-clip `.cube`, ffmpeg `lut3d` + working space,
   WebGL preview shader, cube endpoint. *Prove the loop before layering grades.*
3. **Correct layer** (§5) + never-worse guardrail.
4. **`pass2b.white_reference`** live (§2.3) + wire into correct-layer WB.
5. **Match layer / grade-groups** (§6) — the auto-consistency advantage.
6. **Look layer + default gallery** (§7): presets → reference → LUT.
7. **Arc** (§8): EDSO intent categorical + delta table + intensity dial.
8. **Soft-local** (§9).
9. **EDSO steer/ask/explain** (§10).
10. **UI** (§12): panel, gallery, dials, before/after, NL box.
11. **Export bundle** (§11): CDL/LUT + XML/EDL round-trip.
12. **Re-grade triggers + caches** (§11) hardened.

Each step is independently shippable and must keep `frontend` building + render
parity green.

## 14. QA / acceptance (per PR)
- [ ] `color_stats` populated for a fresh ingest; values sane on a known clip.
- [ ] **Parity:** preview frame == export frame within tolerance on a graded
      test clip (§4).
- [ ] Correct layer **never-worse**: no crushed→noise, no further-clipped
      highlights on edge clips.
- [ ] Grade-groups cluster same-scene shots; anchor = highest `total_quality`.
- [ ] All three look modes resolve to a rendered grade; **no auto look pick**.
- [ ] Arc: intensity dial 0 = flat, 1 = full; LLM emits categories only.
- [ ] NL steering maps to dial/CDL edits; LLM emits no raw color numbers.
- [ ] Export bundle opens in Resolve with correct working space (round-trip).
- [ ] `frontend` builds; design-system checklist green (orange ≤ ~2/view).
- [ ] No new recurring LLM cost except on-demand steering + the live
      `white_reference` pass2b field.

## 15. Honest limits (do not over-promise — strategy §2/§11)
- We **raise the floor, not the ceiling**: better than the ~90% who don't grade
  well, not better than a senior colorist.
- **No crisp local work** (soft-only) — no hard power windows/sky replacement.
- **Cannot relight / recover blown or crushed detail / fix mixed color temp** —
  that's generative relighting, out of scope.
- Can be **confidently wrong** on ambiguous footage → the ask-user loop + the
  three manual modes + never-worse guardrail are the mitigations.

## 16. Risks / notes
- **Parity is the make-or-break** — commit to the one-baked-cube contract (§4)
  and gate every grade feature behind it; do not ship a preview that diverges
  from export.
- **ffmpeg color management** (working space, bit depth around `lut3d`) is
  finicky — test on log + rec709 sources; stamp working space explicitly.
- **`white_reference`** only helps the hard minority (no-people / strongly
  colored scenes); skin + gray-world + never-worse is the workhorse.
- **Re-grade cost** — keep the resolver deterministic + cached so timeline edits
  don't jank.
- **Reframing already exists** — reuse `framing`/`characteristics`; do not
  duplicate the crop solver here.
