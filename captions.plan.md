# Captions — executable plan

**Goal:** auto-captions that beat captions.ai/Opus by being *perception-driven*:
we don't style text on top of pixels we don't understand — we place, time, and
animate captions against an **edit we've already perceived**. Right safe zone,
right word emphasis, right beat, correctly paced — automatically. The UI is a
two-tier gallery ("Suggested for this edit" over "Standards"), not a 30-knob
inspector.

This plan is written to be implemented from a separate chat. It references exact
tables/fields/modules. Reuse the **color-grading playbook** wherever possible
(one deterministic resolver → identical preview + export); see
`color_grading.plan.md` and `backend/app/services/l3/grade/*` for the pattern.

---

## SS0. Why we win (the moat)

Competitors caption a video they can't see. We already computed, per cut, the
things that make captions good:

- **`caption_zones`** — normalized subject-clear boxes (never cover the face).
- **`readability_ms`** — how long a line needs to be readable.
- **word-level timings** + filler flags — true karaoke timing, auto-drop "um/uh".
- **`speaker` / `subject_box` / `shot_size`** — per-speaker identity + placement.
- **`nrg` + loudness + audio onsets** — emphasis and beat-synced motion.
- **`palette` / `color_stats` + chosen grade** — always-legible, on-brand colour.
- **`content_type` + aspect + welds + `pace`/retime** — format-fit styling, synced
  to the *actual* edited timeline.

The "Suggested" row is only possible because we perceived the edit; it is
structurally hard to copy.

---

## SS1. Locked decisions (from ideation)

1. **Two tiers.** A top **"Suggested"** row (4–5 bundles generated *for this
   edit*, each = font + animation + placement + colour), above a **"Standards"**
   catalog (universal building blocks).
2. **Minimal steering.** No heavy NL steering in v1. Selection + a light refine
   in Standards is the whole surface.
3. **No auto-apply.** Captions start OFF; the edit is not captioned until the
   user picks a style.
4. **Static previews** (no live animation in tiles for v1).
5. **Preview on a real frame, not a solid swatch.** A solid black/white card
   can't fairly show styles of varied colour (white styles vanish on white,
   dark on black). Instead:
   - **One shared representative frame** across all tiles (apples-to-apples).
   - Chosen via `caption_zones` + `total_quality` (clean safe area, good
     on-camera moment).
   - Caption rendered **in the real safe zone with its real scrim/outline**, using
     **real words** from the edit's first punchy line.
   - **Frozen at the animation's "peak" pose** (e.g. emphasized word scaled +
     colour-flipped for Pop; filled bar for Highlight-box; half-filled word for
     Karaoke) + a small **motion label** ("Pop · Beat-synced", "Karaoke", …).
   - **Graceful fallback:** if the frame is too busy/low-contrast, use a
     darkened/blurred version of the *same* frame (still their footage, now
     readable) rather than a neutral card.
6. **Standards tiles ride the same representative frame** for visual consistency
   and to keep showing real placement.
7. **Selecting a Suggested style pre-fills the Standards controls** — Suggested =
   a smart starting point; Standards = refine from there.

---

## SS2. Signals & data sources (exact)

| Need | Where it lives | Notes |
|---|---|---|
| Per-word text + start/end + filler flag | `transcripts.segments[].words[]`, `transcripts.fillers` (jsonb) | from L1 `transcript.py` (`Word{start_ms,end_ms,text,is_filler}`) |
| Beats / onsets / tempo / loudness | `audio_features.onsets_ms`, `bpm`, `is_musical`, `rms_db` (prosody), `silence_intervals` | beat-sync + loudness-driven emphasis |
| Safe caption boxes per cut | `cut_records.caption_zones` (List[[x,y,w,h]] normalized) | already baked by pass2b |
| Readability hold time per cut | `cut_records.readability_ms` | |
| Subject box / shot size / framing | `cut_records.framing` (`subject_box`, `shot_size`) | placement & size |
| Speaker + appearance | `cut_records.speaker`, `characteristics` | per-speaker identity |
| Energy / semantics | `cut_records` `nrg`/pace tags, `channel` (said/done/shown), `label`/`summary` | emphasis + suppress non-speech |
| Camera movement | `cut_records.camera` | damp animation on shaky/pan |
| Colour | L1 `color_stats.palette`, chosen grade `look` | colour + contrast |
| Cut → word alignment | `cut_records.word_span` + `src_in_ms/src_out_ms` | slice the file's word list to the cut |
| Format / aspect | document `format.aspect`, footage `content_type` | style archetype + safe-area |
| Retime/pace, welds, continuity | edit doc + `cut_records.continuity` | keep captions in sync across edits |

**Word→cut alignment:** for each timeline segment, load the source file's
`transcripts.segments[].words[]`, keep words whose `[start_ms,end_ms]` fall in
`[src_in_ms, src_out_ms]` (prefer this over `word_span` indices unless the
implementer confirms `word_span` maps cleanly), drop `is_filler` words by
default, then map source-ms → program-ms through the segment (and any retime).

---

## SS3. Data model

### Caption style bundle (the thing a tile represents)
```
CaptionStyle {
  style_id: str
  label: str                    # "Bold Impact", "Clean Minimal", ...
  font: { family, weight, case: "as-is"|"upper", tracking, max_lines, max_chars_per_line }
  animation: { preset, intensity, beat_sync: bool, emphasis: "semantic"|"loudness"|"none" }
  placement: { anchor: "lower_third"|"center"|"top"|"dynamic"|"speaker", safe_area: bool, stability_ms }
  colour: { source: "white"|"black_box"|"match_grade"|"palette_accent"|"custom",
            fill, emphasis_fill, outline, shadow, box }
}
```

### Resolved caption track (baked into `document.resolved`, like grades)
Deterministic resolver output, per program time — the ONE source of truth
preview and export both read:
```
resolved.captions = [
  CaptionEvent {
    prog_start_ms, prog_end_ms,
    lines: [ { words: [ { text, t_in_ms, t_out_ms, emphasized: bool } ] } ],
    box: [x,y,w,h] (normalized, in the chosen caption_zone),
    style_ref: style_id,             # resolved style properties inlined too
    anim: <parametric animation spec, retime-adjusted>,
  }, ...
]
```
Persist the chosen `style` on the document (e.g. `document.captions = {style_id,
enabled, overrides}`), same place `document.look` lives for grading. The
resolver runs server-side in the same PUT /document re-resolve path (see
`backend/app/routers/edit_threads.py::put_document` + `render/tasks.resolve_document`).

---

## SS4. Architecture (parity — copy the grading playbook)

- **Resolver** (`backend/app/services/l3/captions/resolver.py`, new): pure
  function `(document, cut_records, transcripts, audio_features) -> resolved.captions`.
  No I/O; measurement fetched once per resolve (mirror `grade.measure`).
- **Preview:** a **DOM/canvas overlay** in the program monitor
  (`frontend/src/components/preview/`) that reads `resolved.captions` and
  animates in real time. Cheap, fully animatable, instant. (Overlay the server
  track exactly like `resolvedGrades` is overlaid in `composite-preview.tsx`.)
- **Export:** bake `resolved.captions` → **ASS/libass** subtitle burned by the
  ffmpeg compositor (`backend/app/services/render/compositor.py`). ASS is
  fast, industry-standard, and expresses karaoke + transforms. Reserve a
  **per-frame overlay render** path only for kinetic styles ASS can't express.
- **Parity contract:** preview and export must render the same track. Constrain
  v1 animation vocabulary to what ASS can express (see SS13); anything richer is
  phase 2 via frame render.

---

## SS5. The style bundle — catalogs

### Fonts (curate ~6, self-host; prefer SIL OFL so we can embed/burn freely)
- **Condensed impact:** Anton / Bebas Neue (hype, all-caps).
- **Bold geometric sans:** Poppins / Montserrat ExtraBold.
- **Neutral workhorse:** Inter Tight.
- **Rounded friendly:** Nunito.
- **Editorial:** Fraunces (variable serif) or Archivo Expanded.
- **Marker/handwritten:** Permanent Marker / Caveat.
Use **variable fonts** where possible (weight/width as a continuous axis — also
an animation lever). Confirm licensing before adding any non-OFL font.

### Animations (parametric presets)
Pop · Karaoke-fill · Typewriter (word-by-word) · Word-bounce · Highlight-box ·
Subtle-fade · Slide-up. Each defined as enter/emphasis/exit curves + an
`intensity` scalar. **Easing is the premium signal** — use spring/overshoot
curves, never linear.

### Placement
Lower-third · Centered · Top · **Dynamic (safe-zone)** · Speaker-follow.

### Colour
White + soft shadow · Black-box · **Match grade** · **Accent-from-palette** ·
High-contrast pop. All auto-checked for contrast against the footage.

---

## SS6. Tier 1 — "Suggested" (generated per edit)

**Generation recipe** (`backend/app/services/l3/captions/suggest.py`, new):
1. Derive a **base bundle** from edit signals: `content_type` + `nrg` + `palette`
   + aspect + speaker count → the natural best fit. This is **Suggestion #1
   "Auto."**
2. Produce **4 more** by rotating along curated design axes (font archetype,
   motion intensity, colour source, placement anchor, case/weight) toward
   distinct moods, **format-gated** so even the boldest is tasteful:
   - **Auto** — best fit, safe.
   - **Bold / Hype** — condensed all-caps, beat-pop, accent emphasis.
   - **Clean / Minimal** — light weight, subtle fade, white + soft shadow.
   - **Editorial / Premium** — serif or tight sans, karaoke-fill, grade-matched.
   - **Playful / Kinetic** — rounded, bounce/word-pop, palette colours.
3. **Guarantees on every variant:** safe-zone placement (`caption_zones`),
   contrast vs footage, `readability_ms` pacing. No variant can be illegible or
   cover the subject.
4. **Rationale string** per suggestion ("Bold impact — high-energy reel, single
   speaker"), derived from the signals used. Shown under the tile.
5. **Cache per edit version** (stable across sessions; recompute when the edit
   materially changes). A **"Reshuffle"** action re-rotates the 4 variants
   (pin Auto).

---

## SS7. Tier 2 — "Standards" (universal catalog)

The manual building blocks, with landscape/portrait defaults pre-set:
- **Placement** (portrait → center-lower + platform-safe gutter; landscape →
  lower-third), **Fonts** (the curated ~6), **Colours**, **Animations** (SS5).
- Selecting a Suggested style **pre-fills** these controls.

---

## SS8. Preview tiles (static)

Per SS1.5–1.6: one shared representative frame, caption in the real safe zone
with real scrim, real words, frozen at animation peak pose + motion label,
graceful darkened/blurred fallback. Applies to both Suggested and Standards
tiles. Frame extraction: reuse existing proxy/frame decode (see
`l1/color_stats._decode_rgb_frame_at` for the pattern) at the chosen
`hero_ts_ms` of the selected representative cut.

---

## SS9. Placement engine

- **Sit in the largest/most stable `caption_zone`.** Never on `subject_box`.
- **Stability > reactivity:** compute one placement per **run of welded cuts**;
  only move when the safe zone genuinely changes, with **hysteresis** so it never
  twitches per cut. (Continuity/weld info from `cut_records.continuity`.)
- **Shot-size aware:** CU → tuck into a safe band; wide → can nestle near subject;
  portrait → center-lower; landscape → lower-third.
- **Platform safe-areas:** auto-avoid TikTok/Reels UI gutters per aspect/target
  (this is a concrete, visible win — competitors get covered by the like button).
- **Overlay collision:** we own the edit doc — place around split-screen cells,
  name tags, logos already on screen.
- **Size** from aspect + zone size + `readability_ms`.

---

## SS10. Timing & animation engine

- **Word reveal** from per-word `[t_in,t_out]` (karaoke/pop/typewriter land on the
  spoken word).
- **Emphasis** = semantic keyword (from `label`/`summary`/LLM) scaled by
  `nrg`/`rms_db` loudness → the pop lands on the *right* word.
- **Beat-sync:** snap animation keyframes to `audio_features.onsets_ms` (and
  `bpm` when `is_musical`) for reels.
- **Readability:** `readability_ms` governs hold/clear; also line-break decisions
  (`max_chars_per_line`, `max_lines`).
- **Camera-aware damping:** calm motion on `camera:shaky/pan`, livelier on static.
- **Retime-aware:** map word times through the resolved timeline so captions stay
  locked after trims/retimes/welds.
- **Fillers dropped** by default (from `transcripts.fillers` / `is_filler`).
- **Non-speech suppression:** no captions on `channel != said` beats.

---

## SS11. Colour system

- Default from **grade look + `palette`**: choose fill/outline/shadow that clears
  a contrast threshold against the footage in the caption zone.
- `palette_accent` emphasis colour pulled from a vibrant, non-skin palette entry.
- `match_grade` ties caption colour to the chosen grade so it feels designed in.
- Always compute a legibility floor (outline/shadow/box auto-added if contrast is
  insufficient).

---

## SS12. Export (ASS/libass) — v1 vocabulary

Map the resolved track to ASS:
- `\pos`/`\an` placement (from resolved `box`), `\fn` font, `\fs` size,
  `\bord`/`\shad` outline/shadow, `\c`/`\3c`/`\4c` colours, box via `\4a`+bord.
- **Karaoke:** `\k`/`\kf` per word from word timings.
- **Motion:** `\t(...)` for scale/colour emphasis, `\fad` fade, `\move` slide.
- Spring/overshoot approximated with **segmented `\t`** (ASS has no native
  springs). Anything richer (true kinetic typography) → **phase-2 per-frame
  overlay render**.
- Confirm the compositor's ffmpeg build has `libass` (`subtitles`/`ass` filter).

---

## SS13. Backend pieces (new)

```
backend/app/services/l3/captions/
  __init__.py
  suggest.py      # SS6: per-edit suggestion generation + rationale + cache
  resolver.py     # SS3/SS4: document + signals -> resolved.captions (pure)
  styles.py       # SS5 catalogs (fonts/animations/placements/colours) + bundles
  placement.py    # SS9 safe-zone/stability/safe-area/collision solver
  timing.py       # SS10 word/beat/emphasis/readability/retime timing
  colour.py       # SS11 contrast + palette/grade-derived colour
  ass_export.py   # SS12 resolved -> ASS text
```
- Wire the resolver into `render/tasks.resolve_document` + `put_document`
  re-resolve, storing `resolved.captions`.
- Endpoints (mirror `routers/grade.py`): `GET /api/captions/suggestions` (per
  thread/version → 4–5 bundles + rationale + representative-frame ref),
  `GET /api/captions/fonts|catalog` (Standards options). Font files served/
  self-hosted for both web preview and ffmpeg burn.

---

## SS14. Frontend pieces (new)

```
frontend/src/components/captions-view.tsx       # two-tier gallery (Suggested / Standards)
frontend/src/components/preview/caption-overlay.tsx  # animated DOM/canvas overlay in monitor
frontend/src/stores/  # captions selection state (mirror edit-doc-store look handling)
frontend/src/lib/resolve-captions.ts (opt)      # client-side animation interpolation from resolved.captions
```
- Overlay reads `document.resolved.captions` (baked server-side), same overlay
  pattern as grades in `composite-preview.tsx`.
- Style selection persists via the same debounced/serialized save the grade
  panel now uses (see `color-grade-view.tsx` `applyLook`/`flushLook` and the
  store's `commitLook`) — **do NOT re-introduce save-per-tick.**
- Tiles per SS8 (static, shared frame, peak pose + label).

---

## SS15. Phasing

**v1 (ship):**
- Word-timed karaoke/pop + subject-aware placement (`caption_zones`) +
  readability pacing + filler drop + non-speech suppression.
- Suggested row (5 bundles + rationale, cached) + Standards catalog, no
  auto-apply, static tiles on shared frame.
- ~6 curated fonts, ~6 animations, placements, colours.
- ASS export with parity.

**Phase 2 (moat deepening):**
- Per-speaker identity (colour/name tags/position from `speaker`+`subject_box`).
- Beat-synced kinetic typography via per-frame render.
- Semantic emphasis tuning, hover-to-animate tiles, "Reshuffle."
- Platform-target presets (TikTok/Reels/Shorts safe-areas + sizing).

---

## SS16. Resolved decisions + minor open items

**RESOLVED (do not re-litigate):**

1. **End-to-end including export is v1 scope.** "Whole pipeline must work" is the
   bar: perceive → place → time → resolve track → **preview overlay AND ffmpeg
   burn from the same track.** Export is NOT deferred.
2. **Animation is deliberately limited in v1** — only effects that render
   *identically* in the DOM/canvas preview and in ASS/libass: **fade, pop/scale
   emphasis, karaoke word-fill, simple slide.** No kinetic typography in v1
   (deferred to phase 2 frame-render). Limited motion is accepted in exchange for
   guaranteed preview↔export parity.
3. **Export tech = ASS/libass burn, CONFIRMED available.** Local ffmpeg 8.0.1 is
   built `--enable-libass` and exposes both the `ass` and `subtitles` filters.
   Ensure the deploy/render image ships the same (fontconfig + libass + the
   self-hosted font files). Fonts must be resolvable by ffmpeg at burn time
   (fontsdir / fontconfig) AND served to the browser for preview.
4. **Word source = time-based.** Slice each source file's
   `transcripts.segments[].words[]` by `[src_in_ms, src_out_ms]`, map to program
   time. Ignore `word_span` unless it's trivially convenient.

**Minor open items (pick a sensible default, don't block):**

- **Representative-frame selection:** default = the cut with the biggest stable
  `caption_zone` among high `total_quality` on-camera cuts, sampled at its
  `hero_ts_ms`.
- **Font set:** lock ~6 SIL OFL fonts (embed/burn freely) — see SS5.
- **Line-break policy:** derive `max_chars_per_line`/`max_lines` from aspect +
  `readability_ms`; default 2 lines, ~24 chars/line portrait, ~40 landscape.

---

## SS17. Risks & mitigations

- **Tile legibility on busy footage** → shared clean-zone frame + real scrim +
  darkened/blurred fallback (SS1.5).
- **Placement jitter** → weld-run stability + hysteresis (SS9).
- **Preview ≠ export** → single resolved track, constrained v1 vocabulary, ASS
  parity (SS4/SS12); test a burn matches the overlay.
- **Save thrash** → reuse the debounced/serialized persistence + `commitLook`-style
  commit (SS14); never save per interaction tick.
- **Font/glyph coverage** → pick script-appropriate font from transcript language.

---

## SS18. Build checklist

- [ ] `captions/styles.py` catalogs + `CaptionStyle` schema.
- [ ] `captions/timing.py` (word/beat/emphasis/readability/retime).
- [ ] `captions/placement.py` (safe-zone/stability/safe-area/collision).
- [ ] `captions/colour.py` (contrast + palette/grade).
- [ ] `captions/resolver.py` → `resolved.captions`; wire into resolve_document + put_document.
- [ ] `captions/suggest.py` (5 bundles + rationale + per-version cache).
- [ ] `captions/ass_export.py` + compositor burn (libass confirmed present; ensure fonts resolvable at burn time via fontsdir/fontconfig).
- [ ] Endpoints: suggestions + catalog + font hosting.
- [ ] FE `caption-overlay.tsx` (animated preview overlay, reads resolved.captions).
- [ ] FE `captions-view.tsx` (Suggested/Standards, static tiles on shared frame, no auto-apply).
- [ ] FE persistence via debounced/serialized save + look-style commit (no per-tick saves).
- [ ] Parity test: exported ASS burn matches the monitor overlay.
