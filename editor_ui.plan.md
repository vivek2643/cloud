# Editor UI refresh — timeline layout + right panel (FE-only)

Ideation → executable plan for the editor's *look & layout*: the timeline
surface (vertical model, toolbar, zoom, tracks, clip colours) and the right AI
Edit panel (monitor, chat, controls). **Frontend only — no render/schema
changes, no backend.** Another chat executes this.

Guiding principle: **declutter + let it breathe.** The right panel is doing too
much in 460px (monitor + render bar + chat + doc-view stacked), which is why
everything feels cramped. The moves below reclaim space and give the monitor and
chat room. We stay a **full N-track NLE**, minimalist black/grey, **orange
(`--accent`) sparse** (see `.cursor/skills/frontend-design`).

Cross-refs:
- Builds on `timeline_nle.plan.md` (already implemented).
- **Reverses `timeline_trim.plan.md` B3** (which removed add-track) — see §1.6.
- The focus accordion (§1.1) supersedes the earlier "fixed lanes + vertical
  scroll" and the literal "one-at-a-time / 30% overlap" idea.

Design guardrails for every task:
- **No hardcoded hex** — colours via `var(--token)` (the current clip hex in
  `edit-project.ts` is a violation to fix, see §1.7).
- **Orange ≤ ~2 spots per view** (primary action + active state only).
- FE-only: the saved spine+ops the renderer sees must not change.
- `npm run build` clean; design-system checklist green.

---

## Part 1 — Timeline surface

### 1.1 Focus accordion (the vertical model)  [FE]
Replace fixed-height lanes + vertical scroll with an accordion so an N-track
stack **never forces vertical scroll**.
- The **focused** track expands to full working height (thumbnails, waveform,
  trim handles, blade). Every **other** track stays visible as a **sliver**
  (~14–16px min, never collapses to invisible): shows label + mute + its clips
  as thin blocks **at true time positions**; the **playhead spans all lanes** so
  cross-track alignment is preserved.
- Focus **follows selection**: clicking a clip expands its track + selects it;
  clicking a sliver focuses that track. Focus vs selection stay **decoupled**
  (arrow-nav moves focus without changing selection).
- **Focus / All toggle**: "All" = equal fit-to-height (every track same size,
  no scroll); "Focus" = accordion. One control (pairs with the vertical side of
  zoom).
- Enhancements: **focus-a-pair** (pin two tracks expanded, e.g. a video + its
  dialogue) and **peek-on-hover** (hovering a sliver temporarily expands it
  without committing focus).
- Quiet height animation (`duration-150..200`), no bounce.

### 1.2 Focus-navigation arrows  [FE]
Add a visible complement to keyboard ↑/↓.
- A compact **▲/▼ pair in a slim left gutter**, vertically **aligned to the
  focused lane** (fixed x-position). ▼ = focus next track, ▲ = focus previous;
  **clamp at ends** (disable at top/bottom, no wrap). Thin accent bar on the
  focused lane's left edge ties the arrows to it.
- **Disambiguate from the existing reorder chevrons** in `TrackHeaderRow`
  (those swap video-layer `z`): keep focus-arrows in the *gutter* (one pair for
  the whole timeline), keep reorder chevrons *inside each header*, and give them
  distinct icons (focus = filled triangles; reorder = thin chevrons). Tooltips:
  "Focus previous/next track (↑/↓)".
- Optional: **focus dots** in the gutter (one per track, current highlighted;
  click a dot to jump focus). Include only if we want the density.

### 1.3 Center-top transport + cut  [FE]
Rebalance the toolbar (currently everything is left-packed):
- **Left:** snap toggle, undo/redo.
- **Center:** prev-frame · **play/pause** · next-frame · **split/cut (⌘K,
  scissors)** — the most-used controls, centered (modern CapCut/Descript feel).
- **Right:** zoom slider (§1.4) + the `MM:SS / MM:SS` timecode.
- Keep play/pause as the one orange element in the toolbar.

### 1.4 Horizontal zoom slider  [FE]
- Replace the three zoom icon-buttons with a **log-scaled slider** (maps
  `pxPerSec` 2→400) placed on the **right, just left of the timecode**.
- Keep `⌘/Ctrl + wheel` for fine zoom; add a tiny **fit** affordance
  (double-click slider = zoom-to-fit) so we don't lose it.

### 1.5 Track-header gap fix  [FE]
The gap between "V1" and the mute icon is because the label is `flex-1` and
stretches the full header width. Fix: **don't stretch the label** — group
`[label] [mute]` left-aligned with a small gap; optionally shrink `HEADER_W` a
touch. Purely cosmetic.

### 1.6 Default tracks + add-track (reverses trim B3)  [FE]
- **DECISION: default tracks = V1 + A1 + V2** (no A2 by default). Always render
  these three as lanes, with empty ones as **drop targets**, so a cutaway always
  has a home and the layout doesn't jump. (Tracks stay derived in
  `edit-project.ts`; an "empty" lane is a UI placeholder until a clip lands. A2
  still appears automatically if a bed is placed / added.)
- Reintroduce a **minimal `+` add-track** (video / audio) in the header column.
- **Note:** this reverses `timeline_trim.plan.md` B3 — update that plan so the
  two don't conflict.

### 1.7 Clip colours → design tokens (answers "green → orange?")  [FE]
Current clip colours are **hardcoded hex** in `frontend/src/lib/edit-project.ts`
(`COLOR_BASE_AUDIO="#2bb673"` green, `COLOR_BASE_VIDEO`, `VIDEO_COLORS[]`,
`AUDIO_COLORS[]` — lines ~72–74, 117, 160, 189, 234). This violates the design
system.
- **Recommendation: do NOT make audio orange.** Orange is the sole accent and
  must stay sparse (≤ ~2 spots/view); flooding every audio block with orange
  kills the accent and clashes with the orange primary/active states. Green is
  off-palette, so it *should* change — but to **neutral greys**, not orange.
- **DECISION: neutral grey.** Move all clip colours to **tokenised neutral
  greys** (video vs audio differentiated by tone/label/waveform, not hue).
- **DECISION: the playhead becomes orange.** Against grey clips a grey/white
  playhead blends in, so make the **playhead solid `--accent`** — it's the single
  most important always-on indicator and the textbook use of the one accent.
- **Orange budget (keep it sparse):** the playhead owns the *loud* orange. To
  avoid competing oranges, render the **selected/active clip as a soft
  `--accent-soft` wash + border**, NOT another full-orange fill. Net loud-orange
  in the timeline ≈ playhead + the play button; selection stays soft.

---

## Part 2 — Right AI Edit panel (declutter + breathe)

### 2.1 Bot's first message (replace the empty-state)  [FE]
- Remove the `EmptyState` card ("Chat about your footage / Ask about your
  clips…") in `ai-edit-panel.tsx`.
- Instead, show a **UI-only assistant greeting bubble** on a fresh thread (not a
  backend turn, not persisted as a real message). Short placeholder for now:
  > "Hi, I'm EDSO. Tell me what you're making and I'll start building the edit —
  > or ask me anything about your footage."
  (Copy is a placeholder; we'll refine later.)

### 2.2 Bigger composer + clearly-orange send  [FE]
- Grow the composer box a lot: `textarea` `rows≥3`, taller `min-height`, larger
  `max-h` auto-grow; more padding in the bordered box. It should read as a
  roomy input, not a one-line strip.
- The send button is already `--accent`, but it drops to `opacity-30` when the
  input is empty (why it looks grey in the screenshot). **Keep it visibly
  orange** — soften/remove the disabled dimming (or dim less), so the primary
  action always reads orange.

### 2.3 Remove the render bar → "Chat" label; relocate Export  [FE]
- Remove the **Preview/Export preset select + inline Render** (`RenderBar` as it
  sits between monitor and chat). Replace that strip with a **thin "Chat"
  section label** above the conversation, and **shrink the block**. (This strip
  is the surface that morphs into player controls on hover — see §2.5.)
- **DECISION: Export moves on top of the video** (a control on the monitor
  itself). Rough placement for now — "we'll adjust the on-video controls later."
  Don't drop render; just relocate it onto the monitor. Preset choice
  (720p/1080p) can hide behind that control or default to export.

### 2.4 Bigger monitor, less grey  [FE]
- Enlarge the program monitor (`CompositePreview`): reduce the panel padding
  around it, let the frame use the full panel width, and trim the surrounding
  `--sidebar` dead space so the video dominates.
- Consider a **"theater"/expand** affordance for a larger view when needed.
- (The black bars inside the frame come from source-vs-delivery aspect
  mismatch — leave those; they're correct letterboxing, not wasted chrome.)

### 2.5 Player controls on hover  [FE]
Free vertical space by removing the always-on transport strip under the monitor.
- **DECISION: morph the "Chat" bar.** The thin "Chat" section label (§2.3) is
  the surface: at rest it reads **"Chat"**; on **hover over the video** it morphs
  into player controls — **play/pause**, **back 10s**, **forward 10s** (change
  the current ±5s → ±10s), and a new **fullscreen** toggle (Fullscreen API on the
  monitor container). Smooth `transition` between the two states; revert to
  "Chat" on hover-out. Keep a thin scrubber (show on hover).
- Rationale (user): the conventional on-video overlay was designed for passive
  players; this surface is repurposed for an editing context, so a morphing bar
  is intentional here — not the standard video-player convention.

---

## Decisions (locked)
1. **Default tracks:** V1 + A1 + V2 (no A2 by default). (§1.6)
2. **Export:** on top of the video (monitor control); on-video placement refined later. (§2.3)
3. **Player controls:** morph the "Chat" bar on video hover (±10s + fullscreen). (§2.5)
4. **Clip colour:** neutral grey; **playhead → orange**; selection = `--accent-soft` wash + border. (§1.7)

## Suggested execution order
1. Part 2 first (fast, high-visible wins): 2.1 greeting → 2.4 bigger monitor →
   2.5 hover controls → 2.3 remove render bar + relocate Export → 2.2 composer.
2. Part 1: 1.5 gap fix → 1.3 center transport → 1.4 zoom slider → 1.7 clip
   colours → 1.6 default tracks + add-track → 1.1 accordion → 1.2 focus arrows.
   (Accordion + focus arrows are the biggest; land the smaller polish first.)

## QA / acceptance (per PR)
- [ ] `frontend` builds; no TS errors.
- [ ] No hardcoded hex anywhere touched — all colours via `var(--token)`.
- [ ] Orange ≤ ~2 spots per view (timeline + panel each).
- [ ] N-track intact: V2/A2 present, add-track works, tracks still derived.
- [ ] Accordion never forces vertical scroll at any track count; slivers stay
      clickable; playhead spans all lanes.
- [ ] Monitor is visibly larger; hover controls include ±10s + fullscreen.
- [ ] Render parity: saved spine+ops unchanged (FE-only).
