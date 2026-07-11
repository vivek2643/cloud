# Timeline trim — lean AI-first web NLE (Timeline-area ONLY)

Follow-up to `timeline_nle.plan.md`. The two phases (P0 foundations + P1 pro
verbs) landed a full desktop-grade timeline. This plan **removes / simplifies**
the pro-desktop muscle that a web, AI-first editor doesn't need, and decides
what (if anything) from P2 is still worth building.

## Scope guardrail (read first)
- **Touch the Timeline surface ONLY.** Do NOT touch Drive, Cuts view, AI panel,
  Preview, Render, Settings, or any backend beyond the one flagged `[BE]` P2
  item. The dead-code items on other surfaces (Share/Upload-Link buttons, unused
  List view, media-card drag mismatch, folder context-menu hook, `compact`
  grid prop) are explicitly **out of scope here**.
- Files in play (all under `frontend/src/`):
  - `components/timeline-editor.tsx` — toolbar, lanes, handlers, keyboard map.
  - `stores/timeline-view.ts` — tool/snap/marks/markers/insert/link/trackMeta.
  - `stores/edit-doc-store.ts` — mutators (`slip`, `slipOp`, `overwriteSpine`,
    `makeRoomInsert`, delete paths).
  - `components/preview/*` — only where solo/mute mixing is read.
- **Guiding principle:** EDSO does the heavy edit; the timeline is a
  **refinement** surface. Keep the high-frequency basics (zoom, scroll, snap,
  undo/redo, trim, move, reorder, split-at-playhead, copy/paste/duplicate,
  frame-nudge, transport, audio gain/mute, filmstrip, waveforms, numeric TC
  inspectors). Cut everything that needs an NLE tutorial.
- **Render parity:** every task here is `[FE]` (removes UI / view-state; the
  saved spine+ops the renderer sees is unchanged) EXCEPT the one P2 fades item,
  which is `[BE]`. Removing overwrite/insert must keep drops producing the same
  ripple-insert spine+ops the backend already understands.
- After each part: `npm run build` in `frontend/` must pass with **no unused
  symbols** and the design-system checklist stays green.

---

## Part A — Remove outright

### A1. Slip + Slide trim tools  [FE]
Advanced source-trim; near-zero web usage.
- `timeline-editor.tsx`: remove `startSlip` (~726) and `startSlide` (~752); the
  toolbar Slip/Slide `IconBtn`s (~1074, ~1077); the `tool === "slip" | "slide"`
  branches in the clip pointer-down dispatch (~1269–1271); the `Y`/`U` keyboard
  cases (~533, ~535); and the `slipSeg`/`slipOp` selectors (~140, ~147).
- `edit-doc-store.ts`: remove `slip` (decl ~111 / impl ~337) and `slipOp`
  (decl ~155 / impl ~543).
- Also drop the `Y`/`U` rows from any shortcut list.

### A2. Roll edit (shared-seam drag)  [FE]
- `timeline-editor.tsx`: remove `startRoll` (~787) and the roll seam handle it's
  wired to (`key={`roll:...`}` block ~1250–1254).

### A3. Insert / Overwrite modes  [FE]
Drops become plain ripple-insert (the current default). Remove the mode entirely.
- `timeline-editor.tsx`: remove the INS/OVR toggle `IconBtn` (~1129–1137); the
  `insertMode`/`toggleInsertMode` selectors (~196–197); replace the three
  `insertMode === "insert" ? makeRoomInsert(...)` / `overwrite` branches
  (~421, ~922–932, ~944) with the unconditional ripple-insert path.
- `edit-doc-store.ts`: remove `overwriteSpine` (decl ~125 / impl ~415). Keep
  `makeRoomInsert` **only if** the ripple-insert drop still calls it; otherwise
  inline/remove.
- `timeline-view.ts`: remove `insertMode` + `toggleInsertMode` (state ~54–55/90,
  actions ~75/129).

### A4. Timeline In / Out marks  [FE]
No loop was ever wired, and source-range marking is a pro workflow. Removing this
also removes the last consumer of overwrite-in-range (handled in A3).
- `timeline-editor.tsx`: remove Set-in / Set-out `IconBtn`s (~1122–1125); the
  `inMarkMs`/`outMarkMs` selectors + `setInMark`/`setOutMark` (~178–179, 191–192);
  the shaded I/O band render (~1192–1198); and the `inMarkMs`/`outMarkMs` entries
  in `snapTargets` (~561–562, dep array ~569).
- `timeline-view.ts`: remove `inMarkMs`/`outMarkMs` state (~48–49/87–88) and
  `setInMark`/`setOutMark` (~71–72/109–120).

### A5. Markers  [FE]
Session-only, no list panel — half-built, low value.
- `timeline-editor.tsx`: remove the Add-marker `IconBtn` (~1119); `addMarker`/
  `removeMarker` selectors (~193–194); the `M` keyboard case (~539) + its dep in
  the effect array (~551); the marker pips passed to the ruler (`onRemoveMarker`
  ~1188) and `markers` in `snapTargets` (dep ~569).
- `timeline-view.ts`: remove `markers` state (~50/88) and `addMarker`/
  `removeMarker` (~72–73/121–127). Remove marker pip rendering in `TimeRuler`.

### A6. Track lock  [FE]
- `timeline-editor.tsx`: remove the lock toggle handler (~971), the `isLocked`
  helper (~323) and its guards, `locked` lane styling (`opacity` ~1233, const
  ~1213), and the lock button + `Unlock`/`Lock` icon imports (~23).
- `timeline-view.ts`: drop `lock` from `TrackMeta` (~19).

### A7. Lane height resize  [FE]
Lanes become a fixed height.
- `timeline-editor.tsx`: remove the height drag handle (~1008–1011) and replace
  `trackMeta[track.id]?.heightPx ?? DEFAULT_LANE_H` (~1008, ~1212) with the
  constant `DEFAULT_LANE_H`. Remove `MIN_LANE_H`/`MAX_LANE_H` if now unused.
- `timeline-view.ts`: drop `heightPx` from `TrackMeta` (~22).

> After A1–A2 the only tool left is `select`. **Collapse tool modes:** remove the
> Blade button too (see B1), then remove the `tool` state, `TimelineTool` type,
> `setTool`, the tool `IconBtn`s (~1068–1077), the `V`/`B` shortcuts (~529–531),
> and the per-tool cursor/dispatch branches. The clip pointer-down becomes the
> plain select/move path unconditionally.

---

## Part B — Simplify (keep the capability, cut the controls)

### B1. Blade tool mode → keep ⌘K only  [FE]
`splitAtPlayhead()` + `Cmd/Ctrl+K` stay (P0.1, ~347). Remove the Blade **tool
mode**: the Blade `IconBtn` (~1071), the `B` shortcut (~531), the
`tool === "blade"` crosshair cursor (~1234) and `bladeClick` split-at-click
branch (~1259–1268). Splitting = playhead + ⌘K, one obvious way.

### B2. Delete → one ripple delete  [FE]
Collapse lift-vs-ripple into a single delete. Keep the spine's natural ripple and
op removal; bind everything to `Delete` (and `Backspace`). Remove the
`Shift+Delete` "ripple" special-case and any "lift leaves a gap" branch so users
don't reason about two behaviors. (Ref P0.2 in the old plan.)

### B3. Drop the "add empty track" buttons  [FE]
> **REVERSED — superseded by `editor_ui.plan.md` §1.6.** The editor UI refresh
> reintroduces default tracks (V1 + A1 + V2) and a minimal `+` add-track control,
> so this removal no longer applies. Kept here for history; do NOT execute B3.

Tracks are **derived** from ops (z/role) — a manual "add empty track" placeholder
is confusing. Remove the +V / +A toolbar buttons and the pending-empty-track
plumbing. New lanes appear automatically when a clip is dropped/placed (existing
`edit-project.ts` derivation stays authoritative). Keep **video-layer reorder**
(z-swap) since V2+ still needs stacking control.

### B4. Track solo → remove; keep mute  [FE]
Keep **track mute** (audio → `gain_db=-120`; video → view-only hide). Remove
**solo**: the solo toggle (~976), `solo` in `TrackMeta` (~18), and the solo
branch in the preview mixer (`use-program-player` / `composite-preview`
`applyTrackMeta`). After A6/A7/B4, `TrackMeta` holds only `mute`.

### B5. A/V link → always linked  [FE]
Keep the spine dialogue coupled to its video (the sensible default). Remove the
per-seg unlink toggle: `toggleLinked`/`unlinkedSegIds` (`timeline-view.ts`
~59/76/91/130–133; `timeline-editor.tsx` selectors ~198–199 and the `linked`
computation ~585), and the `Link2` inspector control. Selecting/trimming a video
seg always moves its dialogue.

---

## Part C — P2 verdict (what's actually required)

The two removals above (A4 marks, A5 markers) **kill the dependency base** for
most of P2. Verdict per original P2 item:

| P2 item | Verdict | Reason |
|---|---|---|
| Markers + marker list | **Drop** | Removed in A5; no persistence, low value. |
| Loop the In/Out range | **Drop** | Depends on In/Out marks (removed A4); loop was never wired. |
| J/K/L shuttle | **Drop** | Pro-desktop muscle; not needed for reels/podcasts. |
| Export range | **Drop** | Depends on In/Out marks; render already does the full timeline. |
| Transitions (crossfade/dip) | **Drop (defer)** | Needs a backend transition op + renderer; not worth it now. |
| **Audio fades (bed fade in/out)** | **BUILD — the one worth it**, `[BE]` | Genuinely useful for music/SFX beds; small, additive. |
| Video fades / dogears | **Drop (defer)** | Rolls into transitions; defer with them. |
| Keyboard-shortcut cheatsheet | **Optional / nice-to-have** | Cheap `?` overlay; the shortcut set is now small, so low priority. |

### C1. Audio fades — the single P2 to build  `[BE]`
- **Backend:** add `fade_in_ms` / `fade_out_ms` to `place_audio` (op schema +
  validation), and apply gain ramps in the renderer's audio graph. **This is the
  only permitted backend change in this plan** — do it as its own PR.
- **Frontend (Timeline):** fade "dogear" handles on the top corners of **audio**
  clip blocks that set `fade_in_ms`/`fade_out_ms`; preview approximates with a
  gain ramp. Ship FE only after (or behind) the backend field so it never
  silently fails to render.
- If the backend work isn't wanted right now, **defer the whole item** — do NOT
  ship a preview-only fade that doesn't export.

### C2. Shortcut cheatsheet (optional)
A `?`-triggered overlay listing the (now much smaller) shortcut set: play/pause,
frame step/nudge, split (⌘K), copy/cut/paste/duplicate, undo/redo, snap toggle,
zoom. Build only if there's spare time; skip otherwise.

---

## Execution order
1. **Part B first** (B1 blade-mode, B3 add-track, B4 solo, B5 unlink, B2 delete)
   — these are pure UI/control removals with the least cross-coupling.
2. **Part A** (A1 slip/slide → A2 roll → then collapse `tool` state per the note;
   A3 insert/overwrite → A4 marks together since overwrite reads the marks; A5
   markers; A6 lock; A7 height). Do A3+A4 in one pass.
3. **Prune `timeline-view.ts`** last: after A/B, delete now-orphaned state
   (`tool`, `TimelineTool`, `setTool`, `inMarkMs`/`outMarkMs`, `markers`,
   `insertMode`, `unlinkedSegIds`, `lock`/`solo`/`heightPx` on `TrackMeta`). Keep
   `pxPerSec`, `scrollLeftPx`, `snapEnabled`, `snapGuideMs`, `clipboard`, and
   `trackMeta.mute`.
4. **Part C1** (audio fades) only if the `[BE]` change is approved — separate PR.
5. **Part C2** (cheatsheet) optional.

## Acceptance
- [ ] `frontend` builds; zero unused-symbol / TS errors after each part.
- [ ] Toolbar shows only: transport, snap toggle, zoom in/out/fit (no
      tool-mode / marks / markers / INS-OVR / add-track buttons).
- [ ] Splitting works via playhead + ⌘K only; one delete behavior.
- [ ] Drops still ripple-insert into the same spine+ops the renderer already
      understands (diff the saved document — no render regression).
- [ ] Zoom/scroll/snap/undo-redo/trim/move/reorder/copy-paste/nudge/audio-gain/
      filmstrip/waveforms/numeric-TC inspectors all still work.
- [ ] `timeline-view.ts` contains no orphaned state/actions.
- [ ] Design system: orange ≤ ~2 spots/view; no shadows; hairline borders.

## Notes / risks
- **Coupling:** overwrite (A3) is the only consumer of In/Out marks (A4) — remove
  together to avoid a dangling `overwriteSpine`.
- **Preview mixer:** solo removal (B4) touches `use-program-player` — verify audio
  still mixes correctly with only mute (no solo) after the change.
- **Derived tracks:** after B3, confirm a fresh drop still auto-creates its lane
  via `edit-project.ts` (no manual track needed).
- Keep `splitAtPlayhead`, `makeRoomInsert` (if still used by ripple-insert),
  `snapValue`, undo/redo, and clipboard intact — they back the retained basics.
