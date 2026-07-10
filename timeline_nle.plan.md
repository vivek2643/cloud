# Timeline → Professional NLE (timeline-area scope)

Turn the current timeline dock into something that feels like a real NLE
(Premiere / FCP / Resolve), **limited to what lives in and around the timeline
area**. Value/parameter editing (color grade, captions, effects, keyframes,
transform/PiP, speed/retime, precise transform numerics) is **out of scope
here** — that belongs to the separate *Inspector* plan (right-rail tabbed
panel). This file is the executable plan for the timeline surface only.

---

## 0. Guiding principle

- **Timeline area = time-domain manipulation**: tracks, clips, trims, blade,
  ripple, snapping, zoom/scroll, markers, clip visuals (thumbnails/waveforms).
- **Do NOT** add parameter editors (grading, captions, effects, transform
  numeric fields) here — those go to the Inspector.
- Keep the app's design system: minimalist black/grey, **orange (`--accent`)
  used sparingly** (primary action + active indicator only), hierarchy via
  spacing/type, hairline `--border`, ghost buttons. No new colors, no shadows.
- **Open, un-boxed shell** — the timeline must NOT sit inside a framed card.
  Remove the outer container box; let it breathe on the page background,
  separated by whitespace (and at most a single hairline top divider). See §2.0
  — do this first; it's the fastest visible win and sets the aesthetic tone for
  everything else.

## 1. Current state (read these first)

- `frontend/src/components/timeline-editor.tsx` — the whole timeline UI:
  ruler, lanes, blocks, trim/move/reorder pointer handlers, inline inspector
  footer, save/revert/history, keyboard transport.
- `frontend/src/lib/edit-project.ts` — pure mapper: spine + operations →
  generic `{ tracks, clips, durationMs }` (the NLE track/clip model). Tracks are
  **derived** (V1 spine, V2+ by z, A1 dialogue, A2+ by role).
- `frontend/src/stores/edit-doc-store.ts` — working document store + all
  mutators (`trim`, `nudge`, `move`, `reorderSeg`, `split`, `remove`,
  `addSegment`, `addOp`, `setGain`, `setOpFrom`, `setOpEdge`, `setOpZ`,
  `removeOp`). Persisted truth = `timeline: EditSegment[]` + `operations:
  EditOperation[]`. Currently only **revert-to-baseline**, no undo stack.
- `frontend/src/stores/transport-store.ts` — frame-snapped clock (30fps):
  `progMs`, `playing`, `seek`, `step`, `togglePlaying`, `snapMs`,
  `formatTimecode`. **No zoom / scroll state.**
- `frontend/src/lib/api.ts` — types. `EditOperationType = place_video |
  place_audio | split_edit | level`. `EditSegment{seg_id,file_id,in_ms,out_ms,
  transform?}`. `EditOperation` has `gain_db`, `z`, `opacity`, `transform`,
  `role`. `LayoutRegion` exists for split/PiP.
- `frontend/src/stores/drive-store.ts` — `files` (for real clip names + proxy
  URLs used by thumbnails/waveforms).
- `frontend/src/components/preview/composite-preview.tsx` +
  `use-program-player.ts` — how media/proxy is loaded (reuse for thumbnails +
  waveform peaks).

### Hard constraint — render parity
The backend renderer only understands the four op types above + spine segments.
Any feature that changes the **rendered output** must map to those (or add a new
backend op + renderer support). Each task below is tagged:
- **[FE]** frontend/preview-only — operates on existing spine/ops, zero backend
  risk. Safe to ship independently.
- **[BE]** requires a backend op-schema + renderer change to actually render;
  do **not** ship as "silently preview-only" without flagging.

---

## 2. Architecture decisions (do these as shared foundations first)

These are prerequisites that later tasks build on. Land them before the feature
tasks.

### 2.0 Open, un-boxed shell  [FE]  ← do this first (quick, sets the tone)
The timeline is currently wrapped in a framed card that "boxes it in". Open it up
for a stylish, minimalist feel — the timeline should read as part of the canvas,
not a widget in a bordered panel.

- **Remove the outer box** in `timeline-editor.tsx` root element: drop
  `rounded-2xl border p-3` + the `background`/`borderColor` box styling
  (currently lines ~409–413). Replace with a plain, transparent container —
  vertical rhythm via spacing only (`space-y-3` is fine, no border, no radius,
  no bg).
- **Soften the dock seam** in `ai-edit-panel.tsx` (~line 429): keep at most a
  single hairline `border-t` (`--border`) as the divider between the workspace
  and the timeline dock, or drop it entirely in favor of whitespace. No boxed
  padding card.
- **De-box the lanes**: today each `Lane` is a filled `--sidebar` rounded
  rectangle and its label is an `--accent-soft` chip. For the open look:
  - Lane track background: use a **very subtle** fill (e.g. a faint `--border`
    wash) or none — rely on the clip blocks themselves for structure. No per-lane
    rounded card.
  - Track label: plain `--muted` text (see 2.5 headers), **not** an orange
    `--accent-soft` chip. Orange stays reserved for the primary action + active
    state only.
  - Keep a single hairline separating lanes if needed, or just tight spacing.
- Result: ruler → lanes → playhead float on the page background; the only
  chrome is hairlines + the clip blocks. Verify against the design-system
  checklist (orange ≤ ~2 spots, spacing does the work, no shadows).

Acceptance: the screenshot's outer rounded border is gone; the timeline reads as
an open, edge-to-edge editing surface with no card framing; orange appears only
on the play/save primary + the active tool/tab.

### 2.0b Strip timeline chrome + move Save / History / Revert to the panel  [FE]  ← do with 2.0
The timeline currently carries a header row ("Timeline" label + "N cuts ·
MM:SS"), the Save / History / Revert buttons, a version-history dropdown, and a
top-right transport timecode readout. **Remove all of that from the timeline**
(keep it clean = just transport buttons + ruler + lanes + clip inspector) and
**relocate Save / History / Revert to the top of the right-side AI Edit panel**,
which already owns `threadId`, `token`, `ensureThread`, and the thread state.

> This was implemented once and reverted to keep the tree clean; the exact edits
> below are known-good. Execute as written.

**A) `frontend/src/components/timeline-editor.tsx` — remove save/chrome:**
- Imports: drop `Save, RotateCcw, History, Loader2, AlertCircle` from the
  `lucide-react` import. Reduce the `@/lib/api` import to just
  `import { type EditOperation } from "@/lib/api";` (remove `saveEditDocument`,
  `type EditDocument`, `listEditVersions`, `getEditVersion`,
  `type EditVersionListItem`).
- Props: change the signature to only `{ ensureThread }: { ensureThread: () =>
  Promise<string | null> }`. Remove `threadId`, `token`, `onSaved`.
- Store selectors: remove `baseVersion`, `commit`, `revert` (revertStore),
  `setWorking`, `isDirty`.
- Local state: remove `saving`, `error`, `showHistory`, `versions`, and the
  `dirty` `useMemo`.
- Functions: remove `save`, `revert`, `openHistory`, `loadVersion` (and any
  `Cmd+S` effect if present).
- JSX removals:
  - the entire **Header** block (the `Timeline` label + `{timeline.length}
    cut(s) · {fmt(total)}` + the History / Revert / Save buttons);
  - the **version-history dropdown** (`{showHistory && (...)}`);
  - the **top-right timecode span** in the transport strip
    (`{formatTimecode(progMs)} / {formatTimecode(total)}`) — this is the "extra
    timing on the top" the user called out;
  - the bottom **error display** block (`{error && (...)}`).
- Keep: transport strip (play/pause, prev/next frame), `TimeRuler`, lanes,
  playhead, and the two clip inspectors (spine + op). `ensureThread` is still
  used inside `onLaneDrop` — keep it. `formatTimecode` is still used by
  `TimeRuler`; `fmt` is still used by the inspectors/blocks — keep both.

**B) `frontend/src/components/ai-edit-panel.tsx` — host Save/History/Revert:**
- Imports: add `Save, History, RotateCcw` to the `lucide-react` import; add
  `saveEditDocument, listEditVersions, getEditVersion, type EditVersionListItem`
  to the `@/lib/api` import.
- Add working-doc selectors + state (near the other `useState` hooks):
  `timeline`, `operations`, `baseVersion`, `commit`, `revert`, `setWorking`,
  `isDirty` from `useEditDocStore`; a `dirty = useMemo(() => isDirty(),
  [timeline, operations, baseVersion])`; and `saving`, `showHistory`,
  `versions` state.
- Add functions (reuse the panel's existing `token`, `threadId`, `ensureThread`,
  `setError`, `setThread`):
  - `doSave()` — `saveEditDocument(id, { base_version, timeline, operations })`,
    then `commit(res.version, res.document)` and update the local thread inline
    (`setThread(prev => prev ? {...prev, document: res.document,
    document_version: res.version} : prev)`). Map `stale`/`409` to the
    "reload to get the latest" message.
  - `doRevert()` — `revert()` + clear error.
  - `openHistory()` — toggle `showHistory`; on open, `listEditVersions` →
    `setVersions`.
  - `loadVersion(v)` — `getEditVersion` → `setWorking(doc.timeline,
    doc.operations)`; close history.
  - a `Cmd/Ctrl+S` keydown effect → `doSave()`.
- Header UI: in the right-hand button group of the panel header, **when
  `threadId` exists**, add (before the New/Close buttons): History (ghost icon),
  Revert (ghost icon, disabled unless `dirty`), Save (ghost button, disabled
  unless `dirty`; show the `Save` icon tinted `var(--accent)` ONLY when dirty,
  spinner while `saving`). Keep New (`Plus`) and Close (`X`).
- Add a collapsible **version-history dropdown** just below the header row and
  above `<CompositePreview />` (hairline `border-b`, `--muted` labels, rows =
  `v{n} · {created_by}` + time; "No versions yet." when empty).
- Remove the now-unused `handleSavedEdit` function and change the timeline
  render to `<TimelineEditor ensureThread={ensureThread} />` (drop `threadId`,
  `token`, `onSaved`).

**C) Design/UX (keep it minimal):**
- Do NOT give Save a full orange fill — use a **tinted accent icon when dirty**
  only, so the panel keeps ≤~2 orange elements per the design system.
- Show the Save/Revert/History cluster only for an active edit session
  (`threadId`), so the empty/first-run state stays clean.
- Tooltips: "Save (⌘S)", "Revert changes", "Version history".

Acceptance: timeline shows no "Timeline"/cut-count header, no Save/History/Revert
buttons, and no top timecode readout; Save/Revert/History live at the top of the
right panel and persist/restore the working doc exactly as before; `⌘S` saves;
`frontend` builds with no unused-symbol errors.

### 2.1 Zoom + scroll model  [FE]
Today `pxPerMs = trackW / total` (fit-to-width only). Replace with an explicit
zoom:
- Add timeline-view state (new light store `stores/timeline-view.ts`, or local
  state in `timeline-editor.tsx` if simpler): `pxPerSec` (zoom), `scrollLeftPx`.
- Derived `pxPerMs = pxPerSec / 1000`; content width = `total * pxPerMs`.
- The lanes + ruler become a **horizontally scrollable** viewport sharing one
  content width; the **track-header column stays fixed** (does not scroll H).
- Actions: zoom in / out (buttons + `Cmd/Ctrl + wheel` centered on cursor),
  **zoom-to-fit** (compute `pxPerSec` from viewport width), **zoom-to-selection**.
- **Auto-scroll**: during playback keep the playhead in view; when it exits the
  right edge, page the scroll.
- Persist last zoom per thread (optional, `localStorage`).

Acceptance: can zoom from whole-project to ~frame level; ruler ticks re-space
via the existing `candidates[]` logic; playhead stays aligned; header column
fixed while lanes scroll.

### 2.2 Undo / redo stack  [FE]  ← highest-value foundational
`edit-doc-store` has no history. Add one:
- Keep `past: Snapshot[]` and `future: Snapshot[]` where `Snapshot =
  { timeline, operations, selected }` (deep-ish copies; the store already clones
  on set).
- Wrap every mutating action to push the *previous* state onto `past` and clear
  `future`. Cap history (e.g. 100). Coalesce rapid same-kind drags (trim/move):
  push once on pointer-DOWN, not per pointer-move.
- Add `undo()`, `redo()`, `canUndo`, `canRedo`. Keep `revert()` (to baseline)
  separate.
- Keyboard: `Cmd/Ctrl+Z` = undo, `Cmd/Ctrl+Shift+Z` (and `Cmd/Ctrl+Y`) = redo.
  Guard against firing while typing in inputs (same guard as existing keydown).

Acceptance: every timeline edit is individually undoable/redoable; a drag is one
undo step, not 50.

### 2.3 Tool modes  [FE]
Local editor state `tool: "select" | "blade" | "slip" | "slide"` (start with
`select` + `blade`; slip/slide in P1). Toolbar buttons + shortcuts (`V` select,
`B` blade). Cursor changes per tool. Blade tool: clicking a clip splits it at
the click x (mapped to ms), not at its middle.

### 2.4 Snapping engine  [FE]
- `snapEnabled` toggle (default on) + shortcut `S`.
- `snapTargets(excludeClipId?)`: collect candidate ms — every clip edge on every
  track, the playhead (`progMs`), markers, and `0`/`total`.
- Helper `snapValue(ms, pxPerMs, thresholdPx=8)` → nearest candidate within
  threshold else `ms`. Use it inside `startTrim`, `startMove`, `startReorder`,
  and drop position math (currently they only call `snapMs` = frame grid).
- Render a thin vertical **snap guide** at the active snap target during a drag.

Acceptance: dragging a clip edge visibly snaps to neighbor edges/playhead with a
guide line; toggling `S` disables it.

### 2.5 Toolbar + track-header scaffolding  [FE]
- **Toolbar row** at top of the dock (above the ruler): tool buttons, snap
  toggle, zoom in/out/fit, add-track, add-marker, set in/out. Ghost styling; one
  row; icons `size={14}`.
- **Track-header column** (~120px, left of lanes, replaces the tiny `V1/A1`
  label chip in `Lane`): per-track name + mute / solo / lock toggles + a
  drag-to-resize height handle. Header column is fixed (see 2.1). Track UI meta
  (solo/lock/height, and mute for video/base) lives in a **UI store**, keyed by
  track id — NOT persisted to the document (except audio mute which already maps
  to `gain_db`). Persist track UI meta per-thread in `localStorage` (nice-to-have).

---

## 3. P0 — Foundational editing (ship first)

> 2.1–2.5 are prerequisites; these are the P0 *features* on top.

1. **Blade / razor at playhead**  [FE]
   - `splitAtPlayhead()`: split the spine segment under `progMs` at `progMs`
     (extend store `split` to accept an absolute ms, not just midpoint). "Split
     all tracks": also split any `place_video`/`place_audio` op straddling
     `progMs` into two ops.
   - Blade-tool click already splits at the click point (2.3).
   - Shortcut: `Cmd/Ctrl+K` (split at playhead).

2. **Ripple vs. lift delete**  [FE]
   - Spine is gapless, so removing a spine seg already ripples — keep as ripple
     delete (`Shift+Delete`).
   - "Lift" (leave a gap) for spine: not natively representable in a gapless
     spine — implement by replacing the seg with a same-length **black/gap seg**
     ONLY if a gap primitive exists; otherwise scope lift to **ops** (just remove
     the op, which already leaves its slot empty). Document the limitation.
   - For ops: ripple delete = remove op + shift later ops on the *same track*
     left by the removed duration; lift = plain `removeOp`.
   - Keys: `Delete` = lift (ops) / ripple (spine); `Shift+Delete` = ripple.

3. **Copy / cut / paste / duplicate**  [FE]
   - Editor-local clipboard of selected clip descriptors (source file + src
     in/out + kind + track hint).
   - Paste at playhead onto the compatible track (video→video op, audio→audio
     op; a spine clip pastes as a new spine seg at the nearest index or as a V2
     op — pick op to avoid disturbing spine timing; document choice).
   - `Cmd/Ctrl+C/X/V`; `Cmd/Ctrl+D` = duplicate in place (+small offset).

4. **Multi-select**  [FE]
   - Selection becomes a `Set<clipId>` (extend beyond the single `selected`
     seg / `selectedOp`). Shift-click adds/removes; marquee drag on empty lane
     space selects intersecting clips.
   - Move/delete/nudge/copy apply to all selected. Inspector footer shows a
     multi-select summary.

5. **Frame-nudge selected clip(s)**  [FE]
   - `,` / `.` nudge selected clip position by ±1 frame; `Shift` = ±10. For
     spine clips this is reorder-only (position is implicit); for ops, adjust
     `from_ms` via existing `setOpFrom`.

6. **Precise in/out/duration on the inline footer inspector**  [FE]
   - The footer inspector already lives in the timeline component — upgrade its
     coarse ±250ms steppers to **typeable timecode fields** (in / out / duration)
     using `formatTimecode` + a parser. Commit on blur/Enter, frame-snapped.
   - (This is footer-of-timeline, not the right-rail Inspector — in scope.)

Acceptance for P0: an editor can zoom, snap, blade at the playhead, ripple/lift
delete, copy-paste-duplicate, multi-select, nudge by frame, type exact in/out,
and undo/redo every step.

---

## 4. P1 — Pro editing verbs + clip visuals

1. **Advanced trim: slip / slide / roll / ripple-trim**  [FE]
   - **Slip**: drag inside a clip changes `src_in/out` together (content shifts,
     position/duration fixed). Ops: adjust `src_in_ms`+`src_out_ms` by the same
     delta. Spine: adjust `in_ms`+`out_ms` together.
   - **Slide**: drag a clip moves it while auto-trimming the two neighbors to
     keep the timeline gapless (spine) / keep durations (ops).
   - **Roll**: drag the shared cut point between two adjacent spine clips (out of
     left = in of right), no total-duration change.
   - **Ripple-trim**: trimming an edge shifts everything downstream (spine is
     naturally ripple; for ops, shift later same-track ops).
   - Bind to slip/slide tool modes (2.3) + roll = drag the seam handle.

2. **Insert vs. overwrite edit; timeline in/out**  [FE]
   - Timeline **In/Out marks** (`I` / `O`) define a program range (store in
     view/UI state). Show as a shaded ruler band.
   - **Overwrite** (`B`-drop / key): dropped/pasted clip replaces content in the
     range. **Insert**: pushes downstream content right (ripple). Applies to ops;
     for spine, insert = splice a seg at the index, overwrite = trim/replace
     overlapping spine segs. Document spine behavior precisely.

3. **Clip visuals**  [FE]
   - **Video thumbnails / filmstrip** on video blocks: sample proxy frames at
     the clip's src range; lay N thumbnails across the block width (density from
     zoom). Reuse proxy URL path from `composite-preview`/`drive-store`. Cache by
     `{fileId, ms}`; lazy-load only visible clips.
   - **Audio waveforms** on audio blocks: decode peaks via Web Audio
     (`OfflineAudioContext`) or a precomputed peaks endpoint if available; draw
     to a `<canvas>`. Cache peaks per file. Flag CPU cost; downsample.
   - **Real clip names**: replace `file_id.slice(0,4)` with the file name from
     `drive-store` (fallback to id).
   - **Duration label** on each block (right-aligned, `--muted`, hidden when the
     block is too narrow).

4. **Track headers: mute / solo / lock / height / add-remove-reorder**  [FE
   for UI; audio-mute is [FE] via gain]
   - Mute: audio → set `gain_db=-120` (already the mute path); video track mute =
     hide layer in preview (view-only).
   - Solo: view/playback flag (UI store); soloing a track mutes others in the
     **preview mixer only** — coordinate with `use-program-player`.
   - Lock: UI-only; locked track ignores pointer edits + drops.
   - Height: drag handle on the header; store per-track height in UI store.
   - Add track: create an extra empty drop-target lane (video/audio); since
     tracks are derived, an "empty" track is a UI placeholder until a clip lands.
   - Reorder video layers: dragging the header restacks `z` (maps to `setOpZ`).

5. **Link / unlink A/V**  [FE]
   - The spine's dialogue is currently hard-coupled to its video seg. Add a
     per-seg **link toggle** (UI meta): when unlinked, selecting/trimming video
     doesn't move the coupled dialogue. (Persisting an unlink needs a doc field —
     if none, keep unlink session-only and document it.)

Acceptance for P1: clips show thumbnails + waveforms + names + durations; track
headers give mute/solo/lock/height; slip/slide/roll/ripple-trim + insert/
overwrite behave like a standard NLE.

---

## 5. P2 — Markers, ranges, shuttle, transitions/fades

1. **Markers + marker list**  [FE for display; persistence needs a doc field]
   - Add/remove markers at the playhead (`M`), colored pips on the ruler,
     click-to-seek. A collapsible marker list (name + TC).
   - Persistence: add `markers?: {ms:number,label?:string,color?:string}[]` to
     the document (small, additive) — or keep session-only first and flag.

2. **Loop the in/out range + J/K/L shuttle**  [FE]
   - Loop-play the timeline In/Out range (from P1) — a transport-store loop flag
     read by `use-program-player`.
   - `J`/`K`/`L` shuttle (reverse / pause / forward, tap `L` to speed up) — extend
     the transport engine's clock rate. `K+J`/`K+L` = slow shuttle.

3. **Export range**  [BE-adjacent]
   - Use the timeline In/Out to set the render range. Needs the render request to
     accept `range_ms` (check `createRender`/backend). If unsupported, defer and
     flag; do not silently ignore.

4. **Fades (audio + video) via clip dogears**  [BE]
   - UI: fade handles ("dogears") on clip corners producing fade-in/out
     durations. **Rendering fades requires backend support** (no fade field on
     ops today). Either add `fade_in_ms`/`fade_out_ms` to `place_video`/
     `place_audio` + renderer, or defer. Preview can approximate (opacity/gain
     ramp) but MUST be flagged as not-yet-rendered until backend lands.

5. **Transitions (crossfade / dip-to-color)**  [BE]
   - Dragging a transition onto a seam. **Needs a backend transition op +
     renderer.** Scope the UI (seam affordance + duration handle) but gate actual
     rendering on backend; otherwise defer entirely. Do not fake it into export.

6. **Keyboard-shortcut cheatsheet**  [FE]
   - A `?`-triggered overlay listing all timeline shortcuts. Keeps discovery
     professional.

> Explicitly deferred to other plans (do NOT build here): color grade, captions,
> per-clip effects/filters, keyframing UI, transform/PiP numeric controls,
> speed/retime controls, pan, gain-automation rubber-band, VU meters,
> safe-area guides, fullscreen monitor. (Inspector / right-rail + backend.)

---

## 6. Backend touchpoints summary (flag for a separate BE task)

Ship the **[FE]** items independently; these need backend before they render:
- **Fades**: `fade_in_ms` / `fade_out_ms` on `place_video` / `place_audio` +
  renderer ramps.
- **Transitions**: new transition op (seam-anchored) + renderer.
- **Gain automation / pan**: only if pulled in from the Inspector plan.
- **Export range**: `range_ms` on the render request.
- **Persisted markers / unlink**: additive document fields
  (`markers[]`, per-seg `linked` flag).

Anything not in this list is doable purely on the frontend against the existing
spine+ops model with no render regressions.

---

## 7. Suggested execution order

1. **Foundations** (2.0 open shell → 2.0b strip chrome + relocate Save/History/
   Revert → 2.1 zoom/scroll → 2.2 undo/redo → 2.4 snapping → 2.3 tools →
   2.5 toolbar+headers).
2. **P0 features** (blade, ripple/lift, copy-paste, multi-select, nudge, precise
   TC fields).
3. **P1** (slip/slide/roll, insert/overwrite, thumbnails+waveforms+names+
   durations, track mute/solo/lock/height, link/unlink).
4. **P2** (markers, loop+shuttle, cheatsheet; then the [BE]-gated fades/
   transitions/export-range once backend is ready).

Each numbered item is independently shippable and should keep the app building
(`npm run build` in `frontend/`) and the design-system checklist green.

## 8. QA / acceptance checklist (per PR)

- [ ] `frontend` builds; no TypeScript errors.
- [ ] No hardcoded hex — colors via `var(--token)`; orange ≤ ~2 spots/view.
- [ ] Undo/redo covers the new action; a drag is one history step.
- [ ] Zoom/scroll: header column fixed, lanes+ruler share content width,
      playhead aligned at all zoom levels.
- [ ] Snapping guide appears and `S` toggles it.
- [ ] No render regression: saving still produces the same spine+ops the backend
      renderer already understands (diff the document for FE-only tasks).
- [ ] Keyboard shortcuts don't fire while typing in inputs/textareas.

## 9. Risks / notes

- **Waveform decode cost** — decode lazily + cache peaks; consider a backend
  peaks endpoint if it janks.
- **Gapless spine** — "lift" (leave a gap) and true insert/overwrite are subtle
  on a gapless spine; document exact chosen semantics so the executor doesn't
  improvise divergent behavior.
- **Solo/mute in preview** must be coordinated with `use-program-player`'s mixer,
  not just the timeline view.
- **Derived tracks** — tracks come from ops/z/role; "add empty track" is a UI
  placeholder until a clip lands there. Keep the derivation in `edit-project.ts`
  authoritative.
