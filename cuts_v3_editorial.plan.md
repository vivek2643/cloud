# Cuts v3 — Editorial pass (motion-as-label, action promotion, conservative removal)

Follow-on to `cuts_v3.plan.md`. Addresses six issues found reviewing the first
real cuts on the frontend. Grounded in actual Reel-trail data (see "Findings").

## Findings (why, not guesses)

Pulled the real Reel-trail cut records + `action_points`:

- **Full coverage is holding.** Every ms of every clip is in some cut; nothing
  is literally dropped. My earlier "coverage refactor" is therefore NOT needed
  and is dropped from scope.
- **Video cuts are almost all dead-air wind-down.** Real labels: "Settle after
  speech line", "Trailing frame after fiber line", "Final smile at clip end".
- **Real action is buried inside speech cuts.** `action_points` cluster at
  timestamps that fall *inside* a speech cut's span (person gestures / acts
  while talking) — so it never surfaces as its own visual moment. This is why
  "the action at the end never comes up": it's covered, but invisible.
- **Camera move/settle boundaries fragment video** into ~1.1s slivers (Reel:
  20 video cuts, avg 1.1s), most of them trailing "settle" fragments.
- **High-confidence junk is already detected** ("And go (cue)", "Pre-roll
  setup") — it's just still shown as a strip rather than hidden.

## North star for this pass

Camera motion is a **LABEL and a selection handle, not a boundary that shreds
the timeline**. A pan is ONE coherent selectable cut; an action is ONE typed
cut; the trailing micro-settle after speech is not its own cut at all. The LLM
keeps meaning; the lattice keeps boundaries; motion classifies spans within it.

Removal stays conservative: hide ONLY high-confidence junk (cue words, clear
pre-roll). Anything doubtful stays visible. Never remove real (even if weak)
content — keep the bar high.

## Changes

### A. Playback fix (DONE)
`cuts-v3-view.tsx`: play from the in-point reliably by gating `play()` on the
`seeked` event and removing the `#t=` media fragment. No re-ingest needed.

### B. Camera motion: stop fragmenting, start labeling
`base_cuts._camera_marks` / `lattice.build_atoms`:

- **Remove HOLD↔MOVE (`camera_move` / `settle`) from the HARD atom-boundary
  set.** Hard boundaries stay: shot cut, disturbance, transition (wipe/
  degenerate), speech edge, clip edge. A pan no longer becomes `[hold][move]
  [settle]`.
- **Classify each video atom by dominant camera behavior**: `hold` | `pan` |
  `push` | `handheld` (already have `camera_desc`; make it the atom's type).
- A continuous pan is therefore ONE atom → one candidate cut, labeled `pan`.
- **Snap a video cut's in-point** to where the motion/atom actually begins
  (the "arrival" into the shot), never mid-jitter — so cuts don't start
  looking broken (issue 3).

### C. Action promotion (issue 4)
- An **action span** — a run of clustered `action_points` / high
  `action_energy` inside the NON-speech remainder — becomes its own typed atom
  `action`, and is a first-class candidate cut.
- Action frames are **always sent to pass 2** (never budget-trimmed) and the
  cut is labeled as an action, never as "settle".
- Action that overlaps speech stays inside the speech cut (never cut under
  speech) — but its existence is noted so the editor can find it.

### D. Conservative removal (issues 2 + 6)
- Add a **junk confidence** to pass-2 output: `high` (cue words, pre-roll,
  dead air) vs `low`/doubtful.
- Frontend: **hide `high`-confidence junk by default**; keep a "show
  discarded" toggle. Doubtful junk stays visible inline.
- **Speech cue trimming**: pass 1 may start/end a speech cut a few words in to
  exclude leading camera cues ("okay", "and go", "take three", "3-2-1") — the
  trimmed words become a high-confidence junk span (hidden), the good cut
  starts clean. Coverage still holds (the trim is a junk cut, not a gap).

## Explicitly OUT of scope (walked back)
- Ripping out the full-coverage invariant / "drop spans". Not needed —
  coverage holds and removal is handled by hide-not-delete.

## Build order
1. B (biggest visible win — transforms the Reel video cuts). Re-ingest.
2. C (finish the action surfacing). Re-ingest.
3. D (cue trim + hide high-confidence junk). Re-ingest.

Each step is verifiable on the frontend before the next. Re-ingest cost is the
main reason to sequence rather than do all at once.
