# Plan: Music freedom + timeline awareness + two-pass audit

## Goal
Give the brain real, *free* control over music, make the timeline fully
knowable, and restructure the finish-time audit so it judges the edit like a
human editor (fit-to-intent, then fit-to-craft) before falling back to specific
machine flags. Everything stays **generic** — no per-format cookbook, no
enumerated failure lists baked into prompts.

## Guiding split (do not violate)
- **Guidance doc** owns *judgment/taste* — free, generic prose the brain reads.
- **Tool handles** own *ms-precision* — opt-in; the brain still decides which
  beat/moment/clip, the handle just nails the frames. Never a deterministic
  pipeline that removes the brain's choices.
- **Read surfaces** own *awareness* — the timeline must be crystal-clear and
  complete; the audit is only as good as what the map exposes.

---

## Phase 1 — Guidance doc §5 "Working with music"  (DONE, verify only)
**File:** `backend/app/services/l3/guidance_doc.md`
Already drafted (§5). Covers: choose/combine multiple options & ignore junk;
place at multiple points / loop; shift so a strong beat lands on the key moment;
cut density (beat/bar/phrase) is a choice; keep loudness steady across the whole
piece + duck under speech; fade/crossfade edges; silence as a tool.
**Action:** read it, tighten wording only. No code. This is the freedom layer;
the phases below make its promises actually executable.

---

## Phase 2 — Make the timeline crystal-clear (awareness prerequisite)
The audit (Phase 5) and the music policies (§5) both depend on the brain being
able to *see* everything. Close the audio-side blind spots in the map/read.

**Files:** `backend/app/services/l3/arrange.py` (`render_program_map`, ~L244),
`backend/app/services/l3/observe.py` (`read_state` ~L358, `_beat_grid` ~L615).

Add to the AUDIO table / `read_state`:
1. **Per-layer loudness** — surface each audio layer's level (we already measure
   `integrated_lufs` per source at ingest; `read_state` already collects it at
   L385). Show it per audio row so the brain can balance for constant loudness
   (§5 "keep it level") instead of balancing blind.
2. **Candidate beds vs. junk** — a clear list of available audio sources marked
   usable/musical (`is_musical`, `bpm`) vs. likely-junk, so "pick the relevant
   ones, ignore junk" (§5) is groundable.
3. **Audio gaps / dead spots** — expose stretches of the program with *no* audio
   layer covering them (silence holes), so the "sound randomly missing in the
   middle" case the audit must catch is visible as data, not guessed.

**Acceptance:** from the Program Map + `read_state` alone, the brain can answer
"how loud is each layer, which sources are usable music, and where is there no
audio at all" without any prose hints. Keep it generic (no role-of-person cols).

---

## Phase 3 — Two music handles (opt-in ms-precision)
**Files:** `backend/app/services/l3/tools.py` (trim handler ~L391, place_audio
handler ~L404, `_beat_grid_ms` ~L320), `backend/app/services/l3/act.py`
(`trim` ~L515, `place_audio`), `backend/app/services/l3/observe.py`
(`snap_to_beats` ~L308 — reuse as-is).

1. **Beat-snap on `trim`.** Add `snap:"beat"` support to the `trim` tool, mirror
   the existing pattern on `place_audio`/`move` (build grid via `_beat_grid_ms`,
   call `observe.snap_to_beats(grid, edge, max_move_ms=_SNAP_CAP_MS)`, apply the
   snapped edge). Snap whichever edge(s) the brain is moving. Update the `trim`
   tool schema note.
2. **Shift-to-align.** A move that slides a placed bed so a chosen onset sits at
   a chosen program moment: compute the offset from `(target_prog_ms − onset_
   prog_ms)`, apply it to the op's `from_ms`/`to_ms` (and/or `src_in_ms`), then
   beat-snap. Prefer a small param/action over a new deterministic pipeline —
   the brain names the beat and the moment, the handle does the arithmetic.
   Reuse `_beat_grid`'s program-time onsets so "the drop" is addressable.

**Acceptance:** brain can land a trim edge exactly on a beat, and slide a bed so
its chosen beat hits a chosen moment, both without hand-computing ms. Handles are
no-ops when no music is present (grid empty → `snapped:False`).

---

## Phase 4 — Musical-structure awareness (unlocks "land the drop on the climax")
Today the brain gets raw `onsets_ms` + `bpm` but **not** where the drop/hook or
phrase boundaries are — so "shift so the hook lands on the climax" (§5) can't be
targeted. This is the highest-value awareness gap.

**Files (ingest):** wherever `audio_features` are computed (search
`integrated_lufs`/`onsets_ms` producers; `observe.py` L155 reads
`af.is_musical,bpm,onsets_ms,integrated_lufs` from the DB). **Surface:**
`observe._beat_grid` (~L615) so structure rides alongside onsets.

Pragmatic, generic detection (approximate is fine — it's a *prior*, not truth):
- **Section boundaries / phrases** from the loudness/energy envelope: segment on
  sustained energy changes (a novelty curve over RMS), snapped to the nearest
  bar. No genre assumptions.
- **The "drop"/strongest moment** = the onset with the largest local energy jump
  (contrast, like the video salience model), reported as a program-time point.

Add these as optional fields on each beat-grid entry (e.g. `sections:[...]`,
`drop_ms`). Keep behind the existing "only when musical source present" gate.

**Acceptance:** for a musical bed, the beat grid reports at least a strongest
moment and coarse section boundaries in program time; Phase-3 shift can target
"the drop." If detection is unreliable for a source, omit the fields (never
fabricate) — the brain falls back to raw onsets.

**Note:** if this phase is deferred, Phases 1–3 still stand; §5's "shift so a
strong beat lands" degrades gracefully to "shift onto any onset."

---

## Phase 5 — Two-pass audit + flags (the core redesign)
Replace the implicit checklist with two open judgments, then the existing flags.
**No enumerated failure modes in the prompt** — the brain judges against its own
prior of "a finished, high-quality video."

**Files:** `backend/app/services/l3/tools.py` (`_verify_before_finish` ~L529,
the finish-gate), `backend/app/services/l3/converse.py` (`_LOOP_SYSTEM` audit
wording), `backend/app/services/l3/observe.py` (`review` ~L1024 — flags home,
already has `category` "ask"/"guidance").

Structure the audit as three stages, in order:

| Stage | Question | Nature | Where |
|---|---|---|---|
| 1. Fit to intent | "Does this edit do what the user asked?" | judged vs the prompt | planning check + `review` "ask" flags (`_requested_feature_flags`) |
| 2. Fit to craft | "Forget the prompt entirely — does this look and sound like a real, high-quality finished video?" | judged blind, vs the brain's own prior | finish-gate prompt (new) |
| 3. Flags | off-cam, beat-drift, overrun, mid-sentence, audio-gap, loudness imbalance | deterministic sharpeners | `review` flags |

**The Pass-2 discipline (the key rule):** a discrepancy found in Pass 2 may pass
**only** if the brain explicitly reasons one of exactly two things:
(a) the user's ask required it, or (b) the material can't support better.
Any other "looks fine anyway" is not allowed — if it proceeds with a known flaw
it must *name* which of the two reasons applies, in one line. This both stops
rubber-stamping (can't wave it through silently) and stops over-editing (can't
"fix" what the material can't support — must acknowledge the ceiling).

**Wiring:**
- In `_verify_before_finish`, add the Pass-2 gate: when the edit changed, prompt
  the brain for the blind craft verdict + (if passing-with-flaw) the one-line
  reason, *before* letting the turn finish. Keep the loop terminating: Pass 2 is
  fix-or-justify (like the length check at L556), not an infinite hard block.
  Pass 1 stays the ask/contract check already there (`review(user_ask=...)`,
  `_requested_feature_flags`). Stage 3 flags stay advisory (L562-579).
- In `_LOOP_SYSTEM`, state the two questions and the two-escape-hatch rule in
  generic language. Do **not** list what to look for in Pass 2.
- Record the Pass-2 verdict + reason in the trace so it's inspectable.

**New Stage-3 flags to add in `review`** (deterministic, from Phase-2 data):
- audio gap / silence hole in the middle of the program,
- loudness imbalance across layers/sections.
Tag them `category:"craft"` (or reuse "guidance") — keep them advisory.

**Acceptance:** finishing a changed edit forces a blind craft verdict; a
sub-finished timeline (e.g. audio missing mid-program) can't be finished without
either fixing it or the brain naming reason (a) or (b). Order is always
intent → craft → flags. Loop still always terminates.

---

## Phase 6 — Parked (render-blocked)
- **Speed video to tempo** (ramps to the beat) and **cut music to phrase** that
  depends on video speed: blocked on video speed not being baked into export
  (`render/compositor.py` = hard cuts only). Leave as a to-do under the export
  work; do not attempt here.

---

## Order & switchability
Suggested build order: **Phase 2 (awareness) → Phase 3 (handles) → Phase 5
(audit) → Phase 4 (structure)**. Phase 1 is done. Each phase is independent
enough to land and verify on its own; Phase 4 is optional and degrades
gracefully.

- No feature-flag needed for guidance/prompt changes; they're additive.
- Phase 3 handles are inert without music (empty grid), so safe to ship.
- Keep Phase 5's Pass-2 gate fix-or-justify (never a hard infinite block) so the
  loop can't hang.

## Tests to add/update
- `backend/scripts/test_observe_act.py`: trim beat-snap; shift-to-align; audio
  loudness/junk/gap surfacing in `read_state`/map; new `review` gap/loudness
  flags.
- `backend/scripts/test_footage_map.py` / arrange tests: Program Map audio rows
  show loudness + gaps.
- Audit: a fixture timeline with a mid-program silence hole must be blocked at
  the finish-gate unless justified.

## Finish
Run the affected test suites, then **commit and push**.
