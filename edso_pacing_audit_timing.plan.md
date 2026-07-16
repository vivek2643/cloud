# EDSO: timeline presentation, brain knowledge, fit-to-window, audit, pacing & speed-ramps

Single consolidated plan (supersedes the earlier draft of this file). Builds on
already-shipped work:
- **LIVE — do not re-implement:** the done-gate `_verify_before_finish`
  (`tools.py:466`, wired at `tools.py:530`), the target-length anchor
  `_extract_target_s` → `brief.target_duration_s` (`converse.py:286/342`), and
  the `beat_transcript` surfacing (verbatim `said_text` + `aud:` quality on the
  beat line). Extend these; don't rebuild them.

Phased by leverage and cost. Group 1 is runtime-only (no re-ingest); Group 2
(speed-ramps) is a gated bundle.

---

## Hard constraints (from the user — do not violate)
- **Everything is a GENERIC model. Never write anything domain-specific.** No
  podcast/talking-head assumptions anywhere — not in prompts, presentation,
  labels, or examples. Speaker/role is optional metadata, never structure. The
  same code must serve b-roll, montage, screen-recording, music-video, etc.
- **Present and FLAG, never DIRECT.** Surface the assembled program + any
  likely-rough flags as facts; never prescribe the specific fix. The brain decides.
- **Speech never speeds.** Speed suppression is at the SOURCE (the taste fence
  stays `1.0/1.0` for speech), not by hiding a UI tag.
- **Deterministic, signal-driven.** No LLM numbers for geometry/pacing (matches
  `post.py` / `cutrecord_map.py` conventions).
- **Keep it simple.** No speculative machinery, no new primitives where existing
  data/tools already answer the need.

---

# GROUP 1 — brain knowledge + presentation + audit (no re-ingest; do first)

## 1. Compositing rules → the brain's mental model (prompt)
**Why:** the brain is currently never told the timeline structure (multiple
tracks, top layer shows, audio mixes/ducks). It infers it. State it once, in
plain language.

- **Where:** `_LOOP_SYSTEM` (`converse.py`), a short block. Optionally mirror a
  one-liner in `guidance_doc.md`.
- **Text (plain, generic — use as-is):**
  > Think of the edit as stacked tracks, like layers in a photo or video editor.
  > The bottom track is the main line — it plays start to finish and sets the
  > total running time. A clip on any track above covers what's below it for as
  > long as it's there: if it fills the frame it hides the track underneath; if
  > it's an inset or side-by-side, you see both. All sound tracks play at once and
  > mix together; when two overlap, the more important one stays up front and the
  > other automatically dips beneath it. Everything lines up on one clock —
  > positions are in program time (where it lands in the finished video).
- No jargon (no "z-order", "opacity", "spine"), no roles, no assumption about what
  the footage is.

## 2. Program Map — present the assembled edit as a table (replace flat render)
**Why:** the complete layered truth is ALREADY computed by `layers.resolve`
(z-ordered `VideoLayer`s + role `AudioLayer`s, each with program windows +
`video_stack_at`/`audio_at`/`prog_to_source`). Today `arrange.render_timeline`
lists V1 then V2 as unaligned lines, so the brain can't see what's stacked over
what. Render the resolved layers as a time-aligned, referable table.

- **Where:** replace `arrange.render_timeline` (`arrange.py:200`) with a
  Program-Map renderer built from `layers.resolve(document)`; it's injected each
  turn (`converse.py:256`). Resolution is pure/cheap, so keep it **always-on**.
- **Shape — two small tables (generic columns only):**
  ```text
  PROGRAM MAP   0:00–0:38   landscape
  VIDEO
    lane  id   z    prog(ms)      dur    layout      source         label
    V1    c1   0    0–6000        6000   full_frame  a1b2:m03@bal   "establishing shot"
    V1    c2   0    6000–15000    9000   full_frame  a1b2:m07@bal   "subject in motion"
    V2    o3   10   4000–9000     5000   pip          raw 9f0c…     "inset detail"
  AUDIO
    lane  id   role     prog(ms)     source        gain   duck   fade
    A1    —    source   0–15000      (main line)    0      —      —
    A2    o7   music    0–38000      bed asset     -12     -6     in 500
  ```
  - Every row carries a **stable id** (`seg_id`/`op_id`/`layer_id`) the brain can
    act on, plus program window, `layout`, source ref, and a **neutral** label.
  - Overlap is visible from the shared clock + `z` (e.g. `V2 o3` over `V1 c1→c2`
    at 4.0–9.0s; music bed under everything). No prose needed.
  - Split video/audio into two blocks (keeps columns meaningful, no null soup).
  - Generic for any footage type; no speaker/role column in the structure.
- **`read_state` (`observe.py:324`) enriched:** report the z-stack (coverage
  layers + layout), not just the spine, so the on-demand look matches the map.
- **Naming:** call this the **Program Map** (the edit ASSEMBLED) to distinguish
  it from the existing **Footage Map** (sources AVAILABLE).
- (Optional, not required) an at-a-glance ASCII lane strip may ride alongside, but
  the table stays the source of truth.

## 3. Word→program time at sub-cut resolution (on demand)
**Why:** cut-level program windows already exist (`read_state.prog_start_ms/
prog_end_ms`); the only gap is knowing where a specific spoken line inside a cut
lands after internal dead-air excision / pace — needed to land an overlay on a
line. This is finishing the resolution down to the word, NOT a new worldview.

- **Where:** an on-demand detail in `read_state` (per requested cut), reusing the
  caption-timing word→program mapping (`captions/timing.py _to_program`) over the
  cut's resolved keep-spans. Also surfaced inside the audit's `played_text`
  (item 6) so the read-back already carries word offsets.
- **Do NOT** add a standalone `locate` sense — fold it into `read_state`/audit.
  (Reverse program→source is already available via `layers.prog_to_source` if a
  future need arises; don't expose a new tool now.)
- Speed-aware for free once Group 2 / B3 lands; before that it's correct for
  order + excision + trim.

## 4. Fit-to-window recipe (guidance doc, generic)
**Why:** overlay-on-a-line and cut-to-the-beat are the SAME generic operation —
fit a clip to a target program window `[A,B]`. Document the recipe; don't build a
"fit" primitive or a calculator.

- **Where:** `guidance_doc.md`, conceptual only.
- **Text (generic, no mechanisms, no math):**
  > To fit a clip to a target program window `[A,B]`: get `[A,B]` from whatever
  > sense exposes it, then adjust the clip's length with trim or pace.
- Exactness comes from the audit loop (item 6) + reading actual lengths back from
  the Program Map / `read_state`, not from the model doing blind arithmetic.

## 5. Length math → tool schemas (not the guidance doc)
**Why:** each tool's effect on length is a property of that tool. Co-locate it.

- **`trim` schema (`tools.py`):** note that trimming shortens the cut by the
  removed source, and that the exact resulting length is read back from
  `read_state`/Program Map (internal keep-spans mean it isn't purely linear).
- **`retime`/pace schema (`tools.py`):** note `pace p → length ≈ length/p`, and
  that speech-pace also shaves dead-air, so the exact result is read back from
  state.
- Already true, keep as-is: pace applies ONLY when the brain calls `retime`; the
  `pace.levels` array is just the menu. Ensure the doc reflects that control stays
  with the brain.

## 6. Program audit (`review`) + flags → extend the LIVE done-gate
**Why:** the brain never re-reads the ASSEMBLED program (only source-order beats),
so rough heads/tails (foreign-speaker/filler lead-ins) slip through. The done-gate
runs `validate`+`diagnose` today but shows no transcript.

- **New sense `review` (`observe.py`):** for each timeline seg in order, return
  `{idx, id, ref, played_ms, played_text}` where `played_text` is the verbatim
  words over the seg's ACTUAL played keep-spans (reuse
  `footage_map._said_text_for_span` over resolved spans), plus
  `{total_ms, target_ms, cut_count}`. Include word→program offsets (item 3).
- **Flags (presented, not directive):** flag a cut whose first/last played
  sentence is spoken by a different speaker than the cut's dominant speaker, or is
  pure filler/backchannel/dead-air lead-in/tail; and flag an overlay whose program
  window clearly overruns/underfills the line it sits over. Emit as
  `diagnose`-style findings (`{severity, anchor, message}`). **Never** append a
  prescribed fix ("trim to +Xms").
- **Wire into the LIVE `_verify_before_finish` / `run_edit_loop`:** on the finish
  branch, when `changed` and the brain hasn't already called `review`/`diagnose`
  this turn, inject the program read-back + flags once (reuse the existing
  `reviewed` guard so it can't loop), then let it finish or refine.
- **Prompt (`_LOOP_SYSTEM`):** one factual line — "Before finishing you'll see the
  assembled program read back with any flagged rough heads/tails; act on what
  serves the ask or finish." No stronger steer.

## 7. More aggressive `sharp` (dead-air) — runtime, no re-ingest
**Why:** `sharp` should be the max-tight option; broad/balanced stay looser.
- Raise the speech trim ceiling so `sharp` shaves ~all detected removable dead-air:
  backend `cutrecord_map._SPEECH_TRIM_MAX` and the mirrored frontend
  `SPEECH_TRIM_MAX` (`cuts-v3-view.tsx`) — keep them **equal**, move ~0.85 → ~1.0.
- Applied at ladder-synth + frontend view-math → runtime, no re-ingest.
- **Note:** dense beats with no gaps won't tighten (that's word content, not
  dead-air) — not in scope.

## 8. Confirm speech speed-suppression (source-level)
- Verify speech yields `levels=[1.0]*5` in `compute_pace_envelope` (`post.py`,
  `kind=="speech"` branch) and that `footage_map._pace_tag` shows `trim<=Xs` for
  speech (never `pace:LO-HIx`). No hiding hacks — the source fence is the mechanism.

## Group 1 tests
- `test_observe_act.py`: Program Map renders both tables from `layers.resolve`;
  a coverage layer appears with correct `z`/`layout`/program window over the
  spine; `read_state` reports the stack; word→program offset is correct after an
  upstream trim.
- `test_tools_loop.py`: `review` read-back reflects excised spans; a
  foreign-speaker head is flagged; an overrunning overlay is flagged; no flag
  prescribes a fix; loop still terminates.
- Sharp: `_SPEECH_TRIM_MAX` bump removes ~all removable dead-air; backend/frontend
  constants stay equal (guard test if one exists).

---

# GROUP 2 — video speed-ramps (all-or-nothing bundle; persist `intrinsic_velocity`)
**Order rule:** B1+B2 are net-negative WITHOUT B3 (they advertise a lever that
doesn't render). Ship B3 (or at least its geometry) with them, or not at all.
Highest cost; mainly pays off on b-roll/montage, low value for talking-head.

## B1. Content-aware taste fences (open the throttle)
**Where:** `pass2.CutRecord.taste_fences` (defaults `1.0/1.0`) →
`compute_pace_envelope(..., min_tasteful_speed, max_tasteful_speed)` (`post.py`) →
`_pace_levels`. Cross-clip normalization against shared
`PACE_LEVEL_TARGETS=(0.5,0.8,1.0,1.3,1.8)` is already correct — only open the fence.
- **Deterministic rule:** speech OR meaningful diegetic audio (`natural_sound`, not
  safely mutable) → `1.0/1.0`. Else (silent/ambient/motion b-roll) → open, e.g.
  `min≈0.5, max≈2.0–2.5`, optionally scaled by motion. Clamp to `SPEED_FLOOR/CEIL`
  (0.25–4.0). Default `1.0/1.0` whenever unsure.
- **Persist `intrinsic_velocity`** (mean action-energy) in the pace envelope so the
  fence rule + `_pace_levels` can be recomputed at map-build and TUNED at runtime
  (user agreed to persist) — avoids re-ingesting every project on each tweak.

## B2. Speak the aligned LEVEL scale to the brain
**Where:** `footage_map._pace_tag`, the `retime` tool doc, `_LOOP_SYSTEM` beat-line
section, `predict`.
- Render video pace as the shared LEVEL scale (e.g. `tempo:L0–L4`) with resulting
  duration per level, not raw multipliers.
- Prompt: one factual sentence — "pace levels are a shared motion scale; the SAME
  level on adjacent shots matches their on-screen speed; higher level = faster +
  shorter." Explains the affordance; does not tell the brain to ramp.
- Extend `predict` to project a speed choice's length delta (today: energy ladder +
  drop/add only).

## B3. Render bakes per-segment speed — BACKEND + FRONTEND (the real cost)
**MVP scope:** constant per-cut speed on MUTED/silent cuts only (B1 keeps
audio-bearing cuts at 1x → no audio time-stretch/pitch); no intra-cut ramps.
- **Backend (`render/compositor.py`):** time-remap segment video (`setpts=PTS/speed`);
  muted audio untouched.
- **Geometry (`observe`/`arrange`/`predict`):** `played_ms = span/speed` so timeline
  length, `predict`, preview, and export ALL agree. (Today `act.retime` stamps
  `speed` but geometry stays 1x — fix that.)
- **Frontend parity (`resolve-timeline.ts` + `composite-preview.tsx`):** apply the
  same `played = span/speed` + time-remap in preview. The dial/pace-mode UI already
  reads `pace.levels` (`cuts-v3-view.tsx paceRate`), so once geometry applies speed
  the **frontend pacing dropdown reflects it automatically** — but WITHOUT this the
  dropdown lies too.
- Intra-cut ramps: explicitly LATER.

## Group 2 tests
- Fences: audio-bearing → `1.0/1.0`; silent b-roll → spread, cross-clip-aligned
  levels (same level = same apparent motion); `intrinsic_velocity` persisted +
  recomputable at map-build.
- Render/geometry: a chosen speed changes preview AND export identically; timeline
  length, `predict`, and word→program (item 3) all reflect speed.

---

## Sequencing
1. **Group 1** (rules + Program Map + word→program + fit recipe + tool-schema math +
   audit + aggressive sharp + suppression check). Runtime, no re-ingest. Do now.
2. **Group 2** (speed-ramps) — only as a bundle with the B3 render/geometry/preview;
   persist `intrinsic_velocity` to stay runtime-tunable.

## Acceptance
- Brain is told the compositing model in plain, generic language; nothing in the
  system is domain-specific.
- The Program Map (video+audio tables from `layers.resolve`) is always-on, shows
  every layer with a stable id + program window + layout, and makes stacking/overlap
  obvious; `read_state` matches it.
- Word→program time is available at sub-cut resolution on demand (no new sense).
- Fit-to-window lives as a generic recipe in the guidance doc; trim/pace length math
  lives in the tool schemas; pace applies only on an explicit `retime` call.
- Brain sees the assembled program + flags before finishing; flags never prescribe.
- `sharp` shaves ~all removable dead-air; backend/frontend constants stay equal.
- (Group 2) Opening the fence yields spread, cross-clip-aligned speed levels surfaced
  as a level scale with per-level durations; a chosen speed changes preview AND export
  identically and the frontend pacing dropdown reflects it.
