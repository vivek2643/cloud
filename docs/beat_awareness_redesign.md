# Beat awareness redesign — PICTURE + SOUND, one generic substrate

## Why this exists

The arranger ("brain") keeps producing edits that show the **wrong person** (the
silent listener while someone off-screen talks) and defaults to a talking-head
spine even when asked for something else. Every prior fix added more facts, yet
the failure persisted. The root cause is **not** missing awareness — it is how
the awareness is *structured and presented*:

1. We ship the brain **two** representations — a fat, scannable **said-line
   index** (`footage_map`) and a neutral continuous timeline (`source_awareness`).
   The said-line index is **sound-shaped by construction**, so under a "lay the
   spine" task the brain walks it and produces a single-camera, speech-driven cut.
2. A `said` beat is written **speaker-first** (`… G2 off-cam shows:G1 …`), which
   invites the false equation *"this is G2's line → placing it shows G2."* In a
   two-camera shoot the same audio is captured on both cameras, so half the lines
   in any one clip show the **other** person. Walking one clip therefore shows the
   listener for half the edit.
3. The fact that fixes it — *the same words are also available on the camera that
   shows the speaker* — lives in a **distant** `COVERAGE GROUPS` block, referenced
   only by a `dup:tgN` id the brain has to go look up. It never confronts the
   choice at the moment it places.

The same model, given the same facts but arranged as a **narrow, meaning-first
slice**, diagnoses the bug in seconds. The gap is representation + process, not
data.

## Design principles (agreed during ideation)

- **Only two facts are robust, and we assert nothing beyond them.**
  - **Fact #1 (per clip, continuous, rock-solid):** at every instant we know who
    is *on screen* (`PICTURE`) and what is *heard* (`SOUND`), identity-resolved,
    plus shot size and quality. Needs no other clip to be true.
  - **Fact #2 (cross-clip, speech only):** the same spoken words occur in several
    clips. This is a bare **co-occurrence** — it carries **no** claim of
    simultaneity, "angle", or "take".
- **Never classify angle vs. take.** It is impossible from the data and only
  exists for speech. The brain (same brain that reasons well) classifies the
  situation itself from the neutral facts; we never pre-decide it.
- **No privileged axis.** Neither picture nor sound is "the spine". Time (each
  clip's own clock) is the only neutral ordering. Editing is segmented by
  **meaning** (thought / action / reveal beats), not by sensor change — so the
  primary object stays a *meaning-segmented beat index*, not a raw sensor
  timeline.
- **No merged cross-clip spine.** Merging co-occurring beats into one ordered
  sequence would require deciding "same session vs. separate take" — the
  forbidden classification. We keep each clip in its own (safe) source-time order.
- **Generic, not genre-specific.** The same substrate must describe a one-camera
  vlog, a screen-recording tutorial, a multi-take ad read, an action/montage
  sequence, and a multi-camera interview with **no special-casing**.

## The primary object: a de-biased, multi-channel BEAT index

Keep the meaning-segmented beat index (`footage_map`) as the **primary** awareness
— but fix its two real defects (sound-dominance in the listing; scattered
coverage) and make every beat read neutrally.

### Beat line format

Lead with what actually lands on screen (`PIC`), make sound a co-equal peer
(`SND`), then the content. Example (real beat from thread `5ff9a12a`, group
`tg21`):

```
1aedb093:m24  PIC:G1 (med, q.60)  SND:G2 speaking  "…that freedom he gave you…"
              ·alt-PIC:G2 → 48c93cef:m13 (med, q.73)   ·6:19 7.3s  ·nrg:broad…sharp
```

- `PIC:` — whose face / what scene is on screen (from the reconciled cast
  `shows:`), + shot size + quality. **This is the first thing read**, so placing a
  beat can no longer be mistaken for "showing the speaker".
- `SND:` — who is heard (identity-resolved speaker) and the audio state
  (`speaking` / `silence` / `ambient`). A peer of `PIC`, not the subject.
- content — the words (speech) or the action/graphic gist (non-speech).
- `·alt-PIC:` — **Fact #2 folded onto the beat**: every other picture the *same
  sound* is available as, each a neutral `ref (shot, q)`. Stated as a fact, never
  "use this". Absent when there is no co-occurrence.
- tail — timecode, play length, zoom ladder (`nrg`), run id — unchanged.

### Non-speech beats are equal citizens

`done` / `shown` beats render with the **same shape and richness** as `said`, so a
visual-driven edit is first-class:

```
1e529bed:m41  PIC:G1 (listening, med, q.90)  SND:silence   "G1 nods, reacts"    ·2:12 6.7s
2652334c:m07  PIC:desk/setup (wide, q.71)     SND:ambient    "room, establishing" ·0:03 4s
```

A montage/action edit reads `PIC` + these beats and ignores `SND`; a dialogue
edit reads `SND` and matches `PIC` via `·alt-PIC`. Neither axis is privileged.

### What "dedupe" means here (safe)

We dedupe the **information**, not the **sequence**: the "who does each camera
show" fact moves out of the distant `COVERAGE GROUPS` block and onto each beat as
`·alt-PIC`. Clip-time order (the only safe order) is untouched. The same beat may
still appear in two cameras' lists, but that is now **harmless** — each copy leads
with its true `PIC` and carries its `alt-PIC`, so walking one clip no longer hides
the swap. (The double-listing was only dangerous when both the picture truth and
the options were buried.)

## The secondary sense: the continuous timeline (unchanged role)

`source_awareness` / `clip_timeline` stays as the **drill-down** sense — the
escape hatch for spans that fall *between* beats (a silent reaction, a b-roll
window) and for open-ended queries (`scan_source`, now forgiving). It is **not**
promoted to primary: a raw sensor timeline under-segments where meaning is dense
(a long monologue on a locked shot = one span but many beats) and over-segments
where meaning is thin, and the brain reasons better over readable content lines
than a facet grid.

## How this maps to existing code (mostly a rendering reshape)

- `PIC` = existing reconciled `shows:` (via `relations.oncam_global_by_file`) +
  `framing` + `quality`. Already computed.
- `SND` = existing identity-resolved `speaker` (via `relations.identity_index`,
  resolved off the **raw handle** in `people[]`, per the recent fix) + the audio
  state already present (`said` ⇒ speaking; video beats carry `audio`
  = speech/sound/silent + `mute`).
- `·alt-PIC` = existing coverage-group members (`takes.build_take_groups` →
  `_annotate_dups`), relocated onto the beat line instead of a separate block.
- `done`/`shown` equality = prominence/rendering change in `footage_map`; data
  already exists in cuts-v2.

**No new perception/analysis. No cache-shape change** (render-time only, so no
`TREE_VERSION` bump needed for the moment shape). **No verb changes** — `place`
still couples a clip's picture+sound; `V2` / `split_edit` / `split_screen` are
exactly "put a different `PIC` over this `SND`", which now reads as the obvious
move.

## Implementation phases

1. **Beat line reshape** — `footage_map._moment_line`: emit `PIC:… SND:… content`
   led by picture; keep ids/variants/run/tags. Reuse the existing
   handle-resolution (`_speaker_handle`, `_global_speaker`, `_shown_and_cam`).
   Add the `SND` audio-state token (speaking/silence/ambient) from the beat's
   `audio`/channel.
2. **Fold coverage onto the beat** — surface each co-occurring beat's alternates
   inline as `·alt-PIC:Gx → ref (shot, q)` from the dup-group members; trim the
   separate `COVERAGE GROUPS` block to a short pointer (or drop it) once the
   inline form carries the same facts. Keep it neutral — list alternates, no
   verdict.
3. **`done`/`shown` parity** — ensure non-speech beats render with the same PIC
   /SND/content shape and are not visually subordinate; verify their content gist
   (action/graphic) reads well.
4. **Prompt reshape** — `converse` v3 system: describe the material as beats, each
   with a picture and a sound; editing = "choose the picture and the sound for
   each beat; either can drive the cut." Drop "lay the spoken spine" framing.
   Keep the continuous timeline described as the between-beats escape hatch. No
   take/angle language.
5. **Tests** — `test_footage_map`:
   - a `said` beat on the listener's camera renders `PIC:G1 … SND:G2 …
     ·alt-PIC:G2 → <ref>` (the exact podcast bug shape);
   - a `done`/`shown` beat renders with PIC-first parity and `SND:silence`;
   - alt-PIC is absent when there is no co-occurrence (one-camera case);
   - resolution still keys off the raw handle, not the display label.
6. **Pressure-test (validation, not code)** — run the reshaped map on real
   **non-dialogue** footage to confirm `done`/`shown` beats are rich enough that
   the list doesn't silently skew back to speech.

## Explicit non-goals / what we are NOT doing

- Not classifying clips as angles vs. takes (impossible; forbidden).
- Not merging clips into a single cross-clip ordered spine (would require that
  classification).
- Not adding a new perception pass, embeddings, or model calls.
- Not making the raw sensor timeline the primary awareness.
- Not privileging sound (or picture) as "the" spine.

## Risks & caveats

- **`done`/`shown` richness is load-bearing.** If VLM action/reveal detection is
  weak, the beat index collapses back toward speech and the sound-bias returns
  through the back door. This is the main thing to validate (phase 6) before
  trusting the redesign on non-dialogue material.
- **Density.** Inlining `alt-PIC` on every co-occurring beat adds width; mitigate
  with compact refs and the existing disclosure tiers (full detail via
  `inspect_moment` / Tier-1), and by trimming the now-redundant coverage block.
- **Residual double-listing** across cameras remains (we don't merge); acceptable
  verbosity now that each copy is self-describing.
