# Audio for the Brain — awareness + authoring

## Goal

Make the agentic editor ("EDSO") able to do **anything with audio it needs to** —
perceive the full audio state of an edit, and author music / voiceover / SFX
beds, ducking, gain, fades, crossfades, and beat-aligned cuts. Today the brain
sees almost nothing about audio and cannot place a single bed.

## Principle (house rule)

**The brain has complete freedom. We build *capabilities* and *honest facts* to
assist it — never *behaviors* that decide on its behalf.**

- **Code exposes verbs** — it *can* do anything (place beds, gain, duck, fade,
  crossfade, snap-to-beat, replace). Every verb is neutral: it states *what it
  does*, never *when/how* to use it.
- **Observe surfaces the true state** — what audio *is* on each cut, what beds
  exist, what assets are available, loudness, silence, beat grid. Facts the brain
  can't infer, nothing directional.
- **The brain chooses.** No auto-duck, no auto-normalize, no auto-fade, no
  auto-loop, no forced silence-fill, no blocked seams. If a shaping decision
  exists, it's the brain's, invoked through a verb.
- **`guidance_doc` carries NO audio craft advice.** The brain already knows
  editing craft; writing "duck under dialogue / cut on the beat" only re-adds the
  over-specific flavor we removed. Facts live in observe; craft stays with the
  model.
- **One structural default only:** outlook authoritative routing (below) — kept
  because it's a deterministic *truth* about which feed is clean, and it stays
  fully **overridable**. It's assistance, not a lock.

## Answers baked in (from review)

- **Voiceover is NOT privileged.** VO is just another bed the brain balances with
  `set_gain`/`duck`. No `ROLE_VOICEOVER` dialogue-priority, no automatic ducking
  of music under VO.
- **Beds start/stop hard by default.** No automatic fades. The brain adds a fade
  only via `fade_audio`.
- **No automatic ducking.** A bed ducks only by the `duck_db` the brain sets (via
  the `duck` verb or `place_audio`). `_apply_levels` must be changed so a bed
  with `duck_db=0` never ducks.
- **No automatic loudness normalization.** Per-cut loudness is surfaced as a
  *fact*; the brain evens levels with `set_gain` if it wants to. (Phase 3 dropped.)
- **Bed shorter than its window → surfaced, not auto-handled.** Observe reports
  the length mismatch; the brain decides loop / extend / leave short.
- **`split_edit` inside an outlook group is NOT blocked.** The continuous bed is a
  *fact* in observe, not a locked seam — the brain may J/L over it if it wants.
- **Gaps are just silent.** Not auto-filled with room tone, not a validation
  error. Surfaced honestly; the brain fills a gap only if it cares to.

## Current state (grounded — what already exists vs what's missing)

**Already built (downstream is ready):**
- `layers.resolve` handles a `place_audio` op → `AudioLayer(role, kind,
  source_file_id, src_in/out, prog span, gain_db, duck_db)`.
- `layers._apply_levels` currently **auto-ducks** any bed/sfx overlapping live
  dialogue + applies explicit `level` gain. **This auto behavior must become
  opt-in** (duck only when `duck_db<0`).
- `compositor.py` mixes `audio_layers`: delays each to its prog start, applies
  `gain_db + duck_db`, sums with `amix`. So placed beds already render.
- Roles exist: `ROLE_DIALOGUE`, `ROLE_MUSIC`, `ROLE_SFX`; kinds `spine|bed|
  replace|sfx`.
- Verbs that exist: `set_audio` (mute/unmute), `split_edit` (J/L),
  `place … V2 audio:keep`, `retime` (speech dead-air trim).
- Outlook authoritative routing (one continuous bed across angle switches) — code
  owns it, and it is overridable.

**Missing:**
1. **No `act.place_audio` verb** — the op `layers.resolve` already consumes is
   never created; the brain literally cannot add music/VO/SFX. (Referenced in
   `act.py`'s docstring + `read_state`, but `def place_audio` does not exist.)
2. **No audio awareness in `observe`** — brain sees only per-cut `muted`,
   `channel`, `speaker`, a channels list, `split_edits`. It can't see audio
   source, the continuous-bed marker, placed beds, available audio assets,
   loudness, silence, bed-vs-window length, or a beat grid.
3. **No fine authoring** — gain, duck (as a verb), fades, crossfade, beat-snap.

## Decisions (scope)

- **Assets = user-uploaded audio files only** (`files.file_type='audio'` in the
  project/thread scope). A built-in music/SFX library is a **non-goal** for now
  (clean add later behind the same `place_audio` asset arg).
- **Full surface, phased**: Phase 1 unlocks music/VO/SFX + awareness (the 80%);
  Phase 2 adds gain/duck-verb/fades/crossfade/beat-snap. Everything ends up
  buildable.
- **No pitch shifting, no stereo/pan** (out of scope; `retime` never pitches).
- **No automatic loudness normalization** — replaced by a loudness *fact* + the
  `set_gain` verb.

---

## Phase 1 — place beds + audio digest (unlocks music / VO / SFX)

### 1a. `act.place_audio` (new verb)

New `act.place_audio(document, *, source_file_id, role, from_ms, to_ms,
src_in_ms=0, src_out_ms=None, gain_db=0.0, duck_db=0.0, audio_kind="bed")` —
mirrors `place_video`'s op-creation pattern; appends the op shape
`layers.resolve` already reads:

```python
{"op_id": f"pa_{uuid…}", "type": "place_audio",
 "role": role,                       # "music" | "voiceover" | "sfx"
 "source_file_id": source_file_id,
 "src_in_ms": src_in_ms, "src_out_ms": src_out_ms,   # default full asset
 "from_ms": from_ms, "to_ms": to_ms, # program window
 "gain_db": gain_db, "duck_db": duck_db,   # duck only if brain sets it
 "audio_kind": audio_kind}           # "bed" | "replace" | "sfx"
```

- **`role="voiceover"` maps to a plain `ROLE_DIALOGUE` bed with NO auto-duck
  priority** — VO is balanced by the brain, not privileged by code.
- `src_out_ms=None` → full asset duration (look up via `_durations`).
- Validate: source is an owned audio (or video) file in scope; `to_ms>from_ms`;
  span within asset duration. **Do not auto-loop/auto-stretch** if the window
  exceeds the asset — place what exists and let observe report the shortfall.

### 1b. Tool spec + dispatch (`tools.py`)

Add to `_specs()` and the dispatch in `run_tool` — neutral, *what it does* only:

```
place_audio: "Places an audio bed on a program window [from_ms,to_ms]. role
music|voiceover|sfx; source is an audio (or video) asset by file id; gain_db
sets its level; duck_db (<=0) lowers it (0 = no duck)."
args: source, role, from_ms, to_ms, [src_in_ms, src_out_ms, gain_db, duck_db, kind]
```

Resolve `source` via `_resolve_file` (already handles the `CLIP <file8>`
prefix). Add `place_audio` to `observe.affordances`'s `verbs` list. No "use this
for reels / duck under VO" — nothing directional.

### 1c. Audio digest in `observe`

**Per-cut, in `read_state`** — add to each cut dict:
- `audio_source`: `own` | `group-authoritative` | `replaced` | `muted`
  (derive: muted flag → muted; cut in an outlook `sync_group_id` → group-
  authoritative; else own).
- `natural_sound` (video cuts): from the cut's `pace.natural_sound`.
- `loudness_rel`: per-cut integrated level vs the edit median, from L1
  `audio_features` — a fact so the brain *can see* a camera-to-camera jump (it
  fixes it with `set_gain` only if it wants to).

**New sense `audio_state`** (or extend `read_state`) with a compact digest:
```json
{
  "beds": [{"op_id","role","from_ms","to_ms","gain_db","duck_db","asset","asset_dur_ms","window_ms"}],
  "continuous_beds": [{"from_ms","to_ms","group_id","note":"cuts 3-9 share one authoritative bed; angle switches don't change sound"}],
  "assets": [{"file_id","name","dur_ms","is_musical","bpm"}],   // unused audio-type files
  "channels": ["V1","A1","A2?"]
}
```
- `assets` = `files.file_type='audio'` in `ctx.file_ids`/project not already used
  by a `place_audio` op. Add a `build_context` DB read (or reuse the audio-file
  listing) → `ctx.audio_assets`.
- Each bed carries both `asset_dur_ms` and `window_ms` so a **shortfall
  (window > asset)** is a visible fact — the brain decides loop / extend / leave.
- `continuous_beds` = runs of consecutive main-line cuts sharing a
  `sync_group_id` (from the moment meta) — a **fact, not a lock**. It tells the
  brain the sound is already continuous across the angle switches so it needn't
  "fix" the seam; it may still `split_edit` there if it wants.

### 1d. `affordances`

Add per-cut `can_place_bed_here` window info + a global `audio_assets` count.
`can_add_channel: ["A2"]` already exists. List `place_audio` in `verbs`.

---

## Phase 2 — fine authoring + rhythm

### 2a. Verbs (each mirrors an existing act + a resolve field that mostly exists)

| Verb | Creates / sets | Notes |
|---|---|---|
| `set_gain(target_id, gain_db)` | segment/bed `gain_db` | main-line seg, V2 op, or A2 bed |
| `duck(layer_id, amount_db)` | bed `duck_db` | explicit only; `_apply_levels` no longer auto-ducks |
| `fade_audio(target_id, in_ms, out_ms)` | fade envelope on a layer edge | new resolved field `fade_in_ms/out_ms`; compositor adds `afade`. Beds have no fade unless set |
| `crossfade(seam_seg_id, ms)` | audio crossfade across a seam | new; compositor `acrossfade` at the seam |
| `trim`/`move`/`remove` on A2 beds | extend the existing verbs to target `place_audio` op_ids | today they target segs/V2 ops |
| `replace_audio(target_id, source)` | swap a cut's audio source / override authoritative routing | the escape hatch that makes routing overridable |

### 2b. Beat grid + snap

- `observe` surfaces a **beat grid** as a fact **only when a musical bed or
  musical clip is in play**: BPM + downbeat/onset positions mapped to **program**
  time, from L1 `audio_features.bpm` / `onsets_ms` of the musical source. No grid,
  no fact — the brain doesn't try to snap where there's no music.
- `place`/`move`/`trim` gain a `snap:"beat"` option: code snaps the program edge
  to the nearest grid line (same pattern as `snap_span_to_seams`, new
  `snap_to_beats`). Brain expresses intent; code does the ms.

---

## Files to touch

| File | Change |
|---|---|
| `l3/act.py` | new `place_audio`; new `set_gain`/`duck`/`fade_audio`/`crossfade`/`replace_audio`; extend `trim`/`move`/`remove` to A2 op_ids |
| `l3/tools.py` | specs + dispatch for the new verbs; add to `affordances` verb list; neutral "what it does" wording only |
| `l3/observe.py` | per-cut `audio_source`/`natural_sound`/`loudness_rel`; new `audio_state` digest (beds w/ asset-vs-window length, continuous_beds, assets); `build_context` reads audio assets; beat grid when musical |
| `l3/layers.py` | make duck **opt-in** (`duck_db<0` only, no auto-follow); VO = plain dialogue-role bed (no priority); resolve `fade_*`/crossfade fields |
| `render/compositor.py` | `afade` per layer, `acrossfade` at seams |
| `guidance_doc` | **no audio section** (facts live in observe) |

Downstream `layers.resolve` `place_audio` branch + `compositor` amix/gain/duck
are **already built** — Phase 1 mostly *creates the op* and *surfaces state*.

---

## Verb surface (final — "can do anything")

| Verb | Does | Status |
|---|---|---|
| `set_audio` | mute/unmute source | exists |
| `place_audio` | place a music/VO/SFX bed | **P1 (new)** |
| `set_gain` | set a cut/bed level | P2 |
| `duck` | lower a bed by an explicit amount | P2 |
| `fade_audio` | fade in/out | P2 |
| `crossfade` | audio crossfade a seam | P2 |
| `split_edit` | J/L cut | exists |
| `retime` (speech) | shave dead-air/fillers | exists |
| `trim`/`move`/`remove` (bed) | edit a placed bed | P2 (extend) |
| `replace_audio` | swap source / override authoritative routing | P2 |
| beat-snap (via place/move) | align an edge to the beat grid | P2 |

## Observe surface (final — facts only)

Per-cut: `audio_source`, `muted`, `natural_sound`, `speaker`, `loudness_rel`.
Digest: `beds` (with asset-vs-window length), `continuous_beds` (outlook
authoritative marker, a fact not a lock), `assets` (unused audio files),
`beat_grid` (only when musical), `channels`.

## `guidance_doc`

**No audio section.** Nothing directional, no per-format playbook. The
`continuous_beds` marker and every other non-obvious system fact live in observe;
craft stays with the model. Add a line here later *only* if real testing shows the
brain genuinely fighting the system (e.g. hand-routing a group's audio) — never
speculatively.

## Determinism & fallback

- Brain expresses categorical intent; code computes every ms (duck amount it's
  told, beat snap). No LLM numbers.
- No audio assets in scope → `place_audio` reports "no audio assets available"
  (never fabricates a source). Older docs with no beds behave exactly as today.
- Outlook authoritative routing stays a code default **and** overridable via
  `replace_audio`/`set_audio`; the digest makes it visible. It's the one place
  code sets a default, justified because it's structural truth, and the brain can
  always undo it.

## Verification

1. **Unit**: `place_audio` op → `layers.resolve` yields the AudioLayer; a bed with
   `duck_db=0` does **not** duck under overlapping dialogue; a bed with
   `duck_db<0` does; compositor emits the `amix`/`volume` chain (assert the ffmpeg
   filtergraph).
2. **Digest**: a podcast edit shows `continuous_beds` spanning the outlook run; an
   uploaded-but-unused music file shows in `assets`; a bed placed over a window
   longer than its asset reports the shortfall (`asset_dur_ms < window_ms`).
3. **End-to-end**: brain places a music bed under an intro, sets `duck` and a
   `fade_audio`, exports; the bed is audible, ducks only as set, fades as set;
   preview == export.

## Non-goals

- Built-in music/SFX library (assets are user uploads for now).
- Pitch shifting, stereo/pan, multiband processing.
- Automatic loudness normalization (replaced by loudness fact + `set_gain`).
- Level-2 per-speaker audio routing (separate future track).
