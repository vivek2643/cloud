# Audio sync + authoritative audio — executable plan

**Goal:** make multicam / dual-system audio "just work." When several sources
cover the same moment (multiple camera angles + an external mic), align them on
one shared clock, pick **one authoritative audio** for the moment, and make that
authoritative audio the single source of truth for **everything downstream** —
the speech cuts, the weld (seam) decisions, and the final render's audio bed.
The video angle can switch freely for storytelling; the audio never jumps.

This is written to be implemented from a separate chat. It cites real code.
Reuse the deterministic-keep discipline already in the pipeline: **code owns all
quantitative/structural calls (offsets, boundaries, seams); the LLM owns only
semantic/categorical calls.** Sync detection is a pure DSP algorithm — no LLM.

---

## SS0. Why this exists (the two problems it fixes)

1. **Duplicate / drifting cuts on multicam.** Cuts v3's pass 1 is one call over
   all clips (`pass1.run_pass1`), but each camera has its *own* transcript from
   its *own* mic. The same words produce minutely-different word timestamps,
   filler flags, diarization → per-angle boundaries drift, and pass 1 has to
   *guess* (via token overlap) which clips are the same moment. That guessing is
   the source of the "same transcript, grouped slightly differently" bug.
2. **Broken podcast audio.** The edit couples audio to the shown picture
   (`layers.py` spine = coupled `VideoLayer` + `AudioLayer`, same
   `source_file_id`). Cutting to speaker B's camera drags B's camera audio in
   even while A talks → tone/level jumps per cut.

Both have the **same fix**: within a synced group, derive speech from **one
authoritative audio** and **decouple the audio bed from the video-angle track**.

---

## SS1. Scope

**v1 (this plan):**
- External audio + camera angles that **have** audio → auto-align by audio
  cross-correlation.
- Sync groups are **user-declared** (select the files, hit "Sync"); code
  computes the offsets deterministically, with a manual nudge when confidence is
  low.
- Pure multicam (no external file, each camera has its own audio) → same
  treatment: pick one consistent audio source for the group.
- Authoritative audio at the **group level** (Level 1): one source for the whole
  span (external clean feed if present, else best-sounding camera).

**Deferred (documented hooks, not built now):**
- **Silent-camera case (C):** cameras with no audio → the assistant asks the
  user, a popup declares the group, a slider aligns manually. (No visual
  clap/lip auto-sync in v1.)
- **Per-speaker authoritative audio (Level 2):** route A's mic when A talks, B's
  when B talks (needs a mic↔speaker mapping), still decoupled from picture.
- **Auto-discovery of multicam groups** across a whole project (correlate
  everything and cluster) — v1 is user-declared.
- **Voiceover (case A):** an audio file with no on-camera counterpart becomes the
  editorial *spine* the B-roll edits against. Separate flow, not synced — noted
  in SS11, not built here.
- **Clock-drift correction** on long recordings (piecewise offset).

---

## SS2. Pipeline ordering (the key decision)

```
1. L1 analysis   (per file: transcript+diarization — now for audio too — + sync envelope)
2. SYNC          (project/group level: offsets + groups + authoritative audio)   <-- NEW
3. Cuts v3       (sync-aware: one authoritative speech lattice per group)
4. Brain         (cuts picture over a consistent, code-built audio bed)
5. Resolve/render (audio bed routed to the authoritative source)
```

Sync runs **after analysis** (it needs envelopes/transcripts) and **before
cuts** (so pass 1 receives one authoritative speech lattice + known angle
membership, instead of N drifting transcripts to reconcile). No-multicam / single
file → sync is a **no-op** (each file is its own group) and cuts run exactly as
today.

---

## SS3. Core principles (lock these)

1. **Authoritative audio owns the speech truth.** Within a synced group, ONE
   authoritative transcript/diarization drives: (a) `word_span` boundaries, and
   (b) the audio-derived seam inputs (`same_speaker`, `gap_ms`,
   `bridged_speech_ms`). Never per-camera. This is what makes the drift fix
   *complete*, not approximate.
2. **Audio is decoupled from picture inside a synced group.** The video angle
   switches; the audio bed is built once by code from the authoritative source.
   The brain manages **picture only** in a group — it therefore *cannot* recreate
   the wrong-camera-audio bug (no prompt needed).
3. **Simultaneity is ground truth, not a guess.** Sync (audio correlation) tells
   pass 1 which clips are the same moment (→ outlooks/angles), so pass 1 only has
   to reason about genuinely-sequential retakes.
4. **Deterministic sync.** Offsets/groups are pure DSP; the only human inputs are
   *declaring* the group and the optional manual nudge.
5. **Seam weld is an audio/semantic decision.** When picture is decoupled, the
   visual "shot boundary inside the gap" check (`has_scene_or_transition`) must
   NOT block an audio weld — the brain picks a clean angle on top. (See SS7.)

---

## SS4. Analysis additions (SS "improvement in analysis")

Only two, both reuse-not-reinvent:

1. **Transcribe + diarize spoken audio files.** Today audio-only files run only
   `audio_proxy` + `audio_features` (`pipeline.AUDIO_STAGES`,
   `_orchestrate_audio` — no transcript/diarization). For an external mic to be
   the authoritative source, it needs a transcript with word timings +
   diarization. Add a routed path: for a `file_type == "audio"` upload, run
   `transcript` (and `diarization` when speech is present) in addition to
   `audio_features`. Gate cheaply on `is_musical`/VAD so pure music files skip
   transcription. (Music/SFX analysis is otherwise unchanged — LUFS, bpm,
   onsets, silences, rms already exist and are enough.)
2. **A sync signal (envelope) for cross-correlation.** Reuse the existing
   `audio_features.rms_db` (+ `prosody_hop_ms`) envelope first — it exists for
   every analyzed file. If correlation proves unreliable at that resolution
   (~600 pts), add a denser fixed-hop normalized energy/onset-flux envelope
   (this is what the retired `sync_env` was). **Decision to make during build:**
   reuse `rms_db` vs add `sync_env`; start with reuse.

No new heavy analysis beyond these.

---

## SS5. Sync detection (deterministic)

**Input:** a user-declared set of files (angles + audio). **Output:** per-member
offset on a shared group clock + confidence + a chosen authoritative audio.

1. **Envelope prep:** per file, take the energy envelope (SS4) resampled to a
   common fixed hop, normalized.
2. **Pairwise cross-correlation:** for each pair, the lag maximizing normalized
   cross-correlation = the offset; the peak height / sharpness = **confidence**.
   (Optionally narrow the search with a file-creation-time prior.)
3. **Group solve:** anchor all members to one reference (e.g. earliest start);
   store each member's `offset_ms` on the group clock. A pair with a sharp,
   high peak overlapped in real time (same moment); a diffuse/low peak did not
   (unrelated, or a sequential retake — which correctly stays out of the group).
4. **Confidence gate:** high confidence → silent auto-align; low → surface the
   manual nudge UI (SS10) pre-filled with the computed offset.
5. **Authoritative audio selection (Level 1, code-picked, manual override):**
   prefer a dedicated external audio file if the group has one; else pick the
   best-sounding camera by a simple heuristic (e.g. highest `integrated_lufs`
   that isn't clipping on `true_peak_db`, lowest silence ratio). Store the
   choice; user can override.

Distinguishes multicam (high acoustic correlation at an offset) from retakes
(low correlation) — the waveform analogue of the semantic take-group logic.

---

## SS6. Data model (proposed — verify column names during build)

```sql
-- one synced group of sources on a shared clock
create table sync_groups (
    id            uuid primary key default uuid_generate_v4(),
    project_id    uuid not null references projects(id) on delete cascade,
    authoritative_audio_file_id uuid references files(id),   -- Level 1: one source
    created_by    text,                 -- 'auto' | 'user'
    created_at    timestamptz not null default now()
);

create table sync_group_members (
    group_id      uuid not null references sync_groups(id) on delete cascade,
    file_id       uuid not null references files(id) on delete cascade,
    offset_ms     int not null,         -- position on the group clock
    role          text not null,        -- 'video_angle' | 'audio'
    confidence    real,                 -- correlation peak; null for manual
    aligned_by    text not null,        -- 'auto' | 'manual'
    primary key (group_id, file_id)
);
```
- **Pinning:** the cuts ingest must snapshot which sync result it used so a later
  re-sync doesn't mutate an existing edit — mirror the existing
  `edit_threads.ingest_run_id` pattern (`converse.py` passes `pinned_run`).
  Simplest: `cut_records.sync_group_id` (nullable) links a cut to its group;
  the resolver joins `sync_group_members` for the authoritative source + offset.
  (Alternative: denormalize `audio_source_file_id` + `audio_offset_ms` straight
  onto `cut_records` for a zero-join resolver — pick during build.)
- `cut_records` today: `file_id, src_in_ms, src_out_ms, kind, word_span,
  take_group_id, take_role('take'|'outlook'|'winner'), channel, continuity,
  pace, speaker` (`024_cuts_v3.sql` + `027`/`029`). No sync/offset columns yet →
  add `sync_group_id` (+ optionally the denormalized audio-source fields).

---

## SS7. The core cuts change — authoritative speech lattice into pass 1

This is where sync earns its keep. Today `pass1.load_project_file_rows` builds
one `Lattice` per file via `lattice.load_lattice(fid)` (words from
`transcripts.segments`, turns from `diarize.load_turns`, atoms from
`build_atoms`). For a synced group, replace the N per-angle speech lattices with
**one authoritative speech lattice**:

1. **One transcript per group.** Build the group's `words`/`turns` from the
   **authoritative** source's transcript/diarization only. Re-base word times
   onto the group clock (via the authoritative source's `offset_ms`).
2. **Speech splits decided once.** Pass 1 emits `SpeechCut.word_span` over this
   single word list → identical boundaries for every angle by construction. No
   per-camera duplication, nothing for pass 1 to reconcile.
3. **Angle membership is given, not guessed.** Feed pass 1 the known simultaneity
   (these files are angles of this group) so it does NOT emit `TakeCandidate`s
   for them — sync *produces* those outlooks deterministically. Pass 1 still
   groups genuinely-sequential retakes semantically.
4. **Video atoms stay per-angle.** Each angle contributes its own `build_atoms`
   output (its own motion/scene) — those differences are real (different framing/
   movement), surfaced as angle-options of the shared moment, not duplicates.
5. **Seam inputs from authoritative audio (`pass1._gap_seam`).** Today
   `_gap_seam` reads `words[i].speaker` and word `end_ms`/`start_ms` from the
   per-file lattice. In a group these MUST come from the authoritative
   transcript/diarization (they already will, if the group's single lattice is
   the authoritative one — this falls out of step 1). Explicitly verify
   `same_speaker`, `gap_ms`, `bridged_speech_ms` derive from authoritative words.
6. **Video shot-check in the weld (design call, SS3.5).** `_gap_seam`'s
   `has_scene_or_transition` reads the atoms' break-boundary reasons. With the
   speech beat decoupled from any one angle's picture, a shot boundary inside the
   gap should **not** force a hard audio split. Options: (a) drop the video
   shot-check for synced-group speech welds (weld = same_speaker + gap +
   flagged-break only), or (b) evaluate it only against the authoritative/anchor
   angle. **Recommend (a)** — cleanest, matches "weld is audio/semantic."

**Result:** a "moment" = the shared authoritative speech span; each angle is a
picture option within it; boundaries + grouping are exact. This is the complete
fix for the minutely-different-grouping bug.

---

## SS8. Authoritative audio bed in resolve/render

Make the audio follow the authoritative source, not the shown angle. Today
`layers.py` builds, per spine span, a `VideoLayer` and a **coupled**
`AudioLayer(role=dialogue, kind=spine, source_file_id=seg["file_id"])`.

For a spine segment whose cut belongs to a `sync_group`:
- **Video** stays from the chosen angle (`seg["file_id"]`).
- **Audio** re-routes: `AudioLayer.source_file_id = authoritative_audio_file_id`,
  with `src_in_ms/src_out_ms` = the program span mapped into the authoritative
  source's timeline via the group offsets (program → group clock → authoritative
  src). The angle's own scratch audio is simply **not used** (no separate mute op
  needed — it's never emitted).
- Non-synced segments behave exactly as today (coupled audio).

Effect: across a synced region the audio is one continuous source regardless of
angle cuts. This is your "cut the bad camera audio, ride the clean one," applied
consistently instead of per cut. `_apply_split_edits` (J/L) still operates on the
resulting audio layers if the brain ever asks; by default it won't need to.

The compositor already delays/gains/duck-mixes audio layers, so no compositor
change beyond honoring the re-routed source is expected — verify.

---

## SS9. Brain awareness (structural, not prescriptive)

Per the "no specific guidance" steer: **remove audio from the brain's job in a
synced group** rather than teaching it. The footage map already exposes
moments + angle options (outlooks) and continuity. Additions:
- The brain sees a moment's **angle options** and picks *picture*; the audio bed
  is code-built and not something it routes.
- No new prompt rules, no reliance on `split_edit`. The architecture makes the
  correct audio the only audio.

(If any awareness is added, keep it factual: "this moment has N angles; audio is
the group's authoritative track" — not editorial advice.)

---

## SS10. UI

- **Declare a sync group:** multi-select files (angles + audio) in Drive / the
  project, action "Sync." (v1 = user-declared; auto-discovery deferred.)
- **Result readout:** per member, computed offset + a confidence indicator
  ("synced ✓" vs "check alignment"). Low confidence → open the nudge view.
- **Manual nudge:** stacked waveforms (the `audio_proxy` waveform PNGs already
  exist) with a draggable offset; commit updates `sync_group_members.offset_ms`
  (`aligned_by='manual'`).
- **Authoritative source picker:** shows the code-chosen default (external > best
  camera) with a manual override.
- **Silent-camera path (deferred):** the assistant detects no audio on the
  angles, asks the user, and opens the group-declare + manual-slide flow.

---

## SS11. Adjacent / out-of-scope flows (noted)

- **Voiceover (case A):** an audio file with no on-camera match → not synced;
  becomes the editorial spine (its transcript is the backbone the B-roll edits
  under). Different pipeline path; design separately.
- **Music / SFX:** unchanged — `place_audio` beds + ducking + loudness already
  exist; optional beat/section analysis is its own future plan.

---

## SS12. Edge cases & risks

- **Partial overlap:** an angle covers only part of the group's span → offset
  still valid; the moment simply has fewer angle options where an angle is
  absent. Handle by span intersection, not by assuming full coverage.
- **Clock drift** (long dual-system recordings): a single offset drifts over
  many minutes. v1 = single offset (accept minor end drift); piecewise/resampled
  offset is a later refinement.
- **Wrong grouping by the user:** confidence + nudge guard it; a low-confidence
  auto-align never silently commits.
- **Authoritative audio doesn't cover a speaker** (e.g. a single lav on one
  person): Level 1 still uses it group-wide (may be quieter for others); Level 2
  per-speaker routing fixes this later.
- **Retakes mistaken for multicam:** correlation distinguishes them (low peak →
  not grouped); they remain semantic take-groups.
- **Re-sync after an edit exists:** pinning (SS6) prevents mutation of a live
  edit's beat universe.

---

## SS13. Phasing

**v1 (this plan):** analysis additions (transcribe audio files + sync envelope);
deterministic sync detection for user-declared groups (offsets + confidence +
authoritative pick); authoritative speech lattice into pass 1 (splits + seam
inputs); per-angle video atoms + given angle membership; audio bed re-routed in
resolve; group-level authoritative audio; sync-declare + nudge UI.

**Deferred:** silent-camera manual/visual sync; per-speaker (Level 2) routing;
auto-discovery of groups; voiceover spine; clock-drift correction; optional music
beat/section analysis.

---

## SS14. Open questions (decide during build)

1. **Sync envelope:** reuse `rms_db` or add a denser `sync_env`? (Start: reuse.)
2. **cut_records linkage:** `sync_group_id` + resolver join, vs denormalized
   `audio_source_file_id`/`audio_offset_ms` on `cut_records`? (Lean: `sync_group_id`.)
3. **Where sync runs mechanically:** a standalone project step feeding the cuts
   ingest, or a stage inside `ingest.run_ingest` before `run_pass1`? (Feeds pass 1
   either way.)
4. **Video shot-check in weld:** drop it for synced-group speech welds (SS7.6
   recommend) vs evaluate against the anchor angle.
5. **Group clock reference:** earliest-start source vs the authoritative source
   as t=0.

---

## SS15. Build checklist

- [ ] L1: route `transcript` (+`diarization`) for spoken `audio` files; gate music out.
- [ ] Sync signal: confirm `rms_db` envelope suffices (or add `sync_env`).
- [ ] `sync/detect.py` (new): deterministic pairwise cross-correlation → offsets + confidence.
- [ ] `sync/authoritative.py` (new): Level-1 source pick (external > best camera) + override.
- [ ] Schema: `sync_groups` + `sync_group_members`; `cut_records.sync_group_id`; pinning.
- [ ] Cuts: build one authoritative speech lattice per group (re-based word times); feed pass 1 given angle membership; per-angle video atoms.
- [ ] Cuts: verify `pass1._gap_seam` uses authoritative words/diarization; resolve the video shot-check in welds (SS7.6).
- [ ] Resolve: `layers.py` re-routes synced spine audio to the authoritative source (offset-mapped); non-synced unchanged.
- [ ] Compositor: verify re-routed audio source renders correctly.
- [ ] Brain: expose angle options; audio bed code-built (no new prompt rules).
- [ ] UI: declare group + offset/confidence readout + waveform nudge + authoritative picker.
- [ ] No-multicam regression: single-file projects behave exactly as today (sync no-op).
