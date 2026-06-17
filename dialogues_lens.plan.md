# Dialogues lens — phased plan (Sentence ⇄ Topic)

Goal: a **Dialogues** tab in the drive that turns raw footage into clean,
**drop-straight-to-timeline** speech clips, with a single switch between two
granularities — **Sentence** and **Topic**. No other editing required.

North star: *the cut has to be audio-clean* (no clipped words, no abrupt
starts). If that fails, nothing else matters.

## What we already have (no new perception needed)
- `transcripts.segments` → word-level `{start_ms, end_ms, text, is_filler, speaker}`.
- `transcripts.fillers` → detected `um/uh/...`.
- diarization → per-word `speaker` ("S0"/"S1"…) + turns.
- `audio_features.pause_map`, `rms_db` (coarse), `energy_peaks_ms`, `silence_intervals`.

The only thing missing is a fine-grained **energy envelope around boundaries**
for cut-snapping → computed at stage time from the WAV (don't rely on the stored
coarse `rms_db`).

## Core model: phrases → sentences → topics
- **Phrase (atom, immutable):** run of ONE speaker's words with no internal gap
  > 250ms. Computed once. Never crosses a speaker.
- **Sentence (level 1):** merge same-speaker phrases while gap ≤ `G_sentence`,
  respecting sentence-final punctuation; strip leading/trailing fillers.
- **Topic (level 2):** merge same-speaker sentences into one answer/idea,
  bridging the other speaker's short backchannel ("mhm", "yeah"). Single speaker.
- **Speaker change is ALWAYS a hard boundary.** Two speakers = two adjacent
  clips, never merged (multi-select to take both). Overlap → flag, don't guess.

The switch just selects which precomputed segmentation to serve. Both carry a
`topic_id`, so a Topic clip can expand into its child Sentence clips (accordion).

## DialogueSegment (stored artifact)
```
seg_id          stable (file_id:level:order)
file_id
level           "sentence" | "topic"
order           chronological within file
speaker         "S0" | ... | null
text            joined words (leading/trailing fillers stripped)
src_in_ms       FINAL cut in  (silence-snapped + handle)
src_out_ms      FINAL cut out (silence-snapped + handle)
raw_in_ms       first non-filler word start (unpadded)
raw_out_ms      last  non-filler word end   (unpadded)
fade_in_ms      ~10–20ms audio de-click
fade_out_ms
word_start_idx  index range into the flat word list
word_end_idx
topic_id        groups sentences ↔ topics (both levels carry it)
child_seg_ids   topic → its sentence seg_ids (accordion)
flags           ["overlap","noisy","false_start","trails_off","low_confidence"]
confidence      0..1
has_video / has_audio
```
Maps to the existing `EditSegment` 1:1 (`in_ms=src_in_ms`, `out_ms=src_out_ms`,
`content=text`).

---

## Phases

| Phase | Goal | Output | MVP? |
|---|---|---|---|
| 0 | Schema + contracts | `dialogue_segments` table, types | ✅ |
| 1 | Phrase substrate + **cut-point snapping** (the craft) | `l1/dialogue_segments.py` | ✅ |
| 2 | Sentence segmentation | sentence clips | ✅ |
| 3 | Topic segmentation (heuristic) | topic clips + `topic_id` | next |
| 4 | Pipeline wiring (new idempotent stage) | runs on new uploads | ✅ |
| 5 | API endpoint | serve segments per file/folder | ✅ |
| 6 | Frontend Dialogues tab + Sentence/Topic switch | the lens UI | ✅ |
| 7 | Drop-to-timeline | add select(s) to a sequence | ✅ (basic) |
| 8 | Trust/polish | flags, badges, filler toggle, nudge, search | partial |

### Phase 0 — Schema & contracts
- Migration `017_dialogue_segments.sql`: `dialogue_segments(file_id pk, schema_version, segments jsonb, created_at)` — store both levels in one document (`{sentence:[...], topic:[...]}`), upsert on `file_id` (mirrors `clip_perception`).
- Backend pydantic `DialogueSegment`; frontend `DialogueSegment` type in `api.ts`.

### Phase 1 — Phrase substrate + cut-point snapping  ← spend 80% of effort here
- Build phrases from `transcripts.segments` (gap>250ms or speaker change = boundary).
- Load the WAV (available in the L1 speech track) → compute **fine RMS @ 10ms hop**.
- For each boundary:
  - **IN:** search `[raw_in-150ms, raw_in+60ms]` for the local RMS minimum (silence trough); `src_in = trough`; clamp so it never crosses the previous clip's `raw_out`.
  - **OUT:** search `[raw_out-60ms, raw_out+200ms]`; `src_out = trough`; clamp before next `raw_in`.
  - Default handles ~80–120ms folded into the trough search; fades 10–20ms.
- **Overlap:** adjacent different-speaker spans that overlap → flag `overlap` on both, shrink handles so they don't bleed.
- **Noisy:** if no clear trough (RMS never dips below `speech_ref - X`) → flag `noisy`, fall back to word-timing + fixed handles.
- Verify: cuts land in silence; no clipped words on spot-checked clips.

### Phase 2 — Sentence segmentation
- Merge same-speaker phrases: gap ≤ `G_sentence` (~350ms) AND not sentence-final punctuation; cap `L_max` (~12s), floor `L_min` (~1.2s, merge forward).
- Strip leading/trailing fillers (move `raw_in/out` to first/last non-filler word).
- Emit `level="sentence"` segments via Phase-1 snapping.

### Phase 3 — Topic segmentation (heuristic; LLM upgrade later)
- Group sentences of one speaker; start a new topic on: long pause (>~1.2s), OR discourse marker at sentence start ("so/okay/anyway/now/next/another thing"), OR the answerer changes.
- **Backchannel bridging:** the other speaker's very short utterance (<~1.0s or in a backchannel lexicon) does NOT end the topic; it becomes its own tiny `backchannel`-flagged clip (or is excluded), and the main speaker stays continuous.
- Assign `topic_id`; sentence segments inherit it (accordion). Flag `low_confidence` liberally — heuristic topics are approximate.

### Phase 4 — Pipeline wiring
- New stage `dialogue_segments` in `STAGES`, run at the END of the speech track (WAV + words + speakers in hand; no dependency on the audio track).
- Idempotent via `processing_jobs`; best-effort (failure never breaks L1).
- New uploads get it automatically; existing files need a one-shot backfill (defer, like framing).

### Phase 5 — API
- `GET /files/{file_id}/dialogues` → `{sentence:[...], topic:[...]}` (served from the stored doc; switch is instant, zero recompute).
- Folder view: frontend fetches per ready file; aggregate client-side (simplest MVP).

### Phase 6 — Frontend Dialogues tab
- Add `Dialogues` to `TABS` (drive page) + a `DialoguesView`.
- Segmented **Sentence | Topic** toggle at the top → swaps the served list in place (no tabs, no scroll-dup). Topic tile expands to child sentences (accordion).
- Tile: speaker badge, text, duration, flags; hover-scrub preview; play.
- Search box filters tiles by transcript text.

### Phase 7 — Drop-to-timeline
- Map `DialogueSegment → EditSegment` (`in_ms`, `out_ms`, `content`, `file_id`).
- "Add to timeline" / drag → append to a sequence (reuse `edit-doc-store`); multi-select to add several (and adjacent two-speaker pairs).
- Export (EDL/FCPXML/Premiere XML with markers) → later, but it's the real "drop into my NLE" payoff.

### Phase 8 — Trust & polish
- "Why/flags" badges (overlap/noisy/false-start/trails-off) + confidence.
- Toggles: strip fillers (default on), tighten internal dead-air (off).
- Manual nudge handles on `src_in/out`; false-start detection (repeated phrase stems).

---

## Brutal risks / decisions
- **Coarse stored `rms_db` is not enough** → compute fine RMS at stage time. Non-negotiable for clean cuts.
- **Topic is heuristic first** → will be imperfect; flag confidence, plan a cheap LLM topic-shift pass as the upgrade.
- **Overlap/cross-talk has no pretty answer** → flag honestly, let the editor choose.
- **Whisper word-END drift** → never cut on word timestamps; always snap to audio.
- **Music/noise under speech** → snapping degrades; flag `noisy`, fall back gracefully.

## Suggested MVP slice
Phases 0–2 + 4–7 with **Sentence only** (snapped cuts, tab, tiles, add-to-timeline).
Add Topic (Phase 3) immediately after. This ships a genuinely useful lens fast,
then layers the harder semantic level on top.
