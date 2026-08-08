# Speech Cuts Pipeline — Implementation Plan

**Status:** FINAL PLAN — ready to hand to an implementing chat.
**Branch:** build on **`local-dev-isolation`** (where all seam/vcut work lives, uncommitted). Do **not** branch off `main`.
**Depends on:** the vcut pipeline (`backend/app/services/vcut/`), the seam layer, and the vcut Pass-2 cache/question-bank machinery (all implemented). **Assumes the approved L1 change — capturing Whisper confidence into `transcripts.segments` — is already done** (per-word `probability` and/or per-segment `avg_logprob`/`no_speech_prob`). Treat missing confidence as neutral for older files.

---

## 0. What this replaces

Today the vcut pipeline handles the **non-speech** video channel and, for the **speech** channel, just copies a prior run's `kind='speech'` rows verbatim (`store.copy_prior_speech_cuts` — a stopgap). This plan makes vcut **own its speech channel**: a clean, transcript-first speech-cut pass that forms speech cuts, detects takes vs. outlooks, and picks winners — writing `kind='speech'` cut_records into the same vcut `ingest_run`. `copy_prior_speech_cuts` is retired (kept only as a degraded fallback if the speech pass fails).

---

## 1. One-paragraph summary

For each project we load the word-timed, diarized transcript (+ Whisper confidence), silence/RMS, face tracks, and audio-sync groups — all already in L1. **Audio-sync collapses camera angles ("outlooks") first**, so each spoken performance is one *take instance* regardless of how many cameras shot it. A **Gemini-pro text call** then segments the transcript into coherent spoken beats (as *word-index ranges*, never timestamps), clusters *retakes* into take groups, and returns a rough fluency score + hard flags per take. **Deterministic code** snaps those word ranges to exact ms (word timings + silence + a breath pad), computes a per-take **delivery score** from audio (energy, dynamics, pace, hesitation, ASR confidence), and picks the **winner** per take group (flags gate, scores fuse). The winning take's **outlooks** are kept as coverage with the sync system's **authoritative (hero) audio** routed in. A **minimal frame pass** (1 frame, 2 for long+changing on-camera cuts, 0 for voiceover) enriches each cut visually and labels angle types — reusing the vcut Pass-2 cache + closed question bank. Everything lands as `kind='speech'` cut_records; the existing "Best takes" filter already consumes `take_role='winner'`.

---

## 2. Principles / non-goals

1. **Speech is text-first.** The model decides *which words / beats / takes* and emits **word indices**; deterministic code decides the *exact ms*. Never let the LLM emit timestamps (the #1 way speech cuts go wrong).
2. **Audio owns take-vs-outlook.** Same audio across files = outlook (angle); re-spoken line = retake. This is decided by the existing audio-sync system, **not** vision.
3. **Don't touch L1** beyond the already-approved Whisper-confidence capture (assumed done). Everything here is post-L1.
4. **Don't touch the old l3 pipeline.** Reuse only shared data-access + the vcut cache/bank machinery.
5. **Reuse, don't reinvent:** audio-sync (`sync_groups`), face tracks (`best_crop_ms`), the cut_records take/sync/audio columns, and the vcut Pass-2 cached-frame + closed-question-bank flow.
6. **Cost-first on frames:** ~1 frame per on-camera speech cut, none for voiceover, all batched into one flash-lite call.
7. **Separate pipeline, shared sink:** new code under `backend/app/services/vcut/speech/`; output is `kind='speech'` cut_records in the same `ingest_run`.

---

## 3. Inputs (all already persisted in L1)

| signal | source | used for |
|--------|--------|----------|
| word-timed diarized transcript (`words[]` with `start_ms`/`end_ms`/`speaker`) | `transcripts.segments` | segmentation, boundaries, pace |
| **Whisper confidence** (word `probability` / seg `avg_logprob`) | `transcripts.segments` (approved L1 add) | delivery: clarity/articulation |
| silence / inter-word gaps | derivable from word timings (+ L1 gap logic) | boundary snapping, hesitation |
| RMS envelope (`rms_db`) | `audio_features` | delivery: energy + dynamics |
| face tracks (boxes, `best_crop_ms`, embeddings, speaking) | `face_tracks` | frame pick, on-camera, visual delivery |
| audio-sync groups + authoritative audio | `sync_groups` / `sync_group_members` | outlooks + hero audio |

Read these with direct minimal SELECTs (same pattern `vcut/spans.py` uses for `transcripts`), keeping vcut isolated from l3 business logic. The one gray area is reusing the sync **offset math** — see §12/§18.

---

## 4. The take/outlook model (two orthogonal axes)

Two different groupings, each already a cut_records column:

| | what it is | grouped by | detected by | picked how |
|---|---|---|---|---|
| **Outlook (angle)** | one take, multiple cameras | `sync_group_id` | **same audio** (audio-sync) | keep all (coverage); hero **audio** = authoritative; default picture = best-framed angle |
| **Take (retake)** | the line performed again | `take_group_id` / `take_role` | **different audio, same words** (LLM) | pick **one winner** (delivery+fluency); rest = `alternate` |

**Ordering that keeps them clean — collapse angles first:**

```
1. audio-sync groups angles → collapse each sync group to ONE take instance
   (one transcript timeline per take; its angle files attached)
2. Gemini-pro: segment beats (word-index) + cluster retakes into take groups
   + rough fluency score + incomplete/false_start flags
3. deterministic: snap word ranges → ms; delivery score per take; pick winner
4. attach the winner's outlooks (angles) + route the hero audio
```

Because angles are collapsed before step 2, the LLM never mistakes a second camera for a retake.

---

## 5. Module layout (all new, under `vcut/speech/`)

```
backend/app/services/vcut/speech/
  __init__.py
  params.py         # delivery weights, breath pad, frame thresholds, model ids
  inputs.py         # load transcript/words/confidence/silence/rms/face_tracks/sync per project
  outlooks.py       # audio-sync grouping + authoritative(hero) audio + collapse to take instances
  segment_llm.py    # ONE Gemini-pro call: word-index beats + take clusters + fluency + flags
  boundaries.py     # PURE: word-index range → exact ms (silence + breath pad)
  delivery.py       # PURE: per-take delivery metrics + group-normalization
  select.py         # PURE: winner selection (gate + fuse + argmax)
  frames.py         # pick relevant cuts, extract 1-2 frames, batched flash-lite analysis
  store.py          # build kind='speech' CutRecords (word_span/take/sync/audio/visual) + insert
  orchestrate.py    # run_speech_channel(project_id, ingest_run_id) — called by vcut_ingest
backend/scripts/
  test_speech_boundaries.py   # PURE unit tests (word→ms, breath, no phoneme clipping)
  test_speech_delivery.py     # PURE unit tests (metrics + group-normalization)
  test_speech_select.py       # PURE unit tests (gate/fuse/argmax, winner + alternates)
  test_speech_outlooks.py     # angle collapse + hero-audio selection
```

`boundaries.py`, `delivery.py`, `select.py` are **pure** (no I/O) so the winner is reproducible, tunable, and re-derivable without re-calling the LLM.

---

## 6. Stage 1 — inputs (`inputs.py`)
Per project: for every file with a transcript, load words (`start_ms`/`end_ms`/`speaker`/`probability`), `rms_db`, silence/gaps, and `face_tracks`. Load `sync_groups`/`sync_group_members` for the project. Files with no transcript contribute no speech cuts (they're the non-speech channel's domain).

## 7. Stage 2 — outlooks + hero audio (`outlooks.py`)
- Group files by `sync_group_id` (same audio). Each group = the same performance from N cameras.
- **Hero audio** = the group's `authoritative_audio_file_id` (already chosen by the sync system). Every cut on any angle routes audio to it via `audio_file_id` + `audio_offset_ms` (offset from `sync_group_members`).
- **Collapse to take instances:** pick one representative timeline per group (the authoritative-audio file's transcript) so downstream segmentation/clustering runs once per take, not once per angle. Keep the member list so §12 can fan a winning take back out to all angles.
- No sync group (the common single-camera case) → the file is its own take instance, no hero-audio routing.

## 8. Stage 3 — the speech LLM (`segment_llm.py`, Gemini pro)
**One text-only call** (model = `vcut_speech_model`, default `gemini-3.1-pro-preview` per the "not sonnet, use a pro model" decision) over the collapsed transcripts. Input: numbered, diarized words per take instance. Output (word indices only):

```json
{
  "beats": [                         // coherent spoken beats worth keeping
    { "file_id": "...", "word_span": [i, j], "gist": "one short phrase" }
  ],
  "take_groups": [                   // retakes of the SAME line, across beats/files
    { "beat_ids": [..], "fluency": {beat_id: 0..1}, "flags": {beat_id: ["incomplete"|"false_start"]} }
  ]
}
```

Rules in the prompt:
- Emit **word indices**, never timestamps.
- A `take_group` clusters beats that are attempts at the *same content*. A beat with no sibling is its own singleton group (trivially the winner).
- `fluency` = completeness + clean phrasing + right words (text-judgable). `flags` = `incomplete` (didn't finish the line) / `false_start` (restart/stumble) — these are **gates**, not scores.
- Validate word spans exist and don't overlap within a file (reuse the old pipeline's validation *idea*, reimplemented locally).

## 9. Stage 4 — deterministic boundaries (`boundaries.py`, pure)
Turn each beat's `word_span [i,j]` into exact ms:
- `in_ms = words[i].start_ms`, `out_ms = words[j].end_ms`.
- **Breath pad:** extend `in_ms` earlier and `out_ms` later into adjacent **silence** (up to `BREATH_PAD_MS`), never into the neighboring word — so no phoneme is clipped and the cut breathes.
- Snap edges to the cleanest silence point in the gap (reuse the silence/gap data; conceptually the old `lattice.snap_word_edge`, reimplemented simply).
- Emit `word_span` alongside ms (cut_records stores both; the frontend transcript-sync needs the word span).

## 10. Stage 5 — delivery scoring + winner (`delivery.py` + `select.py`, pure)
**Per-take delivery metrics** over the beat's word span (group-normalized *within* each take group — best-of-these, not an absolute bar):
- `energy` — median `rms_db`.
- `dynamics` — std of `rms_db` (expressiveness).
- `pace` — words/sec scored against a natural band (`PACE_LO`..`PACE_HI`), penalizing rushed/dragging.
- `hesitation` — total intra-span gap >350ms (fewer/shorter = better).
- `asr_confidence` — mean word `probability` (clarity/articulation; neutral if absent).
- *(optional, free)* `visual_delivery` — a coarse facing-camera/framing cue from the one on-camera frame (§11), only if present.

**Winner (per take group):**
```
gate      = 0 if any hard flag (incomplete/false_start) else 1
delivery  = w_e·energy + w_d·dynamics + w_p·pace + w_h·(1−hesitation)
            + w_c·asr_confidence + w_v·visual_delivery        (group-normalized terms)
final     = gate · ( w_fluency·fluency_llm + w_delivery·delivery )
winner    = argmax over the group ; others → take_role="alternate"
```
Flags are multiplicative **gates**, scores are additive **attractors** — same shape as the seam curve. Weights live in `params.py` (tone-biasable later). Because everything is pure + the LLM output is persisted, **winners can be re-picked without re-calling the model** (retune weights → instant).

## 11. Stage 6 — minimal frame analysis (`frames.py`)
Only where vision pays; keep it cheap:
- **Which cuts:** on-camera speech cuts, take candidates (for visual delivery), and one frame per **angle** in a sync group (to label `angle_type`). **Voiceover / off-camera → 0 frames.**
- **How many:** **1 frame** at `face_tracks.best_crop_ms` (clearest face) by default; **2** only when the cut is **long AND visually changing** (duration > `SPEECH_FRAME_LONG_MS` *and* an interior `composition_point`/`shot_point` or real motion). Hard cap 2. When 2, place the second at the detected change, not start/end.
- **One batched flash-lite call**, reusing the vcut Pass-2 **cached-frames + closed question bank** machinery + a speech-only field `angle_type ∈ {wide, medium, close, ots, other}` and `on_camera`.
- **Outputs → cut_records:** `on_camera`, `framing`/`shot_size`, `angle_type`, and `scene_specifics` (subject/action/setting/on_screen_text). The same on-camera frame feeds `visual_delivery` back into §10 at no extra cost.

## 12. Stage 7 — emit cut_records (`store.py`, `kind='speech'`)
Per emitted speech cut, build a `CutRecord` and insert via the existing `ingest_store.insert_cut_records`:
- `kind="speech"`, `channel="said"`, `word_span`, `src_in_ms`/`src_out_ms` (from §9), `label`/`summary` from the beat gist, `speech_quality` = the fused score, `hero_ts_ms` = `best_crop_ms` or beat midpoint.
- **Take axis:** `take_group_id` (stable per cluster) + `take_role` (`winner`/`alternate`).
- **Outlook axis:** for a winning take with sync angles, **emit one cut per angle**, all sharing `sync_group_id`, with `audio_file_id` = authoritative (hero) audio and `audio_offset_ms`/`audio_align_confidence` from the sync member — picture stays the angle, audio is the hero. (These columns already exist and `audio_route` already consumes them at arrange time.)
- **Visual fields** from §11.

**Fallback:** if the speech pass fails wholesale, fall back to `copy_prior_speech_cuts` (today's behavior) so a run never ends up speech-less.

## 13. Orchestration (`orchestrate.py`)
- Add `run_speech_channel(project_id, ingest_run_id)` and call it from `run_vcut_ingest` **in place of** `copy_prior_speech_cuts`, after the video cuts are inserted (same run). Speech LLM is text-only and fast; it can run alongside the video work.
- The speech frame call is its own batched flash-lite call (separate from the video Pass-2 cache, since speech frames are a small, different set).
- Fail-open: a speech-channel exception logs + falls back (§12), never fails the whole run (video cuts already inserted).

## 14. Frontend touchpoints (minimal)
- **None for rendering** — `kind='speech'` cut_records already render; the **"Best takes"** filter already shows `take_role='winner'`.
- **Angle switching** (nice-to-have, not required for v1): the timeline can offer alternate angles via `sync_group_id`; `audio_route` already keeps the hero audio. Ship the data now, wire the toggle later.

## 15. Data model / config
- **No new migration** — every field used (`word_span`, `take_group_id`, `take_role`, `sync_group_id`, `audio_file_id`, `audio_offset_ms`, `audio_align_confidence`, `framing`, `on_camera`, `screen_text`, `scene_specifics`) already exists on `cut_records`. `angle_type` rides `scene_specifics` (additive JSON).
- **Config:** `vcut_speech_model: str = "gemini-3.1-pro-preview"`; speech frame model = `vcut_pass2_model` (flash-lite). Everything else (delivery weights, `PACE_LO/HI`, `BREATH_PAD_MS`, `SPEECH_FRAME_LONG_MS`) in `speech/params.py`.

## 16. Testing
- **Pure units** (no I/O): `boundaries` (word→ms, breath pad never clips a neighbor word, snaps to silence), `delivery` (metrics + group-normalization + neutral-confidence path), `select` (flag gates, fuse, argmax winner + alternates, singleton take = winner).
- **`test_speech_outlooks.py`** — angle collapse to one take instance; hero-audio = authoritative; single-camera = self group.
- **Frames** — voiceover/off-camera → 0 frames; long+changing on-camera → 2, else 1; batched call shape.
- **Live smoke** (isolated `dev` schema/R2): a multi-take talking-head clip (winner picked sensibly, alternates marked) and, if available, a 2-camera synced clip (angles kept, hero audio routed). Confirm `scene_specifics`/`angle_type` on-camera, none on VO.

## 17. Sequencing
1. `inputs.py` + `params.py`.
2. **Pure core first:** `boundaries.py`, `delivery.py`, `select.py` + their tests (lock the algorithm before any I/O).
3. `outlooks.py` (audio-sync collapse + hero audio) + test.
4. `segment_llm.py` (Gemini-pro beats + take clusters + fluency + flags).
5. `store.py` (kind='speech' CutRecords incl. take/sync/audio wiring).
6. `frames.py` (minimal frame analysis, reuse vcut bank/flash-lite).
7. `orchestrate.run_speech_channel` wired into `run_vcut_ingest` (replace `copy_prior_speech_cuts`, keep it as fallback).
8. Tests green repo-wide + pyflakes; live smoke on isolated dev tables.

Nothing merges to `main`; nothing flips `cuts_pipeline`. Validated behind the flag on `local-dev-isolation`.

## 18. Open decisions
- **Sync reuse boundary** — read `sync_groups`/`sync_group_members` directly (data access, in keeping with vcut isolation) vs. importing `l3.sync.store`/`audio_route` (business logic). Lean: read tables directly, reimplement the small offset mapping; only import if it gets fiddly.
- **False-start / mid-line filler removal** — v1 flag-only (gate the take) vs. also trimming the false-start sub-span (jump cut). Default: gate only in v1.
- **Does energy act on speech?** — v1: speech cuts are always clean (breath-padded), energy is video-only. Revisit if you want a "tighten pauses/fillers" speech dial.
- **Cross-file retakes** — the LLM can cluster takes across files (multiple recordings); confirm that's desired vs. within-file only.

## 19. Risks
- **Winner subjectivity / tone** — "best" depends on intended tone; keep fluency and delivery as separate sub-scores with tunable weights, and normalize *within* the take group. Don't bake one "good delivery" definition in.
- **Emphasis-on-the-right-word nuance** — a linear delivery score can't capture prosody-meaning alignment; accepted v2 gap.
- **Sync false negatives** — if audio-sync misses a genuine multicam pair (no overlapping audio), those angles look like separate takes; vision-based angle disambiguation is a documented fallback, not v1.
- **Boundary precision is where quality lives** — most "speech cuts feel off" complaints are breath/silence snapping, not the model. Invest test coverage there.
