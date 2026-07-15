# Plan: give the brain the transcript per cut + tune it toward transcript

## Goal (keep it simple)

The editing brain should see, on **every cut**, **both** the visual summary
(already there) **and** the verbatim transcript (missing today). Then just
**tune its preference toward the transcript** for choosing dialogue. This is a
**rendering-only** change to what the beat line shows — surfacing data that
already exists on `cut_records`/moments.

**Hard constraints (from the user):**
- **No cut-level changes.** Do not touch pass1/pass2/post, cut boundaries,
  identity, on_camera, quality scoring, or what any cut *contains*. Only change
  what `footage_map` *renders* to the brain, and one prompt line.
- **Do not drop the visual summary.** Every cut keeps its visual `label`/`summary`.
  We only reorder (transcript first for speech) and nudge via the prompt.
- **Don't over-provide.** Add only the few fields that genuinely help *choose
  dialogue/takes*; leave the rest in `inspect_moment`/Tier-1.

## Why (grounded findings — verified in code)

- The brain reads the **BEAT INDEX**: one line per cut from
  `footage_map._moment_line` (`backend/app/services/l3/footage_map.py`, ~L737).
- The quoted text on a line is `gist = m["gist"] = label`; `summary` renders as a
  secondary `graphic:"…"` tag. **Both `label` and `summary` are pass-2 VISION
  outputs**, not the transcript (`pass2.py` ~L504-512: `label` = "name what the
  cut SHOWS", `summary` = "describe WHAT IS HAPPENING"). So the brain sees a
  paraphrase and **zero verbatim dialogue** — the root cause of content-guessy edits.
- `total_quality` shows on the line as `q.XX` (in PIC), but **`speech_quality`
  (delivery-only, camera-independent) is on the moment struct and NOT rendered.**

## The transcript source already exists — reuse it

`footage_map._span_detail(file_id, in_ms, out_ms)` (~L988-1024) already pulls the
verbatim transcript window for a span from `dialogue_segments`
(`segments["sentence"]`, sentence granularity, selected by time overlap). It is
used only by `moment_detail` (Tier-1, on demand). **Reuse this exact query/shape**
for the resident line. Do NOT reconstruct text from `cut_records.word_span`.

## What to surface (grounded field audit)

Everything below already exists on the moment/`cut_records`; only rendering changes.

**ADD to the beat line (clear wins):**
- **Verbatim transcript** — every speech beat (`channel == "said"`). The core add.
- **`speech_quality`** — delivery crispness+loudness, 0..1, None if no speech.
  Camera-independent, so it's the same across an outlook group's angles → a clean
  "is this line featureable / which take sounds best" signal. Only the *blended*
  `total_quality` is shown today; surface `speech_quality` as its own short tag
  (e.g. `aud:0.72`, or a `low-audio` flag when it's below some render-time cutoff —
  a display threshold, not a cut change).

**MAYBE (only if one short tag each, else skip):**
- `natural_sound` (bool) — does a video/B-roll cut carry usable ambient audio.
- `junk_confidence` — lets the brain rescue a borderline junk beat.

**DO NOT add to the resident line (keep in `inspect_moment`/Tier-1):**
- `characteristics` (per-person appearance fingerprints), `taste_fences`,
  `readability_ms`, `look`, `framing.subject_box`, `hero_ts_ms`. None help *choose
  dialogue*; adding them is the bloat the user warned against. `salience` peak
  offset already shows as `peak:+X.Xs` — its magnitude is not worth adding.

## Changes

### 1. Attach per-beat verbatim text to the moment (batched, one DB read per file)

In `footage_map.build_clip_tree` (~L265-353), where a file's moments are built:
- Factor the `dialogue_segments` fetch out of `_span_detail` into a cached
  `_sentences_for_file(file_id) -> List[dict]` (call it from `_span_detail` too —
  no behavior change there). Load **once per file**, not per moment.
- For each **speech** moment (`channel == "said"`), join the sentences overlapping
  `[in_ms, out_ms]` (same `_ov` test `_span_detail` uses) into one string; store
  as `m["said_text"]`. Prefix speaker labels only when the beat spans >1 speaker
  (post speaker-change-split, usually one → just the words). Leave absent for
  non-speech beats. Use the pinned run's file transcript (keyed by time, so no
  run-drift concern).

### 2. Render transcript-first on speech lines — KEEP the visual summary

In `footage_map._moment_line` (~L737-788):
- For a speech beat with non-empty `m["said_text"]`: put the **verbatim quote
  first** as the line's primary content, then keep the existing visual gist as a
  short secondary note (e.g. `vis:"…"` via `_short_gist`). **Do not remove it.**
- Keep `summary → graphic:"…"` behavior unchanged (info-dense graphics).
- Non-speech beats: **unchanged** (visual label/summary stays primary; no
  `said_text`).
- Modes: in `compact`/paged mode truncate `said_text` like today's gist (`>80` →
  `…`) — the full text is still reachable via `moment_detail`/`_span_detail`. In
  resident mode show it in full (it's short).
- Leave PIC/SND/on-cam/`alt-PIC`/nrg/pace/`peak`/outlook tags exactly as they are.

### 3. Surface `speech_quality`

In `_moment_line`, for speech beats, render `speech_quality` as a short tag
(e.g. `aud:{speech_quality:.2f}`), alongside the existing `q.XX` total. It's
already on the moment (`m["speech_quality"]`, carried in `build_clip_tree`
~L308). Rendering only.

### 4. Tell the brain the line changed (prompt — one line)

In `converse.py`, the "READING A BEAT LINE" guidance (~L74-86): add one line —
on a speech beat the quoted text is now the **verbatim words spoken**; choose
*dialogue* by reading it, and use the visual note (`vis:"…"`/PIC) for *what's on
screen / how it looks*. `aud:` is the delivery quality of the line. Keep it short.

## Testing

Add to `backend/scripts/test_footage_map.py` (which already tests line rendering,
e.g. `test_said_beat_on_listener_camera_reads_pic_first_with_alt_pic`):
1. Speech beat with known `dialogue_segments` → line contains the **verbatim
   text** AND still contains the visual note (both present); transcript first.
2. Speech beat → line contains the `aud:`/`speech_quality` tag.
3. Action beat (`done`/`shown`) → visual label/summary still primary, no
   `said_text` leaks in.
4. `compact=True` long line → truncates, full text still via `moment_detail`.
5. Speech beat with no `dialogue_segments` row → graceful fallback to today's
   visual gist, no crash.

Run `python backend/scripts/test_footage_map.py`; also run `test_pass1.py` /
`test_pass2.py` for no-regression (neither should be touched).

## Non-goals / guardrails (do NOT do these)

- No new agent tool (no `read_transcript`, no word-level `place`/`trim`).
- No highlight-scoring pre-pass, no embeddings, no extra LLM/vision calls.
- No `word_span` reconstruction; use the `dialogue_segments` sentence-overlap path.
- **No cut-level changes** — pass1/pass2/post/boundaries/identity untouched.
- **Do not drop the visual summary** — every cut keeps it; we only reorder + nudge.
- Do not add the "keep in Tier-1" fields to the resident line.
- Do not dump the whole-file transcript into context — stays per-beat (paging /
  `inspect_moment` already handles long footage).

## Edge cases

- **Junk beats**: keep the terse one-liner (short-circuit at `_moment_line` top
  ~L743); no transcript.
- **Recovered speech cuts** ("(recovered)"): still get `said_text` (a feature —
  the brain sees what was recovered).
- **Multi-speaker leftover in a beat**: `said_text` = sentences in span; prefix
  speakers only when >1 present.
- **Missing `dialogue_segments`** (older files): fall back to the visual gist.
- **`speech_quality` None** (no speech / pre-migration): omit the `aud:` tag.

## Touch list

- `backend/app/services/l3/footage_map.py`
  - add `_sentences_for_file()` cache; refactor `_span_detail` to use it
  - `build_clip_tree`: attach `m["said_text"]`
  - `_moment_line`: transcript-first render (keep visual note) + `aud:` tag
- `backend/app/services/l3/converse.py`: one-line "READING A BEAT LINE" update
- `backend/scripts/test_footage_map.py`: tests above
