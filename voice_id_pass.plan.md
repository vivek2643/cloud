# Voice-ID pass (Option B): cast-blind A/V lip-sync binding, parallel to Pass 2

## 0. TL;DR

Replace the brittle still-frame mouth-motion binding (`identity/speaker_frames.py`
micro-bursts + `identity/speaker_pass.py`) with **one cast-blind VLM pass over
short video+audio clips** that runs **parallel to Pass 2**, followed by a
**trivial post-reconcile code step** that maps clip verdicts to global persons.

- The VLM watches a few short clips (video **with audio**) per voice and answers,
  per clip: *is the person on screen the one actually speaking?* — a native
  audio-visual lip-sync judgment. A listener's clip self-rejects (lips don't
  match the audio), which is exactly the failure that left V0 unbound today.
- The pass needs **no cast table** — only the voice map, diarization turns, sync
  groups, and proxies, all available **before** Pass 2 — so it runs concurrently
  with Pass 2 for ~zero added wall-clock.
- Global identity (which `Px`) is resolved **after** reconcile in a few lines of
  code, by mapping each "speaking" clip to the person `reconcile` says is visible
  in that cut.
- `on_camera` derivation is unchanged (`speaker_person ∈ visible_persons`).

Net effect: **delete** two brittle modules, **add** one cleaner pass, make the
identity layer both more robust (audio-visual) and faster (parallel).

---

## 1. Why (motivation & evidence)

The current voice→face binding fails on multicam dialogue (podcasts, interviews):

- **Single-candidate silent stills → lone-face over-detection.** In a multicam
  podcast each camera frames one person, so `speaker_frames` produces
  single-candidate windows across different cameras. Asked "is this one mouth
  moving?" with no audio, the model rubber-stamps a listener too.
- **Observed (run `c210bff0`, podcast `57b689b3`):** for voice `V0`, the five
  windows shown were `[P2, P1, P1, P0, P0]` and the model named a mouther in
  **every** window → a 2‑2‑1 split → `aggregate_votes` correctly refused to bind
  → `V0` fell to off-camera. **P0's frames *were* sent** (2 windows, 5/5 frames);
  the problem was silent-still ambiguity + no same-instant comparison.
- **Per-mic audio energy does NOT disambiguate here** (validated): `V0` was louder
  on *every* file's mic (gaps mostly ≤1.5 dB) — the audio is effectively shared/
  ambient and confounded by speaking loudness. So a pure-audio shortcut is out for
  this footage.

**The fix:** give the model **video + audio** so lip-sync is native. Hearing the
voice while seeing the lips lets a listener clip self-reject and a talker clip
confidently bind — with no fragile per-frame heuristics.

---

## 2. Design — the two-part split

### Part A (slow, cast-blind, PARALLEL to Pass 2)
For each **global voice** `V`:
1. Pick a handful of `V`'s **clean solo turns** (reusing `clean_windows`).
2. Extract short **video+audio clips** (~3 s) around each, from the **outlook-group
   cameras** covering that moment.
3. One Gemini call per voice: *"For each clip, is the person on screen the one
   speaking this audio? `speaking` / `not_speaking` / `no_face`."*

This depends **only** on pre-Pass-2 data (voice map, turns, sync groups, proxies).
No faces, no `people[]`, no cast. → runs concurrently with Pass 2.

### Part B (instant, after reconcile)
For each voice, take its `speaking` clips; map each clip to the **global person**
`reconcile` says is visible in that cut (`visible_persons`); tally votes →
`owner_by_voice`. Then derive `on_camera = owner(voice) ∈ visible_persons(cut)`,
exactly as `apply.py` does today.

### Sequencing

```
pass1
  └─ voice clustering (voice_of)      ─┐  (both available BEFORE pass2)
  └─ outlook groups + sync offsets    ─┘
        │
        ├───────────── Track A (parallel) ─────────────┐
        │   image_plan → extract frames → PASS 2 → reconcile (global cast + visible_persons)
        │
        └───────────── Track B (parallel) ─────────────┐
            voice_id.select_clips → extract clips (video+audio) → cast-blind VLM pass → per-clip verdicts
        │
   JOIN (both tracks done)
        │
   apply.bind:  verdicts + reconcile.visible_persons  ->  owner_by_voice
        │
   _rewrite_cuts (speaker_person, on_camera)  ->  post.assemble_cut_records
```

---

## 3. Components

### 3.1 NEW `backend/app/services/l3/identity/voice_id.py`

The whole Part A + the Part B aggregation.

**Data structures**
```python
@dataclass
class ClipRequest:
    voice: str
    clip_id: str          # e.g. "V0:c0:1aedb093"
    file_id: str
    start_ms: int         # in THIS file's own clock
    end_ms: int

@dataclass
class ClipVerdict:
    voice: str
    clip_id: str
    file_id: str
    verdict: str          # "speaking" | "not_speaking" | "no_face"
    center_ms: int        # clip midpoint in file clock, for cut lookup in Part B
```

**`select_clips(turns_by_file, voice_of, groups, *, k=K) -> List[ClipRequest]`**
- Reuse `speaker_frames.voice_turns` + `clean_windows` to get each voice's clean
  solo turns.
- Keep the **top-K by length** (longest = most articulation to read).
- For each kept turn, compute a clip window of `~CLIP_MS` centred on the turn's
  loudness peak (`_loudness_peak_ms`) — or the midpoint if no audio — clamped
  inside the turn.
- **Fan out across group cameras:** for each outlook-group member covering that
  moment, emit a `ClipRequest` with the window mapped into that member's own
  clock via the group's sync offset. Single-cam project → just the one file.
- Cap `MAX_CLIPS_PER_VOICE` (e.g. 8) so a chatty voice can't explode the plan.

**`run_voice_id_pass(clip_requests, clips_b64, *, model) -> List[ClipVerdict]`**
- Group requests by voice; one Gemini call per voice.
- Build blocks: for each clip, a `text_block(f"CLIP {clip_id}:")` + a
  `media_block(clip_b64, "video/mp4")` (see §3.3). Skip clips with no extracted
  bytes (never send blank).
- Structured output `VoiceIdOutput{verdicts: [{clip_id, verdict}]}` via the same
  `ingest_gemini.complete_gemini` path Pass 2 uses.
- Map each returned verdict back to its `ClipRequest` (carry `file_id`,
  `center_ms`). Missing verdicts default to `no_face`.

**`bind_from_verdicts(verdicts, visible_persons, cuts_by_file, lattices) -> (owner_by_voice, off_camera_voices)`**
(Part B — runs after reconcile.)
- For each voice, keep clips with `verdict == "speaking"`.
- For each such clip, find the cut in `file_id` whose resolved span covers
  `center_ms` (reuse `speaker_frames._cut_span_ms` / lattice span resolve) and
  read its `visible_persons[(file_id, source_ref)]` — usually one person. Add a
  vote per visible person (single-face clip ⇒ one vote).
- Tally votes per voice. Bind the winner iff `top/opinionated ≥ MIN_VOTE_SHARE`
  (0.5) **and** `top - second ≥ MIN_VOTE_MARGIN` (1); else unbound (honest
  ignorance). Reuse the same majority+margin contract already proven in
  `speaker_pass.aggregate_votes` — port it here.
- `off_camera_voices` = voices with no `speaking` clip resolving a person, or
  unbound.

**Prompt (cast-blind)**
```
You are shown short CLIPS. In each clip you can HEAR one person speaking and you
SEE one person on camera. For each clip decide, from lip-sync (do the lips/jaw
articulate in time with the speech you hear):
  - "speaking"      the on-screen person is the one speaking this audio
  - "not_speaking"  the on-screen person is silent / listening / reacting
  - "no_face"       no mouth is clearly visible/legible to judge
Return exactly one verdict per clip, using the clip_id from its label. Never
guess; if you cannot read the mouth, use no_face.
```

**Constants** (top of module, tunable)
```python
K = 4                    # clean turns sampled per voice
CLIP_MS = 3000           # clip length (~3s); enough articulation for lip-sync
MAX_CLIPS_PER_VOICE = 8  # ceiling after camera fan-out
CLIP_WIDTH_PX = 512      # downscale to bound video tokens
MIN_VOTE_SHARE = 0.5
MIN_VOTE_MARGIN = 1
# GUARD_MS / MIN_WIN_MS reused from speaker_frames (or moved into a shared module).
```

### 3.2 NEW clip extractor — `backend/app/services/l3/video_clips.py`
Mirror `frames.py`'s R2-download-once + thread-pool pattern, but emit **video
clips with audio** instead of stills.

```python
def extract_clip(proxy_path, start_ms, end_ms, out_path, width=CLIP_WIDTH_PX) -> bool:
    # ffmpeg -ss <start> -to <end> -i proxy -vf scale=width:-2 -c:v libx264
    #        -preset veryfast -c:a aac -ac 1 -movflags +faststart out.mp4
    # Keep audio (mono is fine). Short + downscaled => small inline payload.

def extract_clips_from_r2(proxy_key, requests_for_file) -> Dict[str, str]:
    # one R2 GET per file, then one ffmpeg per clip; returns {clip_id: b64_mp4}

def extract_for_clip_requests(reqs, proxy_key_by_file) -> Dict[str, str]:
    # group by file, ThreadPoolExecutor (bounded), returns {clip_id: b64}
```
- Same fail-open contract as `frames.extract_for_planned_frames`: a file missing a
  proxy is skipped; its clips just won't appear.
- **Bound the thread pool** and coordinate with Pass 2's frame extraction so the
  two parallel tracks don't oversubscribe R2/ffmpeg (see §5).

### 3.3 LLM plumbing — `base.py` + `gemini_client.py`
Add a neutral media block so video/audio can ride the existing block pipeline:

`base.py`:
```python
def media_block(data_b64: str, media_type: str) -> Block:
    return {"type": "media", "data": data_b64, "media_type": media_type}
```
`gemini_client._parts_for_content`: add a branch
```python
elif btype == "media":
    raw = base64.b64decode(b["data"])
    parts.append(types.Part.from_bytes(data=raw, mime_type=b["media_type"]))
```
- Short (~3 s, 512px, mono-audio) clips are small enough for **inline bytes**.
  If a voice's clips exceed the request's inline limit (~20 MB), either split into
  multiple calls per voice or upload via the genai **Files API** — note as a
  fallback; not expected for this clip size.
- Confirm `ingest_gemini.complete_gemini` passes arbitrary blocks through to
  `gemini_client` and that structured output (`response_schema`) coexists with
  video parts. (It should — video is just another input Part.)

### 3.4 REFACTOR `backend/app/services/l3/identity/apply.py`
- `run(...)` **stops** calling `speaker_frames.plan_bursts` / `speaker_pass.bind_voices`.
- New inputs: the **clip verdicts** produced by Track B (passed in from
  `ingest.py`). It still owns reconcile + the final composition.
- New flow inside `run`:
  1. `occurrences = _occurrences_from_cuts(cuts)`; `face_result = reconcile.reconcile(...)`.
  2. `owner_by_voice, off_camera_voices = voice_id.bind_from_verdicts(verdicts, face_result["visible_persons"], cuts_by_file, lattices)`.
  3. `_owned_voices_by_person`, `_rewrite_cuts` — **unchanged**.
  4. Compose the same payload `{persons, voice_owner, off_camera_voices}`.
- Signature change: replace `(turns_by_file, audio_by_file, motion_by_file,
  proxy_key_by_file)` with `verdicts: List[voice_id.ClipVerdict]`. (Reconcile
  still needs `cuts`; span lookups need `lattices`.)

### 3.5 WIRE concurrency — `backend/app/services/l3/ingest.py`
- Already computes `voice_of`, `turns_by_file`, `groups`, `proxy_keys` before
  Pass 2 (lines ~205-210). Add sync **offsets** per group member if not already
  derivable from `groups` (needed to map clip windows across cameras).
- Build + launch Track B **before** the Pass 2 batch loop:
  ```python
  clip_reqs = voice_id.select_clips(turns_by_file, voice_of, groups)
  clips_b64 = video_clips.extract_for_clip_requests(clip_reqs, proxy_keys)  # I/O
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=1) as vid_pool:
      verdict_future = vid_pool.submit(
          voice_id.run_voice_id_pass, clip_reqs, clips_b64,
          model=settings.identity_voice_id_model)
      # ... existing Pass 2 batch loop runs here, concurrently ...
      verdicts = verdict_future.result()
  ```
  (Clip *extraction* can also be inside the future so ffmpeg overlaps Pass 2; keep
  the R2/ffmpeg worker budget bounded — see §5.)
- Pass `verdicts` into `apply.run(...)`.
- **Fail-open:** if Track B raises, log and pass `verdicts=[]` → every voice
  unbound → all off-camera. Never fail the ingest for a binding miss (matches the
  plan's "no fabrication, no hard fail on identity" stance).

### 3.6 DELETE / SLIM
- `identity/speaker_pass.py`: **delete** (`run_speaker_pass`, `build_speaker_pass_blocks`,
  the prompt, `bind_voices`; port `aggregate_votes`'s majority+margin logic into
  `voice_id.bind_from_verdicts`).
- `identity/speaker_frames.py`: **delete** `plan_bursts`, `_covering_candidates`,
  `_window_score`, `_burst_ts`, `_burst_offsets`, `Burst`, and the burst
  constants (`K`, `N`, `D_MS`). **KEEP** and reuse (move into `voice_id.py` or a
  small shared `identity/_windows.py`): `voice_turns`, `clean_windows`,
  `_largest_clean_subspan`, `_merge_intervals`, `_other_voice_turns_by_file`,
  `_loudness_peak_ms`, `_cut_span_ms`, `_value_at`.
- Remove the `IDENTITY_DEBUG` burst print from `speaker_pass`; add an equivalent
  `[IDDEBUG]` line in `voice_id` (per voice: clips sent + verdicts + resolved owner).

---

## 4. Config
`backend/app/config.py`:
- `identity_voice_id_model` — default a **video-capable** Gemini (Flash, not
  Flash-*Lite*, unless Lite is confirmed to accept inline video+audio).
- `identity_voice_id_thinking = "low"`.
- (Optional) surface `CLIP_MS`, `K`, `MAX_CLIPS_PER_VOICE` if you want them tunable
  without a code change; otherwise keep as module constants.

---

## 5. Concurrency correctness & resource budget
- **No data race:** Track B reads only immutable pre-Pass-2 inputs (`voice_of`,
  `turns_by_file`, `groups`, `proxy_keys`) and writes only its own local verdicts.
  Track A (Pass 2) reads its own frame set. They share nothing mutable.
- **R2 / ffmpeg oversubscription:** both tracks download proxies and run ffmpeg.
  Cap the combined worker count — e.g. give `video_clips.extract_for_clip_requests`
  a small pool (2-3) and keep `frames.MAX_PARALLEL_FILES` as is, or gate both
  behind one shared `Semaphore`. Proxies are the same files, so consider a small
  **local proxy cache** (download once, reuse for both frames and clips) — a nice
  bonus that also speeds up Pass 2 extraction.
- **Join before `apply`:** `verdict_future.result()` must complete before reconcile
  needs it. Reconcile itself can start as soon as Pass 2 finishes; the bind waits
  for both.

---

## 6. Data model / migrations
- **None required.** The persisted `identity_map` payload keeps its shape
  `{persons, voice_owner, off_camera_voices}`, and `cut_records`
  (`voice_ids`, `speaker_person`, `visible_persons`, `on_camera`) are unchanged.
- (Optional, debug-only) persist per-clip verdicts on the run for inspection —
  not needed for correctness; skip unless useful.

---

## 7. Backward compatibility
- Older ingest runs: untouched (identical payload shape); `footage_map` / brain
  reading unchanged.
- Single-cam projects: one clip per turn, still binds (or off-camera if the
  speaker never faces camera) — no regression.
- Projects with no voiceprints / no clean turns: Track B yields no verdicts →
  every voice off-camera, exactly the honest fallback today.

---

## 8. Testing
Unit:
- `select_clips`: reuses `clean_windows`; asserts top-K by length, clip window
  clamped inside the turn, camera fan-out across a group, `MAX_CLIPS_PER_VOICE`
  cap, single-cam yields one clip/turn.
- `bind_from_verdicts`: `speaking` votes → owner; `not_speaking`/`no_face` don't
  dilute; genuine conflict (two persons each named) → unbound; no `speaking` clip
  → off-camera; the majority+margin thresholds.
- `media_block` / `gemini_client`: a `"media"` block becomes a
  `Part.from_bytes(mime_type="video/mp4")`.
Integration:
- Mock `run_voice_id_pass` to return fixed verdicts; drive `apply.run` end-to-end
  and assert `voice_owner`, `off_camera_voices`, per-cut `speaker_person` /
  `on_camera`.
- Regression: a b-roll-only voice → off-camera; a single-cam talking-head → bound.
Delete/replace: `test_identity_speaker_pass.py` and the burst tests in
`test_identity_speaker_frames.py` (keep the `clean_windows` / interval tests).

---

## 9. Rollout & validation
1. Implement behind the existing ingest path (no flag needed — it's a swap), or a
   `settings.identity_voice_id_enabled` toggle if you want a safe fallback to the
   old path during bring-up.
2. Re-ingest the podcast (`57b689b3`) and confirm on run inspection:
   - both hosts' voices bind (`voice_owner` has 2 entries, not 1),
   - `speaker_person` populated on the V0-only cuts that were `None` before,
   - `on_camera` distribution sane.
3. Spot-check a single-cam project (no regression) and a b-roll-heavy one
   (voices correctly off-camera).
4. Confirm wall-clock: total L3 ingest should not grow materially vs today, since
   Track B overlaps Pass 2 (and the deleted burst pass no longer runs serially).

---

## 10. Open questions (decide during implementation)
- **Model tier for inline video+audio** — confirm Flash-Lite support; if not, run
  this one pass on Flash. Verify inline-bytes size limits for ~3 s / 512px clips.
- **Clip length** (2 s vs 3 s) and **K** (3 vs 4) — tune on the podcast; longer =
  better lip-sync read but more tokens.
- **Cross-camera clock mapping** — confirm the sync offset per group member is
  available at ingest time (from `sync_store` groups) to map clip windows; if the
  lattices are already re-based to the group's authoritative clock, the mapping
  may be identity within a group.
- **Multi-face frames** (crowded shots): current design votes for *all* visible
  persons of a `speaking` clip. If that proves ambiguous, add the Option-A variant
  (send candidate cast faces and ask *which* one) — defer until needed.
- **Shared proxy cache** across Track A/B — worth doing for speed, but optional.
