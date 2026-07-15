# ASD-based Identity Layer (plan)

Replace the fragile, metered **Gemini video "is this person speaking?" pass**
with a deterministic, local **Active-Speaker-Detection (ASD)** signal, and move
person identity from **noisy LLM appearance labels** to **face embeddings**.
Keep the cast table (global IDs + a few human characteristics + associated
voices); make on/off-camera a pure code lookup.

> Executor note: this is a design brief for a fresh chat. Read the "Current
> state" and "Files" sections first; they cite exact modules/functions as they
> exist today. Nothing here is committed.

---

## 1. Why (the problem, restated)

Two independent questions per speech beat:

1. **Who is the voice?** (voice → global person)
2. **Is that person the one visibly speaking in the shown angle?** (on-camera
   vs off-camera)

Today #2 is answered by an LLM eyeballing short **inline video clips**
(`identity/voice_id.py` → Gemini). That is the mis-tooled, expensive, slow,
stochastic piece: `no_face`, batching degradation, ties, and the ~7-minute
identity step all live there. #1's *face* half (`identity/reconcile.py`)
clusters **LLM categorical appearance labels** ("beard: yes", "shirt: blue"),
which over-splits (12 "persons" for a 2-person podcast) because the labels flip
run-to-run.

Both are solved CV problems:

- **ASD** (SyncNet → TalkNet → **Light-ASD**): given video+audio, detect+track
  each face and emit a per-face **speaking timeline** by correlating lip motion
  with the audio. Scores each face independently → in a two-shot it says *which*
  face is talking. Deterministic, cheap, local, generic.
- **Face embeddings** (ArcFace/InsightFace): robust machine identity for
  clustering the same person across cuts/clips. CPU-capable via onnxruntime.

Characteristics ("red-shirt guy") stay — but as **human reference labels on the
cast table**, never as the clustering key. Embeddings decide identity;
characteristics describe it.

## 2. Cost / GPU stance

- The deleted voice-id pass is **metered API $** (inline video to Gemini). ASD
  replaces it with **local compute** you already run in L1 (Whisper +
  diarization already need GPU/heavy CPU). Net: removes an API line + the 7-min
  step; adds marginal compute to a pass you already pay for.
- ASD runs at **L1, per file, persisted, reused across ingests/projects** (same
  memoization every other L1 signal gets) — not per-project, not on Pass 2's
  critical path.
- Face detect/embed (**insightface + onnxruntime**) run on **CPU**. Light-ASD
  runs on the existing `torch` (GPU if present, CPU otherwise). Keep the
  `ml_device` pattern (`_gpu_available()` / `torch_device()`), CPU fallback.

## 3. Target architecture

### L1 (new, per file, on the 1080p proxy — video+audio)

New module `l1/active_speaker.py` producing, per file, a list of **face tracks**:

```
FaceTrack = {
  track_id: int,                     # local to this file
  embedding: list[float],            # mean ArcFace embedding over the track
  frames: [{ t_ms: int, box: [x,y,w,h] }],   # sampled (e.g. 5 fps), proxy px space
  speaking: [{ start_ms, end_ms, score }],   # ASD-positive intervals (merged)
  best_crop_ms: int,                 # timestamp of the sharpest/largest crop
                                     # (for the per-person characterization call)
}
```

Pipeline inside the pass (all local):
1. **Detect + track** faces (SCRFD via insightface; IoU tracker) on the 1080p
   proxy at ~5–10 fps → face tracks.
2. **Embed** each track (ArcFace, mean over the track's crops).
3. **ASD** over each track's crop sequence vs the proxy audio → per-window
   speaking score → merged speaking intervals. **Ship the DETERMINISTIC
   AV-correlation proxy first** (decision): correlate **mouth-region motion**
   (lower third of each face crop) against the **audio envelope**, scored **per
   face track independently** — no external model/weights to vendor, testable
   now. This is NOT the old whole-frame "who moves more" motion-bind (that's why
   it's per-track + mouth-region + audio-correlated). A trained model
   (Light-ASD/TalkNet) is a later fidelity UPGRADE that drops in behind the
   identical `FaceTrack.speaking` seam once we have labeled footage to validate
   it — do not vendor it now.
4. Persist per file (see migration §5). Best-effort/fail-open: no faces → empty
   list, never a hard fail (same contract as diarization).

**Where it runs:** a dedicated procrastinate task `l1_active_speaker(file_id)`
on the `gpu` queue, idempotent via a `processing_jobs` stage `"active_speaker"`,
reading the canonical **`proxies/{file_id}/proxy.mp4`** from R2 (it carries
video+audio and real fps; the client A/B proxies do **not** — proxy_b is
160×90, proxy_a is 1 fps, both unusable for lip-sync). Enqueue it once the
editing proxy exists (chain off `l1_editing_proxy`, or enqueue alongside it and
let the stage no-op/retry until the proxy is present). Do **not** try to fold it
into `_track_motion` — that track runs on proxy_b in the client path.

> Alternative if L1 wiring proves heavy: run the same pass in L3 ingest reading
> the 1080p proxy (like today's Track B), parallel to Pass 2. Loses cross-ingest
> memoization; keep L1 as the default.

### L3 ingest (identity resolution — mostly code)

- **`identity/faces.py`** (new; replaces `identity/reconcile.py`'s LLM-label
  clustering): load all files' face-track embeddings, cluster cross-file by
  cosine (conservative, over-split-safe — mirror `identity/voices.py`'s stance)
  → global persons **P0..Pn**; map `(file_id, track_id) → Px`.
- **visible_persons per cut** (in `faces.py` or `apply.py`): for each cut's
  resolved ms span in its file, the set of Px whose track has a `frames[]` box
  present during the span. Deterministic, from tracks — replaces the
  Pass2-`people`-derived `visible_persons`. (Cap to top-N by mean face area if a
  crowd; reuse the spirit of `CROWD_SIZE`.)
- **`identity/bind_asd.py`** (new; replaces `identity/voice_id.py` entirely):
  for each `(file_id, local_speaker)` diarization turn set, intersect with face
  tracks' `speaking` intervals → the track (→ Px) most consistently ASD-speaking
  during that speaker's turns. Aggregate to the global voice via `voice_of`
  (from `identity/voices.py`, unchanged). Majority+margin (port the
  `MIN_VOTE_SHARE`/`MIN_VOTE_MARGIN` contract) → `owner_by_voice: {V: Px|None}`.
  A voice whose turns never coincide with any ASD-speaking visible track →
  off-camera.
- **`identity/apply.py`** (rewrite `run`): keep `_rewrite_cuts` semantics as just
  fixed — `speaker_person = owner_by_voice[voice_ids[0]]` (dominant voice only,
  no fallthrough); `on_camera = speaker_person in visible_persons(cut)`. Inputs
  now come from `faces.py` + `bind_asd.py` instead of `reconcile` + `voice_id`.
- **Person characterization is DE-SCOPED for now.** No `characterize.py`, no
  dedicated call. The cast table carries only the deterministic globals — person
  id + `owned_voices` (§4).
- **Be precise about what a cut carries as identity.** The per-cut GLOBAL
  identity signal is `visible_persons` = `[P0, P1, ...]`, derived from
  face-track embedding clustering, WITH each person's position/box read straight
  from the track (render e.g. `PIC: P0(left) P1(right)`). That is deterministic
  and is the "same id for same person in every cut" guarantee.
- Pass 2's `people[]` free-text descriptions are a SEPARATE, parallel list and
  are **NOT tagged with a P-id** — there is no reliable, cheap way to bind a
  free-text description to a track/P-id per cut (that's exactly the box-matching
  heuristic we reject). So descriptions are ambient context only; identity and
  angle choice ride entirely on `visible_persons` / `speaker_person` /
  on-off-camera, none of which need a description. Do NOT wire per-cut
  descriptions into identity.
- (If a cast-table label is ever wanted later, the ONLY clean source is:
  backfill a person's display from cuts where EXACTLY ONE person is visible --
  there the lone description unambiguously is that P-id, no box-matching.
  Deferred; do not build now.)

### Pass 2 (left as-is for now)

- **Leave `people[]` in Pass 2 untouched.** Its structured `appearance`
  categories simply stop being an identity signal (embeddings cluster now); its
  free-text `description` + `position` keep flowing as the per-cut human read the
  brain associates to P-ids. Slimming Pass 2 (dropping the now-unused `appearance`
  categories / `PersonLook`) is a valid later cleanup but is NOT part of this
  change — don't couple it in.
- `identity/reconcile.py`'s LLM-label CLUSTERING is what's replaced (by
  `faces.py`), not Pass 2's descriptive output. `post.assemble_cut_records`
  keeps `characteristics=list(cut.people or [])` as the per-cut description; the
  new per-cut identity (`visible_persons` global IDs) rides alongside it.

## 4. Cast table (the deliverable)

Per global person, keep it **small and human**. Recommended fields (tune):

```
person_id: "P0"
display:   "man in red shirt, short dark hair, beard"   # one human phrase
tags: { apparent_gender, hair_or_facial_hair, top_color }   # 3 structured tags
owned_voices: ["V0"]        # from bind_asd (associated voices)
appearance_count / is_major # keep for ordering; cast stays UNCAPPED
```

3 structured tags is the sweet spot — enough for "the red-shirt guy" /
"the woman on the left", not an exhaustive roster. `top_color` (clothing) is the
strongest chat-reference signal within a project even though it's a weak
cross-shoot identity signal — which is fine, identity is the embedding's job.

`footage_map._cast_line` already renders `id (display) [voice:...]`; extend it to
surface the tags, and keep per-beat `PIC:Px` / `SND:Px ON-CAM|OFF-CAM` exactly
as now (those read `visible_persons`/`speaker_person`, which still exist).

## 5. Data model / migration (`041_asd_identity.sql`)

- New per-file storage for face tracks. Either a `face_tracks` table
  (`file_id` FK, `tracks jsonb`, `schema_version`) or a column on an existing
  L1 table. Keep it jsonb, best-effort, nullable — same shape as
  `transcripts.speaker_embeddings`.
- `processing_jobs` gains the `"active_speaker"` stage (no schema change; it's a
  row value).
- `cut_records`: `voice_ids`, `speaker_person`, `visible_persons` already exist
  (migration 039) — reused as-is. Consider dropping the now-unused per-cut
  `characteristics` column (or leave it, write `[]`).
- `identity_map` payload (persisted via `ingest_store.set_identity_map`): keep
  `{persons, voice_owner, off_camera_voices}`; `persons[]` now carries the §4
  fields (embedding-clustered id, tags, owned_voices) instead of the old
  `fingerprint`.

## 6. Delete list (once the above lands)

- `identity/voice_id.py` (VLM speaking pass).
- `identity/windows.py` (clip-window selection — only voice_id uses it; verify).
- `video_clips.py` (clip extraction + Haar face-crop — only Track B uses it;
  verify no other importer).
- `identity/reconcile.py` LLM-label clustering (replaced by `faces.py`); if the
  `Occurrence`/`Person` dataclasses are reused, migrate them, else delete.
- `config.py`: `identity_voice_id_model`, `identity_voice_id_thinking`.
- `llm/base.py::media_block` + the `"media"` branch in
  `gemini_client._parts_for_content` **iff** nothing else uses inline media
  (grep first).
- ingest.py: the whole "Track B" block (`_run_voice_id_track`, `select_clips`,
  `verdict_future`, the pool + its cleanup in `except`).
- Tests: `test_identity_speaker_pass.py`, `test_identity_speaker_frames.py`
  (already dead), voice_id tests, `video_clips` tests.

## 7. Wiring order in `ingest.run_ingest`

Replace the Track-A/Track-B split with:

1. pass 1 (unchanged) → `all_speakers_by_file`, `turns_by_file`.
2. `voice_of = identity_voices.assign_voices(...)` (unchanged).
3. image_plan → Pass 2 (now identity-free) → post batches (unchanged, minus
   people).
4. Load per-file face tracks (from L1). `faces.cluster(...)` → persons +
   `(file,track)→Px`; compute `visible_persons` per cut.
5. `bind_asd.bind(turns_by_file, voice_of, face_tracks, visible_persons)` →
   `owner_by_voice`, `off_camera_voices`.
6. `characterize.run(persons, face_tracks, proxies)` → cast descriptors.
7. `apply.run(pass2_output, lattices, voice_of, persons, visible_persons,
   owner_by_voice)` → rewritten cuts + identity_map payload.
8. `post.assemble_cut_records` (as today; on_camera still feeds total_quality,
   so identity must land before it — unchanged ordering requirement).

## 8. New dependencies

- `insightface` + `onnxruntime` (SCRFD detect + ArcFace embed; CPU wheels).
- **ASD: none to vendor for v1.** The deterministic AV-correlation proxy uses
  only opencv (already a dep) + the audio envelope — no external model/weights.
- (Later, behind the same `FaceTrack.speaking` seam: **Light-ASD** — small MIT
  torch model + weights on the existing `torch==2.6.0`, shipped in the worker
  image like the pyannote/HF flow; TalkNet-ASD the heavier fallback. Only after
  the proxy is validated and found insufficient on real footage.)

## 9. Fallbacks / failure modes (never hard-fail)

- No faces / tiny faces / all-low ASD → the cut's speaker resolves to **off-
  camera / unknown** (honest), never a guess. Same fail-open contract as
  diarization.
- Face-embedding cluster over-splits → conservative merge threshold; over-split
  is the safe direction (two fragments of one person just show as two cast rows;
  tune threshold, don't force merges).
- L1 face pass missing for a file (older ingest) → identity degrades to
  id-less PIC/SND, exactly like a run with no reconciled cast today.

## 10. Testing

- `l1/active_speaker`: unit-test the tracker/merge logic on a tiny synthetic
  (two boxes, one "speaking" interval); mock the models.
- `faces.cluster`: same-embedding → one person; distinct → two; over-split-safe.
- `bind_asd.bind`: turn ∩ speaking interval → correct owner; no overlap →
  off-camera; ambiguous (two tracks speaking) → unbound.
- `apply._rewrite_cuts`: dominant-voice attribution + `on_camera` lookup (extend
  existing `test_identity_apply.py`).
- End-to-end on the podcast (`57b689b3`): expect P0/P1 clustered from faces, V0→
  P0 / V1→P1 via ASD, correct ON-CAM/OFF-CAM on the ~1:00–1:30 stretch that
  motivated this.

## 11. Rollout

1. L1 `active_speaker` + migration + storage (dark; nothing reads it yet).
2. Re-run L1 for the podcast files to populate face tracks (GPU on AWS, like the
   diarization run).
3. Land `faces.py` + `bind_asd.py` + `apply.py` rewrite + `characterize.py`;
   slim Pass 2; delete Track B.
4. Re-ingest podcast → verify cast table + on/off-camera on the frontend.
5. Re-ingest the rest.

## 12. Open questions

- ASD engine: v1 is the deterministic AV-correlation proxy (DECIDED). Whether it
  needs to graduate to Light-ASD (vs TalkNet) is a post-validation call —
  benchmark the proxy on the podcast + one run-and-gun clip; only vendor a model
  if the proxy can't tell co-speaking faces apart.
- Characterization: dedicated per-person Gemini call (recommended, clean,
  ~2–5 calls/project) vs zero-new-call reuse of a slimmed Pass 2 still — decide
  based on whether Pass 2 keeps *any* people signal.
- Face-track sampling fps: 5 fps is likely enough for presence+embedding; ASD
  itself may want denser crops (Light-ASD trained ~25 fps) — the pass can sample
  ASD windows denser than the persisted `frames[]`.
- Cross-project person identity: embeddings make global identity across projects
  possible later; out of scope here (keep it per-ingest for now).
