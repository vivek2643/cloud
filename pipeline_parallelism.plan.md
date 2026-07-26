# Pipeline parallelism & latency plan — target ≈1 min per 10-min video

## 0. Goal & honest target

**Goal:** drive the *complete* pipeline (L1 analysis + L3 ingest) for a ~10-min
multicam project from today's **~6-7 min** down toward **~1 min** wall-clock,
purely by exposing parallelism — no quality regressions.

**Honest read on "under 1 min":** achievable for single-file / short clips and
for GPU-backed multicam *if* we fix the serial bottlenecks below. For a 4-camera
10-min shoot the hard floors are (a) Whisper transcription, (b) the Pass 1 Sonnet
call, and (c) the Pass 2 vision calls. With the changes here those overlap and
run on GPU, so a realistic landing is **~60-120 s**; sub-60 s needs GPU + high
worker concurrency + a faster/streamed Pass 1. This plan gets us there and marks
exactly which knob buys which second.

**Latency budget (target, 10-min / 4-cam, GPU box):**

| Phase                    | Today (serial-ish) | Target (parallel) | How |
|--------------------------|--------------------|-------------------|-----|
| L1 across 4 files        | ~4× per-file (serial on 1 GPU) | ~1× per-file | cross-file fan-out + CPU/GPU split |
| L1 per file (speech)     | Whisper→diar→dlg serial | max(Whisper, diar) | overlap Whisper ∥ diarization |
| L3 pass1 (Sonnet)        | ~15-30 s | ~15-30 s (floor) | faster model / stream (optional) |
| L3 frame extract         | serial before pass2 | overlapped w/ pass2 | pipeline extract→pass2 + proxy cache |
| L3 pass2 (vision)        | 4-wide | 8-12 wide | bump concurrency + cache prefix |
| L3 identity              | **already deterministic** (no LLM) | off critical path | — banked; see note below |
| L3 post + heroes         | serial per file | parallel + cached proxy | bump concurrency, reuse proxy |

**Banked already — the old ~7-min pole is gone.** When this plan was first drafted,
the single biggest win was moving the ~7-min identity **LLM** pass off the critical
path. That work has since shipped: identity is now **deterministic** — L1 active-speaker
(insightface + ASD) plus code (`identity/faces.py`, `bind_asd`, voice assignment from
diarization embeddings), **no LLM call**. It runs after Pass 2 but is cheap and does
no model inference. So this plan now covers *everything else*, and the re-ranked ROI
in §8 reflects that removal (the Option B step is done, not pending).

---

## 1. Measure first — instrumentation (Phase 0, do before tuning)

We cannot tune what we cannot see. Today L1 has `processing_jobs.started_at/
finished_at` per stage, but L3 has only coarse `status` transitions.

- **L3 stage timings:** wrap each stage in `ingest.run_ingest` (pass1, image_plan,
  extract, pass2, identity, post, heroes) with a monotonic timer; accumulate into
  a `timings_ms` dict and persist on the `ingest_run` row (new jsonb column, or
  fold into an existing metadata jsonb — no migration if one exists).
- **Per-batch / per-call timings:** record wall-time of each Pass 2 batch future
  and each model call, so we see straggler batches and rate-limit stalls.
- **A one-line summary log** at run end: `pass1=Xs extract=Xs pass2=Xs(max batch=Ys)
  identity=Xs post=Xs total=Xs`. This is the scoreboard every later phase is judged
  against.
- **L1 report:** a small query over `processing_jobs` timestamps → per-file,
  per-stage durations, so we can see whether Whisper or motion is the pole.

Deliverable: a `scripts/timing_report.py` that prints the per-stage breakdown for
a run id / project id. Everything below is prioritized by what this reveals.

---

## 2. The three axes of parallelism

1. **Across files (L1):** every file is an independent `l1_orchestrate` task.
2. **Across stages (overlap):** stages with no data dependency can run at once
   (e.g. Whisper ∥ diarization; frame-extract ∥ pass2; voice-id ∥ pass2).
3. **Within a stage:** batch/shard concurrency (Pass 2 batches, hero extraction,
   clip extraction) + model-call fan-out.

The plan attacks all three, ordered by ROI in §8.

---

## 3. Current critical path (what's actually serial)

```
UPLOAD
  └─ per file: l1_orchestrate  (queue "gpu", WORKER_CONCURRENCY=1)
        proxy ─┐
               ├─ TRACK speech:  transcript(Whisper) → diarization(pyannote) → dialogue   ← SERIAL chain, the pole
               ├─ TRACK audio:   audio_features(librosa)                                   (CPU)
               └─ TRACK motion:  motion → scene → color                                    (CPU)
        (3 tracks ∥, but speech chain is internally serial)

  ⇒ On a 1-GPU box, files run ONE AT A TIME (GPU_WORKERS=1, concurrency=1).
     A 4-cam project pays 4× the per-file L1 cost back-to-back.

L3 ingest  (queue "ingest", INGEST_CONCURRENCY=6 across projects; WITHIN a project:)
   pass1(Sonnet) → v4_segment(per-file CPU) → image_plan → extract_frames → pass2(4-wide) →
   identity(faces + bind_asd + voice-assign  ← deterministic, NO LLM, cheap) →
   post(assemble) → heroes(per-file R2 download + ffmpeg, serial)
```

**Two hidden serial killers:**
1. **Cross-file L1 is serial** on a single-GPU box (`GPU_WORKERS=1`,
   `WORKER_CONCURRENCY=1`). A 4-file project = 4× per-file L1.
2. **Whisper → diarization is serial** inside the speech track, but pyannote
   diarization does **not** need Whisper's words — only the word-labeling *merge*
   does. Two long poles run back-to-back for no reason.

---

## 4. L1 wins

### 4.1 Overlap Whisper ∥ diarization (per-file, no infra change) — HIGH ROI
`_track_speech` runs `transcript → diarization → dialogue_segments` serially. But
`pyannote` produces speaker turns from the **audio alone**; only the step that
writes a `speaker` onto each transcript word needs both.

- Split into: run **Whisper** and **pyannote diarization** as two concurrent
  sub-tasks off the same WAV; then a cheap **merge** step assigns each word its
  speaker (interval overlap) + writes `speaker_embeddings`; then `dialogue_segments`.
- Saves ~`min(Whisper, diarization)` per file — often 30-50% of the speech track.
- Keep it inside `_track_speech` with a 2-worker pool; both release the GIL.
- Idempotency unchanged (each still records its `processing_jobs` stage).

### 4.2 Cross-file L1 fan-out — HIGH ROI (multicam)
Today files serialize behind one GPU worker. Options (pick per deployment):
- **Multi-GPU box:** `GPU_WORKERS=NUM_GPUS` already fans files across GPUs — make
  sure the ingest box actually has >1 GPU for multicam, or
- **Raise GPU-queue concurrency carefully:** Whisper (CTranslate2) + pyannote can
  time-share one GPU if VRAM allows; bump `WORKER_CONCURRENCY` for the gpu queue
  to 2-3 with a VRAM guard, or
- **Split GPU-bound vs CPU-bound L1 stages into separate queues** (see 4.3) so the
  CPU stages of *all* files run in parallel on CPU workers while the GPU handles
  Whisper/diarization — this de-serializes ~half of L1 without more GPUs.

### 4.3 Split L1 into GPU-bound and CPU-bound tasks — MEDIUM ROI, structural
`motion_dynamics`, `scene_detect`, `color_stats` (OpenCV) and `audio_features`
(librosa) are **pure CPU** — they don't need the GPU at all, yet today they ride
the same single-GPU worker and block the next file.

- Emit them as their own tasks on the **`cpu` queue** (fan out across CPU workers),
  keyed by `file_id`, so every file's CPU stages run concurrently with every other
  file's GPU stages. `dialogue_segments`/voice clustering join on their inputs.
- Keeps the GPU busy only with Whisper + diarization (its actual job).
- Requires an orchestration tweak (l1_orchestrate fans out sub-tasks + a join, or
  a small DAG) — the idempotent `processing_jobs` model already supports partial
  completion, so this is safe.

### 4.4 Whisper throughput — MEDIUM ROI
- Confirm `faster-whisper` runs on GPU (`torch_device()=="cuda"`) with a sane
  `beam_size`/`batch_size`; batched inference + chunked long-audio parallelism can
  cut transcription 2-4×. Tune, don't rewrite.

---

## 5. L3 wins

### 5.1 Shared per-project proxy cache — HIGH ROI, easy
Right now the **same proxies are downloaded from R2 multiple times**: Pass 2 frame
extraction (`frames.extract_for_planned_frames`), Option B clip extraction, and
hero-frame extraction (`_extract_and_upload_heroes`) each `_download_from_r2` the
same files independently.

- Add a small **project-scoped proxy cache** (download each `proxy_key` once to a
  temp dir, hand the local path to every consumer). One GET per file per run
  instead of 3-4.
- Cuts I/O latency across extract + identity + heroes, and makes those stages
  cheaper to run concurrently.

### 5.2 Overlap frame extraction with Pass 2 — MEDIUM ROI
Today: extract **all** frames, then start Pass 2. Instead, extract per-batch and
submit each batch as soon as its frames are ready (producer/consumer): batch N's
model call runs while batch N+1's frames are still extracting. Extraction is
I/O+ffmpeg, Pass 2 is network — they overlap cleanly.

### 5.3 Bump Pass 2 batch concurrency — MEDIUM ROI
`MAX_PARALLEL_PASS2_BATCHES = 4` is conservative (inherited from the old split
passes). With Flash-Lite + a cached prompt prefix, push to **8-12**, bounded by
Gemini RPM/TPM quota. Add explicit rate-limit backoff (we already have empty-retry
in `ingest_gemini`). Gate behind a config knob so it's tunable per quota.

### 5.4 Identity pass — DONE (was the ~7-min pole)
The old ~7-min identity **LLM** burst pass has already been replaced by a
deterministic path: L1 active-speaker (insightface/ASD) + `identity/faces.py`,
`bind_asd`, and voice assignment from diarization embeddings. **No model call, no
critical-path cost.** Nothing to do here except (a) make sure it reads from the
shared proxy cache (§5.1) rather than re-downloading, and (b) since it's now cheap
and deterministic, it can even run *concurrently with post-processing* rather than
strictly before it, if instrumentation (§1) shows it still costs measurable seconds.

### 5.4b Fan out the deterministic per-file L3 loops — MEDIUM ROI, easy
Several L3 steps loop over files serially with **no shared state** — pure CPU/DB,
trivially parallel with a bounded `ThreadPoolExecutor`:
- `_load_signals` (`ingest.py`) — per-file `build_l1_snapshot` DB reads.
- `v4_segment.segment_video` (`ingest.py`) — per-file deterministic segmentation
  (motion/scene/audio → video cuts); today a serial per-file loop between pass1 and
  image_plan. On a multicam project this is N× serial CPU for no reason.
- `extract_stills` within a file (`frames.py`) — per-timestamp ffmpeg subprocesses.
Cap each pool and share the §6 global ffmpeg semaphore so they don't thrash.

### 5.5 Parallelize hero-frame extraction — LOW/MEDIUM ROI
`_extract_and_upload_heroes` loops files serially. It's embarrassingly parallel
(per-file R2 + ffmpeg + upload) — run it in a `ThreadPoolExecutor` (like
`frames.extract_for_planned_frames` already does) and reuse the §5.1 proxy cache.

### 5.6 Pass 1 latency (the floor) — OPTIONAL
Pass 1 is one whole-project Sonnet call; everything downstream waits on it. It
can't be split (it's global semantic grouping). Levers, only if it dominates the
budget after the above: (a) a faster Pass 1 model, (b) stream its output so
image_plan can begin on early cuts, (c) prompt/token trimming. Treat as a floor,
not a quick win.

---

## 6. Worker & queue topology

- **Ingest concurrency:** `INGEST_CONCURRENCY=6` governs *cross-project* overlap.
  Fine for throughput; single-project latency is governed by §5, not this.
- **New `cpu`-queue L1 stages (§4.3):** ensure `CPU_WORKERS` is sized to soak them
  (bump default from 2 → e.g. 4-6 on a big box).
- **GPU queue:** decide the multicam story — more GPUs, or `WORKER_CONCURRENCY=2-3`
  with a VRAM guard, or CPU-stage offload (§4.3). Document the chosen posture in
  `run_workers.sh`.
- **Bounded pools everywhere:** every new `ThreadPoolExecutor` (extract, clips,
  heroes) must have a max_workers cap and, where they contend for R2/ffmpeg, share
  a global semaphore so parallel stages don't thrash disk/network.

---

## 7. Knobs table

| Knob | Location | Today | Proposed | Risk |
|------|----------|-------|----------|------|
| `WORKER_CONCURRENCY` (gpu) | `run_workers.sh` | 1 | 1 (multi-GPU) or 2-3 w/ VRAM guard | VRAM OOM |
| `GPU_WORKERS` | `run_workers.sh` | =#GPUs | =#GPUs (ensure >1 for multicam) | cost |
| `CPU_WORKERS` | `run_workers.sh` | 2 | 4-6 (soak CPU L1 stages) | CPU saturation |
| `INGEST_CONCURRENCY` | `run_workers.sh` | 6 | 6-12 (cross-project) | API quota |
| `MAX_PARALLEL_PASS2_BATCHES` | `pass2_params.py` | 4 | 8-12 | Gemini RPM/TPM |
| speech track | `l1/pipeline.py` | serial | Whisper ∥ diar + merge | correctness of merge |
| CPU L1 stages | `l1/pipeline.py` | on gpu worker | own `cpu` tasks | orchestration complexity |
| proxy download | L3 extract/identity/heroes | N× per file | 1× per file (cache) | temp-disk usage |
| frame extract vs pass2 | `ingest.py` | serial | pipelined | scheduling complexity |
| hero extract | `ingest.py` | serial | thread pool | R2/ffmpeg contention |

---

## 8. Rollout — ordered by ROI (each independently shippable)

1. **Phase 0 — instrumentation (§1).** Ship first; it validates every later phase.
2. **Whisper ∥ diarization (§4.1).** Big per-file win, no infra change.
3. **Shared proxy cache (§5.1).** Easy, cuts I/O across the L3 extract/identity/hero stages.
4. **Cross-file L1 de-serialization (§4.2/§4.3).** Structural; biggest multicam win
   now that the identity pole is already gone.
5. **Bump Pass 2 concurrency + rate-limit backoff (§5.3).**
6. **Fan out deterministic L3 per-file loops (§5.4b).** v4_segment + signal-load +
   stills; cheap CPU wins on multicam.
7. **Pipeline extract→pass2 + parallel heroes (§5.2/§5.5).** Polish.
8. **Pass 1 latency (§5.6).** Only if it still dominates.

(The old "Option B / voice-id ∥ Pass 2" step is removed — identity is already
deterministic and off the critical path; see §5.4.)

---

## 9. Ideal parallel pipeline (target shape)

```
UPLOAD (N files)
  ├─ file1 ─┐
  ├─ file2 ─┤  L1 fans across GPUs/CPU workers:
  ├─ file3 ─┤    GPU: Whisper ∥ pyannote → merge → dialogue
  └─ file4 ─┘    CPU: audio_features ∥ (motion→scene→color)   (all files at once)
        │
   (all L1 done)
        │
   L3:  pass1(Sonnet)  ── floor ──
        │
        └── v4_segment (per-file ∥) → image_plan
        │
        ├──────────── extract frames ⇄ PASS 2 (8-12 wide, pipelined) ────────────┐
        │                                                                          │
   JOIN → identity(faces+bind_asd+voice, deterministic) → post(assemble)
        │                                → heroes(parallel, cached proxy)
        │
   READY
```

Shared across all L3 extraction: **one proxy cache**, **one bounded ffmpeg pool**.

---

## 10. Risks & guardrails

- **VRAM:** any GPU-concurrency bump needs a memory guard; Whisper + pyannote
  co-resident can OOM. Prefer CPU-stage offload (§4.3) before raising GPU
  concurrency.
- **API rate limits:** higher Pass 2 concurrency can hit Gemini RPM/TPM — add
  backoff + a concurrency cap that reads from config, and keep the cached prefix.
- **ffmpeg/R2 thrash:** parallel extraction stages must share a bounded pool +
  proxy cache, or disk/network contention erases the gains.
- **Correctness of the Whisper∥diar merge:** the word→speaker assignment must be
  exactly what the serial path produced (interval-overlap on the same words);
  cover with a unit test on a fixed transcript+turns fixture.
- **Idempotency:** the split CPU/GPU L1 tasks must each keep their `processing_jobs`
  stage semantics so retries still skip finished work.
- **Measure each phase against Phase 0 numbers** — ship only what the scoreboard
  says actually moved wall-clock.
```
