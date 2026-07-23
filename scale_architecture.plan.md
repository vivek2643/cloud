# Scale Architecture — code-level plan for fast, parallel, concurrency-safe analysis

## 0. Goal & scope

Make the analysis pipeline (L1 per-file + L3 ingest) **fast, parallel, and safe under
concurrency** at the code level, sized for **~100 users** (target burst: 5 users × 10
videos ≈ 50 concurrent files, design headroom to ~200), and architected so that **scaling
later = add workers + turn config knobs, never a rewrite**.

**In scope (this plan):** everything that is a *code/design* change — DB connection
architecture, bounded resource pools, proactive LLM rate limiting, idempotency, fairness,
observability, and the concurrency knobs.

**Out of scope (deliberately deferred):** *where/how* to deploy (RunPod vs AWS,
autoscaling, cold starts). This plan assumes "run more worker processes" is the scaling
lever and makes the code ready for that; the deployment substrate is a later decision.

**Companion plan:** the per-stage latency/parallelism *details* (Whisper∥diarization, the
GPU/CPU task split, the L3 fan-outs) live in `pipeline_parallelism.plan.md`. This plan
**owns the concurrency-safety architecture** and **sequences** both workstreams so they
compose. Where a step is fully specified there, this plan references it rather than
duplicating.

**Supabase is now Pro.** Pillar 1 below is rewritten around Pro + the transaction pooler,
with the connection knobs raised accordingly.

---

## Pillar 1 — DB connection architecture (do FIRST; unblocks all concurrency)

### The problem

~40 call sites open a **brand-new** `psycopg.connect(database_url)` per query (the `_pg()`
helper duplicated across every store/task module — `l3/*`, `render/tasks.py`,
`export/*`, `l1/pipeline.py`, `captions/resolver.py`, etc.). Under 50–200 concurrent jobs,
each running many queries, this stampedes the Postgres connection ceiling and the whole
fleet stalls with connection errors — **long before GPU/CPU is the bottleneck.**

### Supabase Pro connection facts (drives the knobs)

Limits are set by **compute tier**, not the plan name. Pro *plan* defaults to **Micro**
compute; raise the ceiling by scaling compute (a Pro-plan setting):

| Compute | Direct conns | Pooler max clients |
|---|---|---|
| Micro (Pro default) | 60 | 200 |
| Small | 90 | 400 |
| Medium | 120 | 600 |
| Large | 160 | 800 |
| XL | 240 | 1,000 |

Two connection routes, and they matter for correctness:

- **Transaction pooler (Supavisor, port `6543`)** — multiplexes *many* short-lived client
  connections onto a small backend pool. **Perfect for our autocommit business queries.**
  ⚠️ Transaction mode has **no session state**: it breaks `LISTEN/NOTIFY` and server-side
  **prepared statements**. psycopg3 prepares statements by default → must disable.
- **Session pooler (`5432`) / direct** — keeps session state; required by **Procrastinate**
  (uses `LISTEN/NOTIFY`) and the **migration runner**.

### The change

1. **New module `app/services/db.py`** — one process-global connection pool:
   - `psycopg_pool.ConnectionPool` over the **transaction pooler URL** (`6543`), opened
     lazily once per process, `min_size` small, `max_size = DB_POOL_MAX`.
   - Connections configured with **`prepare_threshold=None`** (disable server-side prepared
     statements — mandatory under transaction pooling) and `autocommit=True`.
   - Expose a context manager `with db.connection() as conn:` that borrows/returns from the
     pool (never opens a raw socket).
   - Optional `db.query(sql, params)` / `db.execute(sql, params)` convenience wrappers.

2. **Migrate every `_pg()` helper** (the ~40 sites) to borrow from `db.connection()`
   instead of `psycopg.connect(...)`. Keep each module's local `_pg`/helper name as a thin
   alias so call sites don't churn — just re-point the body at the shared pool. Row-factory
   variants (`dict_row`) become a small param on the helper.
   - **Exclude** the two session-state consumers: leave **Procrastinate's connector**
     (`jobs.py`) and the **migration runner** (`db_migrations.py` / `worker.py` schema
     check) on the **session pooler / direct** URL (`5432`).

3. **Config (`config.py` + `.env`):**
   - `database_url` → keep as the **session/direct** URL (Procrastinate + migrations).
   - Add `database_pool_url` → the **transaction pooler** URL (`...pooler.supabase.com:6543`)
     for `db.py`. Fall back to `database_url` if unset (dev).
   - Raise `DB_POOL_MAX` now that we're on Pro + pooling. Budget: keep **total backend
     connections across the whole fleet under ~40–80% of the tier's direct-connection
     cap** (Supabase's own guidance). E.g. Micro (60 direct): with the pooler in front,
     set the Supavisor **pool size** in the dashboard to ~40 and let `DB_POOL_MAX` per
     worker be ~5–10; the pooler absorbs the 200 client slots. Scale compute → raise both.

4. **Dashboard step (one-time, non-code):** set the Supavisor **pool size** in Database
   Settings to the budgeted backend count for the current compute tier. Documented here so
   the implementer flags it; not a code change.

### Why this is first

Every other parallelism win multiplies *connections in flight*. Pooling turns "200
concurrent jobs → 200×N sockets" into "200 jobs → bounded backend pool," which is the
precondition for safely raising every concurrency knob below.

---

## Pillar 2 — Parallelism workstream (make each run fast)

These are the latency wins; **full file-level detail is in `pipeline_parallelism.plan.md`**
(re-ranked there now that the ~7-min identity LLM pole is already removed). Listed here so
this plan is a complete checklist and so the rollout (§8) sequences them against Pillar 1:

- **L1: Whisper ∥ diarization** — pyannote needs only the WAV; run concurrently, merge
  word→speaker after. (`l1/pipeline.py` `_track_speech`.) Highest per-file win.
- **L1: split GPU-bound vs CPU-bound stages onto separate queues** — `motion/scene/color`
  (OpenCV) and `audio_features` (librosa) are pure CPU; emit them as `cpu`-queue tasks
  keyed by `file_id` so every file's CPU work runs in parallel with every file's GPU work.
  Fan-out + join. (`l1/pipeline.py`, `run_workers.sh` queues.)
- **L1: cross-file fan-out** — each file is already its own `l1_orchestrate`; parallelism =
  more workers on `gpu`/`cpu`. Keep the per-file task self-contained + idempotent so this
  scales linearly.
- **L3: shared per-project proxy cache** — extract, identity, and heroes each re-download
  the same proxies 3–4×; download each once to a temp dir and pass the local path around.
- **L3: fan out the deterministic per-file loops** — `_load_signals`, `v4_segment`, and
  per-timestamp `extract_stills`, bounded `ThreadPoolExecutor`.
- **L3: bump Pass 2 batch concurrency** — `MAX_PARALLEL_PASS2_BATCHES` 4 → 8–12 (see
  Pillar 4 for the rate-limit guardrail that makes this safe).
- **L3: pipeline extract → pass2** (submit each batch as its frames land) and
  **parallelize hero extraction**.
- **L3: Pass 1 is the floor** — one whole-project call; don't split (quality/cache
  tradeoff). Only revisit if instrumentation says it dominates.

---

## Pillar 3 — Bounded resource pools (so parallelism can't thrash a box)

Unbounded fan-out trades a connection storm for a CPU/disk/VRAM storm. Every parallel
stage draws from a **capped, process-global pool**:

- **`app/services/limits.py`** — module-global `BoundedSemaphore`s, all config-driven:
  - `FFMPEG_CONCURRENCY` — every ffmpeg subprocess (proxy, demux, motion decode, stills,
    heroes, render) acquires this before spawning. One knob prevents N parallel encodes
    from saturating CPU/IO.
  - `R2_CONCURRENCY` — bound simultaneous R2 GETs/PUTs so extract/identity/heroes don't
    saturate the network.
- Every new `ThreadPoolExecutor` (pass2, frame extract, heroes, CPU-L1 fan-out) has an
  explicit `max_workers` and shares these semaphores where it contends for ffmpeg/R2.
- **VRAM guard** for any GPU-queue concurrency bump (Whisper + pyannote co-resident can
  OOM) — prefer the CPU/GPU queue split (Pillar 2) over raising GPU `WORKER_CONCURRENCY`.

---

## Pillar 4 — Proactive LLM rate limiting (not just reactive retry)

`llm/client.py` already handles *transient* failures well — exponential backoff + full
jitter on `{408,409,429,500,502,503,504,529}` and Anthropic overload/rate-limit error
types. Keep it. What's missing for 100-user bursts is a **proactive** limiter so we respect
provider RPM/TPM *before* hitting the wall (else higher Pass 2 concurrency just becomes a
retry storm):

- A **shared concurrency/token limiter** around `client.complete()` (and the Gemini path in
  `ingest_gemini.py`): a config-driven semaphore (max in-flight LLM calls) and, ideally, a
  simple token-bucket for TPM. Keys: `INGEST_LLM_MAX_INFLIGHT`, per provider.
- This makes `MAX_PARALLEL_PASS2_BATCHES = 8–12` (Pillar 2) *safe* — the batches queue
  behind the limiter instead of overrunning quota.
- Keep the cached prompt prefix (Anthropic ephemeral cache / Gemini CachedContent) so
  higher concurrency doesn't blow up token cost.

---

## Pillar 5 — Idempotency & resumability (retries never corrupt or double-charge)

Foundations exist — extend the discipline to every new task:

- **L1**: the `processing_jobs` per-stage rows already let a retried `l1_orchestrate` skip
  finished stages. The Pillar 2 CPU/GPU task split **must** preserve per-stage
  `processing_jobs` semantics so a crashed fan-out resumes, not restarts.
- **L3**: `ingest_runs.status` gates coarse resume; keep pass/segment/post steps
  individually re-runnable (write-then-advance, dedup on write).
- **Dedup on write**: renders/exports already dedup via `resolved_hash`. Any new derived
  artifact keyed by a content hash.
- **Procrastinate** already gives durable jobs, backoff retries, and **single-writer
  locking per task name** (never two L1s for the same file) — lean on it; don't add ad-hoc
  background threads that escape it.

---

## Pillar 6 — Fairness / multi-tenancy (one user's burst can't starve others)

Target load is bursty and multi-user (5 × 10). FIFO on one queue lets a single 10-video
upload monopolize the fleet and tank everyone's p95:

- **Per-user fair scheduling** — partition or weight the `gpu`/`cpu`/`ingest` queues by
  user, or enqueue with a priority that round-robins across users, so N users' first
  videos all start before any user's tenth.
- **Cap per-user in-flight jobs** (config knob) so one account can't consume the entire
  fleet.
- Keep queues **resource-typed** (already: `gpu`/`ingest`/`render`/`export`; add `cpu`) so
  a burst of one work type doesn't block another.

---

## Pillar 7 — Observability baked into the code now

So that whenever/wherever we deploy, the metrics and alerts already exist (this is
`pipeline_parallelism.plan.md` Phase 0 — do it first within the parallelism workstream):

- **Per-stage timings** persisted: L1 already has `processing_jobs.started_at/finished_at`;
  add an L3 `timings_ms` dict on the `ingest_run` row + a one-line run scoreboard
  (`pass1=Xs extract=Xs pass2=Xs(max batch=Ys) post=Xs total=Xs`).
- **Correlation IDs** on every log line: `user_id / project_id / file_id / ingest_run_id`.
- **`scripts/timing_report.py`** — per-stage breakdown for a run/project id; the scoreboard
  every later tuning phase is judged against.
- **User-visible failure states** — a poisoned job fails loud (retry-capped) and surfaces
  status to the user, never silently retries forever or blocks a queue.

---

## 8. Config knobs (updated for Supabase Pro)

| Knob | Location | Today | Proposed | Bound by |
|---|---|---|---|---|
| `database_pool_url` (txn pooler `6543`) | `config.py`/`.env` | — (raw connect) | set | — |
| `DB_POOL_MAX` (per worker, business queries) | `db.py`/`.env` | n/a | 5–10 | tier direct cap |
| Supavisor pool size (backend conns) | Supabase dashboard | default | ~40% of direct cap (heavy PostgREST) else ~80% | compute tier |
| Procrastinate `DB_POOL_MAX` | `jobs.py` | 2 | 2 (keep tiny; session route) | direct cap |
| `WORKER_CONCURRENCY` (gpu) | `run_workers.sh` | 1 | 1 (+CPU split) or 2–3 w/ VRAM guard | VRAM |
| `CPU_WORKERS` | `run_workers.sh` | 2 | 4–6 (soak CPU-L1 + ffmpeg) | CPU |
| `INGEST_CONCURRENCY` | `run_workers.sh` | 6 | 6–12 (cross-project) | LLM quota |
| `MAX_PARALLEL_PASS2_BATCHES` | `pass2_params.py` | 4 | 8–12 | LLM limiter |
| `INGEST_LLM_MAX_INFLIGHT` | `llm/*`/`.env` | — | set (per provider) | provider RPM/TPM |
| `FFMPEG_CONCURRENCY` | `limits.py`/`.env` | — | = physical cores-ish | CPU/IO |
| `R2_CONCURRENCY` | `limits.py`/`.env` | — | set | network |
| per-user in-flight cap | fairness layer | — | set | fairness |

Every knob is config, not code — scaling is tuning, not rewriting.

---

## 9. Rollout — ordered, each independently shippable

1. **Pillar 1 — DB connection pool + transaction pooler routing.** Ship first; unblocks
   all concurrency. Verify: run a burst; connection count stays bounded (dashboard chart).
2. **Pillar 7 — instrumentation** (parallelism Phase 0). The scoreboard that judges the
   rest.
3. **Pillar 2 — Whisper ∥ diarization** (biggest per-file win, no infra change).
4. **Pillar 3 — bounded ffmpeg/R2 semaphores** (make the next fan-outs safe).
5. **Pillar 2 — shared proxy cache + deterministic L3 fan-outs.**
6. **Pillar 4 — proactive LLM limiter**, then **bump Pass 2 concurrency**.
7. **Pillar 2 — CPU/GPU L1 queue split + cross-file fan-out** (structural; biggest multicam
   win). Requires `cpu` queue in `run_workers.sh` + per-stage idempotency (Pillar 5).
8. **Pillar 6 — fairness** (per-user scheduling + in-flight caps).
9. **Pillar 2 — pipeline extract→pass2 + parallel heroes** (polish).

---

## 10. Guardrails

- **Transaction pooler correctness:** business-query pool MUST disable prepared statements
  (`prepare_threshold=None`); Procrastinate + migrations MUST stay on the session/direct
  route (they need `LISTEN/NOTIFY`). Getting this wrong = subtle runtime errors under load.
- **Connection budget:** keep total backend connections under the tier's cap (Supabase's
  40–80% rule); the pooler absorbs client fan-out, not the backend.
- **Bounded everything:** no unbounded `ThreadPoolExecutor` or ffmpeg fan-out — always a
  cap + shared semaphore, or parallelism just moves the bottleneck to disk/VRAM/quota.
- **Idempotency invariants:** the CPU/GPU L1 split must keep `processing_jobs` stage
  semantics so retries skip finished work; no ad-hoc threads outside Procrastinate.
- **Correctness of Whisper∥diar merge:** word→speaker assignment must exactly match the
  serial path (interval overlap on the same words) — cover with a fixture unit test.
- **Measure each phase against the Pillar 7 scoreboard** — ship only what actually moves
  wall-clock / raises safe concurrency.
- **No quality regressions:** parallelism only reorders *when* work runs, never *what* it
  produces (Pass 1 stays one project-wide call; no silent fallbacks).
```
