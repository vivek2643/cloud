# Deployment Plan — Render + Vercel + RunPod (serverless GPU)

Status: PROPOSED (review before executing). Nothing here is implemented yet.

Goal: ship the app for a 100-user MVP with **push-to-`main` auto-deploy**, keeping
the expensive GPU L1 tier on **RunPod Serverless** (scale-to-zero, $0 idle,
burst autoscale) while everything else runs on cheap always-on hosts. No change
to analysis/brain *outputs* — only *where* the L1 compute physically runs.

---

## 0. Target topology

```
Browser ──HTTPS──> Vercel (Next.js frontend)
                        │  NEXT_PUBLIC_API_URL
                        ▼
                 Render: edso-api (FastAPI/uvicorn)  ──enqueue──┐
                        │                                        │
   ┌────────────────────┼────────────────────────────┐         ▼
   │                    │                              │   Supabase PG
   ▼                    ▼                              │  (Procrastinate
 Render:            Render:                            │   + business DB)
 edso-workers       edso-gpu-dispatcher                │
 queues:            queue: gpu                         ▼
 ingest,cpu,        pulls gpu jobs ──/runsync──> RunPod Serverless (GPU)
 l3,render,export                                  handler = real l1_* task fns
   │                                                     │
   └───────────────── R2 (media) ◄──────────────────────┘
```

| Component | Host | Type | Notes |
|---|---|---|---|
| Frontend (Next.js) | Vercel | — | root dir `frontend/` |
| API (FastAPI) | Render | Web service | `uvicorn app.main:app` |
| CPU/LLM workers | Render | Background worker | queues `ingest,cpu,l3,render,export` |
| GPU dispatcher | Render | Background worker | queue `gpu`, forwards to RunPod |
| GPU L1 compute | RunPod | Serverless endpoint | handler runs the real `l1_*` functions |
| Postgres + queue | Supabase Pro | managed | already provisioned |
| Object storage | Cloudflare R2 | managed | already provisioned |

---

## The core design decision: how Procrastinate `gpu` jobs reach RunPod

RunPod Serverless is **push** (HTTP handler); Procrastinate is **pull** (workers
poll Postgres). We bridge them with a **thin dispatcher** and a **task-body mode
switch**, so we keep every Procrastinate guarantee (retries, `lock`/priority,
correlation logging, idempotency) *and* get scale-to-zero GPU.

Mechanism: a single new setting `GPU_EXECUTION = "local" | "runpod"` (default
`"local"`). Add a **guard at the top of each `gpu`-queue task body**:

```python
# top of l1_orchestrate / l1_editing_proxy / l1_active_speaker
if get_settings().gpu_execution == "runpod":
    return runpod_bridge.run_remote("l1_orchestrate", file_id=file_id, r2_key=r2_key)
```

- **Render `edso-gpu-dispatcher`** runs a normal Procrastinate worker on the
  `gpu` queue with `GPU_EXECUTION=runpod`. Each task body forwards to RunPod via
  `/runsync` and blocks until it returns; a non-2xx / failed job raises, so
  Procrastinate retries exactly as today. High `WORKER_CONCURRENCY` (it only
  waits on network, no CPU/VRAM).
- **RunPod handler** imports the backend with `GPU_EXECUTION=local` and calls the
  **real** `l1_*` function → identical compute to today. Any follow-up enqueues
  the real functions already do (e.g. `l1_editing_proxy` → `l1_active_speaker`,
  `_prepare_from_raw` → `l1_active_speaker`) run inside the handler, which has DB
  access, so they land back on the `gpu` queue → dispatcher forwards them too.
- **Local dev / single-box**: `GPU_EXECUTION=local` (default) → byte-identical to
  today. Zero behavior change off Render.

Why this is low-risk: the compute code is untouched; the guard is a 3-line early
return; the "run on RunPod" path is *additive* and off by default. Outputs are
identical because the same functions run on the same inputs — only the machine
changes.

---

## Phase 0 — Prereqs (config + CORS), no external services

Files:
- `backend/app/config.py`: add
  - `gpu_execution: str = "local"`
  - `runpod_api_key: str = ""`
  - `runpod_endpoint_id: str = ""`
  - `runpod_timeout_seconds: int = 900`
  - prod origins support (keep `cors_origins`; see below).
- `backend/app/main.py`: production CORS. Today CORS is dev/LAN-only
  (`DEV_ORIGIN_REGEX`). Add the Vercel origin(s) from `settings.cors_origins`
  to `allow_origins`, keeping the dev regex for local. Exact prod domain filled
  after Vercel is live (set via `CORS_ORIGINS` env).

Verify: `python -m pyflakes app/config.py app/main.py`; app still boots locally
with `GPU_EXECUTION` unset (defaults to `local`).

---

## Phase 1 — RunPod GPU handler + image

Files:
- `backend/handler.py` (new): RunPod serverless entrypoint.
  ```python
  import runpod
  from app.services.jobs import register_tasks
  def handler(event):
      inp = event["input"]
      task, kwargs = inp["task"], inp.get("kwargs", {})
      if task == "warmup":
          from app.services.l1.transcript import _WhisperEngine
          _WhisperEngine.get()             # load weights, return fast
          return {"ok": True, "warmed": True}
      # GPU_EXECUTION=local here → real compute
      fn = {"l1_orchestrate": ..., "l1_editing_proxy": ..., "l1_active_speaker": ...}[task]
      fn(**kwargs)
      return {"ok": True, "task": task}
  runpod.serverless.start({"handler": handler})
  ```
- `backend/Dockerfile.worker`: reuse as-is for the GPU image (already has torch/
  CUDA, ffmpeg, baked `buffalo_sc`, `HF_HOME=/models`). Change `CMD` to
  `python handler.py`, add `runpod` to `requirements.txt`.
- `backend/requirements.txt`: add `runpod`.

Build + push to your registry (GHCR):
```
docker build -f backend/Dockerfile.worker -t ghcr.io/<you>/edso-gpu:latest .
docker push ghcr.io/<you>/edso-gpu:latest
```

RunPod (dashboard, you): create a **Serverless endpoint** from that image,
GPU 24 GB (A10/A5000-class), set env `GPU_EXECUTION=local`, `HF_TOKEN`,
Supabase + R2 keys; enable **FlashBoot**; idle timeout ~30–60 s; mount a network
volume at `/models` to persist weights across cold starts. Copy the **endpoint
id** for the dispatcher.

Verify: RunPod console "test" with `{"input":{"task":"warmup"}}` returns
`warmed:true`; a real `{"input":{"task":"l1_orchestrate","kwargs":{...}}}` on a
known file writes L1 rows to Supabase.

---

## Phase 2 — The RunPod bridge + task guards

Files:
- `backend/app/services/runpod_bridge.py` (new): `run_remote(task, **kwargs)` →
  POST `https://api.runpod.ai/v2/{endpoint_id}/runsync` with
  `{"input":{"task":task,"kwargs":kwargs}}`, auth `Bearer RUNPOD_API_KEY`,
  timeout `runpod_timeout_seconds`; raise on failure so Procrastinate retries.
  Plus `warm()` → async `/run` with `{"input":{"task":"warmup"}}`, fire-and-forget.
- `backend/app/services/l1/pipeline.py`: add the 3-line `GPU_EXECUTION=="runpod"`
  guard to the top of `l1_orchestrate`, `l1_editing_proxy`, `l1_active_speaker`
  (before `correlation.scope`/compute). Nothing else in these functions changes.

Verify: unit test the guard both ways (mock `run_remote`); full backend suite
green except the known unrelated `test_active_speaker.py`; `GPU_EXECUTION=local`
path unchanged.

---

## Phase 3 — Upload pre-warm hook

File: `backend/app/routers/upload.py`
- In `presign_upload` (and `multipart/create`), fire `runpod_bridge.warm()`
  fire-and-forget (never block/fault the upload). This spins a RunPod worker +
  loads weights *while the client encodes/uploads proxies*, so by the time
  `analysis-proxies/complete` enqueues L1 the worker is hot. Guarded by
  `GPU_EXECUTION=="runpod"` so local dev is unaffected.

Verify: `presign` still returns instantly when RunPod is unreachable (warm is
best-effort); log line confirms a warm ping fired.

---

## Phase 4 — Render Blueprint (`render.yaml`)

File: `render.yaml` (repo root) — 3 services, all auto-deploy on push to `main`:
- `edso-api` (web): build with `backend/Dockerfile.api` (Phase 5) or native
  Python; start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- `edso-workers` (worker): start `bash backend/run_workers.sh` with
  `WORKER_QUEUES=ingest,cpu,l3,render,export`, `GPU_WORKERS=0`,
  `GPU_EXECUTION=local` (it never runs gpu jobs). Mid-tier instance (ffmpeg).
- `edso-gpu-dispatcher` (worker): start a minimal `python worker.py` with
  `WORKER_QUEUES=gpu`, `GPU_EXECUTION=runpod`, `WORKER_CONCURRENCY=16` (network
  wait only), and skip the model warmup (it never loads models — small guard in
  `worker.py`'s `_warmup()` already keyed on the `gpu` queue; add a
  `GPU_EXECUTION` check so the dispatcher doesn't try to load Whisper).
- Env var **groups** for the shared Supabase/R2/LLM secrets; per-service extras
  (`RUNPOD_*` only on the dispatcher).

Note: `run_workers.sh` already runs `migrate.py apply` (advisory-locked) before
forking — migrations apply once on deploy, safe across all boxes.

Verify: `render.yaml` validates; on connect, all 3 services build; workers log
"Applying pending migrations…" then "Worker ready".

---

## Phase 5 — API image

File: `backend/Dockerfile.api` (new): slim CPU image (python:3.x-slim) + ffmpeg +
`requirements.txt`, `CMD uvicorn app.main:app`. (Importing routers pulls torch;
acceptable for MVP — a trimmed requirements split is a later optimization, not a
blocker. Flagged, not silently assumed.)

Verify: image builds; `/` (or a health route) responds; schema check passes on
boot against Supabase.

---

## Phase 6 — Vercel frontend

You (dashboard): project already imported with root `frontend/`. Set env:
- `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `NEXT_PUBLIC_API_URL` = the Render `edso-api` URL
Then set `CORS_ORIGINS` on `edso-api` to the Vercel domain and redeploy.

Verify: frontend loads, logs in (Supabase), hits the API (no CORS error),
upload → analysis → cuts round-trips end to end.

---

## Secrets matrix (you set these in dashboards, never in chat/repo)

| Secret | edso-api | edso-workers | edso-gpu-dispatcher | RunPod | Vercel |
|---|:--:|:--:|:--:|:--:|:--:|
| SUPABASE_URL / SERVICE_KEY | ✓ | ✓ | ✓ | ✓ | — |
| DATABASE_URL / DATABASE_POOL_URL | ✓ | ✓ | ✓ | ✓ | — |
| R2_* (account/key/secret/bucket) | ✓ | ✓ | — | ✓ | — |
| ANTHROPIC_API_KEY (+ GEMINI) | ✓ | ✓ | — | — | — |
| HF_TOKEN | — | — | — | ✓ | — |
| GPU_EXECUTION | local | local | **runpod** | local | — |
| RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID | — | — | ✓ | — | — |
| CORS_ORIGINS (Vercel domain) | ✓ | — | — | — | — |
| NEXT_PUBLIC_SUPABASE_URL / ANON_KEY | — | — | — | — | ✓ |
| NEXT_PUBLIC_API_URL (Render api URL) | — | — | — | — | ✓ |

---

## Rollout order & rollback

1. Phase 0–2 land in repo (behind `GPU_EXECUTION=local`, zero runtime change).
2. Build+push GPU image; create RunPod endpoint; test in isolation (Phase 1).
3. Bring up Render (Phases 4–5) with dispatcher `GPU_EXECUTION=runpod`.
4. Point Vercel at Render (Phase 6).
- **Rollback**: flip the dispatcher's `GPU_EXECUTION` to `local` — but Render has
  no GPU, so the true rollback is RunPod Pods (always-on puller) or a temporary
  GPU box. For MVP, RunPod endpoint issues → jobs just retry (Procrastinate),
  and pre-warm failures are already best-effort. Nothing corrupts.

---

## Rough monthly cost (100-user MVP)

- Render: api (small) + workers (mid, ffmpeg) + dispatcher (tiny) ≈ **$40–90/mo**.
- RunPod Serverless GPU: ~**$0.003–0.006 per L1 clip**, $0 idle; a few $ to low
  tens/mo at MVP volume.
- Vercel: Hobby/Pro **$0–20/mo**. Supabase Pro **$25/mo** (already). R2: pennies
  (zero egress).
- **Total ≈ $70–150/mo** at MVP scale, dominated by always-on Render + Supabase,
  not GPU.

---

## Open items / honest caveats

- **No real NLE / no logged-in browser validation** of the full prod path from
  this environment; first end-to-end must be verified live by you.
- **API image carries torch** (router import chain) — works, but fat; trimming
  is a follow-up.
- **RunPod cold start** is mitigated by pre-warm + FlashBoot + `/models` volume,
  but the *first ever* request after a deploy (new image) is still slow; expected.
- Dispatcher blocks one worker slot per in-flight RunPod job; `WORKER_CONCURRENCY`
  sizes max concurrent GPU jobs — tune to RunPod max-workers.
