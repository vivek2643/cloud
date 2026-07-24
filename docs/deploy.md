# Deploy runbook — Render + Vercel + RunPod

Step-by-step for a first production deploy. Follow the order top to bottom.
Architecture background lives in `deployment.plan.md`; this file is the "what do
I click" version.

Order matters: **RunPod first** (Render's dispatcher needs its endpoint id) →
**Render** → **Vercel** → **smoke test**.

---

## 0. Gather your secrets first (5 min)

Collect these into a password manager. You'll paste them into dashboards later —
never into git or chat.

| Secret | Where to find it |
|---|---|
| `SUPABASE_URL` | Supabase → Project Settings → API → Project URL |
| `SUPABASE_SERVICE_KEY` | Supabase → Project Settings → API → `service_role` secret |
| `SUPABASE_ANON_KEY` | Supabase → Project Settings → API → `anon` public |
| `DATABASE_URL` | Supabase → Settings → Database → Connection string → **Session pooler** (port 5432) URI |
| `DATABASE_POOL_URL` | Same page → **Transaction pooler** (port 6543) URI |
| `R2_ACCOUNT_ID` | Cloudflare → R2 → account id (top of R2 overview) |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | Cloudflare → R2 → Manage API Tokens → Create (Object Read & Write) |
| `R2_BUCKET_NAME` | your bucket name (default in code: `aerodrive`) |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys |
| `GEMINI_API_KEY` | aistudio.google.com → API Keys (Pass 2 uses Gemini by default) |
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read scope) |
| `RUNPOD_API_KEY` | runpod.io → Settings → API Keys |

Also confirm (one-time, already flagged): on Hugging Face, **accept the license**
for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` with the
same account as your `HF_TOKEN`. Without this, diarization fails on RunPod.

---

## 1. RunPod — the GPU L1 endpoint

1. Log in → **Serverless** → **New Endpoint**.
2. Source: choose **"Import Git Repository"** (RunPod builds the image on their
   amd64 infra — no local Docker needed). Authorize GitHub, pick `vivek2643/cloud`.
   - **Dockerfile path**: `backend/Dockerfile.worker`
   - **Build context**: `/` (repo root)
   - Branch: `main`
3. Hardware: a **24 GB GPU** (A10 / A5000 / L4 class is plenty).
4. Scaling:
   - Min workers: `0` (scale to zero → $0 idle)
   - Max workers: start with `3`
   - **Enable FlashBoot**
   - Idle timeout: `60` seconds
5. **Advanced → Network Volume**: create/attach a volume, mount path `/models`
   (persists the ~3–4 GB of model weights across cold starts).
6. **Environment variables** on the endpoint (Settings → Environment):
   ```
   GPU_EXECUTION=local
   HF_TOKEN=...
   SUPABASE_URL=...
   SUPABASE_SERVICE_KEY=...
   DATABASE_URL=...
   DATABASE_POOL_URL=...
   R2_ACCOUNT_ID=...
   R2_ACCESS_KEY_ID=...
   R2_SECRET_ACCESS_KEY=...
   R2_BUCKET_NAME=...
   ```
   (`GPU_EXECUTION=local` is critical here — it makes the handler run the real
   compute instead of forwarding back to RunPod.)
7. Deploy. Wait for the first build (several minutes — it's a CUDA image).
8. **Test**: endpoint → Requests → send
   ```json
   { "input": { "task": "warmup" } }
   ```
   Expect `{"ok": true, "warmed": true}` (the first one is slow — it's loading
   weights; that's the cold start we later hide with pre-warm).
9. Copy two values for later: the **Endpoint ID** (in the endpoint URL/overview)
   and your **RunPod API key**.

---

## 2. Render — the always-on tier (API + workers + dispatcher)

1. Log in → **New +** → **Blueprint**.
2. Connect GitHub, pick `vivek2643/cloud`, branch `main`. Render detects
   `render.yaml` and shows 3 services: `edso-api`, `edso-workers`,
   `edso-gpu-dispatcher`, plus an env group `edso-shared`.
3. **Fill the `edso-shared` env group** (applies to all three services):
   ```
   SUPABASE_URL, SUPABASE_SERVICE_KEY,
   DATABASE_URL, DATABASE_POOL_URL,
   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME,
   ANTHROPIC_API_KEY, GEMINI_API_KEY
   ```
4. **Service-specific vars** (Render will prompt for the `sync: false` ones):
   - `edso-gpu-dispatcher`: `RUNPOD_API_KEY` = your key, `RUNPOD_ENDPOINT_ID` =
     the endpoint id from step 1.
   - `edso-api`: `CORS_ORIGINS` — leave as `http://localhost:3000` for now (you'll
     set the Vercel domain in step 3). Also set `RUNPOD_API_KEY` +
     `RUNPOD_ENDPOINT_ID` here (same values as the dispatcher): the API fires the
     upload pre-warm to RunPod, so it needs the creds too.
5. Click **Apply / Create**. Render builds the image once and starts all three.
   - Watch `edso-workers` logs → you should see "Applying pending migrations…"
     then "Worker ready".
   - `edso-api` should reach **Live** (its health check hits `/health`).
6. Copy the **`edso-api` public URL** (e.g. `https://edso-api.onrender.com`).

If `edso-api` crash-loops on first boot complaining about schema drift, give the
`edso-workers` / preDeploy migration a minute to finish, then it self-heals on
the next restart.

---

## 3. Vercel — the frontend

1. Your project is already imported with **root directory = `frontend/`**.
2. Project → Settings → **Environment Variables**, add (Production):
   ```
   NEXT_PUBLIC_API_URL=https://edso-api.onrender.com   ← the URL from step 2.6
   NEXT_PUBLIC_SUPABASE_URL=...
   NEXT_PUBLIC_SUPABASE_ANON_KEY=...
   ```
3. **Redeploy** the frontend (Deployments → ⋯ → Redeploy) so it picks up the env.
4. Copy your Vercel domain (e.g. `https://edso.vercel.app`).
5. Back in **Render → edso-api → Environment**, set
   `CORS_ORIGINS=https://edso.vercel.app` (comma-separate if you have several),
   and let it redeploy.

---

## 4. Smoke test the whole path

1. Open your Vercel URL, log in (Supabase auth).
2. Open browser devtools → Network; confirm calls to `NEXT_PUBLIC_API_URL`
   succeed (no CORS errors).
3. Upload a short video. Then watch **Render → edso-gpu-dispatcher → Logs**:
   - `runpod: dispatching l1_orchestrate …`
   - RunPod endpoint spins a worker (RunPod dashboard shows it active)
   - dispatcher logs `runpod: l1_orchestrate completed …`
4. The file should progress through analysis and cuts should appear in the UI.
   The **first** upload after a deploy is slow (cold start); subsequent ones are
   fast (pre-warm fired on upload start + FlashBoot + 60s idle reuse).

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| API 500s with schema-drift error | migrations didn't apply — check `edso-api` preDeploy + `edso-workers` logs; redeploy |
| Browser "Failed to fetch" / CORS | `CORS_ORIGINS` on `edso-api` doesn't match the exact Vercel origin (scheme + host, no trailing slash) |
| Dispatcher logs `RUNPOD_ENDPOINT_ID is not set` | set it (and `RUNPOD_API_KEY`) on `edso-gpu-dispatcher` |
| RunPod job FAILED with HF/pyannote error | `HF_TOKEN` missing on the endpoint, or pyannote licenses not accepted |
| L1 never runs, no dispatcher log | `edso-gpu-dispatcher` must have `GPU_EXECUTION=runpod` + `WORKER_QUEUES=gpu` |
| Everything slow every time | check FlashBoot is on and idle timeout > 0; confirm pre-warm fires (`runpod: warm ping sent` in `edso-api` logs on upload) |

## Rollback

- The GPU path is gated: set `edso-gpu-dispatcher`'s `GPU_EXECUTION=local` to stop
  forwarding — but Render has no GPU, so real rollback means either standing up a
  RunPod **Pod** (always-on puller) or a temporary GPU box. For MVP, transient
  RunPod issues just cause Procrastinate retries; nothing corrupts.
- Frontend/API: Render + Vercel both keep previous deploys — one-click "Rollback"
  to the last good build.
