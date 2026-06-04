# Running the ingest worker on a GPU (RunPod / Lambda)

The L1/L2 pipeline worker is **stateless**. It pulls jobs from Supabase Postgres
(via Procrastinate) and reads/writes media on Cloudflare R2 using presigned URLs.
That means you can run it on *any* GPU box with internet access — no quota fight,
nothing to expose. The models auto-select CUDA when a GPU is present
(`app/services/ml_device.py`); on a laptop they fall back to CPU unchanged.

This image is the same on RunPod, Lambda Labs, GCP, or a bare server — only the
launch steps differ.

---

## 1. What the worker needs (env vars)

Pass these at runtime (never bake into the image):

| Var | Required | Notes |
|---|---|---|
| `DATABASE_URL` | ✅ | Postgres conn string for Procrastinate + pgvector. **Use the direct/session connection (port 5432), not the transaction pooler (6543)** — Procrastinate relies on `LISTEN/NOTIFY`. e.g. `postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres` |
| `SUPABASE_URL` | ✅ | `https://<ref>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | ✅ | service-role key |
| `R2_ACCOUNT_ID` | ✅ | Cloudflare R2 account id |
| `R2_ACCESS_KEY_ID` | ✅ | |
| `R2_SECRET_ACCESS_KEY` | ✅ | |
| `R2_BUCKET_NAME` | – | defaults to `aerodrive` |
| `ANTHROPIC_API_KEY` | – | only for the L2 narrative fallback |
| `WORKER_CONCURRENCY` | – | default `1`. A single GPU usually wants `1` (avoid VRAM contention). |
| `WORKER_QUEUES` | – | comma-separated queue filter; empty = all queues |
| `FORCE_CPU` | – | set `1` to pin CPU even on a GPU host (debugging) |

> Tip: copy your existing `backend/.env` values. The container reads these from
> the environment directly.

---

## 2. Build & push the image

Build from the **repo root** (the Dockerfile copies `backend/`):

```bash
docker build -f backend/Dockerfile.worker -t YOUR_REGISTRY/edso-worker:latest .

# Docker Hub
docker push YOUR_REGISTRY/edso-worker:latest
# …or GitHub Container Registry (ghcr.io/<user>/edso-worker:latest)
```

The image is ~6–8 GB (CUDA + torch). First build is slow; pushes are cached.
Model weights (~3–4 GB) download on first run unless you mount a volume at
`/models` (see below) to persist them.

---

## 3. Launch on RunPod (recommended)

RunPod **Pods** are GPU containers you control — no quota, live in a couple of
minutes.

1. Create an account at runpod.io and add credit.
2. **Deploy → Pods → GPU Cloud.** Pick a GPU — good fits for this workload:
   - **RTX A4000 / A5000** or **L4** (16–24 GB) — cheap, plenty for Whisper +
     SigLIP + DINOv2. ~$0.20–0.40/hr.
   - **A10 / RTX 4090** if you want more headroom.
3. **Container image:** `YOUR_REGISTRY/edso-worker:latest`.
4. **Container start command:** leave default (`python worker.py`).
5. **Environment variables:** add every required var from the table above.
6. **(Optional) Volume:** mount a Network Volume at `/models` so weights persist
   across restarts (set once, saves the 3–4 GB re-download each boot).
7. **Ports:** none needed — it's an outbound-only worker.
8. Deploy. Open the pod **Logs**; success looks like:

   ```
   ML device selected: cuda (NVIDIA L4)
   Loading Whisper large-v3-turbo (float16, cuda)...
   Worker ready; concurrency=1 queues=ALL; entering main loop.
   ```

Now upload a video in the app — the job is picked up by this GPU pod. Scale
throughput by deploying **more identical pods**; Procrastinate's locking makes
multiple workers safe on the same queues.

### Scripted launch (optional)
If you give me a RunPod API key I can launch/destroy pods for you via their API
(`runpodctl` / GraphQL) instead of the console.

---

## 4. Launch on Lambda Labs (alternative)

Lambda rents full GPU VMs (cheap, but capacity is sometimes unavailable).

```bash
# After launching an instance and SSHing in (Docker + NVIDIA runtime preinstalled):
docker run -d --gpus all --restart unless-stopped \
  -e DATABASE_URL=... \
  -e SUPABASE_URL=... -e SUPABASE_SERVICE_KEY=... \
  -e R2_ACCOUNT_ID=... -e R2_ACCESS_KEY_ID=... -e R2_SECRET_ACCESS_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  -v /opt/models:/models \
  --name edso-worker YOUR_REGISTRY/edso-worker:latest

docker logs -f edso-worker
```

The same `docker run` works on any GPU server (GCP/EC2 once quota lands, etc.).

---

## 5. Notes & limits (v1)

- **One worker = all jobs.** No CPU/GPU queue split yet; the GPU pod runs the
  whole pipeline. Splitting (cheap CPU box for proxy/transcode, GPU box for
  embeddings) is a later optimization via `WORKER_QUEUES`.
- **Faces (insightface) still run on CPU** via onnxruntime. They're a small part
  of L2; GPU faces would need `onnxruntime-gpu` + provider config — deferred.
- **`DATABASE_URL` must be the direct/session connection**, not the transaction
  pooler — Procrastinate needs `LISTEN/NOTIFY`.
- To run the local CPU worker again, just `python backend/worker.py` as before;
  it auto-selects CPU. You can run local + GPU workers simultaneously.
