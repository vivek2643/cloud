# Running the ingest fleet on one AWS GPU box

The L1 pipeline worker is **stateless**. It pulls jobs from Supabase Postgres
(via Procrastinate) and reads/writes media on Cloudflare R2 using presigned URLs.
That means it runs on *any* GPU box with internet access — no ports to expose.
Models auto-select CUDA when a GPU is present (`app/services/ml_device.py`); on a
laptop they fall back to CPU unchanged.

Target deployment: **one large GPU instance** running a *fleet* of worker
processes for cross-file parallelism. Many videos process at once — each worker
process pulls a different file.

> **Fully automated:** `deploy/aws/terraform/` provisions the box and boots the
> fleet in one `terraform apply` (it bakes in your `.env`). See its README. The
> steps below are the manual equivalent.

---

## Queues (parallelism model)

Tasks are routed to two queues so compute specializes:

| Queue | Tasks | Runs on |
|---|---|---|
| `gpu` | `l1_orchestrate` (proxy, transcript, shots, SigLIP embeddings, audio features, diarization) | GPU worker processes |
| `cpu` | `render_edl` (ffmpeg) | CPU worker processes |

`backend/run_workers.sh` launches both pools on one box:
- `GPU_WORKERS` processes on the `gpu` queue, each pinned to a physical GPU via
  `CUDA_VISIBLE_DEVICES` (round-robin) so they don't pile onto GPU 0.
- `CPU_WORKERS` processes on the `cpu` queue.

A worker with no `WORKER_QUEUES` set pulls **all** queues (handy for local dev:
`python backend/worker.py`).

---

## 1. Env vars (pass at runtime, never bake in)

| Var | Required | Notes |
|---|---|---|
| `DATABASE_URL` | ✅ | Postgres conn string. **Use the direct/session connection (5432), not the transaction pooler (6543)** — Procrastinate needs `LISTEN/NOTIFY`. |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | ✅ | |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | ✅ | |
| `R2_BUCKET_NAME` | – | defaults to `aerodrive` |
| `ANTHROPIC_API_KEY` | – | L3 edit-time reasoning + vision (the editor/director) |
| `HF_TOKEN` | – | Hugging Face token for the **pyannote** diarization backend (the default). Needed once you've accepted the gated-model licenses (see below). Omit it and diarization falls back to the CPU embedding backend. |
| `DIARIZATION_BACKEND` | – | `pyannote` (default, GPU), `neural` (Resemblyzer, CPU), or `mfcc`. |
| `GPU_WORKERS` | – | # of GPU-queue processes (default = detected GPU count) |
| `CPU_WORKERS` | – | # of render processes (default `2`) |
| `WORKER_CONCURRENCY` | – | per-process concurrency (default `1`; keep low for VRAM) |

Copy your existing `backend/.env` values.

---

## 2. Image

CI builds and pushes the worker image to GHCR on every push to `main`
(`.github/workflows/build-worker.yml`):

```
ghcr.io/<your-gh-user>/cloud-worker:latest
```

The EC2 box just pulls it — no ECR needed. (For a private GHCR package, log in
with a PAT: `echo $GHCR_PAT | docker login ghcr.io -u <user> --password-stdin`.)

---

## 3. Launch on one AWS GPU instance

Recommended for bursty demos: a **g5** instance (NVIDIA A10G 24 GB).
- `g5.2xlarge` — 1 GPU, cheapest, good for a first parallel test.
- `g5.12xlarge` — 4 GPUs → `GPU_WORKERS=4`, four videos ingest at once.
- Use **spot** to cut cost ~60–70%.

One-time host setup (Ubuntu 22.04 "Deep Learning" AMI already has Docker +
NVIDIA toolkit; otherwise install `docker` + `nvidia-container-toolkit`):

```bash
# SSH in, then:
git clone https://github.com/<you>/cloud.git && cd cloud
cp /path/to/your.env .env            # the real secrets
echo $GHCR_PAT | docker login ghcr.io -u <user> --password-stdin   # if private
docker compose -f deploy/aws/docker-compose.yml pull
GPU_WORKERS=4 CPU_WORKERS=2 docker compose -f deploy/aws/docker-compose.yml up -d
docker compose -f deploy/aws/docker-compose.yml logs -f
```

Healthy startup logs (one line per GPU worker):

```
Fleet: NUM_GPUS=4 GPU_WORKERS=4 CPU_WORKERS=2 concurrency=1
ML device selected: cuda (NVIDIA A10G)
Worker ready; concurrency=1 queues=['gpu']; entering main loop.
```

Now upload videos in the app — they fan out across the GPU workers in parallel.

Bare `docker run` alternative (single container runs the whole fleet):

```bash
docker run -d --gpus all --restart unless-stopped \
  --env-file .env -e GPU_WORKERS=4 -e CPU_WORKERS=2 \
  -v /opt/models:/models \
  --name edso-fleet ghcr.io/<you>/cloud-worker:latest bash run_workers.sh
```

Mounting `/models` persists the weights (Whisper/SigLIP/pyannote) across
restarts so reboots don't re-download them.

---

## 3b. Enable the pyannote diarization backend (one-time)

The default `DIARIZATION_BACKEND=pyannote` uses gated Hugging Face models. Once
per HF account:

1. Create a read token at https://huggingface.co/settings/tokens and set
   `HF_TOKEN=hf_...` in your `.env`.
2. Accept the license on each gated model page (click "Agree"):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

The first ingest downloads the weights into `/models` (persisted by the volume).
If `HF_TOKEN` is missing or the licenses aren't accepted, the worker logs a
warning and **falls back to the `neural` (Resemblyzer) backend** — ingest still
succeeds, just with the weaker diarizer.

---

## 4. Notes & limits

- **Throughput = `GPU_WORKERS`.** With N GPUs, run N GPU workers (1 per GPU) so
  each loads its own model set (Whisper + SigLIP) without VRAM contention.
- **`DATABASE_URL` must be the direct/session connection** (5432), not the
  pooler — Procrastinate needs `LISTEN/NOTIFY`. Each worker holds a couple of
  connections; mind Supabase's connection ceiling as you scale `GPU_WORKERS`.
- **Local dev** is unchanged: `python backend/worker.py` (no queue filter) pulls
  everything and auto-selects CPU.
