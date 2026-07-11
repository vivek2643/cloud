# Deployment & Cost Analysis (ideation — HELD, revisit later)

Status: **parked**. Captured 2026-07-10 from a live ideation pass. Nothing here is
implemented. Revisit after current pipeline/brain work is complete.

Target: deploy the whole app (frontend, API, workers, analysis, user/data mgmt)
to comfortably serve **1,000 registered / 200 DAU**, and feel fast. Assume each
DAU does **2 projects/day @ ~30 min each → ~12,000 video-min/day (~200 hrs/day)**.

---

## 1. Current state (what already exists)

- **Frontend:** Next.js (App Router) — runs locally, **not deployed**.
- **API:** FastAPI (`uvicorn app.main:app`) — **not deployed anywhere** (gap; Terraform only does workers).
- **Workers:** `deploy/aws/terraform` → **2× g5.2xlarge spot** (A10G 24GB) running
  `run_workers.sh` (queues: `gpu` L1, `ingest` Cuts v3, `l3`/`cpu`) via docker-compose,
  pulling jobs from Postgres. **Static fleet, no autoscaling.**
- **Queue:** procrastinate on Supabase Postgres. Tiny pools (`DB_POOL_MAX=2`) because
  the Supabase pooler caps connections (~15 on small tiers). **Connection limits are
  the quiet bottleneck.**
- **DB / Auth / user+data mgmt:** Supabase (Postgres + Auth).
- **Storage:** Cloudflare R2 (zero egress — keep).
- **Heavy deps:** faster-whisper large-v3-turbo, SigLIP, pyannote diarization (~7GB
  weights) on GPU; Anthropic/OpenAI/Gemini backbones for L3.

---

## 2. The reframe: this is a video pipeline, not a web app

"Serve 1,000 users" barely stresses the web tier (presigned URLs + DB reads).
Capacity = **video-seconds processed per minute**, dominated by **GPU L1**. Latency =
**how fast a fresh upload clears the GPU queue**.

Three tiers, three scaling profiles → deploy differently:

| Tier | Nature | Scales with | Home |
|---|---|---|---|
| Frontend | static/SSR | users (cheap) | Vercel / CF Pages |
| API + L3/`ingest` workers | network-bound (DB, R2, LLM) | RPS + queue depth (cheap) | Fargate / Cloud Run (autoscale) |
| **L1 GPU workers** | GPU compute, 7GB models | **video volume (expensive)** | autoscaling EC2 GPU (the real decision) |

---

## 3. THE HEADLINE: LLM cost dominates, not GPU

Measured from real ingest runs (`claude-sonnet-5`, 47 runs). Avg tokens/project:
input 109,554 · output 31,580 · cache_read 18,285 · cache_write 74,942.

| Cost line | Per project | At 400 projects/day |
|---|---|---|
| **LLM (Claude ingest)** | **~$1.09** | **~$436/day ≈ $13,100/mo** |
| GPU L1 compute (est.) | ~$0.05–0.10 | ~$600–1,000/mo |
| API + web + storage | negligible | ~$100–300/mo |

**~$65/mo per active user in Claude alone.** Unit economics must be fixed BEFORE scaling.

### Cost-optimization levers (ordered by impact × safety)
1. **Anthropic Batch API — flat 50% off.** Ingest is async-friendly. Tier it:
   real-time for the first interactive ingest, batch for re-ingests/bulk. ~$6.5k/mo saved. Quality-neutral.
2. **Caching audit — likely net-NEGATIVE today.** Write 75k vs read only 18k (0.24×).
   The per-shard *scoped render* (see §6) gives each shard its own prefix → we write a
   cache that's rarely re-read, paying the 1.25× write premium for ~nothing. Cache only
   the truly-shared prefix (system prompt + repeated transcripts) or disable for
   single-run ingests. ~$3.3k/mo. Quality-neutral.
3. **Output-token diet.** Output ($15/M) is the single biggest line ($0.47/proj); 31.6k
   tokens is a lot. Tighten schemas/summaries/labels. ~$2–3k/mo. Needs care (keep every
   field the brain consumes).
4. **Haiku for pass 1** (text grouping); keep Sonnet for pass 2 (vision). Needs A/B.

Realistic landing: **~$1.09 → ~$0.40–0.55/project (~$5–6.5k/mo)**; levers 1–2 are free.

---

## 4. Gemini estimate (cost confident, quality needs A/B)

2026 rates. Per-project using ~185k input-equiv, ~32k output:

| Model | In /M | Out /M | Per project | 400/day (monthly) | vs Sonnet |
|---|---|---|---|---|---|
| Claude Sonnet (current) | $3.00 | $15.00 | $1.09 | ~$13,100/mo | — |
| Gemini 2.5 Pro | $1.25 | $10.00 | ~$0.55 | ~$6,600/mo | ~50% cheaper |
| Gemini 2.5 Flash | $0.30 | $2.50 | ~$0.14 | ~$1,700/mo | ~87% cheaper |
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | ~$0.03 | ~$370/mo | ~97% cheaper |

- **Batch API = another 50% off** on top (Flash batch ≈ $850/mo, Pro batch ≈ $3,300/mo).
- **⚠️ 200k-context cliff (Pro only):** input ~185k sits at the edge; larger projects
  tip to $2.50/$15 and creep back toward Sonnet cost. **Flash has no cliff** (flat $0.30) —
  an advantage for large projects.

### Quality (relative to Sonnet = 100%)
| Pass | Needs | Gemini 2.5 Pro | Gemini 2.5 Flash |
|---|---|---|---|
| pass1 (grouping, cue/false-start, strict "categories-only" contract) | meticulous instruction-following + editorial nuance | ~88–95% (w/ tuning) | ~70–85% ⚠️ |
| pass2 (identity, shot_size, framing, take/outlook/winner) | multimodal image judgment | ~95–105% | ~80–90% |

- **Gemini edge:** multimodal + long context. Sonnet edge: strict-contract discipline
  (exactly what "deterministic keep, semantic cull" leans on; Flash is riskiest there).
- **Options:** all-Pro (~50% cut, near-parity, watch cliff) · all-Flash (~85–90% cut,
  validate pass1 via A/B) · **hybrid** = Gemini pass2 + Sonnet/Gemini-Pro pass1 (best $/quality).

### Native video (tried; verdict: NOT worth it)
Empirically increases cost a lot for unclear benefit — and the token math agrees: a
30-min clip @1fps ≈ 460k+ video tokens vs ~185k with sampled frames (3–5× input cost,
before audio). Benefit is thin because **L1 already extracts deterministically** what
dense video would make the LLM re-derive at token prices. **Keep sampled frames + L1
signals.** Only *targeted* density (extra frames where a judgment is ambiguous) is worth
considering — never blanket video.

---

## 5. GPU sizing (needs one real g5 calibration run)

- CPU/local measured **0.21× realtime** (~4.75 wall-s per video-s). Bottleneck stages:
  scene_detect, transcript, proxy, audio_features. On A10G expect ~**3–6×** (transcript/audio
  → GPU; proxy/scene_detect → NVDEC + vCPU). **Must measure on one g5 box to lock the math.**
- Throughput @ S≈5: **~2 GPUs avg, ~6–10 at peak** (peak hour ≈ 15% of daily).
- Two independent pressures: (1) throughput, (2) per-job "fast" latency — a 30-min job
  can't be fast on one GPU (~6 min @ S=5) → must chunk (see §7). Static 2-box fleet fails
  both (slow at peak, idle-burning at night) → **autoscale on queue depth**.

---

## 6. Recommended architecture (all-AWS, balanced warm pool)

- **Frontend → Vercel.**
- **API → ECS Fargate + ALB**, autoscale on RPS. Uploads go **direct to R2 via presigned URLs**.
- **L3/`ingest` workers → Fargate**, autoscale on **queue depth** (pure LLM calls, cheap).
- **GPU L1 → autoscaling EC2 GPU** (ECS-on-EC2 ASG or EKS+Karpenter) on queue depth,
  **spot + 1 warm box**. Bake 7GB weights into AMI/image (no cold model-load tax).
  Evolution of the existing Terraform.
- **Queue/DB → fix connections first:** Supabase transaction-mode pooler (Supavisor) for
  API + bounded worker pools, bump tier; dedicated Postgres for the queue if worker count
  climbs; revisit SQS/Redis only if Postgres-as-queue is outgrown.
- **Cross-cutting:** extend GHCR CI to API image; queue-depth/latency dashboards + Sentry;
  R2 lifecycle rules; DB backups; RLS/security pass pre-launch.

### "Runs quick" levers (platform-independent)
1. No cold GPU: warm pool ≥1 + baked weights. 2. Per-video parallelism (§7).
3. Direct-to-R2 uploads + CDN-fronted API. 4. Queue pickup is already instant
   (LISTEN/NOTIFY) → real latency = autoscaling responsiveness.

---

## 7. Chunking & continuity (safe if you chunk the RIGHT layer)

**Chunk L1 (signal extraction) — this is where the latency win is:**
- **File-level parallelism (lossless):** different clips of a project on different workers.
  Clips are independent → zero loss. Do this first.
- **Intra-file chunking (one long recording):** safe IF:
  - Transcript: cut on **silence/VAD boundaries + small overlap**, never mid-word.
  - Scene/motion: overlap seams + dedupe (don't invent/drop a boundary at the cut).
  - **⚠️ Diarization is the ONE real continuity trap** — speaker IDs are global. Either run
    diarization whole-file (it's cheap, ~1.7s avg) or re-cluster speaker embeddings across
    chunks. Prefer whole-file.

**Do NOT chunk L3 by clip.** The semantic passes must see the whole project (take groups =
same line across clips, grouping, cut-to-cut continuity). This is exactly what the pass-2a
co-located sharding already handles — its parallelism is co-located sharding, not time-chunking.

→ **Aggressively chunk large projects at L1; keep L3's global view intact. No info/continuity lost.**

---

## 8. Suggested phasing (when we resume)
1. **Free cost wins:** batch API on the re-ingest path + caching audit (measure read/write ratio).
2. **L1 file-level parallelism** + one **g5 calibration run** (lock realtime factor + fleet math).
3. **Intra-file chunking** (silence-aligned, whole-file diarization) + **autoscaling infra**
   (Fargate API/L3 + GPU ASG on queue depth + Supavisor pooler).
4. Frontend → Vercel; observability, CI, backups, RLS/security pass.
5. (Optional) Gemini A/B (hybrid: Gemini pass2 + Sonnet/Pro pass1) once quality is validated.

Open decisions: LLM unit-cost target (drives batch/model choice); real GPU realtime factor;
pricing/margin per project given ~$1 LLM cost.
