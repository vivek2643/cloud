# L1 speedup — phased plan

L1 is CPU-bound on this machine (no CUDA; Whisper `large-v3-turbo` dominates) and
runs its stages **serially** in one Procrastinate task. Two levers:

- **Phase A — parallelize** the independent stages in-process (free, helps on any
  multi-core host, CPU or GPU).
- **Phase B — run the worker on a GPU host on AWS** (the big win: Whisper fp16 +
  NVENC proxy → seconds). Same code, different host.

## Dependency graph (what can overlap)

```
proxy ─┐ (needs raw)               demux ─┐ (needs raw; already overlaps proxy)
       └─ duration gate                   └─ wav
                                          │
   ┌──────────────── after proxy+demux ───┼─────────────────┐
   │ SPEECH track        │ AUDIO track     │ VIDEO track     │
   │ transcript (Whisper)│ audio_features  │ motion_dynamics │   (all 3 in parallel)
   │   → diarization     │   → beat_cut    │                 │
   └─────────┬───────────┴────────┬────────┴─────────────────┘
             └── join ──► dialogue_cut (needs transcript+diar+audio_features)
```

The two heaviest stages — **transcript (Whisper)** and **motion (optical flow)** —
become concurrent. C extensions (CTranslate2, numpy, OpenCV) release the GIL, so
Python threads give real parallelism here.

## Phases

| Phase | Goal | Work | Risk / guard | Verify |
|---|---|---|---|---|
| **A — Parallelize** | overlap the independent stages | 3 thread tracks (speech / audio / video), **each its own psycopg connection**; `wait(all)` then `dialogue_cut`; keep `processing_jobs` idempotency + best-effort semantics + error propagation | psycopg conns aren't thread-safe → per-track conn; no two tracks write the same row concurrently | imports; threaded harness w/ stubbed stages; idempotent re-run skips done |
| **B1 — Access** | confirm we can launch a GPU | `--profile edso`, us-east-1; check EC2 perms + **G/VT on-demand quota** (L-DB2E81BA); pick `g4dn.xlarge` (T4 16GB) + Deep Learning AMI | new-account GPU vCPU quota may be 0 → request increase / fall back | `sts`, `service-quotas`, `describe-images` |
| **B2 — Launch** | a reachable GPU box | keypair + SSH-only security group (my IP); launch DLAMI instance | cost: stop when idle; tag for cleanup | `describe-instances` running + SSH |
| **B3 — Setup** | deps on the box | verify CUDA torch; ffmpeg w/ NVENC; `pip install -r backend/requirements.txt`; clone repo; copy `.env` (scp, never echo) | NVENC needs the right ffmpeg build; faster-whisper CUDA needs cuDNN | `torch.cuda.is_available()`, `ffmpeg -encoders | grep nvenc` |
| **B4 — Worker** | process jobs on GPU | run `procrastinate worker` on queues `gpu,l3`; confirm `torch_device()==cuda`, whisper fp16, NVENC proxy | only ONE worker should serve the gpu queue to avoid double-processing | enqueue/process a real clip; compare wall time |

## Cost control (Phase B)
- `g4dn.xlarge` ≈ $0.53/hr on-demand. **Stop the instance when idle** (don't leave
  it running). Tag `Project=cloud-l1-gpu` for easy find/terminate.
- Single GPU worker serves `gpu` (L1/L2) + `l3`; the Mac can stop serving the gpu
  queue while the GPU box is up.

## Out of scope (now)
- Autoscaling / spot fleets / containerized deploy — a single managed instance is
  enough to make L1 "really quick"; revisit if throughput needs it.
