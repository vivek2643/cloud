#!/usr/bin/env bash
#
# Launch a fleet of workers on ONE box (the simple "one large GPU instance"
# deployment). Cross-file parallelism = many worker processes, each pulling a
# different video from the queue.
#
#   - GPU_WORKERS processes pull the "gpu" queue (L1 ingest), each pinned
#     round-robin to a physical GPU via CUDA_VISIBLE_DEVICES so they don't all
#     pile onto GPU 0.
#   - One ingest worker pulls the "ingest" queue (Cuts ingest -- Anthropic
#     image/text calls). It's a network/API call, not GPU compute, so it runs
#     at high concurrency in a single process and never competes for VRAM with
#     the L1 ingest workers.
#   - CPU_WORKERS processes pull the "cpu"/"l3"/"render"/"export" queues
#     (ffmpeg renders + exports, L3 auto-editor); no GPU.
#
# Env:
#   GPU_WORKERS         default = number of detected GPUs (min 1)
#   CPU_WORKERS         default = 2
#   NUM_GPUS            override GPU count (else auto-detected via nvidia-smi)
#   WORKER_CONCURRENCY  per-process concurrency (default 1; safe for VRAM)
#   INGEST_CONCURRENCY  concurrent Cuts ingest calls (default 8; see below)
#
# scale_architecture.plan.md Pillar 4/8: INGEST_CONCURRENCY's default moved
# 6->8 -- safe now that llm/client.complete() has a proactive per-provider
# in-flight limiter (INGEST_LLM_MAX_INFLIGHT_{ANTHROPIC,GEMINI}), which bounds
# real concurrent LLM calls regardless of how many projects this process is
# juggling. CPU_WORKERS and gpu WORKER_CONCURRENCY stay at their current
# defaults: the plan's own proposed bumps for those ("soak CPU-L1", "+CPU
# split") depend on the L1 CPU/GPU queue split in pipeline_parallelism.
# plan.md, which hasn't shipped -- raising them now wouldn't soak anything,
# just add idle workers pulling from queues nothing routes CPU-only L1 work
# to yet.
#
# Usage:
#   cd backend && ./run_workers.sh
#   # or in the container:
#   docker run --gpus all --env-file .env -e GPU_WORKERS=4 -e CPU_WORKERS=2 \
#       IMAGE bash run_workers.sh
set -euo pipefail
cd "$(dirname "$0")"

# --- detect GPUs ---------------------------------------------------------
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
  else
    NUM_GPUS=0
  fi
fi
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1

GPU_WORKERS="${GPU_WORKERS:-$NUM_GPUS}"
CPU_WORKERS="${CPU_WORKERS:-2}"
INGEST_CONCURRENCY="${INGEST_CONCURRENCY:-8}"
export WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"

echo "Fleet: NUM_GPUS=$NUM_GPUS GPU_WORKERS=$GPU_WORKERS CPU_WORKERS=$CPU_WORKERS concurrency=$WORKER_CONCURRENCY ingest_concurrency=$INGEST_CONCURRENCY"

# --- apply pending migrations ---------------------------------------------
# The one gated deploy step that's allowed to write schema (see
# migration_runner.plan.md) -- runs once here, before any worker forks, so
# every worker.py process below starts against an up-to-date schema. Safe to
# invoke even if this script is ever run concurrently on multiple boxes: the
# applier takes a Postgres advisory lock, so only one call actually applies
# and the rest find nothing pending.
echo "Applying pending migrations..."
python scripts/migrate.py apply || exit 1

pids=()
cleanup() { echo "Stopping fleet..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

# --- GPU (ingest) workers ------------------------------------------------
for ((i = 0; i < GPU_WORKERS; i++)); do
  gpu=$(( i % NUM_GPUS ))
  echo "  gpu-worker $i -> CUDA_VISIBLE_DEVICES=$gpu queue=gpu"
  CUDA_VISIBLE_DEVICES="$gpu" WORKER_QUEUES="gpu" python worker.py &
  pids+=($!)
  # Stagger starts so N processes don't hammer HF model download at once.
  sleep 2
done

# --- Ingest worker: Cuts ingest (network-bound Anthropic calls, no GPU) --
# One process serves the "ingest" queue with internal concurrency so many
# projects' image/text calls overlap. Decoupled from the "gpu" queue so it
# never contends for VRAM with L1 ingest.
echo "  ingest-worker -> queue=ingest concurrency=$INGEST_CONCURRENCY"
CUDA_VISIBLE_DEVICES="" WORKER_QUEUES="ingest" WORKER_CONCURRENCY="$INGEST_CONCURRENCY" python worker.py &
pids+=($!)

# --- CPU workers: L3 auto-editor + render/export (ffmpeg, no GPU) ---------
# export_options.plan.md Phase 0: "render" (render_edit) and "export"
# (build_export) are both ffmpeg/CPU-bound, so they fold into this same pool
# rather than a dedicated worker -- before this, render_edit/build_export jobs
# were enqueued but never picked up by any worker (queue="render"/"export" was
# never in WORKER_QUEUES anywhere). Running L3 here keeps the network-bound
# editor calls off the GPU ingest workers.
for ((j = 0; j < CPU_WORKERS; j++)); do
  echo "  cpu-worker $j -> queue=cpu,l3,render,export"
  CUDA_VISIBLE_DEVICES="" WORKER_QUEUES="cpu,l3,render,export" python worker.py &
  pids+=($!)
done

echo "Fleet up (${#pids[@]} processes). Ctrl-C to stop."
wait
