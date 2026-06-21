#!/usr/bin/env bash
#
# Launch a fleet of workers on ONE box (the simple "one large GPU instance"
# deployment). Cross-file parallelism = many worker processes, each pulling a
# different video from the queue.
#
#   - GPU_WORKERS processes pull the "gpu" queue (L1 ingest), each pinned
#     round-robin to a physical GPU via CUDA_VISIBLE_DEVICES so they don't all
#     pile onto GPU 0.
#   - One L2 worker pulls the "l2" queue (Gemini perception). L2 is a network/
#     API call, not GPU compute, so it runs at high concurrency in a single
#     process and never competes for VRAM with the ingest workers.
#   - CPU_WORKERS processes pull the "cpu" queue (ffmpeg renders); no GPU.
#
# Env:
#   GPU_WORKERS        default = number of detected GPUs (min 1)
#   CPU_WORKERS        default = 2
#   NUM_GPUS           override GPU count (else auto-detected via nvidia-smi)
#   WORKER_CONCURRENCY per-process concurrency (default 1; safe for VRAM)
#   L2_CONCURRENCY     concurrent Gemini calls in the L2 worker (default 6)
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
L2_CONCURRENCY="${L2_CONCURRENCY:-6}"
export WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"

echo "Fleet: NUM_GPUS=$NUM_GPUS GPU_WORKERS=$GPU_WORKERS CPU_WORKERS=$CPU_WORKERS concurrency=$WORKER_CONCURRENCY l2_concurrency=$L2_CONCURRENCY"

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

# --- L2 worker: Gemini perception (network-bound, no GPU) ----------------
# One process serves the "l2" queue with internal concurrency so many clips'
# Gemini calls overlap. Decoupled from the "gpu" queue so it never contends
# for VRAM with L1 ingest.
echo "  l2-worker -> queue=l2 concurrency=$L2_CONCURRENCY"
CUDA_VISIBLE_DEVICES="" WORKER_QUEUES="l2" WORKER_CONCURRENCY="$L2_CONCURRENCY" python worker.py &
pids+=($!)

# --- CPU workers: L3 edit orchestrator (network-bound Claude calls) -------
# Renders were removed, so the old "cpu" render queue is dead; these processes
# now serve the L3 editor's "l3" queue (kept "cpu" too for forward-compat).
# Running L3 here keeps minutes-long Opus loops off the GPU ingest workers.
for ((j = 0; j < CPU_WORKERS; j++)); do
  echo "  cpu-worker $j -> queue=cpu,l3"
  CUDA_VISIBLE_DEVICES="" WORKER_QUEUES="cpu,l3" python worker.py &
  pids+=($!)
done

echo "Fleet up (${#pids[@]} processes). Ctrl-C to stop."
wait
