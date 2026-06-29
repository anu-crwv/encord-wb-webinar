#!/bin/bash
# Lightweight runner for the OFFLINE Trossen eval (eval/offline_eval.py).
# Pure client on a stock python:3.11 image (no torch/CUDA, no Isaac Sim) — weave +
# wandb + openpi-client + pandas + imageio + moviepy all resolve cleanly here
# (they conflict inside the nvcr pytorch image). imageio-ffmpeg bundles ffmpeg.
set -exo pipefail

export HOME=/data/.home-offline PIP_CACHE_DIR=/data/.home-offline/.pip
mkdir -p "$HOME" "$PIP_CACHE_DIR"
WAM_EVAL_SRC="${WAM_EVAL_SRC:-/data/src/dreamzero-wam/eval}"
DZ_HOST="${DZ_HOST:-dreamzero-trossen-inference}"; DZ_PORT="${DZ_PORT:-8001}"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir \
    "weave>=0.52.0" "wandb>=0.18.0" \
    "openpi-client==0.1.1" "websockets==13.1" msgpack msgpack-numpy \
    numpy pandas pyarrow imageio imageio-ffmpeg pillow "moviepy>=1.0.3,<2.0"

export PYTHONPATH="${WAM_EVAL_SRC}"

# Wait for the DreamZero Trossen server (scale the deployment to 1 before launching).
for i in $(seq 1 60); do
  if timeout 5 bash -c "</dev/tcp/$DZ_HOST/$DZ_PORT" 2>/dev/null; then echo "[offline] server TCP open ($i)"; break; fi
  echo "[offline] server probe $i..."; sleep 10
done

python "${WAM_EVAL_SRC}/offline_eval.py"
echo "=== offline runner done ==="
