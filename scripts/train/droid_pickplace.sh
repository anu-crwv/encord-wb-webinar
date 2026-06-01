#!/bin/bash
# DreamZero DROID pick-place fine-tune — artifact-driven, W&B-tracked.
#
# Thin wrapper around `python -m wam.train`. All config is via env vars (see wam/train.py);
# the defaults are a single-GPU LoRA smoke that consumes the W&B Registry artifacts and
# logs the fine-tuned checkpoint back as an artifact with full lineage.
#
# Scale up later by overriding e.g. NUM_GPUS / MAX_STEPS / PER_DEVICE_BATCH_SIZE / TRAIN_ARCHITECTURE.
set -euxo pipefail

export NUM_GPUS=${NUM_GPUS:-1}
export MAX_STEPS=${MAX_STEPS:-300}
export TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE:-lora}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
export SAVE_STEPS=${SAVE_STEPS:-100}

cd "${WAM_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec python -m wam.train
