#!/bin/bash
# Encord/Trossen fine-tune — artifact-driven, W&B-tracked.
#
# Thin wrapper around `python -m wam.train` (identical mechanism to droid_pickplace.sh, just named
# for the trossen embodiment so runs/artifacts don't read "droid"). All config is via env vars
# (see wam/train.py); the manifest sets DATA_CONFIG=dreamzero/trossen_relative, the v4 dataset dir,
# TRAIN_ARCHITECTURE, MAX_STEPS, OUTPUT_DIR (run name), etc.
set -euxo pipefail

export NUM_GPUS=${NUM_GPUS:-1}
export MAX_STEPS=${MAX_STEPS:-300}
export TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE:-lora}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
export SAVE_STEPS=${SAVE_STEPS:-100}

cd "${WAM_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec python -m wam.train
