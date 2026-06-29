#!/usr/bin/env bash
# DreamZero **Trossen** WebSocket inference server entrypoint.
#
# Trossen analogue of the DROID dz_server_entrypoint.sh. Differences:
#   - PYTHONPATH puts dreamzero-wam's groot FIRST (it has the `trossen`
#     embodiment + modality config) and upstream dreamzero second (for
#     eval_utils + socket_test_optimized_AR, which our server imports).
#   - runs eval/server/trossen_policy_server.py (3 cams / 16-dim packed action).
#   - materializes the LoRA checkpoint from the W&B artifact at boot.
#
# Baked-in paths the checkpoint's config.json requires (already on the
# dreamzero-checkpoints PVC from training):
#   /checkpoints/wam/models/wan2-1-i2v-14b-480p/{*, Wan2.1_VAE.pth, models_clip_*.pth, models_t5_*.pth}
#   /checkpoints/wam/models/umt5-xxl
set -euo pipefail

export HOME=/data/.home-trossen-server
export TMPDIR=/data/.home-trossen-server/.tmp
export PIP_CACHE_DIR=/data/.home-trossen-server/.pipcache
export HF_HOME=/data/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"

WAM_REPO_ROOT="${WAM_REPO_ROOT:-/data/src/dreamzero-wam}"
DREAMZERO_UPSTREAM="${DREAMZERO_UPSTREAM:-/data/dreamzero-src}"
LORA_ARTIFACT="${LORA_ARTIFACT:-encord-wb-physical-ai/wam-finetune-webinar/dreamzero-trossen-lora:v2}"
MODEL_DIR="${MODEL_DIR:-/checkpoints/wam/eval/dreamzero-trossen-lora-v2}"

# HOME/.local persists on the PVC across restarts. A prior run may have left a
# GUI opencv-python (needs libGL, absent here) — purge all opencv variants so the
# headless build below is the one that provides cv2.
python -m pip uninstall -y opencv-python opencv-contrib-python \
    opencv-python-headless opencv-contrib-python-headless >/dev/null 2>&1 || true

# --- deps (mirror the DROID server; amd64/RTX path) ---
COMMON_DEPS="hydra-core==1.3.2 peft==0.5.0 transformers==4.51.3
    diffusers==0.30.2 einops==0.8.1 imageio==2.34.2 imageio-ffmpeg
    loguru msgpack msgpack-numpy tianshou==0.5.1
    websockets==13.1 openpi-client==0.1.1
    ftfy lark tyro dm_tree timm albumentations==1.4.18
    multi-storage-client[boto3,msal,observability-otel]==0.33.0
    wandb>=0.18.0"
if [ "${DZ_ARM64:-0}" = "1" ]; then
    python -m pip install --no-cache-dir --quiet $COMMON_DEPS \
        "opencv-python-headless>=4.8.0" "av>=12.0.0,<16"
else
    # opencv-python-headless (not opencv-python) — the base image lacks libGL.so.1
    # and we run as non-root, so the GUI build's `import cv2` fails on libGL.
    python -m pip install --no-cache-dir --quiet $COMMON_DEPS \
        "av==15.0.0" "opencv-python-headless" "decord==0.6.0"
fi

# --- clone upstream if not staged (provides eval_utils + socket_test machinery) ---
if [ ! -d "$DREAMZERO_UPSTREAM" ]; then
    echo "[dz-trossen] upstream not at $DREAMZERO_UPSTREAM; cloning"
    git clone --depth 1 https://github.com/dreamzero0/dreamzero.git "$DREAMZERO_UPSTREAM"
fi
python -m pip install --no-cache-dir --quiet --no-deps -e "$DREAMZERO_UPSTREAM" || true

# --- materialize the LoRA checkpoint from the W&B artifact (idempotent) ---
if [ ! -f "$MODEL_DIR/experiment_cfg/conf.yaml" ]; then
    echo "[dz-trossen] downloading LoRA artifact $LORA_ARTIFACT -> $MODEL_DIR"
    python - <<PYEOF
import wandb
api = wandb.Api()
api.artifact("$LORA_ARTIFACT").download(root="$MODEL_DIR")
print("[dz-trossen] LoRA artifact downloaded")
PYEOF
fi

# --- verify the baked-in base-model paths exist (fail fast with a clear msg) ---
for p in /checkpoints/wam/models/wan2-1-i2v-14b-480p \
         /checkpoints/wam/models/wan2-1-i2v-14b-480p/Wan2.1_VAE.pth \
         /checkpoints/wam/models/umt5-xxl; do
    if [ ! -e "$p" ]; then
        echo "[dz-trossen] FATAL: required base-model path missing on PVC: $p" >&2
        echo "[dz-trossen] (download artifacts wan2-1-i2v-14b-480p:v0 + umt5-xxl:v0 to these paths)" >&2
        exit 1
    fi
done

# groot FIRST (trossen embodiment) then upstream (eval_utils, socket_test).
export PYTHONPATH="${WAM_REPO_ROOT}:${DREAMZERO_UPSTREAM}:${PYTHONPATH:-}"
export ATTENTION_BACKEND=TE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

echo "[dz-trossen] starting server: model=$MODEL_DIR port=${DZ_PORT:-8001}"
nvidia-smi -L || true

exec torchrun --nproc_per_node=1 --standalone \
    /scripts/trossen_policy_server.py \
        --port "${DZ_PORT:-8001}" \
        --model-path "$MODEL_DIR" \
        --enable-dit-cache \
        --timeout-seconds 36000
