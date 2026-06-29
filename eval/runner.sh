#!/bin/bash
# Runner for the Trossen DreamZero sim eval on Isaac Lab-Arena.
# Runs INSIDE the Isaac Sim image (nvcr.io/nvidia/isaac-sim:6.0.0-dev2) as root
# (namespace PSA = baseline). Runtime-installs IsaacLab + Arena (no custom image),
# registers our Trossen embodiment + DreamZero policy, and runs Arena's eval_runner.
#
# Env knobs:
#   DZ_HOST / DZ_PORT     DreamZero policy server (default localhost:8001)
#   EVAL_JOBS_CONFIG      jobs config JSON
#   WAM_EVAL_SRC          this repo's eval/ dir on the pod (default /data/src/dreamzero-wam/eval)
#   ARENA_DIR             where to clone/cache Arena (+submodules) — keep on the PVC to avoid re-clone
set -exo pipefail

DZ_HOST="${DZ_HOST:-localhost}"
DZ_PORT="${DZ_PORT:-8001}"
WAM_EVAL_SRC="${WAM_EVAL_SRC:-/data/src/dreamzero-wam/eval}"
EVAL_JOBS_CONFIG="${EVAL_JOBS_CONFIG:-${WAM_EVAL_SRC}/jobs_configs/trossen_pnp_dreamzero_jobs_config.json}"
ARENA_DIR="${ARENA_DIR:-/data/wam/arena/IsaacLab-Arena}"
TROSSEN_DIR="${TROSSEN_DIR:-/data/wam/arena/trossen_ai_isaac}"
ISAAC_PY=/isaac-sim/python.sh
[[ -x "$ISAAC_PY" ]] || { echo "FATAL: $ISAAC_PY not found/executable (need root on the isaac-sim image)"; exit 1; }

export HOME=/root PIP_CACHE_DIR=/data/wam/arena/.pipcache
mkdir -p "$PIP_CACHE_DIR" "$(dirname "$ARENA_DIR")"
command -v git >/dev/null || { apt-get update -qq && apt-get install -y --no-install-recommends git ffmpeg >/dev/null; }

# --- Arena + IsaacLab (submodules) + Trossen assets, cached on the PVC ---
# Arena's .gitmodules use git@github.com SSH URLs; no ssh in the image -> rewrite to https.
git config --global url."https://github.com/".insteadOf "git@github.com:"
if [[ ! -d "$ARENA_DIR/.git" ]]; then
  git clone --depth 1 ${ARENA_REF:+--branch "$ARENA_REF"} \
      https://github.com/isaac-sim/IsaacLab-Arena.git "$ARENA_DIR"
fi
# Only IsaacLab is needed (skip the large Isaac-GR00T submodule — we don't use isaaclab_arena_gr00t).
# Idempotent: safe to re-run if a prior attempt left submodules uninitialized.
if [[ ! -f "$ARENA_DIR/submodules/IsaacLab/source/isaaclab/setup.py" ]]; then
  git -C "$ARENA_DIR" submodule update --init --depth 1 submodules/IsaacLab
fi
[[ -d "$TROSSEN_DIR/.git" ]] || git clone --depth 1 \
    https://github.com/TrossenRobotics/trossen_ai_isaac.git "$TROSSEN_DIR"

# Mirror Arena's Dockerfile install: register editable IsaacLab source packages,
# then run IsaacLab's OWN installer (isaaclab.sh -i) which pulls the correct pinned
# deps (warp-lang==1.12.0, lazy_loader, usd-core, ...). A bare `--no-deps` loop is
# NOT enough — those deps are required at import. Marker-gated so re-runs are fast.
# NOTE: pip installs land in the EPHEMERAL container (/isaac-sim site-packages),
# not the PVC, so every fresh pod must reinstall. The pip cache lives on the PVC
# (PIP_CACHE_DIR) so downloads/builds are fast on subsequent pods. Use --cache-dir.
ln -sfn /isaac-sim "$ARENA_DIR/submodules/IsaacLab/_isaac_sim" 2>/dev/null || true
for d in "$ARENA_DIR"/submodules/IsaacLab/source/isaaclab*/; do
  [[ -d "$d" ]] && "$ISAAC_PY" -m pip install -q --no-deps -e "$d" || true
done
# Canonical IsaacLab dependency install (warp-lang==1.12.0, lazy_loader, usd-core, ...).
ISAACLAB_PATH="$ARENA_DIR/submodules/IsaacLab" bash "$ARENA_DIR/submodules/IsaacLab/isaaclab.sh" -i || true
# Arena packages WITH their declared deps (NOT --no-deps).
"$ISAAC_PY" -m pip install -q -e "$ARENA_DIR" || true
"$ISAAC_PY" -m pip install -q -e "$ARENA_DIR/isaaclab_arena_environments" 2>/dev/null || true

# Our adapter/embodiment deps (openpi-client = websocket client + image_tools).
"$ISAAC_PY" -m pip install --no-cache-dir -q \
    "openpi-client==0.1.1" "websockets==13.1" "msgpack" "msgpack-numpy" \
    "weave>=0.52.0" "wandb>=0.18.0" "moviepy>=1.0.3,<2.0" "proglog<=1.0.0" tqdm \
    "mediapy" "imageio" "imageio-ffmpeg" "pillow" || true

# Our eval package (embodiment + policy adapter) on PYTHONPATH.
export PYTHONPATH="${WAM_EVAL_SRC}:${ARENA_DIR}:${PYTHONPATH:-}"
export TROSSEN_MOBILE_AI_USD="${TROSSEN_MOBILE_AI_USD:-$TROSSEN_DIR/assets/robots/mobile_ai/mobile_ai.usd}"

# Pre-flight: wait for the DreamZero policy server (14B load takes ~6 min; non-fatal if
# still down). Skipped for render-only debug entrypoints that don't hit the server.
if [ "${WAM_SKIP_SERVER_PROBE:-0}" != "1" ]; then
  for i in $(seq 1 40); do
    if timeout 5 bash -c "</dev/tcp/$DZ_HOST/$DZ_PORT" 2>/dev/null; then echo "[runner] server TCP open ($i)"; break; fi
    echo "[runner] server probe $i failed; retrying in 10s..."; sleep 10
  done
fi

cd "$ARENA_DIR"
# Entrypoint: run_trossen_eval.py (default) or e.g. debug_cameras.py via WAM_EVAL_ENTRY.
ENTRY="${WAM_EVAL_ENTRY:-run_trossen_eval.py}"
"$ISAAC_PY" "${WAM_EVAL_SRC}/${ENTRY}" \
    --eval_jobs_config "$EVAL_JOBS_CONFIG" \
    --enable_cameras --headless

echo "=== runner done ==="
