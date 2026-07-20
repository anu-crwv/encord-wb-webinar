#!/usr/bin/env bash
# Launch a weave-traced per-joint amplitude diag (eval/deploy/trossen-diag-amp-job.yaml) on the cluster.
#
# Usage: scripts/eval/run_amp_diag.sh <tag> <model_dir> [data_root] [task] [resume_run_id] [model_label]
#   tag           job suffix (trossen-diag-<tag>)
#   model_dir     checkpoint dir on /checkpoints (LoRA or full; server auto-detects)
#   data_root     LeRobot dataset root (the eval EPISODE comes from here). default: encord_trossen_v4
#   task          DIAG_TASK substring (e.g. "coffee", "yellow cylindrical"). default: coffee
#   resume_run_id W&B run id to RESUME INTO so weave traces attach to it (e.g. the training run).
#                 empty -> a fresh job_type=eval run. Tracing is ON by default (WAM_EVAL_NO_WEAVE=1 disables).
#   model_label   WEAVE_MODEL_LABEL for the leaderboard (sanitized to an identifier by the eval).
#
# Requires KUBECONFIG set and the repo staged at /data/src/dreamzero-wam on the cluster.
set -euo pipefail
TAG="${1:?tag}"; MODELDIR="${2:?model_dir}"
DATA="${3:-/data/wam/datasets/encord_trossen_v4}"; TASK="${4:-coffee}"
RUNID="${5:-}"; LABEL="${6:-$TAG}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
MANIFEST="$HERE/eval/deploy/trossen-diag-amp-job.yaml"
kubectl -n dreamzero delete job "trossen-diag-${TAG}" --ignore-not-found >/dev/null 2>&1 || true
sed -e "s|__TAG__|${TAG}|g" -e "s|__MODELDIR__|${MODELDIR}|g" -e "s|__DATA__|${DATA}|g" \
    -e "s|__TASK__|${TASK}|g" -e "s|__RUNID__|${RUNID}|g" -e "s|__LABEL__|${LABEL}|g" \
    "$MANIFEST" | kubectl -n dreamzero apply -f -
echo "launched trossen-diag-${TAG}  model=${MODELDIR}  data=${DATA}  task='${TASK}'  resume_run='${RUNID}'"
