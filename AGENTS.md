# Repository Guidelines

## Project Structure & Module Organization

This repository is a W&B-native fine-tuning harness for DreamZero. The upstream model and training code lives in `groot/`; keep changes there minimal unless you are modifying DreamZero internals. The additive orchestration layer is in `wam/`, including `wam/train.py`, artifact helpers, and bootstrap scripts under `wam/artifacts/`. Dataset conversion and utility scripts live in `scripts/data/`, `scripts/train/`, and `scripts/encord/`. Kubernetes manifests are in `deploy/cks/`, split into `bootstrap/`, `train`, and `infra/`. Runbooks and architecture notes are in `docs/`; evaluation placeholders are in `eval/`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements-train.txt`: install the additive training dependencies on top of the expected NVIDIA PyTorch image.
- `python scripts/data/convert_lerobot_to_gear.py --dataset-path /path/to/dataset`: generate GEAR/DreamZero metadata for a LeRobot v2 dataset.
- `kubectl -n dreamzero apply -f deploy/cks/infra/stager.yaml`: start the staging pod used to copy the repo to the data PVC.
- `kubectl apply -f deploy/cks/bootstrap/00-models-download.yaml`: register base model artifacts.
- `kubectl apply -f deploy/cks/bootstrap/01-pickplace-subset.yaml`: build and register the DROID pick-place subset.
- `MAX_STEPS=2` in `deploy/cks/train/droid-pickplace-train-single.yaml`: smoke setting before longer training runs.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, explicit imports, and `from __future__ import annotations` in new typed modules. Prefer `pathlib.Path` for filesystem paths and keep environment-variable knobs uppercase near module-level configuration. Preserve existing patterns such as `WAM_*` for harness settings, `*_artifact` for W&B helpers, and descriptive script names like `build_pickplace_subset.py`.

## Testing Guidelines

There is no formal test suite checked in. Validate data scripts against a small local or staged dataset and confirm generated `meta/*.json`, `tasks.jsonl`, and `episodes.jsonl` match the expected schema. For training changes, run a smoke job with `MAX_STEPS=2`, then inspect pod logs and the W&B run lineage before launching a full run.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add validated single-GH200 ZeRO-3 CPU-offload training path`. Follow that style: start with a verb and describe the concrete change. Pull requests should include the purpose, changed commands or manifests, relevant W&B artifact/run links, and smoke or full-run results. Include screenshots only when W&B lineage, metrics, or Kubernetes status views clarify the change.

## Security & Configuration Tips

Do not commit kubeconfigs, W&B API keys, downloaded model weights, datasets, or generated checkpoints. Keep secrets in Kubernetes, such as the `wandb-api-key` secret in the `dreamzero` namespace, and keep large artifacts on PVCs or in W&B artifacts.
