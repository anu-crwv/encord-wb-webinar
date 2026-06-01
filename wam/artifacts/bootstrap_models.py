#!/usr/bin/env python3
"""Download the DreamZero base model weights to the checkpoints PVC and register them as W&B artifacts.

For each base model we:
  1. snapshot_download the HF repo to a canonical PVC path (/checkpoints/wam/models/<name>),
  2. log a W&B *reference* artifact pointing at that PVC path (type="model"), so we get
     versioning + lineage WITHOUT re-uploading 70GB of weights, and
  3. link it into the model Registry collection.

The reference points at the shared PVC, which every training Job mounts at the same path — so
`use_artifact(...)` gives the training run a real input-lineage edge and resolves to local bytes,
and training never has to hit Hugging Face again.

Pass --upload to instead log a fully-managed artifact (uploads the bytes to W&B) for a model that
needs to be portable off-cluster.

Standalone (stdlib + huggingface_hub + wandb) so it runs inside a slim Kubernetes Job via configmap.

Usage:
    python bootstrap_models.py                         # all three base models, reference artifacts
    python bootstrap_models.py --only umt5-xxl         # just one
    python bootstrap_models.py --only umt5-xxl --upload
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Install huggingface_hub: pip install huggingface_hub")

# --- defaults (mirror wam/config.py) ------------------------------------------
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "wam-finetune-webinar")
MODEL_REGISTRY = os.environ.get("WAM_MODEL_REGISTRY", "wandb-registry-model")
MODELS_DIR = os.path.join(os.environ.get("WAM_CHECKPOINTS_ROOT", "/checkpoints/wam"), "models")

BASE_MODELS = {
    "wan2-1-i2v-14b-480p": {
        "hf_repo": "Wan-AI/Wan2.1-I2V-14B-480P",
        "role": "dit_backbone",
        "notes": "Wan2.1 I2V 14B DiT backbone + VAE + CLIP image encoder + T5 text encoder.",
    },
    "umt5-xxl": {
        "hf_repo": "google/umt5-xxl",
        "role": "tokenizer",
        "notes": "UMT5-XXL tokenizer used by the Wan text encoder.",
    },
    "dreamzero-agibot": {
        "hf_repo": "GEAR-Dreams/DreamZero-AgiBot",
        "role": "pretrained_policy",
        "notes": "Pretrained DreamZero policy; pretrained_model_path for new-embodiment LoRA fine-tuning.",
    },
}


def snapshot_with_retry(retry_wait: int = 320, **kwargs) -> str:
    attempt = 0
    while True:
        attempt += 1
        try:
            return snapshot_download(**kwargs)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if any(s in msg for s in ("429", "too many requests", "rate limit")):
                print(f"[models] rate limited; waiting {retry_wait}s (attempt {attempt})", file=sys.stderr)
                time.sleep(retry_wait)
                continue
            raise


def register(name: str, spec: dict, args) -> None:
    local_dir = Path(MODELS_DIR) / name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[models] === {name}  <-  {spec['hf_repo']} ===")
    print(f"[models] downloading to {local_dir} ...")
    path = snapshot_with_retry(
        repo_id=spec["hf_repo"], repo_type="model", local_dir=str(local_dir), max_workers=4,
    )
    size_gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e9
    print(f"[models] downloaded {name} ({size_gb:.1f} GB)")

    if args.no_wandb:
        print("[models] --no-wandb set; skipping artifact logging.")
        return

    import wandb

    metadata = {
        "hf_repo": spec["hf_repo"],
        "role": spec["role"],
        "notes": spec["notes"],
        "pvc_path": str(local_dir),
        "size_gb": round(size_gb, 2),
        "storage": "upload" if args.upload else "reference",
    }
    run = wandb.init(entity=args.entity, project=args.project, job_type="register-model",
                     name=f"register-{name}", config=metadata)
    art = wandb.Artifact(name=name, type="model", metadata=metadata,
                         description=f"{spec['notes']} Source: {spec['hf_repo']}.")
    if args.upload:
        art.add_dir(str(local_dir))                      # uploads bytes to W&B (portable)
    else:
        art.add_reference(f"file://{local_dir}", max_objects=200000)  # pointer to PVC (no upload)
    logged = run.log_artifact(art)
    logged.wait()
    print(f"[models] logged artifact {args.entity}/{args.project}/{name}:{logged.version}")
    target = f"{MODEL_REGISTRY}/{name}"
    run.link_artifact(logged, target_path=target)
    print(f"[models] linked to Registry: {target}")
    run.finish()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=list(BASE_MODELS), help="Register a single model instead of all")
    p.add_argument("--upload", action="store_true", help="Upload bytes to W&B (managed) instead of reference artifact")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--no-wandb", action="store_true", help="Download only; skip artifact logging")
    args = p.parse_args()

    names = [args.only] if args.only else list(BASE_MODELS)
    for name in names:
        register(name, BASE_MODELS[name], args)
    print(f"\n[models] done: {names}")


if __name__ == "__main__":
    main()
