"""Central configuration for the DreamZero → W&B training harness.

Everything that the artifact-bootstrap and (later) the training entrypoint need to
agree on lives here so there is a single source of truth. Values can be overridden
by environment variables so the same code runs locally, in a notebook, and inside a
Kubernetes Job without edits.
"""

from __future__ import annotations

import os

# --- W&B target ----------------------------------------------------------------
# All runs + artifacts land here. Kept constant so dataset → training → eval
# lineage is co-located and comparable in one project.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "wam-finetune-webinar")

# W&B Registry collections (org-level). Base weights + trained checkpoints go to
# the "model" registry; DROID dataset variants go to the "dataset" registry.
# link_artifact target format is "wandb-registry-<registry>/<collection>".
MODEL_REGISTRY = os.environ.get("WAM_MODEL_REGISTRY", "wandb-registry-model")
DATASET_REGISTRY = os.environ.get("WAM_DATASET_REGISTRY", "wandb-registry-dataset")

# --- PVC / filesystem layout ---------------------------------------------------
# On the CoreWeave cluster these are the two RWX shared-vast PVCs mounted into
# every Job. Locally they can be pointed anywhere via env vars.
#   dreamzero-checkpoints  -> /checkpoints   (model weights)
#   dreamzero-data         -> /data          (datasets, caches)
CHECKPOINTS_ROOT = os.environ.get("WAM_CHECKPOINTS_ROOT", "/checkpoints/wam")
DATA_ROOT = os.environ.get("WAM_DATA_ROOT", "/data/wam")

MODELS_DIR = os.path.join(CHECKPOINTS_ROOT, "models")
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")

# --- Base models to register ---------------------------------------------------
# name             -> the W&B artifact / registry collection name (stable handle)
# hf_repo          -> Hugging Face source
# role             -> how the training harness consumes it (used as artifact metadata)
BASE_MODELS = {
    "wan2-1-i2v-14b-480p": {
        "hf_repo": "Wan-AI/Wan2.1-I2V-14B-480P",
        "hf_repo_type": "model",
        "role": "dit_backbone",
        "notes": "Wan2.1 I2V 14B DiT backbone + VAE + CLIP image encoder + T5 text encoder.",
    },
    "umt5-xxl": {
        "hf_repo": "google/umt5-xxl",
        "hf_repo_type": "model",
        "role": "tokenizer",
        "notes": "UMT5-XXL tokenizer used by the Wan text encoder.",
    },
    "dreamzero-agibot": {
        "hf_repo": "GEAR-Dreams/DreamZero-AgiBot",
        "hf_repo_type": "model",
        "role": "pretrained_policy",
        "notes": "Pretrained DreamZero policy; pretrained_model_path for new-embodiment LoRA fine-tuning.",
    },
}

# --- DROID dataset -------------------------------------------------------------
DROID_HF_REPO = os.environ.get("WAM_DROID_REPO", "GEAR-Dreams/DreamZero-DROID-Data")
DROID_HF_REPO_TYPE = "dataset"

# The three camera views in the DROID LeRobot dataset.
DROID_VIDEO_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
)


def model_local_dir(name: str) -> str:
    """Canonical on-PVC path for a base model's weights."""
    return os.path.join(MODELS_DIR, name)
