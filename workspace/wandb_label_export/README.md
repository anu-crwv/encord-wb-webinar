# Encord labels to W&B

This folder contains the minimal workflow for exporting labels from Encord and logging them to Weights & Biases.

## 1. Export labels from Encord

You need two Encord values:

- `PROJECT_HASH`: the Encord project hash.
- `ENCORD_SSH_KEY_FILE`: the local private key file used by the Encord SDK.

Run from this folder:

```bash
uv run --script export_encord_labels.py \
  --project-hash PROJECT_HASH \
  --ssh-key-file /path/to/encord_ssh_private_key \
  --output-json encord_labels.json
```

## 2. Log the export to W&B

The repo defaults are:

- W&B entity: `encord-wb-physical-ai`
- W&B project: `wam-finetune-webinar`

Because you already ran `wandb login`, the script should use your existing W&B credentials.

```bash
uv run --script log_encord_labels_to_wandb.py \
  --labels-json encord_labels.json \
  --entity encord-wb-physical-ai \
  --project wam-finetune-webinar \
  --artifact-name encord-labels
```

This creates:

- a W&B run with job type `encord-label-export`
- a W&B Table called `encord_label_preview`
- a dataset artifact called `encord-labels:latest`
- a local summary file at `wandb_export/encord_label_summary.json`

If you also have the original images locally, pass the folder root to get bounding-box overlays in the W&B Table:

```bash
uv run --script log_encord_labels_to_wandb.py \
  --labels-json encord_labels.json \
  --image-root /path/to/images
```

The image overlay currently supports Encord bounding boxes. Other shapes are counted and kept in the raw JSON artifact.
