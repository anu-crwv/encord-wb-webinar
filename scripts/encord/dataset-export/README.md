# Dataset Export

Exports an Encord dataset of 3-camera data groups to a W&B dataset artifact.

The script downloads only video items, writes a LeRobot/DROID-style `dataset/` folder, and logs it to W&B.

## Setup

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
aws sso login --profile encord-robotics
```

Edit W&B settings:

```text
scripts/encord/wandb_config.yaml
```

Edit artifact description and tags:

```text
scripts/encord/dataset-export/dataset_export_config.yaml
```

## Run

```bash
AWS_PROFILE=encord-robotics uv run --script scripts/encord/dataset-export/export_encord_dataset_to_wandb.py \
  --dataset-hash <encord_dataset_hash> \
  --limit 3
```

For full export, omit `--limit`.

Configured tags are logged as W&B artifact tags, and the `latest` alias is applied automatically.

Local output:

```text
exports/encord-dataset-export/<timestamp>/
```

W&B output:

```text
<source_artifact_name>:vN
```
