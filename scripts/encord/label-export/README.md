# Label Export

Exports one Encord project's labels and source dataset metadata to W&B.

## Setup

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
```

Edit run notes:

```text
scripts/encord/label-export/export_metadata.yaml
```

Edit W&B settings:

```text
scripts/encord/wandb_config.yaml
```

## Export Labels To W&B

```bash
uv run --script scripts/encord/label-export/export_single_view_labels_to_wandb.py \
  --metadata-yaml scripts/encord/label-export/export_metadata.yaml
```

Writes local files to:

```text
exports/encord-label-export/<timestamp>/
```

Logs:

- source dataset artifact
- labels artifact
- preview table with `language_instruction`

## Convert Captions To DROID Layout

```bash
uv run --script scripts/encord/label-export/export_encord_captions_to_droid.py --help
```

Use this for local DROID/LeRobot-style label files, not W&B versioning.
