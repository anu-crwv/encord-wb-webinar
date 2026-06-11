# Label Export

Exports one Encord project's single-view caption labels to a W&B labels artifact that overlays the
3-camera dataset artifact.

The exporter reuses the source episode parquet files from S3 for state/action/timing data, then rewrites
the task and language annotation columns from Encord captions. It does not fabricate state/action values.

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
  --metadata-yaml scripts/encord/label-export/export_metadata.yaml \
  --source-artifact-ref encord-source-data:v0 \
  --limit 3
```

For a full export, omit `--limit`.

Writes local files to:

```text
exports/encord-label-export/<timestamp>/
```

Logs:

- labels artifact with `dataset/data/...` and `dataset/meta/...`
- preview table with `language_instruction`

The W&B artifact type is `dataset` because W&B artifact names cannot change type after creation, and this
overlay is materialized as a dataset fragment.

The labels artifact is intended to be materialized together with the source dataset artifact:

```text
encord-source-data:vN + encord-single-view-labels:vM => local dataset/
```

The source dataset artifact provides:

```text
dataset/videos/...
```

The labels artifact provides:

```text
dataset/data/chunk-000/episode_000000.parquet
dataset/meta/info.json
dataset/meta/tasks.jsonl
dataset/meta/episodes.jsonl
```

Each output parquet preserves the source columns such as `action`, `observation.state`, `timestamp`,
and `frame_index`, then fills `task_index` and the `annotation.language.language_instruction{,_2,_3}`
columns from the Encord caption.

## Convert Captions To DROID Layout

```bash
uv run --script scripts/encord/label-export/export_encord_captions_to_droid.py --help
```

Use this for local DROID/LeRobot-style label files, not W&B versioning.
