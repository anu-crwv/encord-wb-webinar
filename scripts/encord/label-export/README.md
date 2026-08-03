# Label Overlay Export

Exports one Encord project's single-view caption labels to a W&B label overlay artifact that materializes
with the 3-camera source dataset artifact.

The exporter reuses the source episode parquet files from S3 for state/action/timing data, then rewrites
the task and language annotation columns from Encord captions. It does not fabricate state/action values.

## Supported Project Shape

This exporter supports both single-video/single-view caption projects and data-group caption projects.
Each Encord label row must have a language-instruction classification and resolve to one episode.

Single-video rows can resolve `episode_path` and `source_parquet_uri` from the row's backing storage item
metadata. Data-group rows only need the top-level group storage item to carry `episode_path`; the exporter
matches that path to the required `--source-artifact-ref` and derives the original source/parquet URIs from
that source artifact's per-camera child metadata.

The exporter does not inspect data-group child items live in Encord. The W&B source dataset artifact is the
source of truth for data-group video source metadata.

## Setup

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
```

Edit run notes and artifact tags:

```text
scripts/encord/label-export/label_export_config.yaml
```

Edit W&B settings:

```text
scripts/encord/wandb_config.yaml
```

## Export Label Overlay To W&B

```bash
uv run --script scripts/encord/label-export/export_single_view_labels_to_wandb.py \
  --metadata-yaml scripts/encord/label-export/label_export_config.yaml \
  --source-artifact-ref encord-source-data:v0 \
  --limit 3
```

For a full export, omit `--limit`.

Writes local files to:

```text
exports/encord-label-export/<timestamp>/
```

Logs:

- label overlay artifact with `dataset/data/...` and `dataset/meta/...`
- configured tags as W&B artifact tags, with the `latest` alias applied automatically
- preview table with `language_instruction`

The W&B artifact type is `dataset` because W&B artifact names cannot change type after creation, and this
overlay is materialized as a dataset fragment.

The label overlay artifact is intended to be materialized together with the source dataset artifact:

```text
encord-source-data:vN + encord-captions:vM => local dataset/
```

The source dataset artifact provides:

```text
dataset/videos/...
```

The label overlay artifact provides:

```text
dataset/data/chunk-000/episode_000000.parquet
dataset/meta/info.json
dataset/meta/tasks.jsonl
dataset/meta/episodes.jsonl
```

Each output parquet preserves the source columns such as `action`, `observation.state`, `timestamp`,
and `frame_index`, then fills `task_index` and the `annotation.language.language_instruction{,_2,_3}`
columns from the Encord caption.
