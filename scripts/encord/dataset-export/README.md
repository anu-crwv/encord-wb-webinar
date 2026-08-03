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

For large video exports, put W&B's local artifact working directory on a volume with enough free space:

```bash
WANDB_DATA_DIR=/Volumes/big-disk/wandb-data AWS_PROFILE=encord-robotics uv run --script scripts/encord/dataset-export/export_encord_dataset_to_wandb.py \
  --dataset-hash <encord_dataset_hash>
```

The exporter prints progress while registering local files with the artifact and emits an upload/finalization heartbeat every
60 seconds. Adjust it with `--wandb-upload-heartbeat-seconds`, or set it to `0` to disable.

Local output:

```text
exports/encord-dataset-export/<timestamp>/
```

W&B output:

```text
<source_artifact_name>:vN
```

## Upload Existing Shared S3 Cache

To register the local shared S3 cache in W&B without calling Encord or copying the cache into W&B storage, use:

```bash
uv run --script scripts/encord/dataset-export/upload_cached_s3_to_wandb.py \
  exports/encord-dataset-export/_cache/s3/ego-data-collection-encord/raw-feed/some-prefix \
  --dry-run --max-episodes 5
uv run --script scripts/encord/dataset-export/upload_cached_s3_to_wandb.py \
  exports/encord-dataset-export/_cache/s3/ego-data-collection-encord/raw-feed/some-prefix
```

This uses the same W&B config and dataset export tags/description as the Encord dataset exporter:

```text
scripts/encord/wandb_config.yaml
scripts/encord/dataset-export/dataset_export_config.yaml
```

Artifact paths are relative to the selected folder by default, so the selected folder's internal structure is preserved. The artifact entries are `s3://...` references by default, so W&B stores the manifest/source links rather than staging or uploading another copy of the cached files. To upload the whole shared S3 cache, select `exports/encord-dataset-export/_cache/s3`.

The local paths must be under `exports/encord-dataset-export/_cache/s3`, where the next path segment is the S3 bucket name. Use `--s3-cache-root` if your cache root is different. By default, `--no-reference-checksum` avoids an S3 object metadata scan while creating the W&B artifact.

When `episode_*` folders are present, incomplete episodes are skipped by default. A complete episode must include
`meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, and at least 3 visible video
files. Use `--max-episodes` for smoke tests so the script still uploads whole episodes.

Encord-only options from the dataset exporter, such as `--dataset-hash`, `--limit`, and `--unsigned-s3`, are accepted as no-ops for command compatibility.

## Recover an Encord Project from R2 MCAPs

`recover_encord_project_from_r2_mcaps.py` reads an Encord project's data groups, maps each original
`raw-feed/trossen-data/.../episode_*` path to its exact
`r2://trossen-robotics-data/trossen-data-mobile/.../episode_*.mcap` object, and runs a streaming pipeline:

- R2 downloads use concurrent threads and reuse size-matched files in the shared R2 cache.
- Completed MCAP downloads are immediately handed to separate spawned extraction processes.
- Each extractor stream-copies the three compressed camera topics into MP4 without re-encoding.
- Only episodes with all three validated videos receive a completion marker and enter `file_map.json`.
- Original Encord item metadata, group metadata, camera roles, and source identifiers are preserved for re-upload.
- Missing camera-role metadata is inferred from canonical item paths, and duplicate camera references are
  resolved deterministically. Repairs and source warnings are included in `project_recovery_manifest.json`.

Load the ignored local credential file, then run a one-episode smoke test:

```bash
set -a
source .env.r2-recovery
set +a

uv run --script scripts/encord/dataset-export/recover_encord_project_from_r2_mcaps.py \
  08411c84-7e66-4ad9-a63d-d948f9e821a1 \
  --episode-path-contains episode_000258/ \
  --limit 1
```

For the full project, omit the filter and limit. Defaults use 16 download threads and 4 extraction
processes; adjust them with `--download-workers` and `--extract-workers`.

MCAP cache:

```text
exports/encord-dataset-export/_cache/r2/trossen-robotics-data/
```

Upload-ready output:

```text
exports/encord-dataset-export/recovered/r2/trossen-robotics-data/<project_hash>/
```

The output mirrors the original episode paths, includes the four `meta/*` files required by the local
uploader, and writes `file_map.json` for restoring custom three-camera data groups.

Upload the recovered files and rebuild those groups in an existing Encord storage folder:

```bash
uv run --script scripts/encord/data-registration/upload_cached_s3_folder_to_encord.py \
  <encord_folder_hash> \
  --data-dir exports/encord-dataset-export/recovered/r2/trossen-robotics-data/08411c84-7e66-4ad9-a63d-d948f9e821a1
```

The uploader is resumable: it reuses UUIDs for files already in the folder, uploads only missing files,
builds groups once all seven items are present, and skips groups that already exist.
