# Encord Scripts

Small utilities for registering Encord data, creating data groups, exporting datasets and labels, and versioning those exports in W&B.

Set your Encord key once before scripts that call Encord:

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_key
```

AWS-backed scripts default to profile `encord-robotics`.

## Data Registration

Use Encord Cloud Synced Folders to ingest the S3 data, then update item metadata from `registration.json`:

```bash
cd scripts/encord/data-registration
uv run --script build_registration_json.py --output registration.json
uv run --script update_metadata_schema.py registration.json --dry-run
uv run --script update_metadata_schema.py registration.json
uv run --script update_cloud_synced_folder_metadata.py registration.json --dry-run
uv run --script update_cloud_synced_folder_metadata.py registration.json
```

`update_cloud_synced_folder_metadata.py` recursively lists the Cloud Synced Folder, matches Encord storage items to `registration.json`, merges `clientMetadata`, skips items already up to date, and writes a report.

### Data Registration Scripts

`inspect_s3_bucket.py`  
Prints a quick view of the S3 folder structure, file extensions, and sample objects.

`build_registration_json.py`  
Builds `registration.json` from S3 object paths plus metadata sidecars such as `info.json` and parquet summaries. We now mainly use this JSON as a metadata source, not for uploading files.

`update_metadata_schema.py`  
Reads `registration.json` and creates/updates Encord metadata schema fields, including enum values for task, environment, source family, camera, and related fields.

`update_cloud_synced_folder_metadata.py`  
Applies metadata from `registration.json` to files already loaded by a Cloud Synced Folder. Use `--dry-run` first. Progress prints every 2,000 listed items by default; change with `--progress-interval`.

`load_registration_json.py`  
Older direct-registration path for loading a JSON into an Encord storage folder. Prefer Cloud Synced Folders for this dataset.

`create_metadata_schema.py`  
Compatibility wrapper. Prefer `update_metadata_schema.py`.

Useful options:

```bash
uv run --script update_cloud_synced_folder_metadata.py --dry-run
uv run --script update_cloud_synced_folder_metadata.py --progress-interval 5000
uv run --script update_cloud_synced_folder_metadata.py --report-json metadata_report.json
```

Generated `*_report*.json` files are audit logs and can be regenerated.

## W&B Versioning

We use two W&B artifact families that overlay into one LeRobot/DROID-style dataset.

### Dataset Artifact

The dataset artifact versions the source videos.

- `v0`: 2-3 episode smoke test with metadata.
- `v1`: random 500 episode sample from an Encord dataset.
- `v2`: curated dataset.

The dataset export starts from an Encord dataset whose rows are 3-video data groups. It resolves each video back to S3 from Encord client metadata, downloads the videos locally, and logs them to W&B under LeRobot/DROID-style paths such as:

```text
dataset/.../videos/...
```

It should also log source lineage: Encord dataset hash, data group UUIDs, video storage item UUIDs, S3 paths, and client metadata.

Run:

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key

uv run --script scripts/encord/dataset-export/export_encord_dataset_to_wandb.py \
  --dataset-hash <encord_dataset_hash> \
  --limit 3 \
  --alias v0
```

The script writes local files under `exports/encord-dataset-export/<timestamp>/` and logs `encord-source-data:vN` to W&B. Inside the artifact, videos are stored at:

```text
dataset/videos/chunk-000/observation.images.exterior_image_1_left/episode_000000.mp4
dataset/videos/chunk-000/observation.images.wrist_image_left/episode_000000.mp4
dataset/videos/chunk-000/observation.images.wrist_image_right/episode_000000.mp4
```

### Labels Artifact

The labels artifact versions label/caption/metadata trials.

Each labels artifact contains the missing LeRobot/DROID pieces, such as:

```text
dataset/.../meta/...
dataset/.../data/...
```

Each labels artifact must record the exact dataset artifact it overlays:

```yaml
source_dataset_artifact: encord-source-data:vN
```

### Training

A labels artifact alone is not trainable. To train, materialize both artifacts:

```text
encord-source-data:vN + encord-labels:vM => local dataset/
```

The episode paths must match across both artifacts. W&B artifact lineage is the source of truth for which dataset version was used with which label version.
