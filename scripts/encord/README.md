# Encord Scripts

Small utilities for inspecting S3 data, building metadata from file paths and sidecars, and applying that metadata to Encord.

Run scripts from `scripts/encord/data-registration` with `uv run --script <script>.py`. Set your Encord key once before scripts that call Encord:

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_key
```

AWS-backed scripts default to profile `encord-robotics`.

## Recommended Workflow

Use Encord Cloud Synced Folders to ingest the S3 data, then update item metadata from `registration.json`:

```bash
uv run --script build_registration_json.py --output registration.json
uv run --script update_metadata_schema.py registration.json --dry-run
uv run --script update_metadata_schema.py registration.json
uv run --script update_cloud_synced_folder_metadata.py registration.json --dry-run
uv run --script update_cloud_synced_folder_metadata.py registration.json
```

`update_cloud_synced_folder_metadata.py` recursively lists the Cloud Synced Folder, matches Encord storage items to `registration.json`, merges `clientMetadata`, skips items already up to date, and writes a report.

## Scripts

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

## Useful Options

```bash
uv run --script update_cloud_synced_folder_metadata.py --dry-run
uv run --script update_cloud_synced_folder_metadata.py --progress-interval 5000
uv run --script update_cloud_synced_folder_metadata.py --report-json metadata_report.json
```

Generated `*_report*.json` files are audit logs and can be regenerated.
