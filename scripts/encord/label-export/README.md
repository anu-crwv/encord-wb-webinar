# Encord Labels To W&B

Version one Encord project's source dataset and label export in W&B.

## Config Files

Edit label/version notes here:

```text
scripts/encord/label-export/export_metadata.yaml
```

Edit stable W&B settings here:

```text
scripts/encord/wandb_config.yaml
```

`source_artifact_ref` is optional in `export_metadata.yaml`. Set it only when a new label export should point at an existing dataset artifact version:

```yaml
source_artifact_ref: encord-source-data:v0
```

## Run

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key

uv run --script scripts/encord/label-export/export_single_view_labels_to_wandb.py \
  --metadata-yaml scripts/encord/label-export/export_metadata.yaml
```

The script finds the project's attached dataset. If the project has zero or multiple datasets, it fails with a clear error.

## Outputs

Local files are written under `exports/encord-label-export/<timestamp>/`:

- `source_dataset_manifest.json`
- `source_dataset_items.json`
- `encord_labels.json`
- `encord_data_metadata.json`
- `label_preview_rows.json`
- `wandb_lineage.json`

W&B receives:

- `encord-source-data:vN` as a `dataset` artifact, with `source_dataset_manifest.json` and `source_dataset_items.json`
- `encord-single-view-labels:vN` as a `labels` artifact, with `encord_labels.json` and `label_preview_rows.json`
- `encord_single_view_labels` preview table

`source_dataset_items.json` is the dataset lineage record. Each item includes the Encord data hash, data title/type, Encord storage item UUID, source S3 URI when present, and full Encord client metadata.

The label export run uses the source dataset artifact, so W&B lineage shows which dataset version produced which label version.
