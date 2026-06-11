# Data Groups

Set Encord auth first:

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
```

## Create Groups From Raw Folder

Creates 3-camera video data groups with metadata JSONs in a carousel. By default it upserts into folder `1fd8f6ec-afd1-47c0-8498-07448a9dc8e9` and skips episodes already present.

```bash
uv run --script scripts/encord/data-groups/create_data_groups_from_raw_folder.py --no-limit
```

Create only 5 by default:

```bash
uv run --script scripts/encord/data-groups/create_data_groups_from_raw_folder.py
```

Create a fresh folder instead:

```bash
uv run --script scripts/encord/data-groups/create_data_groups_from_raw_folder.py \
  --output-folder-name test-data-groups
```

## Convert Single-Video Dataset

Takes an Encord dataset of single videos, finds matching data groups in folder `1fd8f6ec-afd1-47c0-8498-07448a9dc8e9`, and creates a new dataset backed by those groups.

```bash
uv run --script scripts/encord/data-groups/create_group_dataset_from_single_video_dataset.py \
  --source-dataset-hash <single_video_dataset_hash> \
  --output-dataset-title "v0 Data Groups"
```

Use `--debug` if matching looks wrong.
