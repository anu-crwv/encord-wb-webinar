# Encord x W&B Versioning

We use two W&B artifact families that overlay into one LeRobot/DROID-style dataset.

## Dataset Artifact

The dataset artifact versions the source videos.

- `v0`: 2-3 episode smoke test with metadata.
- `v1`: random 500 episode sample from an Encord dataset.
- `v2`: curated dataset.

The dataset export starts from an Encord dataset whose rows are 3-video data groups. It resolves each video back to S3 from Encord client metadata, downloads the videos locally, and logs them to W&B under LeRobot/DROID-style paths such as:

```text
dataset/.../videos/...
```

It should also log source lineage: Encord dataset hash, data group UUIDs, video storage item UUIDs, S3 paths, and client metadata.

## Labels Artifact

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

## Training

A labels artifact alone is not trainable. To train, materialize both artifacts:

```text
encord-source-data:vN + encord-labels:vM => local dataset/
```

The episode paths must match across both artifacts. W&B artifact lineage is the source of truth for which dataset version was used with which label version.
