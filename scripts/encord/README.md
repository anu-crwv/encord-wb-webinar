# Encord → W&B

This directory contains one supported workflow for the webinar dataset:

```text
prepared Encord data-group dataset + caption project
                         ↓
                export_to_wandb.py
                         ↓
          one train-ready W&B dataset artifact
```

The exporter never downloads or re-uploads source videos. Each video remains in S3 and is added to the
W&B artifact as an external `s3://` reference. Only the source `info.json` and per-episode Parquet data
are read; generated train-ready Parquet and JSON files live in an auto-deleted temporary directory for
the duration of the run. There is no persistent export cache.

## Prerequisites

- `uv` and Python 3.11 or 3.12.
- An Encord SSH private-key file with access to the dataset and project.
- AWS credentials with read access to the referenced source objects. The standard boto3 credential chain
  is used; pass `--aws-profile` only when a named profile is required.
- W&B authentication and write access to the destination entity/project.

The Encord dataset and caption project must already exist. Dataset registration, data-group creation,
caption generation, recovery, migration, and video re-encoding are deliberately outside this public
workflow.

## Export

First validate the complete export without creating a W&B run:

```bash
uv run --script scripts/encord/export_to_wandb.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <dataset-id> \
  --project-hash <project-id> \
  --wandb-entity <entity> \
  --wandb-project <project>
```

Publish the same validated shape by adding `--apply`:

```bash
uv run --script scripts/encord/export_to_wandb.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <dataset-id> \
  --project-hash <project-id> \
  --wandb-entity <entity> \
  --wandb-project <project> \
  --apply
```

The artifact collection defaults to `encord-train-ready` with the `latest` alias. Use `--artifact-name`
or repeat `--alias` to override them. `--limit 3` provides a small validation or publication smoke run.
For Encord's US deployment, pass `--domain https://api.us.encord.com`.

## Dataset contract

This exporter intentionally supports the webinar's Trossen dataset rather than claiming to handle every
possible Encord project:

- The project has exactly one attached dataset, matching `--dataset-hash`.
- Every dataset row is an Encord data group containing exactly one `cam_high`, `cam_left_wrist`, and
  `cam_right_wrist` child.
- Child metadata contains `camera_name`, `episode_path`, and an S3 `source_uri`. The group or first camera
  contains `source_parquet_uri`; otherwise the standard episode path is used to derive it.
- Each project label row contains a numbered `Language Instruction`; the first number is the task caption.
- Source Parquet contains `action`, `observation.state`, `timestamp`, and `frame_index`. State/action
  vectors must contain at least the webinar's 16-value Trossen layout.
- Source `meta/info.json` declares all three camera features and one consistent FPS across every episode.

Missing captions/cameras/metadata, duplicate hashes or episode paths, inconsistent FPS, and incompatible
vectors fail the export instead of silently skipping data.

## Artifact layout

```text
dataset/
├── videos/chunk-000/observation.images.<camera>/episode_000000.mp4  # S3 reference
├── data/chunk-000/episode_000000.parquet                            # generated
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    ├── stats.json
    ├── relative_stats_dreamzero.json
    ├── modality.json
    └── embodiment.json
encord_export_manifest.json
```

The manifest records the Encord dataset/project IDs, data-group and storage-item IDs, label IDs, source
URIs, W&B destination, artifact paths, schema version, and aggregate counts. It never contains credentials.

## Test

The regression suite is self-contained and performs no Encord, S3, or W&B network writes:

```bash
uv run --script scripts/encord/tests/test_export_to_wandb.py
```
