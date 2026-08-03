# Encord → W&B

This directory contains the useful Encord curation workflow from the webinar without the old recovery,
cache, migration, and bulk-operations scripts:

```text
raw Encord videos + metadata
              ↓  create_data_groups_from_raw_folder.py
three-camera data-group dataset
              ↓  captioning/create_caption_ontology.py
caption ontology + dataset-attached project
              ↓  captioning/create_captions_from_metadata.py
deterministic task captions
              ↓  captioning/gemini_caption_agent/main.py (optional refinement)
reviewed caption project
              ↓  export_to_wandb.py
one train-ready W&B dataset artifact
              ↓  register_wandb_videos.py (optional)
externally referenced Encord video dataset

create_sampled_dataset.py provides an optional, reproducible
metadata-balanced branch before creating the corresponding caption project.
```

The data-group builder reuses existing Encord storage items; it does not transfer media. The exporter
never downloads or re-uploads source videos either. Each video remains in S3 and is added to the W&B
artifact as an external `s3://` reference. Only the source `info.json` and per-episode Parquet data are
read; generated train-ready Parquet and JSON files live in an auto-deleted temporary directory for the
duration of the run. There is no persistent export cache.

## Prerequisites

- `uv` and Python 3.11 or 3.12.
- An Encord SSH private-key file with access to the dataset and project.
- A Gemini API key when running the caption agent.
- AWS credentials with read access to the referenced source objects when exporting or inferring active
  arms. The standard boto3 credential chain is used; pass `--aws-profile` only when a named profile is
  required.
- W&B authentication and write access to the destination entity/project.
- An Encord S3 integration that can access the referenced bucket when registering W&B references back in
  Encord.

Commands validate their inputs before the first write and then execute immediately. The scripts use the
public PyPI Encord SDK and accept
`--domain https://api.us.encord.com` for the US deployment.
Registration recovery, migration, re-encoding, and local-cache repair remain outside the public workflow.
The focused [`captioning/README.md`](captioning/README.md) contains the shortest project-setup path.

## 1. Build three-camera data groups

The source storage folder must contain one `cam_high`, `cam_left_wrist`, and `cam_right_wrist` video per
episode plus at least one JSON/JSONL metadata item. The builder joins items by `episode_path`, validates
camera uniqueness and consistent task/date metadata, creates the custom Encord layout, and produces one
grouped dataset. Existing groups in `--output-folder-id` are reused by episode path.

Create the grouped dataset:

```bash
uv run --script scripts/encord/create_data_groups_from_raw_folder.py \
  --ssh-key-file /path/to/encord-key \
  --source-folder-id <raw-folder-id> \
  --output-folder-name "Trossen data groups" \
  --output-dataset-title "Trossen grouped demonstrations"
```

To populate existing resources instead, pass `--output-folder-id` and/or `--dataset-hash`. `--limit 3`
creates a real three-episode dataset after the whole source folder passes validation.

## 2. Create the caption ontology and project

[`captioning/caption_ontology.json`](captioning/caption_ontology.json) is the complete public ontology
contract: three optional free-text classifications named `Language Instruction 1/2/3`. Create just the
ontology with:

```bash
uv run --script scripts/encord/captioning/create_caption_ontology.py \
  --ssh-key-file /path/to/encord-key
```

Create the ontology and a new project attached to the grouped dataset:

```bash
uv run --script scripts/encord/captioning/create_caption_ontology.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <grouped-dataset-id> \
  --project-title "Trossen Captioning"
```

Omit the project flags to create only the ontology. Pass `--workflow-template-hash` with the project flags
when the organization already has a suitable caption-review workflow template. The script creates a new
project because Encord binds the ontology at project creation; it does not replace an existing project's
ontology.

To reuse an existing caption ontology for another dataset:

```bash
uv run --script scripts/encord/captioning/create_caption_project.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <grouped-dataset-id> \
  --ontology-hash <caption-ontology-id> \
  --project-title "Trossen Captioning"
```

Both project-creation paths verify the dataset and require all three Language Instruction
classifications before creating the project. Pass `--workflow-template-hash` when the project should use
an existing review workflow template.

## 3. Create captions from task metadata

The deterministic captioner restores the webinar's known task mapping from the editable
[`captioning/task_captions.yaml`](captioning/task_captions.yaml). Add tasks or edit their canonical,
paraphrased, and arm-aware instructions without changing Python. The script resolves `task_name` from the
backing item's metadata, `episode_path`, row title, or child metadata, then writes `Language Instruction
1/2/3`. Unknown task names and partial caption rows are reported as hard failures.

```bash
uv run --script scripts/encord/captioning/create_captions_from_metadata.py \
  --ssh-key-file /path/to/encord-key \
  --project-hash <caption-project-id>
```

The command saves immediately. `--overwrite` replaces only the three Language Instruction fields; other
classifications and objects on the label row are preserved. Pass `--caption-map path/to/tasks.yaml` to
use a different mapping.

The default third instruction says `the robot arm`. To restore the webinar's useful motion-aware variant,
add `--infer-active-arm`:

```bash
uv run --script scripts/encord/captioning/create_captions_from_metadata.py \
  --ssh-key-file /path/to/encord-key \
  --project-hash <caption-project-id> \
  --infer-active-arm
```

This reads only the `observation.state` and `action` columns from each source Parquet object into memory.
It compares robust motion scores for the left and right seven-joint slices and writes `the left arm`,
`the right arm`, or `both robot arms` into `Language Instruction 3`. No Parquet is written to disk or
cached. All episodes must pass before any labels are saved.

## 4. Refine captions with Gemini

The restored Gemini agent keeps the original Encord Runner, prompt, structured response, label-writing,
and workflow-routing behavior. Its public cleanup removes multi-worker sharding, worker locks, persistent
video/proxy caches, failure journals, and startup retry tuning.

Configure its public placeholders, then validate and run:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py check
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run
```

The selected Encord video is downloaded only into the agent library's temporary task context, uploaded to
Gemini, and removed when the task finishes. The Gemini upload is also deleted after processing by default.
See [`captioning/gemini_caption_agent/README.md`](captioning/gemini_caption_agent/README.md) for the exact
project contract, configuration, validation, caption rules, and workflow routes. Use this after the
metadata captioner when visual inspection should improve or reject the deterministic baseline.

## 5. Create a metadata-balanced sample

The sampler selects rows hierarchically across metadata fields and creates a new dataset containing the
same backing Encord items. Selection is deterministic with the default seed and never copies media.

```bash
uv run --script scripts/encord/create_sampled_dataset.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <source-dataset-id> \
  --target-dataset-size 100
```

By default it balances first across `task_name`, then `collection_datetime`. Repeat `--stratify-by` to
choose another order or use `--seed` to select another reproducible sample. Create or attach the
corresponding caption project before exporting that sampled dataset.

## 6. Export

Create the train-ready W&B artifact:

```bash
uv run --script scripts/encord/export_to_wandb.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <dataset-id> \
  --project-hash <project-id> \
  --wandb-entity <entity> \
  --wandb-project <project>
```

The artifact collection defaults to `encord-train-ready` with the `latest` alias. Use `--artifact-name`
or repeat `--alias` to override them. `--limit 3` publishes a small three-episode artifact.
For Encord's US deployment, pass `--domain https://api.us.encord.com`.

## 7. Register W&B video references back in Encord

The reverse bridge is useful when a W&B artifact version represents the video selection that should be
reviewed in Encord. It reads the artifact manifest, requires one external reference for each of the three
webinar cameras per episode, and registers those existing S3 objects through an Encord cloud integration:

```bash
uv run --script scripts/encord/register_wandb_videos.py \
  --artifact-ref <entity>/<project>/encord-train-ready:latest \
  --s3-region <aws-region> \
  --ssh-key-file /path/to/encord-key \
  --integration-name <encord-s3-integration-name>
```

`--limit 3` registers three complete episodes after the whole artifact validates. The script never calls
the W&B artifact download API and never reads video bytes from S3. W&B supplies the reference URIs and
Encord's existing integration registers the corresponding permanent HTTPS object URLs. The resulting
dataset contains the referenced videos and provenance metadata; it is not a reconstruction of the
generated LeRobot Parquet files in the artifact.

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

## Smoke checks

Start with `--help`. For a small real run, use `--limit 3` on grouping, export, or reference registration,
or choose a small `--target-dataset-size` when sampling. These commands create real Encord resources or a
real W&B artifact, so use disposable titles for smoke runs. No synthetic test harness or fake service
clients are shipped in this public workflow.
