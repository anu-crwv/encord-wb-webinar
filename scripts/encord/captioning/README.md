# Captioning

Creates Encord language-instruction labels from dataset metadata.

## Setup

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
aws sso login --profile encord-robotics
```

## Supported Project Shapes

`create_captions_v1.py` supports both single-video dataset rows and data-group dataset rows. For data groups,
it reads the group item metadata first, then falls back to child item metadata to resolve `task_name`, episode
metadata, and the source parquet.

`create_captions_from_metadata.py` supports existing single-video and data-group projects. It writes one
`Language Instruction` classification. For videos, the classification is applied across the full video range.
For data groups, the classification is written at group level. Task names are resolved from top-level metadata
first, then `episode_path`, then the row title, with child metadata used only as a last fallback.

## Create V1 Caption Project

Creates a new Encord project from a dataset hash, attaches the default ontology, and writes:

- `Language Instruction 1`: canonical task instruction
- `Language Instruction 2`: safe paraphrase
- `Language Instruction 3`: arm-aware instruction from source parquet state/action

```bash
AWS_PROFILE=encord-robotics uv run --script scripts/encord/captioning/create_captions_v1.py \
  <encord_dataset_hash> \
  "Project title"
```

Use `--dry-run --limit 3` for a smoke check before creating the project.

If S3 access is public/anonymous, use:

```bash
uv run --script scripts/encord/captioning/create_captions_v1.py \
  <encord_dataset_hash> \
  "Project title" \
  --unsigned-s3
```

Source parquets are cached in the main worktree's shared S3 cache:

```text
/Users/encordsf/Desktop/encord-wb-webinar/exports/encord-dataset-export/_cache/s3/
```

## Other Scripts

`create_captions_from_metadata.py` applies one `Language Instruction` classification to an existing single-video
or data-group project from dataset metadata.

`create_empty_language_instruction.py` adds empty language fields to an existing project.

`teleport_workflow_tasks.py` moves workflow tasks between Encord workflow stages.
