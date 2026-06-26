# Captioning

Creates Encord language-instruction labels from dataset metadata.

## Setup

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
aws sso login --profile encord-robotics
```

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

`create_empty_language_instruction.py` adds empty language fields to an existing project.

`teleport_workflow_tasks.py` moves workflow tasks between Encord workflow stages.
