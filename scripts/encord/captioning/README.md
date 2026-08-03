# Captioning projects

This folder covers the complete caption-project setup: create the ontology, attach a dataset, seed three
language instructions, and optionally refine them with Gemini. Commands validate their inputs and then
execute immediately.

## Start a new captioning project

The bundled [`caption_ontology.json`](caption_ontology.json) defines three optional text fields named
`Language Instruction 1`, `Language Instruction 2`, and `Language Instruction 3`.

Create that ontology and a project attached to an existing dataset in one command:

```bash
uv run --script scripts/encord/captioning/create_caption_ontology.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <dataset-id> \
  --project-title "Robot Captioning"
```

Omit `--dataset-hash` and `--project-title` to create only the ontology.

## Reuse an existing caption ontology

Create another project without duplicating the ontology:

```bash
uv run --script scripts/encord/captioning/create_caption_project.py \
  --ssh-key-file /path/to/encord-key \
  --dataset-hash <dataset-id> \
  --ontology-hash <ontology-id> \
  --project-title "Robot Captioning"
```

The command verifies that the dataset exists, is non-empty, and the ontology contains all three caption
fields before creating the project. Either creation command accepts
`--workflow-template-hash <template-id>` for an existing caption-review workflow.

## Seed captions

[`task_captions.yaml`](task_captions.yaml) is the editable mapping from task metadata to three instruction
variants:

```bash
uv run --script scripts/encord/captioning/create_captions_from_metadata.py \
  --ssh-key-file /path/to/encord-key \
  --project-hash <project-id>
```

Add `--infer-active-arm` to describe left-, right-, or both-arm motion in the third instruction. Parquet
columns are read directly into memory and are never cached. Use `--overwrite` to replace only the three
caption fields.

For visual refinement and workflow routing, continue with the cleaned
[`gemini_caption_agent`](gemini_caption_agent/README.md).
