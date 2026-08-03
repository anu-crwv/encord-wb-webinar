# Gemini Caption Agent

This is the webinar's Encord task agent for generating robotics captions with Gemini. It keeps the
proven workflow behavior from the original implementation while removing the multi-worker, cache,
proxy-video, and recovery machinery.

The agent runs against an existing Encord project with:

- An agent stage named `Gemini Captioning`.
- Outgoing pathways named `success`, `human_review`, and `failure`.
- Text classifications named `Language Instruction 1`, `Language Instruction 2`, and
  `Language Instruction 3`.
- A `task_name` metadata value on the row's storage item or one of its children.

## Setup

Export the two credentials used by the existing Encord Agents runner and Gemini client:

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
export GEMINI_API_KEY=...
```

They can instead be placed in an ignored `.env` file in this directory or at the repository root.

Copy the public placeholders in `config.yaml` to the project and workflow names you want to run:

```yaml
project_hash: "<project-hash>"
agent_stage_name: "Gemini Captioning"
success_pathway: "success"
failure_pathway: "failure"
metadata_mismatch_pathway: "human_review"
video_layout: "camera_cam_high"
```

## Validate

Check authentication, project access, the workflow stage, ontology classifications, task metadata,
and the selected video layout:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py check
```

To inspect authentication without printing secrets:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py debug-auth
```

## Run

Process the pending tasks currently at the configured agent stage:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run
```

Use another config file when needed:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run \
  --config /path/to/config.yaml
```

The default is intentionally a single local runner with `task_batch_size: 1`. `max_tasks_per_stage`
can limit a smoke run, and `refresh_every` can keep the runner polling. There is no task sharding,
worker lock, persistent failure journal, video cache, or ffmpeg proxy.

## Task flow

For each pending task, the agent:

1. Initializes the Encord label row.
2. Routes already-complete rows to `success` unless `overwrite` is enabled.
3. Selects the configured video view from the data group.
4. Falls back to another video child when the configured layout is unavailable and marks the row for
   human review.
5. Temporarily downloads the selected Encord asset through `download_asset`.
6. Uploads that video to Gemini and waits for Gemini file processing.
7. Requests a structured three-caption response plus a metadata-mismatch decision.
8. Deletes the uploaded Gemini file unless `keep_uploaded_files` is enabled.
9. Validates all captions before adding any label instances.
10. Saves the Encord label row once and returns the configured workflow pathway.

The temporary Encord asset is removed when the task finishes. No downloaded video or generated proxy is
retained in the repository.

## Gemini output

Gemini must produce:

```json
{
  "language_instruction_1": "detailed whole-episode instruction",
  "language_instruction_2": "short paraphrase of the same instruction",
  "language_instruction_3_action": "action phrase following use the robot arm to",
  "metadata_mismatch": false
}
```

`Language Instruction 3` is assembled in Python so it always begins with
`use the robot arm to ...`.

The agent rejects empty or duplicate captions and wording about cameras, videos, frames, timestamps,
uncertainty, success, or failure. Invalid model output is not saved.

## Routing

- `success`: captions validate, the preferred layout exists, and metadata matches the visible task.
- `human_review`: captions validate, but Gemini reports a clear metadata mismatch or layout selection
  had to fall back.
- `failure`: authentication after startup, video selection/download, Gemini, parsing, validation, or
  label writing fails.

The metadata task name is a quality-control signal. Captions are generated from the video content, not
copied from metadata.
