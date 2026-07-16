# Gemini Caption Agent

Local Encord task agent that captions robotics videos with Gemini and writes the three training language instructions back to Encord.

The agent is intended for an existing Encord project with an agent workflow stage and three text classifications:

- `Language Instruction 1`
- `Language Instruction 2`
- `Language Instruction 3`

It captions from video content, not from the metadata task name. The metadata task name is only used as a QC signal so Gemini can flag rows where the visible task is clearly different from the metadata.

## Setup

Set Encord auth and Gemini credentials:

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
export GEMINI_API_KEY=...
```

Or put the Gemini key in an ignored `.env` file in this folder:

```text
GEMINI_API_KEY=...
```

Edit:

```text
scripts/encord/captioning/gemini_caption_agent/config.yaml
```

## Commands

Validate project access, workflow stage, caption classifications, Gemini auth, task metadata, and sample video selection:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py check
```

Debug local auth without printing secrets:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py debug-auth
```

Run the local task agent:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run
```

Use a different config:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run \
  --config /path/to/config.yaml
```

## Config

Important fields:

- `project_hash`: Encord workflow project to process.
- `agent_stage_name`: agent stage where pending tasks are pulled from.
- `success_pathway`: route for valid captions with no mismatch.
- `failure_pathway`: route for download, Gemini, parsing, or validation failures.
- `metadata_mismatch_pathway`: route for valid captions that need human review.
- `metadata_task_key`: exact client-metadata key for the expected task name.
- `video_layout`: one preferred data-group child layout to send to Gemini.
- `local_video_cache_dir`: local cache for downloaded videos.
- `use_gemini_video_proxy`: create a smaller full-episode MP4 for Gemini upload.
- `gemini_video_proxy_width`: proxy width in pixels.
- `gemini_video_proxy_fps`: proxy frame rate.
- `gemini_video_proxy_crf`: proxy H.264 quality/size tradeoff.
- `overwrite`: whether to replace existing caption answers.
- `task_batch_size`: recommended value is `3`.
- `parallel_worker_count`: total number of local workers intentionally sharing this project/stage.
- `parallel_worker_index`: zero-based shard index for this worker.
- `worker_lock_dir`: local directory for per-shard lock files.

`task_batch_size: 3` is the recommended local-runner default here. Each task performs a video cache lookup or download, Gemini file upload, model call, Encord label save, and workflow route. Drop to `1` when debugging one row at a time, or if Gemini quota/rate limits get noisy.

`use_gemini_video_proxy: true` is recommended. The agent downloads and caches the original selected video, then creates a lower-resolution, lower-fps, full-episode MP4 under the cache's `_gemini_proxy/` folder. Gemini receives the proxy, not the original. This preserves the whole task while making upload and Gemini video processing much faster.

## Parallel Workers

The Encord task runner does not locally claim a task before Gemini work starts. If two identical agent processes run against the same project/stage, they can both see the same pending task. This agent prevents that by sharding tasks deterministically by `data_hash`.

Use single-worker mode by default:

```yaml
parallel_worker_count: 1
parallel_worker_index: 0
```

For two terminals, create two config files:

```yaml
# config_worker_0.yaml
parallel_worker_count: 2
parallel_worker_index: 0
```

```yaml
# config_worker_1.yaml
parallel_worker_count: 2
parallel_worker_index: 1
```

Then run:

```bash
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run \
  --config config_worker_0.yaml
uv run --script scripts/encord/captioning/gemini_caption_agent/main.py run \
  --config config_worker_1.yaml
```

Shard rule:

```text
int(UUID(data_hash)) % parallel_worker_count == parallel_worker_index
```

Tasks outside the worker's shard return `None`, so they remain at the agent stage for the correct worker. The agent also takes a local lock for each project/stage/shard. Starting the same shard twice exits clearly; starting different shard indexes is allowed.

The original video cache and Gemini proxy cache can be shared by the workers. Temporary cache files include the process ID and a random suffix, then atomically replace the final cached file.

Both `check` and `run` load `.env` before creating the Encord task runner. If `ENCORD_SSH_KEY_FILE` is already exported in the shell, that shell value wins over `.env`.

## Task Flow

For each task at `agent_stage_name`:

1. If parallel sharding is enabled and the row does not belong to this worker, return `None`.
2. Initialize the Encord label row.
3. If all three caption classifications already have answers and `overwrite` is false, skip the row and route to `success_pathway`.
4. Resolve the expected task name from the exact `metadata_task_key`.
5. Select the video to send to Gemini.
6. Download the selected video through `download_asset` unless it is already cached.
7. Create or reuse the smaller Gemini proxy video when enabled.
8. Upload the proxy video to Gemini.
9. Ask Gemini for strict JSON with three captions plus a metadata-mismatch decision.
10. Validate the JSON and caption text.
11. Write the three Encord classification answers.
12. Save the label row and route the task.

The agent does not save partial caption sets after Gemini, parsing, or validation failures.

## Video Selection

The agent supports both single-video rows and data-group rows.

For a single-video row, the row storage item itself is sent to Gemini.

For a data group, the agent uses `video_layout` from config. Current Trossen data groups use:

```yaml
video_layout: "camera_cam_high"
```

Selection logic:

1. Look up the configured layout in `label_row.metadata.children`.
2. Match that child metadata to the storage child item.
3. If the layout cannot be found, choose a random video child instead.
4. When a random fallback is used, still generate and save captions, but route the task to `metadata_mismatch_pathway` for human review.

If the data group has no recognizable video children, the task routes to `failure_pathway`.

## Gemini Output

Gemini must return only this JSON shape:

```json
{
  "language_instruction_1": "detailed whole-episode instruction",
  "language_instruction_2": "short paraphrase of the same instruction",
  "language_instruction_3_action": "action phrase for the robot-arm instruction",
  "metadata_mismatch": false
}
```

Caption intent:

- `language_instruction_1`: detailed whole-episode command with visible objects, action, and final state.
- `language_instruction_2`: shorter safe paraphrase of the same task.
- `language_instruction_3_action`: action phrase that can follow `use the robot arm to`.

The prompt tells Gemini not to mention camera, video, frames, timestamps, uncertainty, success, or failure.

The agent builds `Language Instruction 3` in Python instead of trusting Gemini to write the final robot-arm phrase. It always writes `use the robot arm to ...`.

## Caption Validation

The agent rejects Gemini output when:

- JSON is missing or malformed.
- Any caption is empty.
- Any two captions are duplicates.
- `language_instruction_3_action` is empty.
- The generated `Language Instruction 3` does not start with `use the robot arm to `.
- A caption mentions camera/video/frame/timestamp language.
- A caption contains uncertainty wording like `maybe`, `probably`, `appears to`, or `seems to`.
- A caption mentions success or failure.

Rejected rows are not saved and route to `failure_pathway`.

## Metadata Mismatch Routing

Gemini receives the metadata task name and the selected video. It sets `metadata_mismatch=true` only when the visible task is clearly incompatible with the metadata task name.

Examples:

- Metadata says `Coil wire`, but the video shows opening or closing an air fryer tray: mismatch.
- Metadata says `Pour coffee`, and Gemini describes a more detailed coffee-pouring sequence: not a mismatch.
- Object color differences or a partial view alone: not a mismatch.
- If unsure: not a mismatch.

When Gemini flags a mismatch, the agent saves the three captions and routes to `metadata_mismatch_pathway`.

## Label Writing

The agent writes the three caption strings as Encord classification answers:

- `Language Instruction 1`
- `Language Instruction 2`
- `Language Instruction 3`

The language instructions are always added as global classifications on the label row. For data-group tasks, this means the captions live at the group level, not on an individual camera child or frame range.

## Routing Summary

`success_pathway`:

- Captions are valid.
- Gemini does not flag metadata mismatch.
- The requested data-group video layout was found.

`metadata_mismatch_pathway`:

- Captions are valid.
- Gemini flags metadata mismatch, or the configured data-group video layout was missing and a random video child was used.

`failure_pathway`:

- Missing task metadata key.
- No usable video child exists.
- Video download/cache fails.
- Gemini call fails.
- Gemini output fails validation.
