DreamZero training consumes the modality fields below; RL bookkeeping fields are optional parity fields.

| Category | Expected by repo config | Local sample has? | Notes |
|---|---|---:|---|
| Video | `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left` | Partial | Sample has 3 videos named `cam_high`, `cam_left_wrist`, `cam_right_wrist`; needs semantic mapping. |
| State | `state.joint_position`, `state.gripper_position` | Partial | Sample has one packed `observation.state` 16D vector; needs explicit new-embodiment split/config. |
| Action | `action.joint_position`, `action.gripper_position` | Partial | Sample has one packed `action` 16D vector; needs explicit new-embodiment split/config. |
| Language | `annotation.language.language_instruction{,_2,_3}` | Missing | Filled from Encord captions. |
| Timing | `timestamp` | Present | Required for video frame loading. |
| Episode bookkeeping | `episode_index`, `frame_index`, `task_index`, `meta/tasks.jsonl`, `meta/episodes.jsonl` | Present | `task_index` must be remapped to captions. |
| RL fields | `next.reward`, `next.done`, `is_terminal`, `is_first`, `discount` | Missing | Not consumed by active DreamZero modality config. |
| Metadata | `meta/modality.json`, `meta/stats.json`, `meta/relative_stats_dreamzero.json` | Missing | Needed for this harness when training the new embodiment. |
