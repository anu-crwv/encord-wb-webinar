# Evaluation — Trossen DreamZero eval suite (built from scratch)

## Why this exists (the problem)

We fine-tuned a DreamZero (groot) world-action model on a **new embodiment** — the Trossen AI
mobile bimanual robot (`dreamzero-trossen-lora`: 16-dim packed state/action, 3 RGB cameras,
language-conditioned). We needed a way to answer **"how good is the fine-tuned model, and does
better data/training actually improve the policy?"** — comparable across variants, with full
W&B/Weave lineage from dataset → training run → checkpoint → eval.

**Nothing existed for this.** The reference eval (`github.com/anu-wandb/dreamzero-evals`, and the
DROID-only `arhanjain/sim-evals` it builds on) is hard-wired to the **DROID** embodiment — Franka
7-DOF, **8-dim** `oxe_droid` actions, 2 cameras. Our model is **16-dim bimanual** with different
cameras, so none of it could be reused for the rollout/eval. **This whole `eval/` suite was written
from scratch** for the Trossen embodiment — the policy server, the eval drivers, the embodiment +
scene, and the Weave instrumentation — keeping only the *structure/conventions* of the dreamzero-evals
Weave layer (`@weave.op run_episode`, `EvaluationLogger`, `VideoFileClip` media, the leaderboard
convention) so results look familiar in the W&B workspace.

Everything lives in the `wam-finetune-webinar` W&B project.

## Two complementary evals (both from scratch, same Weave structure)

### 1. Offline real-data eval — `offline_eval.py`  ✅ working, primary metric
Replays **real held-out Trossen episodes** (the Encord LeRobot dataset) through the model and scores
**predicted vs ground-truth actions** (`action_mse` / `action_mae` / `gripper_mae`). This evaluates on
exactly the training data distribution → genuinely meaningful action quality, with **no sim-to-real
gap**. It's a pure client (no GPU / no Isaac Sim) that hits the policy server.

- Videos per episode (Weave `VideoFileClip`): `episode_video` = the **real** recorded 3-cam footage,
  `dream_video` = the WAM's predicted video, `side_by_side` = real‖dream synced.
- Latest run: mean `action_mse ≈ 0.117` over 5 tasks (coffee-pour, towels, batteries, glue) on v4.
- Run it: scale the server to 1, then apply `deploy/trossen-offline-eval-job.yaml` (stock `python:3.11`
  CPU job → `offline_runner.sh` → `offline_eval.py`).

### 2. Closed-loop sim eval — `run_trossen_eval.py` (NVIDIA Isaac Lab-Arena)  🚧 in progress
The Trossen robot **acts in a physics sim**: Isaac Sim renders cameras → our policy → 16-dim actions →
env steps → success metric. This is the closed-loop behavior check. We authored the full stack from
scratch on Arena:
- `isaaclab_arena_dreamzero/embodiments/trossen.py` — the **Trossen embodiment** (mobile_ai.usd,
  16-dim joint-position action in the trained order, 3 cameras, 16-dim `state` obs), registered via
  `@register_asset`.
- `isaaclab_arena_dreamzero/environments/trossen_pick_and_place.py` — a **custom scene** with the work
  table **raised into the tall robot's reach** (the stock Franka env puts objects ~1.2 m below the
  Trossen arms) + objects + lighting/HDR + `PickAndPlaceTask`.
- **Open limiter:** the model trained on **real** Trossen photos; Isaac renders synthetic assets, so
  closed-loop *success* may stay low until the rendered scene resembles training (camera-aim + assets +
  domain randomization — the current iteration). Treat sim success as a behavior check; the offline
  eval is the trustworthy action-quality number.

## Shared pieces (from scratch)
- `isaaclab_arena_dreamzero/policy/` — `DreamZeroRemotePolicy` (PolicyBase) + `DreamZeroTrossenAdapter`
  (3-cam + 16-dim wire format), modeled on Arena's `isaaclab_arena_openpi`.
- `server/trossen_policy_server.py` — the **DreamZero Trossen inference server** (3 cams / 16-dim packed
  action) wrapping the upstream roboarena server; packs the WAM dream video into responses
  (`DZ_DREAM_VIDEO=1`). Deployed via `deploy/trossen-inference.yaml` (configmap + Deployment + Service).
- `weave_eval.py` — `weave.init` **inside** `wandb.init` (same project) so traces map to the run
  ([weave-in-workspaces](https://docs.wandb.ai/weave/guides/tools/weave-in-workspaces)) +
  `run.use_artifact(<ckpt>)` lineage + run tags (smoke/full-scale/<steps>/offline/<version>).
- `video_logging.py` — sim 3-cam strip + dream + synced side-by-side mp4s.
- `debug_cameras.py` — render-only introspection (link/object world positions + camera poses + frames)
  for calibrating the sim scene.

## Verified contracts
| Concern | Contract |
| --- | --- |
| Wire client | `openpi_client.websocket_client_policy.WebsocketClientPolicy`; server uses the roboarena `endpoint` routing key |
| Server response | `(H, 16)` action chunk (`actions`/`action`); `H >= open_loop_horizon` (default 5) |
| Trossen modality | video `video.{exterior_image_1_left,wrist_image_left,wrist_image_right}`; packed `state.state`(16) / `action.action`(16), q99-normalized; images sent at 640×480 |
| 16-dim packing | `[left_joint_0..5, left gripper, right_joint_0..5, right gripper, linear_vel, angular_vel]` |
| Checkpoint load | `GrootSimPolicy(embodiment_tag=TROSSEN, model_path=…)`; tag + projector `trossen:32` + `trossen_relative` registered |

## Deploy / run
- Server: `deploy/trossen-inference.yaml` (RTX node, v4 checkpoint, `DZ_DREAM_VIDEO=1`). Scale to 1 before any eval.
- Offline eval: `deploy/trossen-offline-eval-job.yaml` (CPU node).
- Sim eval: `deploy/trossen-eval-job.yaml` (Isaac Sim image, RTX node) → `runner.sh` → `run_trossen_eval.py`.
- Sim debug render: `deploy/trossen-cam-debug-job.yaml` → `runner.sh` (`WAM_EVAL_ENTRY=debug_cameras.py`).

## Reference clones (not committed)
`/Users/anu/Projects/{IsaacLab-Arena,trossen_ai_isaac,dreamzero-upstream}` were cloned for API/asset
reference (Arena env/embodiment patterns, `mobile_ai.usd`, the upstream roboarena server).
