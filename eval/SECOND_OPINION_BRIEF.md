# Second-opinion brief: DreamZero Trossen finetune + Isaac-Lab sim eval

**Purpose.** We want an independent review of a robot-learning eval problem. A world-action
model finetuned on real bimanual-robot data produces a **correct predicted "dream" video**
in a physics simulator, but its **action output fails to drive the simulated arm to the
task** (the arm hovers near its rest pose). We have deeply instrumented this and ruled out
several causes. We want your read on our diagnosis and what to try next. You do **not** have
our repo, so everything needed is inlined below (with code snippets and numbers).

---

## 0. Cast of characters / links

- **Model family: DreamZero** (a.k.a. "groot" internally) — an open world-action model.
  Repo: `https://github.com/dreamzero0/dreamzero`. It is a **Wan2.1 image-to-video (I2V)
  diffusion transformer (DiT, ~14B, 40 layers)** that **jointly flow-matches video latents
  and robot-action tokens** in one coupled denoising loop, plus a small per-embodiment MLP
  action decoder. Initialized from a pretrained "dreamzero-agibot" checkpoint; text encoder
  is UMT5-XXL.
- **Working reference sim eval: `arhanjain/sim-evals`** —
  `https://github.com/arhanjain/sim-evals`. An Isaac-Lab closed-loop sim eval for **DROID**
  (Franka 7-DOF) policies. DreamZero's DROID model reportedly **worked** here (arm did the
  tasks in sim). We built our Trossen eval in the same spirit but from scratch.
- **Our robot: Trossen AI mobile bimanual** (two 6-DOF arms + 2 grippers + mobile base).
  Isaac asset `mobile_ai.usd` from `https://github.com/TrossenRobotics/trossen_ai_isaac`.
- **Sim framework: NVIDIA Isaac Lab-Arena** (`https://github.com/isaac-sim/IsaacLab-Arena`),
  Isaac Sim 6.0.
- **Experiment tracking:** Weights & Biases + Weave.

---

## 1. What we finetuned and on what data

**Task/goal.** Finetune DreamZero on a partner's (Encord) **real Trossen bimanual dataset**,
then evaluate how good the finetuned policy is.

**Dataset (LeRobot v2.0 format).**
- **400 episodes, 852,846 frames, 30 fps**, `robot_type: trossen`.
- 3 RGB cameras, each **640×480 (4:3)**: `exterior_image_1_left`, `wrist_image_left`,
  `wrist_image_right`.
- **16-dim state and action**, packed as:
  `[L_joint_0..5 (6), L_gripper (1), R_joint_0..5 (6), R_gripper (1), base_linear_vel (1), base_angular_vel (1)]`
- The dataset is **one workspace** (a fixed tabletop with tools/props). Base velocities are
  ≈0 throughout (the base doesn't move during manipulation).

**Real per-dim state ranges (from 4 episodes, 8,900 frames)** — useful later:
```
dim     min     p1      p99     max
Lj0   -0.265  -0.159   0.810   0.839
Lj1   -0.000   0.007   2.756   2.789   <- left shoulder swings ~0..2.8 during reach
Lj2    0.006   0.006   1.922   2.062
Lj3   -1.067  -0.754   1.569   1.575
...
Lgrip -0.001   0.000   0.032   0.037   <- gripper open ~0.03-0.04
linv  -0.008  -0.002   0.002   0.006   <- base ~stationary
angv  -0.023  -0.011   0.011   0.025
```

**Finetune configuration.**
- **LoRA** (`train_architecture=lora`), **rank 4, alpha 4**, `save_lora_only=true`.
- `learning_rate=1e-4`, warmup_ratio 0.05, weight_decay 1e-5, bf16 + tf32.
- `per_device_train_batch_size=1`, **max_steps=3000** ("full-scale"), multi-node 2×GH200
  with DeepSpeed **ZeRO-3** (CPU offload). train_loss 0.53 → 0.21.
- Model/head config: `model/dreamzero/action_head=wan_flow_matching_action_tf`,
  transform `dreamzero_cotrain`.
- **Training-time overrides (from our launcher):** `num_frames=33`, **`action_horizon=24`**,
  `num_views=3`, **`image_resolution_width=320`, `image_resolution_height=176`**.
- **Data config `trossen_relative.yaml`** — note **absolute** actions:
  ```yaml
  # 16-dim packed state+action (single keys state.state / action.action), 3 cameras.
  # Absolute actions: with single packed keys there is no shared sub-key between state and
  # action, so relative-action stats don't apply.
  relative_action: false
  relative_action_per_horizon: false
  relative_action_keys: []
  max_chunk_size: 5
  defaults: [dreamzero/base_48_wan_fine_aug_relative]  # inherits transforms below
  ```
- **Image transforms the model expects (from `base_48_wan_fine_aug_relative`):**
  ```yaml
  crop_cfg:   VideoCrop  scale: 0.95           # center crop in eval
  resize_cfg: VideoResize height: 256 width: 480   # NOTE base=480x256; train override=320x176
  normalize:  VideoNormalize mean:[0.5,0.5,0.5] std:[0.5,0.5,0.5]   # -> [-1,1]
  # state.state / action.action are q99-normalized: 2*(x-q01)/(q99-q01)-1
  action_horizon: 48   # base (train overrode to 24)
  ```
- Embodiment tag `trossen`; per-embodiment action projector index. **NOTE:** at inference the
  DiT **hardcodes `embodiment_id = 0`** (see §7), so the Trossen projector must be index 0.

**Checkpoint.** `dreamzero-trossen-lora:v4`. (v2 was broken — ZeRO-3 + `save_lora_only`
re-read a partitioned state_dict and dropped 4 projector matrices to shape `(0,)`; fixed by
filtering the already-gathered state_dict; re-trained → v4 complete.)

---

## 2. The model architecture (why "dream right, action wrong" is even possible)

This is the crux. We traced the inference path:

- The backbone is an **`IdentityBackbone`** — there is **no separate action expert**. A single
  **WAN video DiT jointly denoises video latents AND action tokens** in one loop. Action
  tokens attend (blockwise-causal) to the first conditioning image latent + current/previous
  video-block latents + same-block state.
- **Actions are decoded by a thin per-embodiment MLP** (`CategorySpecificMLP`) reading the
  same DiT latents.
- **Classifier-free guidance (CFG) asymmetry** — the key finding. In the denoise loop the
  model computes both conditional and unconditional predictions for **both** video and action,
  but originally applied CFG **only to video**:
  ```python
  flow_pred_cond,   flow_pred_cond_action   = predictions[0]   # conditional
  flow_pred_uncond, flow_pred_uncond_action = predictions[1]   # unconditional
  # VIDEO gets guidance:
  flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)  # cfg_scale=5.0
  # ACTION originally used the conditional prediction ALONE (== CFG 1.0, no guidance):
  #   noisy_input_action = scheduler.step(model_output=flow_pred_cond_action, ...)
  ```
- Other inference knobs: `num_inference_steps=16`, `cfg_scale=5.0`, `sigma_shift=5.0`,
  `NUM_DIT_STEPS=8` (only 8 of 16 steps actually run the DiT; the rest reuse a cached
  prediction via cosine similarity — the agent that read the code noted this step-skipping
  degrades the **action** more than the video).

**Hypothesized mechanism:** on out-of-distribution (sim-rendered) inputs, the CFG-amplified
**video/dream stays coherent**, but the **un-guided action** regresses toward the *conditional
mean* of the finetune data — which for an arm at episode start is "stay near rest / tiny
motions." Hence dream correct, arm hovers.

---

## 3. The two evals we built (from scratch)

`arhanjain/sim-evals` and the DreamZero DROID eval are hard-wired to DROID (Franka 7-DOF,
8-dim `oxe_droid`, 2 cams), so none of it was reusable for our 16-dim bimanual Trossen. We
built both evals from scratch, keeping only the Weave logging conventions.

### 3a. Offline real-data eval — WORKING, primary metric
Replays **real held-out Trossen episodes** through the model and scores **predicted vs
ground-truth actions** (`action_mse`/`mae`, `gripper_mae`). Pure client (no sim). **Latest:
mean `action_mse ≈ 0.117`** over 5 tasks. This validates that the served model + q99
unnormalization produce correct actions **on real images**.

### 3b. Closed-loop Isaac Lab-Arena sim eval — the problem child
Robot acts in a physics sim: Isaac renders 3 cameras → our policy adapter → DreamZero server →
16-dim action → env steps → success metric. Stack we authored:

- **Trossen embodiment** (`mobile_ai.usd`): 16-dim absolute joint-position action, 3
  `TiledCamera`s, 16-dim packed state obs. Action term:
  ```python
  # absolute joint-position targets, unit scale, no default offset, order preserved
  JointPositionActionCfg(asset_name="robot", joint_names=_LEFT_ARM, preserve_order=True, use_default_offset=False)
  # (+ left_gripper, right_arm, right_gripper, and a base-velocity term frozen scale=0)
  # actuators: arm stiffness 400, damping 80 (matches DROID sim-evals)
  ```
- **Custom pick-and-place env**: a maple work-table raised to the arm's working height +
  a cube + a bowl pinned to reachable spots; dome light + HDR background; `fix_root_link=True`
  so the base can't drift.
- **Policy adapter + server**: sends 3 RGB views (resize-with-pad) + 16-dim state + prompt;
  the server wraps DreamZero's roboarena inference and returns a `(H,16)` action chunk;
  `open_loop_horizon=5` (replay 5, refetch).
- **Weave tracing**: `weave.init()` inside `wandb.init()` (same W&B project) so rollouts map to
  the run; per-episode `EvaluationLogger` with `episode_video`/`dream_video`/`side_by_side`
  `VideoFileClip` media (real 3-cam strip ‖ WAM dream), success + object_moved scores.

Control timing: sim runs `decimation=4`, `sim.dt=0.005` → **50 Hz control**.
Sim camera intrinsics: `PinholeCameraCfg(focal_length=2.1, horizontal_aperture=5.376)` →
**~104° HFOV** (copied from DROID/sim-evals; the real Trossen uses RealSense D405 ~87°).

---

## 4. Problems we already found and fixed (sim eval)

1. **Cameras rendered black.** Root cause: cameras mounted on link prims with placeholder
   rotations pointed into empty space. Fix: mount at the **real Trossen camera frames** baked
   into `mobile_ai.usd`. Every `*_color_optical_frame` sits at rot `(-0.5,0.5,-0.5,0.5)`
   relative to its camera link, and `cam_high_link` carries a built-in ~37° downward pitch.
   Parent at the real camera links with that optical rotation + `convention="ros"` →
   training-like views. (Verified: exterior cam shows table + cube + bowl.)
2. **Objects unreachable / on the floor.** The table was ~1.2 m below the arms; objects
   scattered or spawned past the table edge. Fix: raise the table to the arm working height
   (surface ~z=1.0) and pin the cube + bowl to explicit reachable spots on the table interior.
3. **Whole robot drifted mid-rollout.** The base was a free root; arm reaction forces (from the
   flailing policy) shoved it around, so the camera wandered off the workspace and knocked the
   cube off. Fix: `fix_root_link=True` (base action already frozen). Verified stable.
4. **Arm collapsed to all-zeros at reset (out-of-distribution start).** Arena's reset writes
   the joint *state* from `init_state` but leaves the joint position *targets* at 0, so the
   stiff PD actuators snapped the arms bolt-straight before the policy acted. The model then
   computed its first action from a pose it never saw in training (real episodes start bent:
   per-arm `[0, 1.047, 0.523, 0.628, 0, 0]`). Fix: after each reset, force joint state **and**
   target to the real rest pose, then recompute the observation:
   ```python
   robot.write_joint_state_to_sim(jp_rest, zeros)
   robot.set_joint_position_target(jp_rest)
   robot.write_data_to_sim()
   obs = env.unwrapped.observation_manager.compute()
   ```
   Verified: proprio at step 0 now exactly matches the dataset's first frame.

---

## 5. The core unsolved problem and how we isolated it

**Symptom:** the WAM **dream video is correct** (it imagines the reach-and-grasp), but the
**arm under-commits** — it moves to a slightly-bent pose near rest and holds; the cube/bowl are
never grasped; `success_rate = 0` across episodes.

We ran three isolating experiments:

### 5a. Is the action reversed / mis-applied? → NO (plumbing is correct)
We **replayed a real episode's ground-truth 16-dim actions open-loop** into the sim (no model,
no server). The arm reproduced the real motion:
```
[replay] START gripper=(0.643, 0.323, 1.322)  cube=(0.70, 0.10, 1.06)
[replay] step 20  gt_j1=+2.30  gt_grip=+0.013 -> gripper=(0.60, 0.50, 1.03)  # descended to table
[replay] SUMMARY gripper_z: start=1.322 min=1.025 -> descended 0.297 m  (DESCENDS = plumbing OK)
[replay] cube end pos=(0.70, 0.10, 0.99)   # cube got pushed
```
So the action pathway (sign, joint order, scale, actuators) is faithful. Feeding correct
actions makes the arm reach.

### 5b. Does the proprio state drift out-of-distribution under load? → NO
We logged the per-step proprio during a closed-loop rollout and flagged any dim outside the
real `[p1,p99]` band. **Every step: `OOD dims: NONE (all in real range)`.** The state we feed
the model stays in-distribution the whole rollout.

### 5c. Does the model's action head under-commit? → YES (this is the failure)
Logging the model's commanded targets in closed loop, the **left shoulder (Lj1)** — which must
swing ~1.05 → ~2.4 to descend to the table — stays low:
```
no-CFG (before fix):  Lj1 target ~1.0-1.35  (avg ~1.13), per-step delta ~0.1-0.4 then holds
real reach needs:     Lj1 up to ~2.4-2.8
```
So the arm hovers because the model *commands* it to hover on synthetic pixels.

### 5d. We tried fix #1 (add CFG to the action) → helped ~20%, not enough
We changed the denoise loop to guide the action like the video (the unconditional action
prediction already existed, just discarded):
```python
flow_pred_action = flow_pred_uncond_action + self.action_cfg_scale * (
    flow_pred_cond_action - flow_pred_uncond_action)   # action_cfg_scale defaults to 5.0
# and use flow_pred_action in scheduler.step(...)
```
Result (same scene, same rest start, CFG=5.0 vs none):
```
Lj1 (left shoulder) avg:   no-CFG ~1.13   ->   CFG=5.0 ~1.37   (real reach needs ~2.4)
Rj1 (right shoulder):      ~2.0 both
```
CFG measurably increased commitment (~+20% on the left shoulder) but the arm still hovers,
`success=0`. Interpretation: **#1 is a real but secondary contributor; not the dominant
blocker.** Guidance amplifies a task signal that is mostly *absent* on synthetic pixels.

---

## 6. What is DIFFERENT between our Trossen setup and the WORKING DROID sim-evals

We investigated `arhanjain/sim-evals` (where a DreamZero DROID model worked closed-loop). This
**disproved** several of our own hypotheses:

| Aspect | sim-evals DROID (works) | Ours (Trossen, hovers) |
|---|---|---|
| Action space | **absolute joint position**, `JointPositionActionCfg`, `use_default_offset=False`, scale 1 | **same** |
| Actuators | stiff PD, stiffness 400 / damping 80 | **same** (matched) |
| Controller | direct joint targets, **no IK/OSC** | **same** |
| Norm | server-side only | **same** |
| Control rate | **15 Hz** (decimation 8, dt 1/120) | **50 Hz** (decimation 4, dt 0.005) — **DIFF** |
| Open-loop horizon | 8 | 5 — DIFF |
| Images to model | `resize_with_pad(224,224)`, **2 cams** | resize_with_pad→ model 480×256 (base), **3 cams** — DIFF |
| Camera HFOV | ~104° (Zed-like) matched to DROID | ~104° in sim but real D405 ~87° — **DIFF (mismatch vs OUR real cams)** |
| Training data | DROID: **~76k+ trajectories, huge visual diversity** | Trossen finetune: **~400 episodes, one workspace** — **DIFF (big)** |
| Data fps | ~15 Hz | **30 Hz** (we run sim at 50 Hz) — DIFF |

So the "delta-EE vs absolute-joint" theory is **wrong** (both are absolute joint). The real
differences are: **(a) tiny/narrow finetune data**, **(b) camera FOV vs OUR real cameras**,
**(c) control-rate mismatch (50 vs 30 Hz; horizon 5 vs trained chunk)**, **(d) a
possibly-different train vs eval image resolution** (trained at 320×176 per our launcher; the
eval transform stack targets 480×256/256×480), **(e) 3 cams vs 2**.

---

## 7. Our current five candidate explanations (ranked)

1. **CFG asymmetry (action un-guided).** Mechanism above. *Tested: real but only ~20% effect;
   secondary.*
2. **Narrow finetune → brittle action decoder; robust dream.** The dream rides the web-scale
   pretrained WAN backbone (robust to visual shift); the action is a thin per-embodiment MLP
   finetuned on ~400 episodes of one workspace (brittle). DROID's massive diversity is why its
   action head committed on synthetic pixels and ours doesn't. *Untested; structural.*
3. **Camera FOV / render mismatch.** Our sim cams are ~104° HFOV, but the real Trossen uses
   RealSense D405 (~87°); plus synthetic textures/lighting. The action conditions on VAE/CLIP
   latents of the first frame → OOD latents → weak action, while the dream tolerates it.
   *Untested fix: set sim focal ≈2.8 (87°) + domain-match the scene.*
4. **Control-rate & denoising-step mismatches.** Sim 50 Hz vs data 30 Hz; horizon 5 vs trained
   chunk; and `NUM_DIT_STEPS=8/16` step-caching reportedly degrades the action more than the
   video. *Untested.*
5. **Per-embodiment projector index + q99 stats.** Inference hardcodes `embodiment_id=0`; if
   Trossen's projector isn't index 0 the action MLP reads wrong weights (dream unaffected). And
   q99/q01 from the checkpoint's `metadata.json` scale the absolute actions; if narrower than
   the finetune motion, commands shrink. *Caveat: offline eval (`0.117`) uses the same path and
   is correct, so these are unlikely as the sole cause — but they compound with the render
   shift.*

Plus an open detail: **train image resolution 320×176 vs eval transform 480×256/256×480** — a
possible train/eval preprocessing mismatch we haven't fully reconciled.

---

## 8. Specific questions for you (the second opinion)

1. Is our **mechanism for "dream correct, action wrong" (CFG asymmetry + narrow-finetune
   brittleness)** sound? Anything we're missing about *why the two heads diverge on OOD pixels*?
2. Given CFG=5.0 helped only ~20%, is it worth **pushing CFG higher (10–15)** on a
   flow-matching *action* head, or does that just add overshoot/artifacts? Any principled way
   to guide the action more without instability (e.g., guidance only on early denoising steps,
   or conditioning the action on the *clean/denoised* video latents rather than joint denoise)?
3. Is the most likely dominant cause **the tiny finetune dataset (400 eps, one workspace)** —
   i.e., is this fundamentally a data/robustness problem that no sim-side fix will close, and
   the honest answer is "offline eval is the metric; sim is a visual demo"?
4. How much would you expect the **train (320×176) vs eval (480×256) resolution**, **FOV
   (104° vs 87°)**, **control-rate (50 vs 30 Hz)**, and **step-caching (8/16 DiT steps)**
   mismatches to matter individually? Which would you fix first?
5. Any better **isolation experiment** than what we ran (GT-replay, proprio-OOD, action-CFG)?
   E.g., we can condition the action denoise on GT/clean latents (`lazy_joint_forward_causal_gt_cond`
   exists) to test whether the action is purely image-latent-limited. Worth it?
6. Is there a **cheap "domain-match" recipe** you'd prioritize (match FOV, real wood-table
   texture, real-ish objects, lighting matched to the training photos) that tends to unlock
   real-data world-models on synthetic renders — or is that usually a dead end for single-
   workspace finetunes?

**What we consider settled (please challenge if you disagree):** model training is good (dream
correct; offline `action_mse 0.117`); sim action plumbing is correct (GT replay grasps);
proprio is in-distribution. The failure is specifically the **action head under-committing on
synthetic images**.
