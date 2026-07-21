#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Weave-traced Trossen sim eval — same structure as github.com/anu-wandb/dreamzero-evals
(deploy/cks/scripts/droid_weave_eval.py), driven over the Isaac Lab-Arena env.

Structure (mirrors the DROID eval exactly):
  * weave.init(project) inside an active wandb.run (weave-in-workspaces mapping).
  * @weave.op run_episode(episode_idx, instruction, max_steps): drives one rollout
    on the Arena env with our DreamZeroRemotePolicy, captures the sim 3-cam strip +
    the WAM "dream" video, writes episode_N.mp4 / _dream.mp4 / _sbs.mp4, and RETURNS
    a dict whose episode_video / dream_video / side_by_side keys hold moviepy
    VideoFileClip media (so the videos render on the run_episode trace).
  * One EvaluationLogger(model=<ckpt>, dataset=trossen-sim-<task>, name=<task>);
    per episode: log_prediction(inputs, output=run_episode_result) -> log_score(...)
    -> finish(); log_summary(...) at the end.

No wandb.Video — all media + tracing go through Weave, like the reference repo.

Run inside the Isaac Sim image (eval/runner.sh):
  python.sh run_trossen_eval.py --eval_jobs_config <cfg.json> --enable_cameras --headless
"""

from __future__ import annotations

import json
import os

import weave

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner import enable_cameras_if_required, get_policy_from_job, load_env
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.evaluation.job_manager import JobManager

# Built lazily on the main thread in main(); run_episode (a @weave.op) reads these
# globals rather than taking unserializable env/policy as inputs (same pattern the
# DROID eval uses with _ENV/_CLIENT).
_ENV = None
_POLICY = None
_PREV_SUCCESSES = 0.0

# Indoor HDR maps to cycle through for domain randomization (narrows the sim-to-real
# visual gap by varying the background + ambient light each episode).
_HDR_POOL = [
    "home_office_robolab", "wooden_lounge_robolab", "brown_photostudio_robolab",
    "photo_studio_robolab", "kiara_interior_robolab", "garage_robolab",
    "billiard_hall_robolab", "carpentry_shop_robolab",
]


def _domain_randomize(env, episode_idx: int):
    """Per-episode domain randomization (gated by WAM_DOMAIN_RAND): vary the dome-light
    intensity + color and swap the HDR background each episode so the rendered scene
    isn't a single fixed synthetic look. The env is built once for all episodes, so the
    build-time HDR/light variations can't re-sample -- we drive it directly here, in the
    same per-episode hook as the rest-pose seed. Returns a refreshed observation or None."""
    # NB: treat "0"/"false"/"no" as OFF -- a bare `not os.environ.get(...)` leaves DR ON for
    # WAM_DOMAIN_RAND=0 because "0" is a truthy string in Python (this silently overrode the
    # domain-MATCH HDR/light every episode). Only "1"/"true"/"yes" enable randomization.
    if os.environ.get("WAM_DOMAIN_RAND", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    try:
        import random

        import omni.usd
        from pxr import Gf, UsdLux

        from isaaclab_arena.assets.registries import HDRImageRegistry

        rng = random.Random(4242 + episode_idx)
        stage = omni.usd.get_context().get_stage()

        # Resolve an HDR texture for this episode (best-effort).
        hdr_name = _HDR_POOL[episode_idx % len(_HDR_POOL)]
        tex = None
        try:
            tex = HDRImageRegistry().get_hdr_by_name(hdr_name)().texture_file
        except Exception:  # noqa: BLE001
            pass

        intensity = rng.uniform(500.0, 1500.0)
        warm, cool = rng.uniform(0.92, 1.0), rng.uniform(0.90, 1.0)
        n = 0
        for p in stage.Traverse():
            if p.GetTypeName() == "DomeLight":
                L = UsdLux.DomeLight(p)
                L.GetIntensityAttr().Set(intensity)
                L.GetColorAttr().Set(Gf.Vec3f(1.0, warm, cool * 0.97))
                if tex:
                    L.GetTextureFileAttr().Set(tex)
                n += 1
        for _ in range(3):
            env.unwrapped.sim.step(render=True)
            env.unwrapped.scene.update(env.unwrapped.physics_dt)
        print(f"[run_trossen_eval] DR ep{episode_idx}: hdr={hdr_name} intensity={intensity:.0f} lights={n}", flush=True)
        return env.unwrapped.observation_manager.compute()
    except Exception as e:  # noqa: BLE001
        print(f"[run_trossen_eval] domain randomization failed: {e}", flush=True)
        return None


def _clip(path):
    try:
        from moviepy.editor import VideoFileClip

        return VideoFileClip(path, audio=False) if path and os.path.exists(path) else None
    except Exception as e:  # noqa: BLE001
        print(f"[run_trossen_eval] VideoFileClip wrap failed: {e}", flush=True)
        return None


def _metrics_plain() -> dict:
    env = _ENV.unwrapped
    if hasattr(env.cfg, "metrics") and env.cfg.metrics is not None:
        try:
            return metrics_to_plain_python_types(env.compute_metrics())
        except Exception as e:  # noqa: BLE001
            print(f"[run_trossen_eval] compute_metrics failed: {e}", flush=True)
    return {}


def _seed_rest_pose(env, obs):
    """Force the arms to the REAL Trossen rest pose after reset, and return a refreshed
    observation. Arena's reset writes the joint *state* from init_state but leaves the
    joint position *targets* at 0, so the stiff actuators yank the arms straight to
    all-zeros before the policy acts -> the model's first action is computed from an
    out-of-distribution proprio it never saw in training. Here we write both the joint
    state AND the position target to the rest pose, then recompute the observation so
    the policy starts in-distribution."""
    try:
        import torch

        from isaaclab_arena_dreamzero.embodiments.trossen import _REST_JOINT_POS

        def _as_torch(x):
            if isinstance(x, torch.Tensor):
                return x
            import warp as wp

            return wp.to_torch(x)

        robot = env.unwrapped.scene["robot"]
        names = list(robot.joint_names)
        jp = _as_torch(robot.data.joint_pos).clone()
        for nm, val in _REST_JOINT_POS.items():
            if nm in names:
                jp[:, names.index(nm)] = float(val)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp)
        robot.write_data_to_sim()
        fresh = env.unwrapped.observation_manager.compute()
        return fresh if isinstance(fresh, dict) and fresh else obs
    except Exception as e:  # noqa: BLE001
        print(f"[run_trossen_eval] rest-pose seed failed: {e}", flush=True)
        return obs


@weave.op(name="run_episode")
def run_episode(episode_idx: int, instruction: str, max_steps: int) -> dict:
    """One Trossen rollout on the Arena env. Returns a Weave-renderable summary
    with episode_video / dream_video / side_by_side VideoFileClip media + scores."""
    global _PREV_SUCCESSES
    env, policy = _ENV, _POLICY
    video = policy._video  # TrossenVideoLogger: captures sim + dream frames per step

    policy.set_task_description(instruction)
    policy.reset()  # clears action cache + video buffers
    obs, _ = env.reset()
    obs = _seed_rest_pose(env, obs)  # start at the real rest pose (see helper)
    dr_obs = _domain_randomize(env, episode_idx)  # per-episode DR (gated by WAM_DOMAIN_RAND)
    if dr_obs is not None:
        obs = dr_obs

    steps_done = 0
    for t in range(max_steps):
        action = policy.get_action(env, obs)  # captures sim+dream frames internally
        obs, _, terminated, truncated, _ = env.step(action)
        steps_done = t + 1
        done = bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)
        trunc = bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated)
        if done or trunc:
            break

    # Write the three videos for this episode (sim strip / dream concat / synced sbs).
    paths = video.build_episode_videos(episode_idx) if video is not None else {}

    # Per-episode success from the env's cumulative metrics (delta in success count).
    m = _metrics_plain()
    n_ep = float(m.get("num_episodes", episode_idx + 1) or (episode_idx + 1))
    cum_succ = float(m.get("success_rate", 0.0)) * n_ep
    ep_success = max(0.0, cum_succ - _PREV_SUCCESSES)
    _PREV_SUCCESSES = cum_succ

    return {
        "episode_idx": episode_idx,
        "instruction": instruction,
        "steps_done": steps_done,
        "success": bool(round(ep_success)),
        "cumulative_success_rate": float(m.get("success_rate", 0.0)),
        "object_moved_rate": float(m.get("object_moved_rate", 0.0)),
        "episode_video": _clip(paths.get("episode_video_path")),
        "dream_video": _clip(paths.get("dream_video_path")),
        "side_by_side": _clip(paths.get("side_by_side_path")),
        "n_steps": int(paths.get("n_steps", steps_done)),
        "n_dream_frames": int(paths.get("n_dream_frames", 0)),
    }


def main() -> None:
    global _ENV, _POLICY

    parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(parser)
    args_cli, _ = parser.parse_known_args()

    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        eval_jobs_config = json.load(f)
    enable_cameras_if_required(eval_jobs_config, args_cli)
    # JobManager converts each job's arena_env_args dict -> Arena CLI arg list + defaults.
    job = JobManager(eval_jobs_config["jobs"]).all_jobs[0]
    instruction = job.language_instruction
    num_episodes = int(job.num_episodes or 10)
    max_steps = int(os.environ.get("WAM_EVAL_MAX_STEPS", "400"))

    # W&B run + Weave (same project) + model-artifact lineage, BEFORE the sim.
    from isaaclab_arena_dreamzero.weave_eval import finish_eval_tracing, init_eval_tracing

    run = init_eval_tracing()

    try:
        with SimulationAppContext(args_cli):
            # Register our Trossen embodiment now the sim app (and isaaclab) is up.
            import isaaclab_arena_dreamzero.embodiments  # noqa: F401  (register trossen embodiment)
            import isaaclab_arena_dreamzero.environments  # noqa: F401  (register trossen_pick_and_place)

            _ENV = load_env(job.arena_env_args, job.name)
            _POLICY = get_policy_from_job(job)

            from weave import EvaluationLogger

            model_label = os.environ.get("WEAVE_MODEL", "dreamzero-trossen-lora")
            art = os.environ.get("LORA_ARTIFACT", "")
            if ":" in art:
                model_label = f"dreamzero-trossen-lora:{art.rsplit(':', 1)[-1]}"
            # weave >=0.51 requires model/dataset/name to be identifiers ([A-Za-z0-9_], leading letter/_).
            _san = lambda s: (__import__("re").sub(r"[^0-9A-Za-z_]", "_", str(s)) or "x")
            eval_logger = EvaluationLogger(model=_san(model_label), dataset=_san(f"trossen_sim_{job.name}"), name=_san(job.name))

            print(f"[run_trossen_eval] {num_episodes} episodes, max_steps={max_steps}, model={model_label}", flush=True)
            for ep in range(num_episodes):
                out = run_episode(episode_idx=ep, instruction=instruction, max_steps=max_steps)
                pred = eval_logger.log_prediction(
                    inputs={"episode_idx": ep, "instruction": instruction}, output=out
                )
                pred.log_score(scorer="success", score=bool(out["success"]))
                pred.log_score(scorer="cumulative_success_rate", score=out["cumulative_success_rate"])
                pred.log_score(scorer="object_moved_rate", score=out["object_moved_rate"])
                pred.log_score(scorer="n_steps", score=int(out["n_steps"]))
                pred.finish()
                print(f"[run_trossen_eval] episode {ep}: success={out['success']} steps={out['n_steps']} "
                      f"videos(sim={out['episode_video'] is not None},dream={out['dream_video'] is not None},"
                      f"sbs={out['side_by_side'] is not None})", flush=True)

            final = _metrics_plain()
            eval_logger.log_summary({
                "num_episodes": num_episodes,
                "success_rate": float(final.get("success_rate", 0.0)),
                "object_moved_rate": float(final.get("object_moved_rate", 0.0)),
            })
            print(f"[run_trossen_eval] done. final metrics: {final}", flush=True)
    finally:
        finish_eval_tracing(run)


if __name__ == "__main__":
    main()
