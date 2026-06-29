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
            eval_logger = EvaluationLogger(model=model_label, dataset=f"trossen-sim-{job.name}", name=job.name)

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
