#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Render SIM CO-TRAINING data: replay real Trossen joint trajectories in the Isaac sim,
render the 3 cameras per frame, and emit LeRobot episodes with SIM videos + the REAL
parquet labels UNCHANGED (state/action/language/task_index).

Why: the closed-loop failure is visual/dynamics OOD + covariate shift, NOT missing labels.
This produces (sim_observation -> exact real action) pairs — trustworthy labels (geometry,
timing, and action are the real trajectory's; only the pixels are sim) — so co-training on
real+sim teaches the action head to act on rendered observations. (User rec: "render the
same trajectory under sim appearance while preserving exact geometry/timing/actions".)

Sharded + task-scoped so we can fan out across many RTX GPUs and match the scene object to
the task (cylinders->corn_can, batteries->yellow_block). Renders v8 episodes whose task
matches WAM_RENDER_TASK and (episode_index % NUM_SHARDS == SHARD_INDEX). Scene/object come
from the jobs config (EVAL_JOBS_CONFIG). Kinematic replay (set joint state + 1 render step;
no physics settle) for speed. Output LeRobot episodes under WAM_COTRAIN_OUT/shard-<SHARD>.

Run in the Isaac image via eval/runner.sh with WAM_EVAL_ENTRY=render_sim_cotraining.py and
WAM_SKIP_SERVER_PROBE=1 (no policy server needed)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner import enable_cameras_if_required, load_env
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import JobManager
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

SRC = os.environ.get("WAM_COTRAIN_SRC", "/data/wam/datasets/encord_trossen_v8")
OUT = os.environ.get("WAM_COTRAIN_OUT", "/data/wam/datasets/sim_cotrain")
TASK = os.environ.get("WAM_RENDER_TASK", "")            # task-string substring filter ("" = all)
STRIDE = int(os.environ.get("WAM_RENDER_STRIDE", "1"))  # frame stride (1 = native 30fps)
MAX_FRAMES = int(os.environ.get("WAM_RENDER_MAX_FRAMES", "1200"))
SHARD = int(os.environ.get("SHARD_INDEX", "0"))
NSHARD = int(os.environ.get("NUM_SHARDS", "1"))
MAX_EPS = int(os.environ.get("WAM_RENDER_MAX_EPS", "0"))  # cap episodes this shard renders (0 = no cap)
CHUNK = 1000
CAMS = ("exterior_image_1_left", "wrist_image_left", "wrist_image_right")
FPS = 30


def _chunk(i: int) -> str:
    return f"chunk-{i // CHUNK:03d}"


def main() -> None:
    import torch
    import imageio.v2 as imageio

    parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(parser)
    args_cli, _ = parser.parse_known_args()
    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        cfg = json.load(f)
    enable_cameras_if_required(cfg, args_cli)
    job = JobManager(cfg["jobs"]).all_jobs[0]

    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{SRC}/meta/tasks.jsonl") if l.strip()}
    want_ti = None if not TASK else {i for i, t in tasks.items() if TASK.lower() in t.lower()}
    eps = [json.loads(l) for l in open(f"{SRC}/meta/episodes.jsonl") if l.strip()]

    # Select this shard's episodes (task-filtered).
    sel = []
    for e in eps:
        i = int(e["episode_index"])
        if i % NSHARD != SHARD:
            continue
        pq = f"{SRC}/data/{_chunk(i)}/episode_{i:06d}.parquet"
        try:
            ti = int(pd.read_parquet(pq, columns=["task_index"])["task_index"].iloc[0])
        except Exception:
            continue
        if want_ti is not None and ti not in want_ti:
            continue
        sel.append(i)
    if MAX_EPS > 0:
        sel = sel[:MAX_EPS]
    print(f"[render] shard {SHARD}/{NSHARD} task='{TASK}' object={cfg['jobs'][0]['arena_env_args'].get('pick_up_object')} "
          f"-> {len(sel)} episodes; out={OUT}/shard-{SHARD}", flush=True)
    if not sel:
        print("[render] no episodes in this shard; done", flush=True)
        return

    def _as_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        import warp as wp
        return wp.to_torch(x)

    with SimulationAppContext(args_cli):
        import isaaclab_arena_dreamzero.embodiments  # noqa: F401
        import isaaclab_arena_dreamzero.environments  # noqa: F401
        from isaaclab_arena_dreamzero.embodiments.observations import (
            LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEFT_GRIPPER_JOINT, RIGHT_GRIPPER_JOINT)
        from isaaclab_arena_dreamzero.policy.trossen_adapter import DreamZeroTrossenAdapter
        from openpi_client import image_tools

        env = load_env(job.arena_env_args, job.name)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        names = list(robot.joint_names)
        adapter = DreamZeroTrossenAdapter()
        h, w = adapter.target_image_size
        outdir = f"{OUT}/shard-{SHARD}"

        def set_pose(s16: np.ndarray) -> None:
            jp = _as_torch(robot.data.joint_pos).clone()
            for k, jn in enumerate(LEFT_ARM_JOINTS):
                jp[:, names.index(jn)] = float(s16[k])
            jp[:, names.index(LEFT_GRIPPER_JOINT)] = float(s16[6])
            for k, jn in enumerate(RIGHT_ARM_JOINTS):
                jp[:, names.index(jn)] = float(s16[7 + k])
            jp[:, names.index(RIGHT_GRIPPER_JOINT)] = float(s16[13])
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            robot.set_joint_position_target(jp)
            robot.write_data_to_sim()

        n_done = 0
        for i in sel:
            ch = _chunk(i)
            full = pd.read_parquet(f"{SRC}/data/{ch}/episode_{i:06d}.parquet")
            state = np.stack(full["observation.state"].to_numpy()).astype(np.float32)
            idxs = list(range(0, min(len(state), MAX_FRAMES * STRIDE), STRIDE))
            capt = {cam: [] for cam in CAMS}
            for t in idxs:
                set_pose(state[t])
                u.sim.step(render=True)
                ex = adapter.extract(u.observation_manager.compute(), 0)
                capt[CAMS[0]].append(image_tools.resize_with_pad(ex.exterior_image_1_left, h, w))
                capt[CAMS[1]].append(image_tools.resize_with_pad(ex.wrist_image_left, h, w))
                capt[CAMS[2]].append(image_tools.resize_with_pad(ex.wrist_image_right, h, w))
            # write parquet (subsampled to idxs so labels align 1:1 with rendered frames)
            os.makedirs(f"{outdir}/data/{ch}", exist_ok=True)
            full.iloc[idxs].reset_index(drop=True).to_parquet(f"{outdir}/data/{ch}/episode_{i:06d}.parquet")
            for cam in CAMS:
                vd = f"{outdir}/videos/{ch}/observation.images.{cam}"
                os.makedirs(vd, exist_ok=True)
                imageio.mimsave(f"{vd}/episode_{i:06d}.mp4",
                                [f.astype(np.uint8) for f in capt[cam]],
                                fps=max(1, FPS // STRIDE), codec="libx264")
            n_done += 1
            if n_done % 5 == 0 or n_done == len(sel):
                print(f"[render] shard {SHARD}: {n_done}/{len(sel)} episodes ({len(idxs)} frames last)", flush=True)
        # per-shard episode index list (for assembly)
        os.makedirs(outdir, exist_ok=True)
        with open(f"{outdir}/rendered_eps.json", "w") as f:
            json.dump(sel, f)
        print(f"[render] shard {SHARD} DONE: {n_done} episodes -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
