#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 Phase-3 (2x2 modality swap) — STAGE 1: real frames + sim renders at REAL poses.

To build a CONTROLLED image/state swap we anchor both modalities to the same pose
sequence. Step 0 (CPU pod) saved real_states (K,16), gt_chunks (K,24,16), and meta
(dataset root, episode, sampled timesteps ts, camera order).

This stage (Isaac image, has imageio + ffmpeg):
  1. loads the K REAL camera frames (3 views) from the dataset mp4s at ts, and
  2. drives the sim robot to each real joint pose, settles, and captures the 3 SIM
     camera views + the settled/MEASURED 16-dim sim state via the SAME adapter the
     eval uses. Both image sets are resize_with_pad'd to 480x640, exactly like
     DreamZeroTrossenAdapter.pack_request.

Writes real_frames.npy (K,3,480,640,3), sim_frames.npy (K,3,480,640,3),
sim_states.npy (K,16). Because sim_state is the settled/measured pose while
real_state is the commanded one, Stage 2's RS-vs-RR comparison also probes the
commanded-vs-measured state distinction.

Run via eval/runner.sh with WAM_EVAL_ENTRY=diag_render_at_poses.py and
WAM_SKIP_SERVER_PROBE=1 (no policy server needed).
"""

from __future__ import annotations

import json
import os

import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner import enable_cameras_if_required, load_env
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import JobManager
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

D = os.environ.get("WAM_2X2_DIR", "/data/wam/diag2x2")


def _load_real_frames(meta: dict, h: int, w: int):
    """Read the K real frames (3 views) from the dataset mp4s at meta['ts']."""
    import imageio.v2 as imageio
    from openpi_client import image_tools

    root, ep, ts, cams = meta["root"], int(meta["episode"]), meta["ts"], meta["cams"]
    per_cam = {}
    for c in cams:
        path = f"{root}/videos/chunk-000/observation.images.{c}/episode_{ep:06d}.mp4"
        allf = [np.asarray(f)[:, :, :3].astype(np.uint8) for f in imageio.get_reader(path, "ffmpeg")]
        per_cam[c] = [image_tools.resize_with_pad(allf[min(t, len(allf) - 1)], h, w) for t in ts]
    # (K, 3, h, w, 3) in the trained camera order
    return np.stack([np.stack([per_cam[c][k] for c in cams]) for k in range(len(ts))]).astype(np.uint8)


def main() -> None:
    import torch

    parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(parser)
    args_cli, _ = parser.parse_known_args()
    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        cfg = json.load(f)
    enable_cameras_if_required(cfg, args_cli)
    job = JobManager(cfg["jobs"]).all_jobs[0]

    meta = json.load(open(f"{D}/meta.json"))
    real_states = np.load(f"{D}/real_states.npy")  # (K, 16)
    K = real_states.shape[0]

    from isaaclab_arena_dreamzero.policy.trossen_adapter import DreamZeroTrossenAdapter
    h, w = DreamZeroTrossenAdapter.target_image_size  # (480, 640)

    # Real frames first (pure CPU, no sim needed).
    real_frames = _load_real_frames(meta, h, w)
    np.save(f"{D}/real_frames.npy", real_frames)
    print(f"[render] real_frames {real_frames.shape} saved; rendering {K} sim poses", flush=True)

    def _as_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        import warp as wp
        return wp.to_torch(x)

    with SimulationAppContext(args_cli):
        import isaaclab_arena_dreamzero.embodiments  # noqa: F401
        import isaaclab_arena_dreamzero.environments  # noqa: F401
        from isaaclab_arena_dreamzero.embodiments.observations import (
            LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEFT_GRIPPER_JOINT, RIGHT_GRIPPER_JOINT,
        )
        from openpi_client import image_tools

        env = load_env(job.arena_env_args, job.name)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        names = list(robot.joint_names)
        adapter = DreamZeroTrossenAdapter()

        def set_pose(s16: np.ndarray) -> None:
            jp = _as_torch(robot.data.joint_pos).clone()
            for i, jn in enumerate(LEFT_ARM_JOINTS):
                jp[:, names.index(jn)] = float(s16[i])
            jp[:, names.index(LEFT_GRIPPER_JOINT)] = float(s16[6])
            for i, jn in enumerate(RIGHT_ARM_JOINTS):
                jp[:, names.index(jn)] = float(s16[7 + i])
            jp[:, names.index(RIGHT_GRIPPER_JOINT)] = float(s16[13])
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            robot.set_joint_position_target(jp)
            robot.write_data_to_sim()

        sim_frames, sim_states = [], []
        for k in range(K):
            set_pose(real_states[k])
            for _ in range(6):  # settle so render + measured state reflect the pose
                u.sim.step(render=True)
                u.scene.update(u.physics_dt)
            obs = u.observation_manager.compute()
            ex = adapter.extract(obs, 0)
            f3 = np.stack([
                image_tools.resize_with_pad(ex.exterior_image_1_left, h, w),
                image_tools.resize_with_pad(ex.wrist_image_left, h, w),
                image_tools.resize_with_pad(ex.wrist_image_right, h, w),
            ]).astype(np.uint8)
            sim_frames.append(f3)
            sim_states.append(np.asarray(ex.state, dtype=np.float32))
            dstate = float(np.abs(np.asarray(ex.state)[:14] - real_states[k][:14]).mean())
            print(f"[render] pose {k:02d}: |sim_state-real_state|(arms+grip)={dstate:.4f}", flush=True)

        np.save(f"{D}/sim_frames.npy", np.stack(sim_frames))
        np.save(f"{D}/sim_states.npy", np.stack(sim_states))
        print(f"[render] saved sim_frames {np.stack(sim_frames).shape} "
              f"sim_states {np.stack(sim_states).shape}", flush=True)


if __name__ == "__main__":
    main()
