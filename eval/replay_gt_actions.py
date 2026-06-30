#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Open-loop replay of a REAL episode's ground-truth actions in the Trossen sim.

Isolates the action-application path from the model entirely: we feed the dataset's
own 16-dim actions (the real reach-grasp trajectory) straight into the env as joint
targets, step the sim, and track the gripper height + save frames. If the sim arm
descends to the table and grasps (reproducing the real motion), the action plumbing is
correct and the eval's weak motion is the visual domain gap. If the arm instead moves
AWAY / up, the action is being reversed/mis-applied in the sim.

Run like the eval: python.sh replay_gt_actions.py --eval_jobs_config <cfg> --enable_cameras --headless
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

OUT = "/data/wam/eval_videos/debug"
GT = "/data/src/dreamzero-wam/eval/gt_actions_ep0.npy"


def _to_np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import warp as wp

        return wp.to_torch(x).detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


def main() -> None:
    import torch

    parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(parser)
    args_cli, _ = parser.parse_known_args()
    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        cfg = json.load(f)
    enable_cameras_if_required(cfg, args_cli)
    job = JobManager(cfg["jobs"]).all_jobs[0]
    os.makedirs(OUT, exist_ok=True)

    gt = np.load(GT).astype(np.float32)  # (T, 16) real joint-target trajectory
    print(f"[replay] loaded GT actions {gt.shape}", flush=True)

    with SimulationAppContext(args_cli):
        import isaaclab_arena_dreamzero.embodiments  # noqa: F401
        import isaaclab_arena_dreamzero.environments  # noqa: F401
        from isaaclab_arena_dreamzero.embodiments.trossen import _REST_JOINT_POS

        env = load_env(job.arena_env_args, job.name)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        names = list(robot.joint_names)
        bnames = list(robot.body_names)

        # Seed the real rest pose (same as the eval start).
        jp = (robot.data.joint_pos if isinstance(robot.data.joint_pos, torch.Tensor)
              else __import__("warp").to_torch(robot.data.joint_pos)).clone()
        for nm, val in _REST_JOINT_POS.items():
            if nm in names:
                jp[:, names.index(nm)] = float(val)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp)
        robot.write_data_to_sim()

        grip_i = bnames.index("follower_left_gripper_left") if "follower_left_gripper_left" in bnames else None
        cube = u.scene["rubiks_cube_hot3d_robolab"] if "rubiks_cube_hot3d_robolab" in u.scene.keys() else None

        def gripper_z():
            return float(_to_np(robot.data.body_pos_w)[0][grip_i][2]) if grip_i is not None else float("nan")

        def gripper_xyz():
            p = _to_np(robot.data.body_pos_w)[0][grip_i]
            return (float(p[0]), float(p[1]), float(p[2]))

        z0 = gripper_z()
        cube_p = _to_np(cube.data.root_pos_w)[0] if cube is not None else None
        print(f"[replay] START gripper={tuple(round(v,3) for v in gripper_xyz())} cube={None if cube_p is None else tuple(round(float(v),3) for v in cube_p)}", flush=True)

        # Apply GT actions open-loop. Subsample (real fps != sim control) and hold each a
        # few steps so the arm can track. Track gripper height vs the real motion.
        T = gt.shape[0]
        stride = max(1, T // 220)
        idxs = list(range(0, T, stride))
        save_at = {0: "f000", len(idxs) // 4: "f025", len(idxs) // 2: "f050", 3 * len(idxs) // 4: "f075", len(idxs) - 1: "f100"}
        zs = []
        for k, i in enumerate(idxs):
            act = torch.from_numpy(gt[i]).to(dtype=torch.float32, device=u.device).reshape(1, -1)
            for _ in range(2):
                env.step(act)
            z = gripper_z()
            zs.append(z)
            if k % 20 == 0 or k == len(idxs) - 1:
                gx = gripper_xyz()
                print(f"[replay] step {k:3d}/{len(idxs)} gt_j1={gt[i,1]:+.2f} gt_grip={gt[i,6]:+.3f} -> gripper=({gx[0]:+.2f},{gx[1]:+.2f},{gx[2]:+.2f})", flush=True)
            if k in save_at:
                try:
                    import imageio
                    cam = u.scene["exterior_image_1_left"]
                    img = _to_np(cam.data.output["rgb"])[0]
                    imageio.imwrite(os.path.join(OUT, f"gtreplay_{save_at[k]}.png"), img.astype(np.uint8))
                except Exception as e:
                    print(f"  save {save_at[k]} failed: {e}", flush=True)

        zmin, zmax = min(zs), max(zs)
        print(f"\n[replay] SUMMARY gripper_z: start={z0:.3f} min={zmin:.3f} max={zmax:.3f}", flush=True)
        print(f"[replay] gripper descended by {z0 - zmin:+.3f} m below start "
              f"({'DESCENDS toward table = plumbing OK' if (z0 - zmin) > 0.15 else 'does NOT descend = action likely reversed/mis-applied'})", flush=True)
        if cube is not None:
            cp = _to_np(cube.data.root_pos_w)[0]
            print(f"[replay] cube end pos=({cp[0]:+.2f},{cp[1]:+.2f},{cp[2]:+.2f})", flush=True)
        print(f"[replay] frames saved under {OUT} (gtreplay_*.png)", flush=True)


if __name__ == "__main__":
    main()
