#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""One-shot camera/scene introspection for the Trossen Arena env — diagnoses the
black sim render. Builds the env, resets, then dumps:
  * scene asset keys (table/object/bowl/robot/cameras)
  * robot body (link) names + WORLD positions (so we know where cam_high_link etc.
    actually are, and where to aim cameras)
  * each camera obs: shape + min/max/mean (all ~0 == black) + saves the first frame
    as a PNG under /data/wam/eval_videos/debug/
No policy/server — just the env + render. Run like the eval:
  python.sh debug_cameras.py --eval_jobs_config <cfg.json> --enable_cameras --headless
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
    parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(parser)
    args_cli, _ = parser.parse_known_args()
    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        cfg = json.load(f)
    enable_cameras_if_required(cfg, args_cli)
    job = JobManager(cfg["jobs"]).all_jobs[0]

    os.makedirs(OUT, exist_ok=True)
    with SimulationAppContext(args_cli):
        import isaaclab_arena_dreamzero.embodiments  # noqa: F401  (register trossen embodiment)
        import isaaclab_arena_dreamzero.environments  # noqa: F401  (register trossen_pick_and_place)

        env = load_env(job.arena_env_args, job.name)
        obs, _ = env.reset()
        obs, _ = env.reset()  # 2nd cycle so materials/render settle
        u = env.unwrapped

        print("\n===== SCENE KEYS =====", flush=True)
        try:
            print(list(u.scene.keys()), flush=True)
        except Exception as e:
            print("scene.keys err:", e, flush=True)

        print("\n===== JOINTS (post-reset) =====", flush=True)
        try:
            robot = u.scene["robot"]
            jn = list(robot.joint_names)
            jp = _to_np(robot.data.joint_pos)[0]
            print("  cfg.init_state.joint_pos:", dict(robot.cfg.init_state.joint_pos), flush=True)
            for i, nm in enumerate(jn):
                if i < jp.shape[0]:
                    print(f"    {nm:36s} = {jp[i]:+.3f}", flush=True)
        except Exception as e:
            print("  joints err:", e, flush=True)

        print("\n===== ROBOT LINKS (name -> world xyz) =====", flush=True)
        try:
            robot = u.scene["robot"]
            names = list(robot.body_names)
            pos = _to_np(robot.data.body_pos_w)[0]  # (num_bodies, 3) for env 0
            for i, nm in enumerate(names):
                if i < pos.shape[0]:
                    p = pos[i]
                    print(f"  {nm:36s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})", flush=True)
        except Exception as e:
            print("robot links err:", e, flush=True)

        print("\n===== RIGID OBJECTS (key -> world xyz) =====", flush=True)
        obj_pos = {}
        for k in list(u.scene.keys()):
            try:
                o = u.scene[k]
                if hasattr(o, "data") and hasattr(o.data, "root_pos_w"):
                    p = _to_np(o.data.root_pos_w)[0]
                    obj_pos[k] = p
                    print(f"  {k:36s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})", flush=True)
            except Exception:
                pass

        print("\n===== CAMERA SENSOR WORLD POSES (pos + forward dir) =====", flush=True)

        def _fwd(qwxyz):
            # rotate camera -Z (opengl forward) by quaternion (w,x,y,z)
            w, x, y, z = qwxyz
            v = np.array([0.0, 0.0, -1.0])
            t = 2 * np.cross([x, y, z], v)
            return v + w * t + np.cross([x, y, z], t)

        cam_world = {}  # cname -> (pos, quat_wxyz)
        for cname in ("exterior_image_1_left", "wrist_image_left", "wrist_image_right"):
            try:
                cam = u.scene[cname]
                p = _to_np(cam.data.pos_w)[0]
                q = _to_np(cam.data.quat_w_world)[0]
                cam_world[cname] = (p, q)
                f = _fwd(q)
                print(f"  {cname:24s} pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"quat_wxyz=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}) "
                      f"fwd=({f[0]:+.2f},{f[1]:+.2f},{f[2]:+.2f})", flush=True)
            except Exception as e:
                print(f"  {cname}: pose err {e}", flush=True)

        # ---- Suggested OffsetCfg.rot so each camera AIMS AT the workspace ----
        # Convention-free: the world camera quat is a fixed left-multiple of the
        # offset (K = q_world_old * offset_old^-1, constant for this link pose), so
        #   offset_new = offset_old * inv(q_world_old) * q_lookat_world .
        # offset_old is the current cfg value (same for all three cameras).
        print("\n===== SUGGESTED CAMERA OFFSETS (rot wxyz, opengl) =====", flush=True)

        def _qmul(a, b):
            w1, x1, y1, z1 = a
            w2, x2, y2, z2 = b
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])

        def _qinv(q):
            q = np.asarray(q, float)
            return np.array([q[0], -q[1], -q[2], -q[3]]) / float(q @ q)

        def _mat2quat(R):
            t = np.trace(R)
            if t > 0:
                s = np.sqrt(t + 1.0) * 2
                w = 0.25 * s
                x = (R[2, 1] - R[1, 2]) / s
                y = (R[0, 2] - R[2, 0]) / s
                z = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
            q = np.array([w, x, y, z])
            return q / np.linalg.norm(q)

        def _lookat_quat(cam_pos, target, up=(0.0, 0.0, 1.0)):
            fwd = np.asarray(target, float) - np.asarray(cam_pos, float)
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            up = np.asarray(up, float)
            right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
            cup = np.cross(right, fwd)
            # opengl camera basis columns in world: X=right, Y=up, Z=back(-fwd)
            R = np.column_stack([right, cup, -fwd])
            return _mat2quat(R)

        OFFSET_OLD = np.array([0.5, -0.5, 0.5, -0.5])  # current cfg value (all 3 cams)
        cube = obj_pos.get("rubiks_cube_hot3d_robolab")
        bowl = obj_pos.get("bowl_ycb_robolab")
        pts = [p for p in (cube, bowl) if p is not None]
        target = (np.mean(pts, axis=0) if pts else np.array([1.0, 0.0, 0.78]))
        print(f"  workspace target (object centroid) = ({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})", flush=True)
        for cname, (p, q) in cam_world.items():
            q_des = _lookat_quat(p, target)
            off_new = _qmul(_qmul(OFFSET_OLD, _qinv(q)), q_des)
            off_new = off_new / np.linalg.norm(off_new)
            chk = _fwd(_qmul(_qmul(q, _qinv(OFFSET_OLD)), off_new))  # predicted new world fwd
            print(f"  {cname:24s} rot=({off_new[0]:+.4f},{off_new[1]:+.4f},{off_new[2]:+.4f},{off_new[3]:+.4f})"
                  f"  -> predicted_fwd=({chk[0]:+.2f},{chk[1]:+.2f},{chk[2]:+.2f})", flush=True)

        print("\n===== CAMERA OBS STATS =====", flush=True)
        cam = obs.get("camera_obs", {}) if isinstance(obs, dict) else {}
        if not cam:
            print("  NO 'camera_obs' group in obs! keys=", list(obs.keys()) if isinstance(obs, dict) else type(obs), flush=True)
        for key, val in (cam.items() if hasattr(cam, "items") else []):
            a = _to_np(val)
            a0 = a[0] if a.ndim == 4 else a
            stat = f"shape={a.shape} dtype={a.dtype} min={float(a.min()):.2f} max={float(a.max()):.2f} mean={float(a.mean()):.2f}"
            black = "  <<< ALL BLACK" if float(a.max()) < 1.0 else ""
            print(f"  {key:28s} {stat}{black}", flush=True)
            try:
                import imageio
                img = a0.astype(np.uint8) if a0.dtype != np.uint8 else a0
                imageio.imwrite(os.path.join(OUT, f"{key}.png"), img)
            except Exception as e:
                print(f"    save {key} failed: {e}", flush=True)
        print(f"\n[debug_cameras] frames saved under {OUT}", flush=True)


if __name__ == "__main__":
    main()
