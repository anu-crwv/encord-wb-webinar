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
        import isaaclab_arena_dreamzero.embodiments  # noqa: F401  (register trossen)

        env = load_env(job.arena_env_args, job.name)
        obs, _ = env.reset()
        obs, _ = env.reset()  # 2nd cycle so materials/render settle
        u = env.unwrapped

        print("\n===== SCENE KEYS =====", flush=True)
        try:
            print(list(u.scene.keys()), flush=True)
        except Exception as e:
            print("scene.keys err:", e, flush=True)

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
        for k in list(u.scene.keys()):
            try:
                o = u.scene[k]
                if hasattr(o, "data") and hasattr(o.data, "root_pos_w"):
                    p = _to_np(o.data.root_pos_w)[0]
                    print(f"  {k:36s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})", flush=True)
            except Exception:
                pass

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
