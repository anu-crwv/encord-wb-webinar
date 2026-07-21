#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop KICKSTART experiment: does a motion-bearing seed unlock sustained,
task-directed control after handover? (Diagnosis: cold-start/bootstrapping failure.)

Three phases per rollout:
  1. SEED  (t < N)        : execute a scripted/known-good action; STILL call DreamZero
                            every step (warm its temporal context) but DISCARD its action.
  2. HOLD  (N <= t < N+K)  : hold the arm still for K frames; still warm DreamZero.
  3. POLICY(t >= N+K)      : hand over — execute DreamZero's action.

The claim under test is NOT "does the arm move during the seed" but "does the model
CONTINUE to reduce gripper->object distance AFTER external control stops." We therefore
separate distance reduction by phase and report post-handover progress strictly.

Env knobs (swept across jobs):
  WAM_SEED_MODE   scripted_toward | scripted_neutral | scripted_away | ep35_gt   (default scripted_toward)
  WAM_KICKSTART_STEPS  N seed steps (default 8; 0 = plain AR baseline)
  WAM_HOLD_STEPS       K hold steps after seed (default 0)
  WAM_SEED_ACTIONS     path to ep35 seed actions .npy (for ep35_gt mode)

Seed directions (scripted, joint-space, absolute targets built from the current pose):
  toward  : ramp left shoulder Lj1 + elbow Lj2 down/in  -> gripper descends toward the table/object
  neutral : oscillate Lj0 (shoulder yaw)                -> visible motion, ~no net progress ("motion-only")
  away    : ramp Lj1 up                                 -> gripper rises away from the object

Run via eval/runner.sh with WAM_EVAL_ENTRY=run_trossen_kickstart.py (needs the policy server).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner import enable_cameras_if_required, get_policy_from_job, load_env
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import JobManager
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

MAX_STEPS = int(os.environ.get("WAM_EVAL_MAX_STEPS", "150"))
N_EPISODES = int(os.environ.get("WAM_KICK_EPISODES", "3"))
SEED_ACTIONS_PATH = os.environ.get("WAM_SEED_ACTIONS", "/data/wam/arladder/ep35_seed_actions.npy")
# List of [seed_mode, N_seed, K_hold] configs run SEQUENTIALLY in one job (one Arena setup,
# one warm server -> no cross-job server contention). Default = core confirming set.
_DEFAULT_CONFIGS = [["scripted_toward", 0, 0], ["scripted_toward", 8, 0], ["ep35_gt", 8, 0]]
CONFIGS = json.loads(os.environ.get("WAM_KICK_CONFIGS", json.dumps(_DEFAULT_CONFIGS)))


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
    instruction = job.language_instruction
    pick_name = job.arena_env_args.get("pick_up_object", "corn_can_hope_robolab")

    import statistics as st
    seed_traj = None
    if any(c[0] == "ep35_gt" for c in CONFIGS):
        seed_traj = np.load(SEED_ACTIONS_PATH).astype(np.float32)
        print(f"[kick] loaded ep35 seed actions {seed_traj.shape} from {SEED_ACTIONS_PATH}", flush=True)
    print(f"[kick] configs (seed,N,K): {CONFIGS}  episodes/config={N_EPISODES} max_steps={MAX_STEPS}", flush=True)

    with SimulationAppContext(args_cli):
        import isaaclab_arena_dreamzero.embodiments  # noqa: F401
        import isaaclab_arena_dreamzero.environments  # noqa: F401
        from isaaclab_arena_dreamzero.embodiments.observations import pack_trossen_state_16d
        from isaaclab_arena_dreamzero.embodiments.trossen import _REST_JOINT_POS

        env = load_env(job.arena_env_args, job.name)
        policy = get_policy_from_job(job)
        policy.set_task_description(instruction)
        u = env.unwrapped
        robot = u.scene["robot"]
        names = list(robot.joint_names)
        bnames = list(robot.body_names)
        grip_i = bnames.index("follower_left_gripper_left") if "follower_left_gripper_left" in bnames else None
        obj = u.scene[pick_name] if pick_name in u.scene.keys() else None
        print(f"[kick] pick={pick_name} obj={'ok' if obj else 'MISSING'} "
              f"grip={'ok' if grip_i is not None else 'MISSING'}", flush=True)

        def grip_xyz():
            return _to_np(robot.data.body_pos_w)[0][grip_i].astype(float) if grip_i is not None else np.full(3, np.nan)

        def obj_xyz():
            return _to_np(obj.data.root_pos_w)[0].astype(float) if obj is not None else np.full(3, np.nan)

        def seed_action(mode: str, N: int, t: int, cur16: np.ndarray) -> np.ndarray:
            a = cur16.copy()
            frac = (t + 1) / max(N, 1)
            if mode == "ep35_gt":
                return seed_traj[min(t, len(seed_traj) - 1)]
            if mode == "scripted_toward":
                a[1] = cur16[1] + 0.9 * frac      # Lj1 shoulder down
                a[2] = cur16[2] + 0.4 * frac      # Lj2 elbow
            elif mode == "scripted_away":
                a[1] = cur16[1] - 0.5 * frac      # raise up
            elif mode == "scripted_neutral":
                a[0] = cur16[0] + 0.25 * math.sin(2 * math.pi * (t / max(N, 1)) * 2.0)  # yaw wiggle
            return a

        def seed_pose(obs):
            jp_t = (robot.data.joint_pos if isinstance(robot.data.joint_pos, torch.Tensor)
                    else __import__("warp").to_torch(robot.data.joint_pos)).clone()
            for nm, val in _REST_JOINT_POS.items():
                if nm in names:
                    jp_t[:, names.index(nm)] = float(val)
            robot.write_joint_state_to_sim(jp_t, torch.zeros_like(jp_t))
            robot.set_joint_position_target(jp_t)
            robot.write_data_to_sim()
            fresh = u.observation_manager.compute()
            return fresh if isinstance(fresh, dict) and fresh else obs

        def run_config(mode: str, N: int, K: int):
            rows = []
            for ep in range(N_EPISODES):
                policy.reset()
                obs, _ = env.reset()
                obs = seed_pose(obs)
                d0 = float(np.linalg.norm(grip_xyz() - obj_xyz()))
                d_end_seed = d_end_hold = d_5policy = d0
                cos_toward = []
                prev_g = grip_xyz()
                d_final = d0
                for t in range(MAX_STEPS):
                    cur16 = _to_np(pack_trossen_state_16d(u))[0].astype(float)
                    model_out = policy.get_action(env, obs)  # ALWAYS call -> warm DreamZero context
                    if t < N:
                        act_np = seed_action(mode, N, t, cur16)
                        action = torch.from_numpy(np.asarray(act_np, np.float32)).reshape(1, -1).to(u.device)
                        phase = "seed"
                    elif t < N + K:
                        action = torch.from_numpy(cur16.astype(np.float32)).reshape(1, -1).to(u.device)
                        phase = "hold"
                    else:
                        action = model_out
                        phase = "policy"
                    obs, _, term, trunc, _ = env.step(action)
                    g = grip_xyz()
                    d = float(np.linalg.norm(g - obj_xyz()))
                    d_final = d
                    if phase == "policy":
                        ov, gv = obj_xyz() - prev_g, g - prev_g
                        nv = np.linalg.norm(gv) * np.linalg.norm(ov)
                        if nv > 1e-6:
                            cos_toward.append(float(np.dot(gv, ov) / nv))
                    prev_g = g
                    if t == N - 1:
                        d_end_seed = d
                    if t == N + K - 1:
                        d_end_hold = d
                    if t == N + K + 4:
                        d_5policy = d
                    done = bool(term.any()) if hasattr(term, "any") else bool(term)
                    if (bool(trunc.any()) if hasattr(trunc, "any") else bool(trunc)) or done:
                        break
                red_seed = d0 - d_end_seed
                red_pol_total = d_end_hold - d_final
                red_pol5 = d_end_hold - d_5policy
                mean_cos = float(np.mean(cos_toward)) if cos_toward else float("nan")
                rows.append(dict(red_seed=red_seed, red_pol_total=red_pol_total,
                                 red_pol5=red_pol5, mean_cos=mean_cos, d0=d0, d_final=d_final))
                print(f"[kick] {mode} N={N} K={K} ep{ep}: d0={d0:.3f}->seed {d_end_seed:.3f}"
                      f"->hold {d_end_hold:.3f}->final {d_final:.3f} | seed_red={red_seed:+.3f} "
                      f"POST-HANDOVER={red_pol_total:+.3f} (first5={red_pol5:+.3f}) cos={mean_cos:+.2f}", flush=True)
            pol = [r["red_pol_total"] for r in rows]
            cos = [r["mean_cos"] for r in rows if r["mean_cos"] == r["mean_cos"]]
            return dict(mode=mode, N=N, K=K, n=len(rows),
                        seed_red=st.mean([r["red_seed"] for r in rows]) if rows else float("nan"),
                        post=st.mean(pol) if pol else float("nan"),
                        pos=sum(1 for x in pol if x > 0.02), tot=len(pol),
                        cos=st.mean(cos) if cos else float("nan"))

        summaries = []
        for mode, N, K in CONFIGS:
            print(f"\n[kick] ===== running config seed={mode} N={N} K={K} =====", flush=True)
            summaries.append(run_config(mode, int(N), int(K)))

        print("\n[kick] ================= KICKSTART SUMMARY =================", flush=True)
        print(f"  {'seed':16s} {'N':>3s} {'K':>3s} {'seed_red':>9s} {'POST_HANDOVER':>14s} {'pos':>6s} {'cos':>6s}", flush=True)
        for s in summaries:
            print(f"  {s['mode']:16s} {s['N']:>3d} {s['K']:>3d} {s['seed_red']:>+9.3f} "
                  f"{s['post']:>+14.3f} {s['pos']:>3d}/{s['tot']:<2d} {s['cos']:>+6.2f}", flush=True)
        best = max((s for s in summaries), key=lambda s: (s['post'] if s['post'] == s['post'] else -9), default=None)
        if best and best["post"] > 0.02:
            print(f"\n[kick] VERDICT: BOOTSTRAP WORKS — seed={best['mode']} N={best['N']} K={best['K']} gives "
                  f"post-handover reduction {best['post']:+.3f} m ({best['pos']}/{best['tot']} rollouts). "
                  f"An inference-side initializer can drive sustained task-directed control.", flush=True)
        else:
            print(f"\n[kick] VERDICT: NO SUSTAINED POST-HANDOVER PROGRESS in any config — the cold-start is "
                  f"deeper than a one-time motion seed (needs persistent seed / short-context regime / retrain).", flush=True)


if __name__ == "__main__":
    main()
