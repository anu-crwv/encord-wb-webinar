#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""DAgger round-1 ON-POLICY relabel harness — the correction round-0 (open-loop clone) can't do.

Round-0 (v10) cloned the expert's OWN clean trajectories, so at test time the policy — conditioned
on its OWN autoregressive context — drifts into states the demos never covered (from rest it stalls
~0.32 m from the object and mildly retracts). DAgger fixes exactly this: roll out with the POLICY in
the loop so we VISIT the policy's state distribution, and at every visited state record the EXPERT's
corrective action as the label. Retraining on (policy-context observation -> expert action) closes the
covariate-shift/context gap.

Control: at each step we ALWAYS call the policy (warms its temporal context with every frame, like
the kickstart), then EXECUTE the expert action with probability BETA else the policy action
(DAgger-beta mixing; BETA=0.5 default gives coverage of both on-path and policy-drifted states). The
label recorded at every step is the REACTIVE expert corrective action for the current geometry:
  * object not yet lifted -> approach: pregrasp above object (open), descend+close when aligned/low
  * object lifted (grasped) -> carry to bin, release when over it
This reactive labeler (vs the expert's sequential waypoint machine) gives a sensible action for ANY
state, including the off-manifold ones the policy induces.

Writes LeRobot episodes (same schema as scripted_expert_trossen.py) when WAM_EXPERT_OUT is set, so the
existing v11 assembler can consume them. Needs the policy server (BETA<1). Run via eval/runner.sh with
WAM_EVAL_ENTRY=dagger_rollout_relabel.py. Env knobs mirror the expert plus:
  WAM_DAGGER_BETA     prob of executing the EXPERT action each step (default 0.5; 0 = pure on-policy)
  WAM_DAGGER_EPISODES episodes (default 12)
  WAM_ARM_JITTER      per-episode rest-pose arm jitter (rad) for state diversity (default 0.10)
"""

from __future__ import annotations

import json
import os

import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner import enable_cameras_if_required, get_policy_from_job, load_env
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import JobManager
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

N_EPISODES = int(os.environ.get("WAM_DAGGER_EPISODES", "12"))
OUT = os.environ.get("WAM_EXPERT_OUT", "")
MAX_STEPS = int(os.environ.get("WAM_MAX_STEPS", "160"))
BETA = float(os.environ.get("WAM_DAGGER_BETA", "0.5"))
GRIP_OPEN = float(os.environ.get("WAM_GRIP_OPEN", "0.05"))
GRIP_CLOSED = float(os.environ.get("WAM_GRIP_CLOSED", "0.0"))
PREGRASP_Z = float(os.environ.get("WAM_PREGRASP_Z", "0.12"))
GRASP_Z = float(os.environ.get("WAM_GRASP_Z", "0.0"))
LIFT_Z = float(os.environ.get("WAM_LIFT_Z", "0.14"))
SETTLE_STEPS = int(os.environ.get("WAM_SETTLE_STEPS", "45"))
IK_GAIN = float(os.environ.get("WAM_IK_GAIN", "0.4"))
IK_DAMP = float(os.environ.get("WAM_IK_DAMP", "0.05"))
RESET_JITTER = float(os.environ.get("WAM_RESET_JITTER", "0.04"))   # object xy jitter (m)
ARM_JITTER = float(os.environ.get("WAM_ARM_JITTER", "0.10"))       # rest-pose arm jitter (rad)
CHUNK = 1000
CAMS = ("exterior_image_1_left", "wrist_image_left", "wrist_image_right")
FPS = 30
DOWN_QUAT = np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float64)


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
    instruction = job.language_instruction
    pick_name = cfg["jobs"][0].get("arena_env_args", {}).get("pick_up_object", "corn_can_hope_robolab")
    dest_name = cfg["jobs"][0].get("arena_env_args", {}).get("destination_location", "bowl_ycb_robolab")
    print(f"[dagger] pick={pick_name} dest={dest_name} beta={BETA} eps={N_EPISODES} "
          f"max_steps={MAX_STEPS} out={OUT or '(none)'}", flush=True)

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        try:
            import warp as wp
            return wp.to_torch(x).detach().cpu().numpy()
        except Exception:
            return np.asarray(x)

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
        from isaaclab_arena_dreamzero.embodiments.trossen import _REST_JOINT_POS
        from isaaclab_arena_dreamzero.policy.trossen_adapter import DreamZeroTrossenAdapter
        from openpi_client import image_tools

        env = load_env(job.arena_env_args, job.name)
        obs, _ = env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        names = list(robot.joint_names)
        bnames = list(robot.body_names)
        obj = u.scene[pick_name] if pick_name in u.scene.keys() else None
        dest = u.scene[dest_name] if dest_name in u.scene.keys() else None
        adapter = DreamZeroTrossenAdapter()
        h, w = adapter.target_image_size

        policy = None if BETA >= 1.0 else get_policy_from_job(job)
        if policy is not None:
            policy.set_task_description(instruction)

        larm_dof = [names.index(j) for j in LEFT_ARM_JOINTS]
        lgrip_dof = names.index(LEFT_GRIPPER_JOINT)
        rarm_dof = [names.index(j) for j in RIGHT_ARM_JOINTS]
        rgrip_dof = names.index(RIGHT_GRIPPER_JOINT)
        grip_bodies = [b for b in bnames if b.startswith("follower_left") and "gripper" in b]
        if not grip_bodies:
            grip_bodies = [next(b for b in bnames if b.startswith("follower_left") and b.endswith("link_6"))]
        tcp_bidx = [bnames.index(b) for b in grip_bodies]
        print(f"[dagger] TCP bodies={grip_bodies} pick_obj={'ok' if obj else 'MISSING'} "
              f"dest={'ok' if dest else 'MISSING'}", flush=True)

        rest16 = np.zeros(16, dtype=np.float32)
        for k, j in enumerate(LEFT_ARM_JOINTS):
            rest16[k] = _REST_JOINT_POS[j]
        rest16[6] = GRIP_OPEN
        for k, j in enumerate(RIGHT_ARM_JOINTS):
            rest16[7 + k] = _REST_JOINT_POS[j]
        rest16[13] = _REST_JOINT_POS[RIGHT_GRIPPER_JOINT]

        def ee_pose():
            bp = _np(robot.data.body_pos_w)[0]
            p = bp[tcp_bidx].mean(axis=0).astype(float)
            q = _np(robot.data.body_quat_w)[0][tcp_bidx[0]].astype(float)
            return p, q

        def left_jacobian():
            J = _np(robot.root_physx_view.get_jacobians())
            Jb = np.mean([J[0, b - 1] for b in tcp_bidx], axis=0)
            return Jb[:, larm_dof]

        def ik_step(cur_larm, tgt_pos):
            p, _ = ee_pose()
            err = tgt_pos - p
            Ju = left_jacobian()[0:3, :]  # position-only (matches the expert default)
            JT = Ju.T
            dq = JT @ np.linalg.inv(Ju @ JT + (IK_DAMP ** 2) * np.eye(3)) @ err
            dq = np.clip(dq, -0.2, 0.2)
            return cur_larm + IK_GAIN * dq

        def build_action(larm_tgt, grip):
            a = rest16.copy()
            a[0:6] = larm_tgt
            a[6] = grip
            return a

        def cur_larm():
            return _np(robot.data.joint_pos)[0][larm_dof].astype(float)

        def obj_xyz():
            return _np(obj.data.root_pos_w)[0].astype(float)

        def state16():
            jp = _np(robot.data.joint_pos)[0]
            return np.concatenate([jp[larm_dof], [jp[lgrip_dof]], jp[rarm_dof],
                                   [jp[rgrip_dof]], [0.0, 0.0]]).astype(np.float32)

        def corrective(gp, op, place, lifted):
            """Reactive expert target+grip for ANY state (dominant case: approach from rest)."""
            if not lifted:
                d_xy = float(np.linalg.norm(gp[:2] - op[:2]))
                d_z = float(gp[2] - op[2])
                if d_xy > 0.03 or d_z > PREGRASP_Z * 0.6:
                    return op + np.array([0, 0, PREGRASP_Z]), GRIP_OPEN      # go above object
                close = (d_xy < 0.03 and d_z < 0.04)
                return op + np.array([0, 0, GRASP_Z]), (GRIP_CLOSED if close else GRIP_OPEN)
            d_bin = float(np.linalg.norm(gp[:2] - place[:2]))
            if d_bin > 0.06:
                return place + np.array([0, 0, LIFT_Z]), GRIP_CLOSED         # carry to bin
            return place + np.array([0, 0, PREGRASP_Z]), GRIP_OPEN           # release over bin

        def step(a16):
            act = torch.zeros((u.num_envs, 16), device=u.device, dtype=torch.float32)
            act[0] = torch.tensor(np.asarray(a16, np.float32), device=u.device)
            return env.step(act)

        def reset_scene(ep, rng):
            o, _ = env.reset()
            if obj is not None:
                p = _as_torch(obj.data.root_pos_w).clone()
                if RESET_JITTER > 0:
                    p[0, 0] += float(rng.uniform(-RESET_JITTER, RESET_JITTER))
                    p[0, 1] += float(rng.uniform(-RESET_JITTER, RESET_JITTER))
                q = _as_torch(obj.data.root_quat_w).clone()
                if os.environ.get("WAM_LAY_OBJECT", "1") == "1":
                    q[0] = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=q.device, dtype=q.dtype)
                obj.write_root_pose_to_sim(torch.cat([p, q], dim=-1))
                obj.write_data_to_sim()
            # jitter the rest pose so the policy's cold-start states span a region (diverse labels)
            jitter_larm = np.array(
                [_REST_JOINT_POS[j] + (float(rng.uniform(-ARM_JITTER, ARM_JITTER)) if ARM_JITTER > 0 else 0.0)
                 for j in LEFT_ARM_JOINTS], dtype=np.float32)
            jp = _as_torch(robot.data.joint_pos).clone()
            for k in range(6):
                jp[:, larm_dof[k]] = float(jitter_larm[k])
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            robot.set_joint_position_target(jp)
            robot.write_data_to_sim()
            # settle the object while HOLDING the jittered pose (commanding rest16 here would drag
            # the arm back to rest and erase the jitter before the rollout starts)
            hold16 = build_action(jitter_larm, GRIP_OPEN)
            for _ in range(SETTLE_STEPS):
                step(hold16)
            return u.observation_manager.compute()

        rng = np.random.default_rng(int(os.environ.get("WAM_EP_SEED_BASE", "7000")))
        if policy is not None:
            policy.reset()
        results, written, n_written = [], [], 0
        for ep in range(N_EPISODES):
            if policy is not None:
                policy.reset()
            obs = reset_scene(ep, rng)
            o0 = obj_xyz()
            place = _np(dest.data.root_pos_w)[0].astype(float) if dest is not None else o0 + np.array([0.25, -0.45, 0.0])
            capt = {c: [] for c in CAMS}
            rec_state, rec_action = [], []
            n_expert = 0
            min_dist = 1e9
            for t in range(MAX_STEPS):
                ex = adapter.extract(obs, 0)  # SAME obs the policy sees this step (consistent context)
                larm = cur_larm()
                gp, _ = ee_pose()
                op = obj_xyz()
                lifted = bool(op[2] - o0[2] > 0.05)
                tgt, grip = corrective(gp, op, place, lifted)
                expert_a = build_action(ik_step(larm, tgt), grip).astype(np.float32)
                # record (obs at t, EXPERT label) — DAgger relabel of the visited state
                if OUT:
                    capt[CAMS[0]].append(image_tools.resize_with_pad(ex.exterior_image_1_left, h, w))
                    capt[CAMS[1]].append(image_tools.resize_with_pad(ex.wrist_image_left, h, w))
                    capt[CAMS[2]].append(image_tools.resize_with_pad(ex.wrist_image_right, h, w))
                    rec_state.append(state16())
                    rec_action.append(expert_a)
                # ALWAYS warm the policy; execute expert w.p. BETA else policy (DAgger-beta)
                pol_a = None
                if policy is not None:
                    pol_a = _np(policy.get_action(env, obs))[0].astype(np.float32)
                if policy is None or rng.random() < BETA:
                    exec_a = expert_a
                    n_expert += 1
                else:
                    exec_a = pol_a
                obs, _, term, trunc, _ = step(exec_a)
                min_dist = min(min_dist, float(np.linalg.norm(ee_pose()[0] - obj_xyz())))
                done = (bool(term.any()) if hasattr(term, "any") else bool(term)) or \
                       (bool(trunc.any()) if hasattr(trunc, "any") else bool(trunc))
                if done:
                    break
            n = len(rec_state)
            results.append(dict(ep=ep, steps=n, min_dist=min_dist, frac_expert=n_expert / max(1, n)))
            print(f"[dagger] ep{ep}: steps={n} min_ee_obj={min_dist:.3f}m frac_expert_exec={n_expert/max(1,n):.2f}", flush=True)
            if OUT and rec_state:
                _write_episode(OUT, n_written, rec_state, rec_action, capt, instruction, imageio)
                written.append(int(n_written)); n_written += 1

        print(f"[dagger] ===== DONE: {len(written)} relabel episodes, "
              f"{sum(r['steps'] for r in results)} transitions, mean min_ee_obj="
              f"{np.mean([r['min_dist'] for r in results]):.3f}m =====", flush=True)
        if OUT and written:
            with open(f"{OUT}/demos.json", "w") as f:
                json.dump({"written": written, "n": len(written), "task": job.name,
                           "instruction": instruction, "kind": "dagger_r1_relabel", "beta": BETA}, f)
            print(f"[dagger] wrote {len(written)} relabel episodes -> {OUT}", flush=True)


def _write_episode(out, ep, states, actions, capt, instruction, imageio):
    """One LeRobot episode (sim videos + expert-relabel state/action + language). Matches
    scripted_expert_trossen._write_episode so the v11 assembler consumes it unchanged."""
    import pandas as pd
    ch = f"chunk-{ep // CHUNK:03d}"
    os.makedirs(f"{out}/data/{ch}", exist_ok=True)
    n = len(states)
    df = pd.DataFrame({
        "action": [np.asarray(a, dtype=np.float32) for a in actions],
        "observation.state": [np.asarray(s, dtype=np.float32) for s in states],
        "timestamp": [i / FPS for i in range(n)],
        "frame_index": list(range(n)),
        "episode_index": [ep] * n,
        "index": list(range(n)),
        "task_index": [0] * n,
        "annotation.language.language_instruction": [instruction] * n,
    })
    df.to_parquet(f"{out}/data/{ch}/episode_{ep:06d}.parquet")
    for cam in CAMS:
        vd = f"{out}/videos/{ch}/observation.images.{cam}"
        os.makedirs(vd, exist_ok=True)
        imageio.mimsave(f"{vd}/episode_{ep:06d}.mp4",
                        [f.astype(np.uint8) for f in capt[cam]], fps=FPS, codec="libx264")


if __name__ == "__main__":
    main()
