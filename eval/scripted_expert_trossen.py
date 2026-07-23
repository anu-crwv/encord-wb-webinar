#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Scripted IK pick-place EXPERT for the Trossen Arena env — the foundation of the DAgger fix.

Serves three jobs:
  1. POSITIVE CONTROL: proves the eval scene is solvable (a competent policy CAN pick+place),
     which none of our learned checkpoints established.
  2. CORRECTIVE LABELER: given ANY robot/object state (including the drifted, off-manifold
     states the learned policy visits in closed loop), returns the expert 16-dim action that
     makes progress toward the task. This is what open-loop sim co-training lacked.
  3. CLEAN DEMO SOURCE: records (sim observation -> expert action) with CORRECT object grounding
     (the arm actually reaches the object that is rendered), fixing the object-placement bug that
     undermined the v9 co-train.

Left arm only (the task arm); right arm held at rest; base frozen. Damped least-squares IK on
the left EE (follower_left_link_6) drives a waypoint state machine: pre-grasp -> descend -> close
-> lift -> over-bin -> release. Physics stepping (env.step with joint-position targets) so the
gripper actually grasps. Records LeRobot episodes when WAM_EXPERT_OUT is set.

Run in the Isaac image via eval/runner.sh with WAM_EVAL_ENTRY=scripted_expert_trossen.py and
WAM_SKIP_SERVER_PROBE=1 (NO policy server needed). Tunables (env):
  WAM_EXPERT_EPISODES   episodes to run (default 4)
  WAM_EXPERT_OUT        if set, write LeRobot episodes here (demo/DAgger data); else eval-only
  WAM_GRIP_OPEN/CLOSED  gripper carriage open/closed joint values (default 0.044 / 0.0)
  WAM_GRASP_Z / PREGRASP_Z / LIFT_Z   grasp/approach/lift heights rel. object (m)
  WAM_IK_GAIN / IK_DAMP  DLS IK step gain + damping
  WAM_RESET_JITTER      per-episode object-xy jitter (m) so demos vary (default 0.04)
  WAM_MAX_STEPS         max steps/episode (default 220)
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

N_EPISODES = int(os.environ.get("WAM_EXPERT_EPISODES", "4"))
OUT = os.environ.get("WAM_EXPERT_OUT", "")
MAX_STEPS = int(os.environ.get("WAM_MAX_STEPS", "400"))
GRIP_OPEN = float(os.environ.get("WAM_GRIP_OPEN", "0.05"))
GRIP_CLOSED = float(os.environ.get("WAM_GRIP_CLOSED", "0.0"))
PREGRASP_Z = float(os.environ.get("WAM_PREGRASP_Z", "0.12"))
GRASP_Z = float(os.environ.get("WAM_GRASP_Z", "0.0"))    # grasp AT the settled object center (secure)
LIFT_Z = float(os.environ.get("WAM_LIFT_Z", "0.14"))     # modest lift (reachable + keeps grasp)
SETTLE_STEPS = int(os.environ.get("WAM_SETTLE_STEPS", "45"))  # let the object fall+settle before targeting
IK_GAIN = float(os.environ.get("WAM_IK_GAIN", "0.4"))
IK_DAMP = float(os.environ.get("WAM_IK_DAMP", "0.05"))
USE_ORIENT = os.environ.get("WAM_IK_ORIENT", "0") == "1"  # position-only IK by default (orient term
# with a wrong DOWN_QUAT was contorting the arm + overshooting; bent rest pose is ~downward already)
RESET_JITTER = float(os.environ.get("WAM_RESET_JITTER", "0.04"))
POS_TOL = float(os.environ.get("WAM_POS_TOL", "0.03"))
CHUNK = 1000
CAMS = ("exterior_image_1_left", "wrist_image_left", "wrist_image_right")
FPS = 30
# gripper points straight down (w,x,y,z); matches controller.py DOWNWARD_ORIENTATION
DOWN_QUAT = np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float64)


def _q_err(q_cur: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    """Orientation error (world) as a rotation vector, from current->target quats (w,x,y,z)."""
    def norm(q):
        return q / (np.linalg.norm(q) + 1e-9)
    qc, qt = norm(q_cur), norm(q_tgt)
    # relative quat q_err = q_tgt * conj(q_cur)
    w0, x0, y0, z0 = qt
    w1, x1, y1, z1 = qc * np.array([1, -1, -1, -1])
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    v = np.array([x, y, z])
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros(3)
    ang = 2.0 * np.arctan2(n, w)
    if ang > np.pi:
        ang -= 2 * np.pi
    return (v / n) * ang


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
    dest_name = cfg["jobs"][0].get("arena_env_args", {}).get("destination_location", "blue_sorting_bin")
    print(f"[expert] pick={pick_name} dest={dest_name} out={OUT or '(eval-only)'} "
          f"grip open/closed={GRIP_OPEN}/{GRIP_CLOSED}", flush=True)

    # Weave/W&B results tracing (same convention as run_trossen_eval.py): a leaderboard entry
    # under model='scripted_ik_expert' so the positive-control results render in the workspace.
    from isaaclab_arena_dreamzero.weave_eval import WeaveEvalLogger
    wlog = WeaveEvalLogger(model_label=os.environ.get("WEAVE_MODEL", "scripted_ik_expert"),
                           task_name=job.name)
    WEAVE_ON = wlog.el is not None
    WV_DIR = os.environ.get("WAM_EVAL_VIDEO_DIR", "/data/wam/eval_videos/scripted_expert")

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
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        names = list(robot.joint_names)
        bnames = list(robot.body_names)
        obj = u.scene[pick_name] if pick_name in u.scene.keys() else None
        dest = u.scene[dest_name] if dest_name in u.scene.keys() else None
        adapter = DreamZeroTrossenAdapter()
        h, w = adapter.target_image_size

        # left-arm DOF indices (in the articulation's joint order) + EE body index
        larm_dof = [names.index(j) for j in LEFT_ARM_JOINTS]
        lgrip_dof = names.index(LEFT_GRIPPER_JOINT)
        rarm_dof = [names.index(j) for j in RIGHT_ARM_JOINTS]
        rgrip_dof = names.index(RIGHT_GRIPPER_JOINT)
        # Control the GRIPPER FINGERTIP (TCP), not the wrist link_6: the fingers extend ~0.12 m
        # past link_6, so IK-ing link_6 to the object jams the fingers into the table (first test:
        # link_6 stalled 0.164 m above). TCP = mean of the left-arm gripper finger bodies.
        grip_bodies = [b for b in bnames if b.startswith("follower_left") and "gripper" in b]
        if not grip_bodies:  # fallback to wrist
            grip_bodies = [next(b for b in bnames if b.startswith("follower_left") and b.endswith("link_6"))]
        tcp_bidx = [bnames.index(b) for b in grip_bodies]
        print(f"[expert] TCP bodies={grip_bodies} idx={tcp_bidx} larm_dof={larm_dof} lgrip_dof={lgrip_dof}", flush=True)
        _Jshape = _np(robot.root_physx_view.get_jacobians()).shape
        print(f"[expert] jacobian shape={_Jshape} nbodies={len(bnames)} ndof={len(names)} "
              f"(fixed-base -> expect rows={len(bnames)-1})", flush=True)
        try:
            _lim = _np(robot.data.joint_pos_limits)[0][lgrip_dof]
            print(f"[expert] left gripper carriage limits={np.round(_lim,4)} (open={GRIP_OPEN} closed={GRIP_CLOSED})", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[expert] gripper-limit probe skipped: {_e}", flush=True)

        n_dof = len(names)
        rest16 = np.zeros(16, dtype=np.float32)
        for k, j in enumerate(LEFT_ARM_JOINTS):
            rest16[k] = _REST_JOINT_POS[j]
        rest16[6] = GRIP_OPEN
        for k, j in enumerate(RIGHT_ARM_JOINTS):
            rest16[7 + k] = _REST_JOINT_POS[j]
        rest16[13] = _REST_JOINT_POS[RIGHT_GRIPPER_JOINT]

        def ee_pose():
            """TCP position = mean of the gripper finger bodies; orientation from the first."""
            bp = _np(robot.data.body_pos_w)[0]
            p = bp[tcp_bidx].mean(axis=0).astype(float)
            q = _np(robot.data.body_quat_w)[0][tcp_bidx[0]].astype(float)
            return p, q

        def left_jacobian():
            """6x6 position/orientation jacobian of the TCP w.r.t. the 6 left-arm DOFs
            (fixed base -> jacobian body row = body index - 1). Mean over finger bodies."""
            J = _np(robot.root_physx_view.get_jacobians())  # [N, nbodies-1, 6, ndof]
            Jb = np.mean([J[0, b - 1] for b in tcp_bidx], axis=0)  # [6, ndof]
            return Jb[:, larm_dof]  # [6, 6]

        def ik_step(cur_larm: np.ndarray, tgt_pos: np.ndarray, tgt_quat: np.ndarray) -> np.ndarray:
            p, q = ee_pose()
            perr = tgt_pos - p
            J = left_jacobian()                  # 6x6 (pos rows 0:3, orient rows 3:6)
            if USE_ORIENT:
                err = np.concatenate([perr, _q_err(q, tgt_quat)])  # 6
                Ju = J
            else:
                err = perr           # 3 — position-only: let orientation float
                Ju = J[0:3, :]       # 3x6
            JT = Ju.T
            dq = JT @ np.linalg.inv(Ju @ JT + (IK_DAMP ** 2) * np.eye(Ju.shape[0])) @ err
            dq = np.clip(dq, -0.2, 0.2)  # per-step joint-delta cap -> smooth, no overshoot
            return cur_larm + IK_GAIN * dq

        def build_action(larm_tgt: np.ndarray, grip: float) -> np.ndarray:
            a = rest16.copy()
            a[0:6] = larm_tgt
            a[6] = grip
            return a

        def step_action(a16: np.ndarray):
            act = torch.zeros((u.num_envs, 16), device=u.device, dtype=torch.float32)
            act[0] = torch.tensor(a16, device=u.device)
            env.step(act)

        def cur_larm():
            return _np(robot.data.joint_pos)[0][larm_dof].astype(float)

        def obj_xyz():
            return _np(obj.data.root_pos_w)[0].astype(float)

        def reset_scene(ep):
            env.reset()
            # Seat the object in a low, graspable pose + jitter xy so demos vary. A standing cylinder
            # (top ~1.14) exceeds the arm's marginal downward reach and blocks the top-down fingers;
            # laying it on its side keeps it low (~radius height) and reliably graspable. WAM_LAY_OBJECT=0
            # to keep the native upright pose.
            if obj is not None:
                rng = np.random.default_rng(4000 + ep)
                p = _as_torch(obj.data.root_pos_w).clone()
                if RESET_JITTER > 0:
                    p[0, 0] += float(rng.uniform(-RESET_JITTER, RESET_JITTER))
                    p[0, 1] += float(rng.uniform(-RESET_JITTER, RESET_JITTER))
                q = _as_torch(obj.data.root_quat_w).clone()
                if os.environ.get("WAM_LAY_OBJECT", "1") == "1":
                    q[0] = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=q.device, dtype=q.dtype)  # 90deg about y -> on its side
                obj.write_root_pose_to_sim(torch.cat([p, q], dim=-1))
                obj.write_data_to_sim()
            # let the object fall + settle on the table BEFORE we read its pose for waypoints,
            # else we target the spawn height (~0.05 m too high) and the fingers miss/knock it.
            for _ in range(SETTLE_STEPS):
                step_action(rest16)

        results = []
        for ep in range(N_EPISODES):
            reset_scene(ep)
            o0 = obj_xyz(); p0, _ = ee_pose()
            pick = obj_xyz()
            place = _np(dest.data.root_pos_w)[0].astype(float) if dest is not None else pick + np.array([0.25, -0.45, 0.0])
            # waypoint list: (target_xyz, grip, min_hold_steps, name)
            wps = [
                (pick + [0, 0, PREGRASP_Z], GRIP_OPEN, 6, "pregrasp"),
                (pick + [0, 0, GRASP_Z],    GRIP_OPEN, 8, "descend"),
                (pick + [0, 0, GRASP_Z],    GRIP_CLOSED, 22, "grasp"),   # longer hold -> secure grip
                (pick + [0, 0, LIFT_Z],     GRIP_CLOSED, 10, "lift"),
                (place + [0, 0, LIFT_Z],    GRIP_CLOSED, 12, "carry"),
                (place + [0, 0, PREGRASP_Z], GRIP_CLOSED, 8, "over_bin"),
                (place + [0, 0, PREGRASP_Z], GRIP_OPEN, 8, "release"),
            ]
            if ep == 0:
                tcp0, _ = ee_pose()
                print(f"[expert] ep0 GEOM: tcp_rest={np.round(tcp0,3)} object={np.round(pick,3)} "
                      f"bin={np.round(place,3)} tcp->obj={np.linalg.norm(tcp0-pick):.3f}m", flush=True)
            capt = {c: [] for c in CAMS}
            cap_ex = []  # exterior-cam frames for the Weave rollout video
            rec_state, rec_action = [], []
            t = 0
            min_dist = 1e9
            max_obj_z = -1e9
            for (tp, grip, hold, nm) in wps:
                tp = np.asarray(tp, dtype=np.float64)
                held = 0
                while t < MAX_STEPS:
                    larm = cur_larm()
                    tgt = ik_step(larm, tp, DOWN_QUAT)
                    a16 = build_action(tgt, grip)
                    # record BEFORE stepping: (obs at t, action taken) — matches training convention
                    if OUT or WEAVE_ON:
                        ex = adapter.extract(u.observation_manager.compute(), 0)
                        exf = image_tools.resize_with_pad(ex.exterior_image_1_left, h, w)
                        if WEAVE_ON:
                            cap_ex.append(np.asarray(exf, np.uint8))
                        if OUT:
                            capt[CAMS[0]].append(exf)
                            capt[CAMS[1]].append(image_tools.resize_with_pad(ex.wrist_image_left, h, w))
                            capt[CAMS[2]].append(image_tools.resize_with_pad(ex.wrist_image_right, h, w))
                            st = np.concatenate([larm, [_np(robot.data.joint_pos)[0][lgrip_dof]],
                                                 _np(robot.data.joint_pos)[0][rarm_dof],
                                                 [_np(robot.data.joint_pos)[0][rgrip_dof]], [0.0, 0.0]]).astype(np.float32)
                            rec_state.append(st)
                            rec_action.append(a16.astype(np.float32))
                    step_action(a16)
                    t += 1
                    held += 1
                    p, _ = ee_pose()
                    o_now = obj_xyz()
                    d = float(np.linalg.norm(p - o_now))
                    min_dist = min(min_dist, d)
                    max_obj_z = max(max_obj_z, float(o_now[2]))
                    reached = np.linalg.norm(p - tp) < POS_TOL
                    if held >= hold and (reached or held >= hold + 45):
                        break
                if ep == 0:
                    pf, _ = ee_pose()
                    print(f"[expert] ep0 wp={nm:9s} tgt={np.round(tp,3)} got={np.round(pf,3)} "
                          f"err={np.linalg.norm(pf-tp):.3f}m obj={np.round(obj_xyz(),3)} "
                          f"grip={grip:.3f} t={t}", flush=True)
            of = obj_xyz()
            lifted = float(of[2] - o0[2])
            max_lift = float(max_obj_z - o0[2])
            to_bin = float(np.linalg.norm(of[:2] - place[:2]))
            reached = bool(min_dist < 0.05)
            picked = bool(max_lift > 0.05)          # grasped + lifted at some point (primary milestone)
            placed = bool(lifted > 0.02 and to_bin < 0.15)
            success = picked
            results.append(dict(ep=ep, lifted=lifted, max_lift=max_lift, to_bin=to_bin, min_dist=min_dist,
                                reached=reached, picked=picked, placed=placed, steps=t))
            print(f"[expert] ep{ep}: reach={reached}(min {min_dist:.3f}m) PICK={picked}(max_lift {max_lift:+.3f}m) "
                  f"place={placed}(to_bin {to_bin:.3f}m, end_lift {lifted:+.3f}m) steps={t}", flush=True)

            if OUT and rec_state:
                _write_episode(OUT, ep, rec_state, rec_action, capt, instruction, imageio)

            # Log this episode to Weave (leaderboard prediction + rollout video).
            if WEAVE_ON:
                vid = None
                if cap_ex:
                    os.makedirs(WV_DIR, exist_ok=True)
                    vid = f"{WV_DIR}/{job.name}_ep{ep:03d}.mp4"
                    try:
                        imageio.mimsave(vid, cap_ex, fps=FPS, codec="libx264")
                    except Exception as e:  # noqa: BLE001
                        print(f"[expert] weave video write failed: {e}", flush=True); vid = None
                wlog.log_episode(
                    inputs={"episode_idx": ep, "task": job.name, "instruction": instruction},
                    output={"reached": reached, "picked": picked, "placed": placed,
                            "min_ee_obj_m": round(float(min_dist), 4), "max_lift_m": round(max_lift, 4),
                            "end_lift_m": round(lifted, 4), "to_bin_m": round(to_bin, 4), "steps": t},
                    scores={"reach": reached, "pick": picked, "place": placed,
                            "min_ee_obj_m": float(min_dist), "max_lift_m": float(max_lift)},
                    video_path=vid,
                )

        nr = sum(1 for r in results if r["reached"])
        npick = sum(1 for r in results if r["picked"])
        npl = sum(1 for r in results if r["placed"])
        print(f"[expert] ===== DONE: reach {nr}/{len(results)}, PICK {npick}/{len(results)}, place {npl}/{len(results)}; "
              f"mean min_ee_obj={np.mean([r['min_dist'] for r in results]):.3f}m =====", flush=True)
        if WEAVE_ON:
            n = max(1, len(results))
            wlog.log_summary({"episodes": len(results), "reach_rate": nr / n, "pick_rate": npick / n,
                              "place_rate": npl / n, "mean_min_ee_obj_m": float(np.mean([r["min_dist"] for r in results]))})
        wlog.finish()


def _write_episode(out, ep, states, actions, capt, instruction, imageio):
    """Write one LeRobot episode (sim videos + expert state/action + language)."""
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
