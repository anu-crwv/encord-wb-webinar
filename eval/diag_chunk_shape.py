#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""GPU diagnostic: WHERE inside the predicted 24-step action chunk does the reach live?

Deployment (DreamZeroRemotePolicy) executes only the first ``open_loop_horizon``
(=5) rows of every predicted chunk, then re-plans. The server, however, returns
the full 24-step chunk (verified live). So if the model loads its reach into the
*later* chunk indices (6..23), deployment discards exactly the motion that matters
and the arm hovers -- regardless of model quality, data, or visual OOD.

This is the cheap, no-sim test of that hypothesis. We run the model in the SAME
autoregressive path as the server (feed its own dreamed video latents back) over a
real held-out episode, and for every prediction record the full (H,16) chunk and
the current state. We then measure, over the ARM joints (Lj0..5, Rj0..5):

  disp_pred[k] = mean_j | pred[k, j] - state_now[j] |     (absolute-target model)
  disp_gt[k]   = mean_j | gt[t+k, j]  - state_now[j] |

Read:
  * disp_pred ramps ~linearly and disp_pred[4]/disp_pred[H-1] ~= 5/24 ~ 0.21
        => chunk is evenly loaded; executing 5/24 is benign; the hover is NOT
           truncation -> look at closed-loop covariate shift / state OOD (Tier 1+).
  * disp_pred[0..4] ~= 0 then ramps only after k>=5 (back-loaded)
        => the reach is in the DISCARDED steps; raising open_loop_horizon should
           unlock motion (confirm in the closed-loop sweep, Tier 0.1b).
  * disp_pred[H-1] << disp_gt[H-1] everywhere (flat, tiny)
        => head genuinely under-commits even over the full chunk (data/head problem).

Also compares the SHAPE to GT (does the real trajectory move immediately or ramp?),
so a front-loaded GT vs back-loaded prediction (a phase/timing error) is visible.

READ-ONLY: loads the policy + dataset, runs inference, prints tables, optional Weave.

Run (single GPU), from the repo root:
    MODEL_DIR=/checkpoints/wam/eval/dreamzero-trossen-lora-v6 \
    WAM_TROSSEN_DATA=/data/wam/datasets/encord_trossen_v6 \
    DIAG_TASK=coffee \
    torchrun --standalone --nproc_per_node=1 eval/diag_chunk_shape.py
"""

from __future__ import annotations

import os

os.environ.setdefault("ATTENTION_BACKEND", "TE")
os.environ.setdefault("ENABLE_DIT_CACHE", "false")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

# Reuse the proven helpers from the amplitude diag so the forward path is byte-identical.
from diag_gt_cond import (
    CAMS,
    DIM_NAMES,
    FRAMES_PER_CHUNK,
    build_obs,
    extract_action,
    episode_for_task,
    init_mesh,
    load_frames,
    reset_ar_state,
)

MODEL_DIR = os.environ.get("MODEL_DIR", "/checkpoints/wam/eval/dreamzero-trossen-lora-v6")
DATA_ROOT = os.environ.get("WAM_TROSSEN_DATA", "/data/wam/datasets/encord_trossen_v6")
TASK_SUBSTR = os.environ.get("DIAG_TASK", "coffee")
MAX_CHUNKS = int(os.environ.get("DIAG_MAX_CHUNKS", "24"))
# The deployed adapter executes this many of the returned rows before re-planning.
EXEC_HORIZON = int(os.environ.get("WAM_OPEN_LOOP_HORIZON", "5"))

# Arm reach joints only (exclude grippers idx 6/13 and base velocities 14/15).
ARM_IDX = [i for i, nm in enumerate(DIM_NAMES) if nm.startswith(("Lj", "Rj"))]


def main() -> None:
    torch._dynamo.config.recompile_limit = 800
    mesh = init_mesh()
    rank = dist.get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _trace_run = None
    if rank == 0 and os.environ.get("WAM_EVAL_NO_WEAVE") != "1":
        try:
            import sys as _sys
            _eval_dir = os.path.dirname(os.path.abspath(__file__))
            if _eval_dir not in _sys.path:
                _sys.path.insert(0, _eval_dir)
            from isaaclab_arena_dreamzero.weave_eval import init_eval_tracing, finish_eval_tracing
            _trace_run = init_eval_tracing()
            globals()["_finish_eval_tracing"] = finish_eval_tracing
        except Exception as _e:  # noqa: BLE001
            print(f"[shape] weave tracing unavailable ({_e}); continuing", flush=True)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("trossen"),
        model_path=MODEL_DIR,
        device=device,
        device_mesh=mesh,
    )
    ah = policy.trained_model.action_head
    num_frame_per_block = int(ah.num_frame_per_block)
    action_horizon = int(ah.action_horizon)

    ep, task = episode_for_task(TASK_SUBSTR)
    df = pd.read_parquet(f"{DATA_ROOT}/data/chunk-000/episode_{ep:06d}.parquet")
    gt_action = np.stack(df["action"].to_numpy()).astype(np.float32)
    gt_state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    frames = load_frames(ep)
    T_total = min(len(gt_action), len(frames[CAMS[0]]))
    step_starts = list(range(0, T_total - 1, action_horizon))[:MAX_CHUNKS]

    if rank == 0:
        print(f"[shape] model={MODEL_DIR}", flush=True)
        print(f"[shape] episode {ep} task='{task}' T={T_total} action_horizon={action_horizon} "
              f"exec_horizon(deployed)={EXEC_HORIZON} chunks={len(step_starts)}", flush=True)

    # Autoregressive pass mirroring the server (feed own dreamed latents back).
    reset_ar_state(ah)
    frame_buffers = {cam: [] for cam in CAMS}
    prev_video_pred = None
    # Accumulators for the per-index displacement curves (H bins).
    H = action_horizon
    dp = np.zeros(H, dtype=np.float64)   # sum disp_pred[k]
    dg = np.zeros(H, dtype=np.float64)   # sum disp_gt[k]
    dpd = np.zeros(H, dtype=np.float64)  # sum step-delta |pred[k]-pred[k-1]|
    cnt = np.zeros(H, dtype=np.int64)
    cntg = np.zeros(H, dtype=np.int64)

    for s, t in enumerate(step_starts):
        for cam in CAMS:
            frame_buffers[cam].append(frames[cam][t])
        num_frames = 1 if s == 0 else FRAMES_PER_CHUNK
        obs = build_obs(frame_buffers, num_frames, gt_state[t], task)
        if s == 0:
            latent_video = None
        else:
            latent_video = (prev_video_pred[:, :, -num_frame_per_block:]
                            if prev_video_pred is not None else None)
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = policy.lazy_joint_forward_causal(
                Batch(obs=obs), latent_video=latent_video)
        dist.barrier()
        prev_video_pred = video_pred
        act = extract_action(result_batch)  # (H, 16) absolute joint targets
        state_now = gt_state[t]

        Hh = min(H, act.shape[0])
        for k in range(Hh):
            dp[k] += float(np.mean(np.abs(act[k, ARM_IDX] - state_now[ARM_IDX])))
            cnt[k] += 1
            if k == 0:
                dpd[k] += float(np.mean(np.abs(act[k, ARM_IDX] - state_now[ARM_IDX])))
            else:
                dpd[k] += float(np.mean(np.abs(act[k, ARM_IDX] - act[k - 1, ARM_IDX])))
            tk = t + k
            if tk < T_total:
                dg[k] += float(np.mean(np.abs(gt_action[tk, ARM_IDX] - state_now[ARM_IDX])))
                cntg[k] += 1
        if rank == 0:
            print(f"[shape] chunk {s:02d} t={t:04d} pred={act.shape} "
                  f"disp0={dp[0]/max(cnt[0],1):.3f} "
                  f"disp{EXEC_HORIZON-1}={dp[EXEC_HORIZON-1]/max(cnt[EXEC_HORIZON-1],1):.3f} "
                  f"disp{H-1}={dp[H-1]/max(cnt[H-1],1):.3f}", flush=True)

    if rank != 0:
        return

    disp_pred = dp / np.maximum(cnt, 1)
    disp_gt = dg / np.maximum(cntg, 1)
    step_delta = dpd / np.maximum(cnt, 1)

    print(f"\n[shape] per-index ARM-joint displacement from current state "
          f"(mean over {int(cnt[0])} chunks)\n", flush=True)
    print(f"  {'k':>3s} {'disp_pred':>10s} {'disp_gt':>10s} {'pred/gt':>8s} "
          f"{'step_delta':>11s} {'exec?':>6s}", flush=True)
    for k in range(H):
        r = disp_pred[k] / disp_gt[k] if disp_gt[k] > 1e-6 else float("nan")
        mark = "EXEC" if k < EXEC_HORIZON else "drop"
        print(f"  {k:3d} {disp_pred[k]:10.4f} {disp_gt[k]:10.4f} {r:8.2f} "
              f"{step_delta[k]:11.4f} {mark:>6s}", flush=True)

    # Summary ratios: how much of the full-chunk reach is reached by the executed horizon?
    end_p = disp_pred[H - 1] if disp_pred[H - 1] > 1e-6 else float("nan")
    end_g = disp_gt[H - 1] if disp_gt[H - 1] > 1e-6 else float("nan")
    frac_exec_p = disp_pred[EXEC_HORIZON - 1] / end_p if end_p == end_p else float("nan")
    frac_exec_g = disp_gt[EXEC_HORIZON - 1] / end_g if end_g == end_g else float("nan")
    linear_frac = EXEC_HORIZON / H
    commit = end_p / end_g if end_g == end_g and end_g > 1e-6 else float("nan")

    print(f"\n[shape] SUMMARY (arm joints, absolute targets):", flush=True)
    print(f"  full-chunk reach   pred={end_p:.4f} rad   gt={end_g:.4f} rad   "
          f"commit(pred/gt)={commit:.2f}", flush=True)
    print(f"  progress by executed step {EXEC_HORIZON-1}:  pred={frac_exec_p:.2f}   "
          f"gt={frac_exec_g:.2f}   (linear baseline={linear_frac:.2f})", flush=True)

    # Verdict heuristic.
    verdict = ""
    if commit == commit and commit < 0.4:
        verdict = ("HEAD UNDER-COMMITS over the FULL chunk (pred reach << gt reach) "
                   "-> truncation is NOT the main cause; head/data problem.")
    elif frac_exec_p == frac_exec_p and frac_exec_p < 0.5 * linear_frac:
        verdict = ("BACK-LOADED: <half the linear share of motion is in the executed "
                   f"first {EXEC_HORIZON} steps -> discarded steps 5..{H-1} hold the reach; "
                   "raising open_loop_horizon should unlock motion (confirm in sim).")
    else:
        verdict = ("EVENLY-LOADED: executing 5/24 is benign (pred reaches its linear "
                   "share early) -> the hover is NOT chunk truncation; look at "
                   "closed-loop covariate shift / state OOD next (Tier 1+).")
    print(f"\n[shape] VERDICT: {verdict}", flush=True)

    if _trace_run is not None:
        try:
            import weave
            import wandb
            label = os.environ.get("WEAVE_MODEL_LABEL") or MODEL_DIR.rstrip("/").rsplit("/", 1)[-1]

            def chunk_shape_eval():
                return {
                    "task": task, "model": label, "episode": int(ep),
                    "exec_horizon": EXEC_HORIZON, "action_horizon": H,
                    "reach_pred_rad": round(float(end_p), 4), "reach_gt_rad": round(float(end_g), 4),
                    "commit_ratio": round(float(commit), 4),
                    "frac_reach_by_exec_pred": round(float(frac_exec_p), 4),
                    "frac_reach_by_exec_gt": round(float(frac_exec_g), 4),
                    "linear_baseline": round(float(linear_frac), 4),
                    "disp_pred": [round(float(x), 4) for x in disp_pred],
                    "disp_gt": [round(float(x), 4) for x in disp_gt],
                    "verdict": verdict,
                }
            metrics = weave.op(chunk_shape_eval)()
            wandb.run.summary.update({
                "shape/commit_ratio": float(commit),
                "shape/frac_reach_by_exec_pred": float(frac_exec_p),
                "shape/frac_reach_by_exec_gt": float(frac_exec_g),
                "shape/reach_pred_rad": float(end_p),
                "shape/reach_gt_rad": float(end_g),
            })
            print("[shape] weave trace + metrics logged", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[shape] weave logging failed: {_e}", flush=True)
        finally:
            globals().get("_finish_eval_tracing", lambda r: None)(_trace_run)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    main()
