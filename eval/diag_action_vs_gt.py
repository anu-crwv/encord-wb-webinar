#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose the video-action decoupling: does the served model's PREDICTED action trajectory
track the GROUND-TRUTH joint trajectory, or does it under-commit (stay near rest)?

For a real coffee episode from the v4 dataset, feed the model the REAL frames + REAL state
(teacher-forced state, so we isolate action-prediction quality from compounding rollout error)
and compare the predicted action chunk to the GT actions, PER DIMENSION. If the reach joints'
predicted range/std is << GT (or uncorrelated), the action head under-commits / is misaligned —
even though the dream (video) is correct. Runs in the offline CPU env (openpi_client)."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from openpi_client import image_tools, websocket_client_policy

DATA_ROOT = os.environ.get("WAM_TROSSEN_DATA", "/data/wam/datasets/encord_trossen_v4")
H, W = 480, 640
CAMS = ("exterior_image_1_left", "wrist_image_left", "wrist_image_right")
HORIZON = int(os.environ.get("DIAG_HORIZON", "5"))     # open-loop stride (eval uses 5)
STRIDE = int(os.environ.get("DIAG_STRIDE", "10"))       # query every STRIDE frames
# 16-dim: [Lj0..5, Lgrip, Rj0..5, Rgrip, linv, angv]
DIM_NAMES = ["Lj0", "Lj1", "Lj2", "Lj3", "Lj4", "Lj5", "Lgrip",
             "Rj0", "Rj1", "Rj2", "Rj3", "Rj4", "Rj5", "Rgrip", "linv", "angv"]


def _episode_for_task(task_substr: str):
    """Find the first episode whose task string matches (via meta/tasks.jsonl + episodes.jsonl)."""
    import json
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{DATA_ROOT}/meta/tasks.jsonl") if l.strip()}
    want = [i for i, t in tasks.items() if task_substr.lower() in t.lower()]
    for pq in sorted(glob.glob(f"{DATA_ROOT}/data/chunk-000/episode_*.parquet")):
        df = pd.read_parquet(pq, columns=["task_index"])
        if int(df["task_index"].iloc[0]) in want:
            ep = int(pq.split("episode_")[-1].split(".")[0])
            return ep, tasks[int(df["task_index"].iloc[0])]
    ep = 0
    return ep, tasks.get(0, "?")


def _frames(ep: int):
    import imageio.v2 as imageio
    out = {}
    for cam in CAMS:
        f = f"{DATA_ROOT}/videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
        rd = imageio.get_reader(f, "ffmpeg")
        out[cam] = [np.asarray(fr)[:, :, :3].astype(np.uint8) for fr in rd]
    return out


def main() -> None:
    host = os.environ.get("DZ_HOST", "dreamzero-trossen-inference")
    port = int(os.environ.get("DZ_PORT", "8001"))
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    ep, task = _episode_for_task(os.environ.get("DIAG_TASK", "coffee"))
    print(f"[diag] server {host}:{port} | episode {ep} | task='{task}'", flush=True)

    df = pd.read_parquet(f"{DATA_ROOT}/data/chunk-000/episode_{ep:06d}.parquet")
    gt_action = np.stack(df["action"].to_numpy())            # (T,16) GT commanded targets
    gt_state = np.stack(df["observation.state"].to_numpy())  # (T,16)
    frames = _frames(ep)
    T = min(len(gt_action), len(frames[CAMS[0]]))
    print(f"[diag] T={T} frames; querying every {STRIDE}, chunk horizon {HORIZON}", flush=True)

    sid = f"diag-{os.getpid()}"
    pred_rows, gt_rows = [], []
    for t in range(0, T - HORIZON, STRIDE):
        req = {"prompt": task, "endpoint": "infer", "session_id": sid}
        req["observation/state"] = gt_state[t].astype(np.float64)  # teacher-forced real state
        for cam in CAMS:
            req[f"observation/{cam}"] = image_tools.resize_with_pad(frames[cam][t], H, W)
        resp = client.infer(req)
        act = np.asarray(resp.get("action", resp.get("actions")), dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(-1, 16)
        n = min(HORIZON, act.shape[0], T - t)
        pred_rows.append(act[:n])
        gt_rows.append(gt_action[t:t + n])
    pred = np.concatenate(pred_rows, 0)   # (N,16)
    gt = np.concatenate(gt_rows, 0)       # (N,16)

    print(f"\n[diag] compared {pred.shape[0]} predicted vs GT action rows\n", flush=True)
    print(f"{'dim':6s} {'GT_mean':>9s} {'GT_std':>8s} {'GT_range':>9s} | "
          f"{'PRED_mean':>10s} {'PRED_std':>9s} {'PRED_range':>10s} | {'MSE':>7s} {'std_ratio':>9s} {'corr':>6s}", flush=True)
    for d in range(16):
        g, p = gt[:, d], pred[:, d]
        gstd, pstd = g.std(), p.std()
        grng, prng = g.max() - g.min(), p.max() - p.min()
        mse = float(np.mean((g - p) ** 2))
        ratio = (pstd / gstd) if gstd > 1e-6 else float("nan")   # <1 => model moves this joint LESS than GT
        corr = float(np.corrcoef(g, p)[0, 1]) if gstd > 1e-6 and pstd > 1e-6 else float("nan")
        flag = "  <-- UNDER-COMMIT" if (gstd > 0.1 and ratio < 0.4) else ""
        print(f"{DIM_NAMES[d]:6s} {g.mean():+9.3f} {gstd:8.3f} {grng:9.3f} | "
              f"{p.mean():+10.3f} {pstd:9.3f} {prng:10.3f} | {mse:7.3f} {ratio:9.2f} {corr:+6.2f}{flag}", flush=True)
    print("\n[diag] READ: for joints GT actually moves (GT_std>0.1), std_ratio<<1 or corr~0 => the model's "
          "action head under-commits/misaligns vs GT (video-action decoupling). std_ratio~1 & corr~1 => actions track GT.", flush=True)


if __name__ == "__main__":
    main()
