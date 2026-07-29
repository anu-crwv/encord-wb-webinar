#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrapping test: does a MOVING context restore action DIRECTION in sim?

Tier-0/2×2/probe results established:
  * single-frame inference has healthy MAGNITUDE but weak DIRECTION (corr ~0.22);
  * teacher-forced AR on a real reaching episode is well-directed (corr ~0.86);
  * closed-loop-from-rest never ramps into the reach -> directionless jitter (hover).

So the question is no longer magnitude or appearance; it is whether the model can be
DIRECTED in sim when its context already contains motion. This runs a teacher-forced
autoregressive pass over a dense (small-stride) sequence of poses rendered in sim
(sim_frames) and, as a control, the corresponding real frames, and measures the
per-arm-joint amplitude ratio + DIRECTION correlation vs the dataset GT action chunk.

  obs source ∈ {sim, real} × latent ∈ {observed (None -> encode obs), dreamed (self)}

Read:
  * sim-observed corr ~= real-observed corr (~0.8): a MOVING sim context restores
    direction -> the closed-loop hover is a COLD-START/bootstrapping failure (fixable
    by seeding motion / a receding-observed-history inference regime), NOT a model limit.
  * sim-observed corr << real-observed corr: sim frames degrade the direction-relevant
    representation even WITH motion -> closer scene match / adaptation needed.
  * observed ~= dreamed (expected from v6 normal≈gtcond): dream path is not the problem.

Consumes /data/wam/arladder/{sim,real}_frames.npy, real_states.npy, gt_chunks.npy, meta.json
(Step-0-dense + diag_render_at_poses). READ-ONLY. Run from repo root:
    MODEL_DIR=/checkpoints/wam/eval/dreamzero-trossen-lora-v6 \
    torchrun --standalone --nproc_per_node=1 eval/diag_ar_ladder.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("ATTENTION_BACKEND", "TE")
os.environ.setdefault("ENABLE_DIT_CACHE", "false")

import numpy as np
import torch
import torch.distributed as dist

from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

from diag_gt_cond import CAMS, DIM_NAMES, FRAMES_PER_CHUNK, build_obs, extract_action, init_mesh, reset_ar_state

MODEL_DIR = os.environ.get("MODEL_DIR", "/checkpoints/wam/eval/dreamzero-trossen-lora-v6")
D = os.environ.get("WAM_2X2_DIR", "/data/wam/arladder")
INSTR = os.environ.get("WAM_2X2_INSTR",
                       "pick up the two blue ethernet cables from the rack and plug each into a port on the network switch")
ARM_IDX = [i for i, nm in enumerate(DIM_NAMES) if nm.startswith(("Lj", "Rj"))]


def _weighted(pred: np.ndarray, gt: np.ndarray) -> dict:
    """GT-std-weighted amplitude ratio + direction correlation over arm joints."""
    amps, cors, ws = [], [], []
    for j in ARM_IDX:
        gstd, pstd = float(gt[:, j].std()), float(pred[:, j].std())
        if gstd > 0.05:
            ws.append(gstd)
            amps.append(pstd / gstd)
            cors.append(float(np.corrcoef(gt[:, j], pred[:, j])[0, 1]) if pstd > 1e-6 else 0.0)
    if not ws:
        return {"wamp": float("nan"), "wcorr": float("nan"), "mag": float("nan")}
    w = np.asarray(ws)
    mag = float(np.mean([np.mean(np.abs(pred[:, j])) for j in ARM_IDX]))
    return {"wamp": float(np.average(amps, weights=w)),
            "wcorr": float(np.average(cors, weights=w)), "mag": mag}


def _ar_pass(policy, ah, frames_k, states_k, gt_chunks, latent_mode, nfpb, horizon):
    """Teacher-forced AR over the pre-rendered frame sequence. Returns (pred, gt) rows
    from chunk 1 onward (chunk 0 is mode-identical / fresh)."""
    reset_ar_state(ah)
    K = frames_k.shape[0]
    buffers = {cam: [] for cam in CAMS}
    prev_pred = first_pred = None
    pred_rows, gt_rows = [], []
    for k in range(K):
        for i, cam in enumerate(CAMS):
            buffers[cam].append(frames_k[k][i])
        num_frames = 1 if k == 0 else FRAMES_PER_CHUNK
        obs = build_obs(buffers, num_frames, states_k[k], INSTR)
        if k == 0:
            latent = None
        elif latent_mode == "observed":
            latent = None                                   # model VAE-encodes the observed frames
        elif latent_mode == "dreamed":
            latent = prev_pred[:, :, -nfpb:] if prev_pred is not None else None
        else:  # repeated_initial
            latent = first_pred[:, :, -nfpb:] if first_pred is not None else None
        dist.barrier()
        with torch.no_grad():
            result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs), latent_video=latent)
        dist.barrier()
        prev_pred = video_pred
        if first_pred is None:
            first_pred = video_pred
        chunk = extract_action(result)
        if k >= 1:
            n = min(horizon, chunk.shape[0], gt_chunks.shape[1])
            pred_rows.append(chunk[:n])
            gt_rows.append(gt_chunks[k][:n])
    if not pred_rows:
        return np.zeros((0, 16), np.float32), np.zeros((0, 16), np.float32)
    return np.concatenate(pred_rows, 0), np.concatenate(gt_rows, 0)


def main() -> None:
    torch._dynamo.config.recompile_limit = 800
    mesh = init_mesh()
    rank = dist.get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sim_frames = np.load(f"{D}/sim_frames.npy")     # (K,3,H,W,3)
    real_frames = np.load(f"{D}/real_frames.npy")   # (K,3,H,W,3)
    real_states = np.load(f"{D}/real_states.npy")   # (K,16)
    gt_chunks = np.load(f"{D}/gt_chunks.npy")        # (K,24,16)
    srcs = {"sim": sim_frames, "real": real_frames}

    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag("trossen"), model_path=MODEL_DIR,
                            device=device, device_mesh=mesh)
    ah = policy.trained_model.action_head
    nfpb = int(ah.num_frame_per_block)
    horizon = int(ah.action_horizon)
    if rank == 0:
        print(f"[ladder] model={MODEL_DIR} K={sim_frames.shape[0]} instr='{INSTR[:50]}...'", flush=True)

    results = {}
    for src in ("real", "sim"):
        for mode in ("observed", "dreamed"):
            pred, gt = _ar_pass(policy, ah, srcs[src], real_states, gt_chunks, mode, nfpb, horizon)
            if rank != 0:
                continue
            m = _weighted(pred, gt) if len(pred) else {"wamp": float("nan"), "wcorr": float("nan"), "mag": float("nan")}
            results[(src, mode)] = m
            print(f"[ladder] obs={src:4s} latent={mode:8s}  wcorr={m['wcorr']:+.3f} "
                  f"wamp={m['wamp']:.3f} mag={m['mag']:.4f}  (rows={len(pred)})", flush=True)

    if rank != 0:
        return

    print("\n[ladder] ===== DIRECTION (weighted corr vs GT reach) =====", flush=True)
    print(f"  {'':10s} {'observed':>10s} {'dreamed':>10s}", flush=True)
    for src in ("real", "sim"):
        ro = results[(src, "observed")]["wcorr"]
        rd = results[(src, "dreamed")]["wcorr"]
        print(f"  obs={src:4s} {ro:10.3f} {rd:10.3f}", flush=True)

    real_o = results[("real", "observed")]["wcorr"]
    sim_o = results[("sim", "observed")]["wcorr"]
    keep = sim_o / real_o if real_o and abs(real_o) > 1e-6 else float("nan")
    print(f"\n[ladder] real-coffee baseline (prior run): wcorr~0.86", flush=True)
    print(f"[ladder] this episode: real-observed wcorr={real_o:.3f}  sim-observed wcorr={sim_o:.3f}  "
          f"(sim keeps {keep:.0%} of real direction)", flush=True)
    if keep == keep and keep > 0.7 and sim_o > 0.5:
        verdict = ("MOVING sim context RESTORES direction -> the closed-loop hover is a COLD-START/"
                   "bootstrapping failure (the model needs motion in context; from rest it never gets it). "
                   "Fix candidates: seed initial motion, receding observed-history inference, or GT/oracle "
                   "video kickstart -- NOT necessarily a retrain.")
    elif sim_o < 0.4:
        verdict = ("sim degrades DIRECTION even WITH motion (sim-observed wcorr low) -> the direction-relevant "
                   "representation is corrupted by sim frames in the multi-frame path. Fix: closer scene match / "
                   "latent-path adaptation.")
    else:
        verdict = (f"PARTIAL: sim keeps {keep:.0%} of real direction; both bootstrapping and sim-latent "
                   f"factors contribute.")
    print(f"\n[ladder] VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    main()
