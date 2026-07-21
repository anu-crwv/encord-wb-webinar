#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 Phase-3 (2x2 modality swap) — STAGE 2: run the 4 cells through the model.

Loads the pose-anchored arrays from Stage 1 (real_frames, sim_frames, real_states,
sim_states, gt_chunks) and, for each of K poses, runs ONE first-chunk forward
(current_start_frame=0, no AR history) for each cell:

    image      state     cell
    real       real      RR   (expected healthy — the training distribution)
    real       sim       RS   (state-source only)
    sim        real      SR   (visual-source only)  <- isolates pure visual shift
    sim        sim       SS   (the closed-loop condition — expected collapsed)

Per cell we report, over the ARM joints:
  * mean_disp      : mean |pred[k] - state_fed| over the chunk  (motion the model commits)
  * first_disp     : |pred[0] - state_fed|                      (first-step magnitude)
  * corr           : per-joint corr(pred, gt_chunk) averaged    (direction vs the real reach)
  * amp_ratio      : std(pred)/std(gt) averaged                 (amplitude vs real)

Interpretation (from the identical checkpoint/instruction/preprocessing):
  * RR healthy, SR collapsed, RS ~healthy   -> visual shift is the cause (high confidence).
  * both SR and RS collapse                 -> image-state consistency / state semantics.
  * SR only partial collapse                -> both visual and state factors contribute.
  * SR healthy with real state              -> revise the visual-OOD diagnosis.

READ-ONLY. Run (single GPU) from the repo root so ``groot``/``diag_gt_cond`` import:
    MODEL_DIR=/checkpoints/wam/eval/dreamzero-trossen-lora-v6 \
    torchrun --standalone --nproc_per_node=1 eval/diag_2x2_modality.py
"""

from __future__ import annotations

import os

os.environ.setdefault("ATTENTION_BACKEND", "TE")
os.environ.setdefault("ENABLE_DIT_CACHE", "false")

import numpy as np
import torch
import torch.distributed as dist

from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

from diag_gt_cond import CAMS, DIM_NAMES, extract_action, init_mesh, reset_ar_state

MODEL_DIR = os.environ.get("MODEL_DIR", "/checkpoints/wam/eval/dreamzero-trossen-lora-v6")
D = os.environ.get("WAM_2X2_DIR", "/data/wam/diag2x2")
INSTR = os.environ.get("WAM_2X2_INSTR", "pick up the object and place it in the tray")
ARM_IDX = [i for i, nm in enumerate(DIM_NAMES) if nm.startswith(("Lj", "Rj"))]

CELLS = [("RR", "real", "real"), ("RS", "real", "sim"), ("SR", "sim", "real"), ("SS", "sim", "sim")]


def _build_obs(frame3: np.ndarray, state16: np.ndarray) -> dict:
    """Single-frame (T=1 -> first chunk) obs in the server's raw input format."""
    obs = {f"video.{cam}": frame3[i][None].astype(np.uint8) for i, cam in enumerate(CAMS)}
    obs["state.state"] = np.asarray(state16, dtype=np.float64).reshape(1, -1)
    obs["annotation.language.action_text"] = INSTR
    return obs


def _metrics(chunk: np.ndarray, state_fed: np.ndarray, gt_chunk: np.ndarray) -> dict:
    n = min(chunk.shape[0], gt_chunk.shape[0])
    ch, gt = chunk[:n], gt_chunk[:n]
    disp = float(np.mean([np.mean(np.abs(ch[:, j] - state_fed[j])) for j in ARM_IDX]))
    first = float(np.mean([abs(ch[0, j] - state_fed[j]) for j in ARM_IDX]))
    corrs, ratios = [], []
    for j in ARM_IDX:
        gstd, pstd = float(gt[:, j].std()), float(ch[:, j].std())
        if gstd > 0.05:  # only joints the real reach actually moves
            ratios.append(pstd / gstd)
            if pstd > 1e-6:
                corrs.append(float(np.corrcoef(gt[:, j], ch[:, j])[0, 1]))
    return {
        "disp": disp, "first": first,
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
        "amp_ratio": float(np.mean(ratios)) if ratios else float("nan"),
    }


def main() -> None:
    torch._dynamo.config.recompile_limit = 800
    mesh = init_mesh()
    rank = dist.get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _trace_run = None
    if rank == 0 and os.environ.get("WAM_EVAL_NO_WEAVE") != "1":
        try:
            import sys as _sys
            _ed = os.path.dirname(os.path.abspath(__file__))
            if _ed not in _sys.path:
                _sys.path.insert(0, _ed)
            from isaaclab_arena_dreamzero.weave_eval import init_eval_tracing, finish_eval_tracing
            _trace_run = init_eval_tracing()
            globals()["_finish_eval_tracing"] = finish_eval_tracing
        except Exception as _e:  # noqa: BLE001
            print(f"[2x2] weave unavailable ({_e})", flush=True)

    real_frames = np.load(f"{D}/real_frames.npy")   # (K,3,480,640,3)
    sim_frames = np.load(f"{D}/sim_frames.npy")      # (K,3,480,640,3)
    real_states = np.load(f"{D}/real_states.npy")    # (K,16)
    sim_states = np.load(f"{D}/sim_states.npy")      # (K,16)
    gt_chunks = np.load(f"{D}/gt_chunks.npy")        # (K,24,16)
    K = real_frames.shape[0]
    frames = {"real": real_frames, "sim": sim_frames}
    states = {"real": real_states, "sim": sim_states}

    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag("trossen"), model_path=MODEL_DIR,
                            device=device, device_mesh=mesh)
    ah = policy.trained_model.action_head

    if rank == 0:
        print(f"[2x2] model={MODEL_DIR} K={K} instr='{INSTR}'", flush=True)

    per_cell = {}
    for name, isrc, ssrc in CELLS:
        agg = {"disp": [], "first": [], "corr": [], "amp_ratio": []}
        for k in range(K):
            reset_ar_state(ah)
            obs = _build_obs(frames[isrc][k], states[ssrc][k])
            dist.barrier()
            with torch.no_grad():
                result, _vp = policy.lazy_joint_forward_causal(Batch(obs=obs), latent_video=None)
            dist.barrier()
            chunk = extract_action(result)  # (24,16)
            m = _metrics(chunk, states[ssrc][k], gt_chunks[k])
            for key in agg:
                agg[key].append(m[key])
        per_cell[name] = {key: float(np.nanmean(v)) for key, v in agg.items()}
        if rank == 0:
            c = per_cell[name]
            print(f"[2x2] {name} (img={isrc:4s} state={ssrc:4s})  "
                  f"disp={c['disp']:.4f} first={c['first']:.4f} "
                  f"corr={c['corr']:+.3f} amp_ratio={c['amp_ratio']:.3f}", flush=True)

    if rank != 0:
        return

    print("\n[2x2] ================= MODALITY-SWAP TABLE =================", flush=True)
    print("  metric = mean arm-joint displacement committed (rad); higher = more motion", flush=True)
    print(f"  {'':10s} {'state=real':>12s} {'state=sim':>12s}", flush=True)
    for isrc in ("real", "sim"):
        rr = per_cell["RR" if isrc == "real" else "SR"]["disp"]
        rs = per_cell["RS" if isrc == "real" else "SS"]["disp"]
        print(f"  img={isrc:6s} {rr:12.4f} {rs:12.4f}", flush=True)
    print("\n  direction corr vs GT reach:", flush=True)
    for isrc in ("real", "sim"):
        rr = per_cell["RR" if isrc == "real" else "SR"]["corr"]
        rs = per_cell["RS" if isrc == "real" else "SS"]["corr"]
        print(f"  img={isrc:6s} {rr:12.3f} {rs:12.3f}", flush=True)

    # Verdict from disp ratios (SR/RR isolates visual; RS/RR isolates state).
    rr, rs = per_cell["RR"]["disp"], per_cell["RS"]["disp"]
    sr, ss = per_cell["SR"]["disp"], per_cell["SS"]["disp"]
    vis_keep = sr / rr if rr > 1e-6 else float("nan")   # sim image, real state: fraction of RR motion kept
    st_keep = rs / rr if rr > 1e-6 else float("nan")    # real image, sim state
    if vis_keep == vis_keep and vis_keep < 0.5 and st_keep > 0.7:
        verdict = (f"VISUAL shift is the cause: sim image collapses motion to {vis_keep:.0%} of RR "
                   f"even with REAL state, while sim state keeps {st_keep:.0%}.")
    elif vis_keep < 0.5 and st_keep < 0.7:
        verdict = (f"IMAGE-STATE / state semantics also matter: sim image keeps {vis_keep:.0%}, "
                   f"sim state keeps {st_keep:.0%} of RR motion.")
    elif vis_keep >= 0.7:
        verdict = (f"REVISE: sim image with real state keeps {vis_keep:.0%} of RR motion — "
                   f"visual shift is NOT the dominant cause.")
    else:
        verdict = (f"MIXED: sim-image-keep={vis_keep:.2f} sim-state-keep={st_keep:.2f}; "
                   f"both visual and state contribute.")
    print(f"\n[2x2] VERDICT: {verdict}", flush=True)

    if _trace_run is not None:
        try:
            import weave, wandb
            def modality_2x2():
                return {"instr": INSTR, "K": int(K),
                        "cells": {n: per_cell[n] for n in per_cell},
                        "sim_image_motion_kept": round(float(vis_keep), 4),
                        "sim_state_motion_kept": round(float(st_keep), 4),
                        "verdict": verdict}
            weave.op(modality_2x2)()
            wandb.run.summary.update({
                "2x2/RR_disp": rr, "2x2/RS_disp": rs, "2x2/SR_disp": sr, "2x2/SS_disp": ss,
                "2x2/sim_image_motion_kept": float(vis_keep),
                "2x2/sim_state_motion_kept": float(st_keep)})
            print("[2x2] weave + metrics logged", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[2x2] weave logging failed: {_e}", flush=True)
        finally:
            globals().get("_finish_eval_tracing", lambda r: None)(_trace_run)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    main()
