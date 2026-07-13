#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""GPU diagnostic: is the DreamZero action head UNDER-FIT, or is the video->action
context at inference simply BAD (AR drift / coupling)?

We run one real Trossen "coffee" episode through the model twice, per chunk:

  (A) NORMAL   autoregressive: feed the model's OWN predicted video latents back
               via ``latent_video`` (the deployed inference path).
  (B) GT-COND  : feed GROUND-TRUTH VAE-encoded future latents via ``latent_video``
               (i.e. replace the possibly-drifted dream with the true video).

Everything else (state is teacher-forced from the dataset, same frames, same
forward function) is held identical, so the ONLY thing that changes between A and
B is the video context that conditions the action head. For each mode we compare
the predicted action chunk against the dataset GT actions, PER JOINT, using the
amplitude (std) ratio and correlation.

READ:
  * gtcond std_ratio >> normal and ~1  => the action head CAN commit; the deployed
    under-commit is caused by bad video context (AR drift / video-action coupling).
  * gtcond still << 1 & low corr        => the action head is genuinely under-fit;
    even a perfect dream does not make it move.

IMPORTANT caveat (encoded below): ``latent_video`` only takes effect when
``current_start_frame != 0`` (wan_flow_matching_action_tf.py:1082). The FIRST chunk
of every episode is therefore always self-conditioned and IDENTICAL in both modes;
we only accumulate the comparison from the 2nd chunk onward.

Run (single GH200 is enough), from the repo root so ``groot`` is importable:

    MODEL_DIR=/checkpoints/wam/eval/dreamzero-trossen-lora-v5 \
    WAM_TROSSEN_DATA=/data/wam/datasets/encord_trossen_v4 \
    DIAG_TASK=coffee \
    torchrun --standalone --nproc_per_node=1 eval/diag_gt_cond.py

This is a READ-ONLY diagnostic: it loads the policy and dataset, runs inference,
and prints a table. It writes nothing.
"""

from __future__ import annotations

import glob
import json
import os

# The action head reads a couple of env knobs at import/construction time; mirror
# the trossen server (eval/server/trossen_policy_server.py:182-184) BEFORE importing
# groot so the model is built with the same attention backend / cache settings.
os.environ.setdefault("ATTENTION_BACKEND", "TE")
os.environ.setdefault("ENABLE_DIT_CACHE", "false")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy, unsqueeze_dict_values

# ---------------------------------------------------------------------------
# Config (env-driven)
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "/checkpoints/wam/eval/dreamzero-trossen-lora-v5")
DATA_ROOT = os.environ.get("WAM_TROSSEN_DATA", "/data/wam/datasets/encord_trossen_v4")
TASK_SUBSTR = os.environ.get("DIAG_TASK", "coffee")
MAX_CHUNKS = int(os.environ.get("DIAG_MAX_CHUNKS", "24"))  # chunks per mode (each = action_horizon steps)

H, W = 480, 640  # native Trossen frame size (height, width); matches trossen server image_resolution
# Trossen video modality_keys order == video_concat_order (base_48_wan_fine_aug_relative.yaml:354-357).
# This order defines the 2x2 tile layout that the world model dreams (see _encode_gt_latents).
CAMS = ("exterior_image_1_left", "wrist_image_left", "wrist_image_right")
FRAMES_PER_CHUNK = 4  # ARDroidRoboarenaPolicy.FRAMES_PER_CHUNK (socket_test_optimized_AR.py:55):
#   feed 1 frame on the first call (forces current_start_frame=0 / fresh start), then 4 frames.
#   Feeding >1 frame on later calls is REQUIRED: videos.shape[2]==1 resets current_start_frame to 0
#   (wan_flow_matching_action_tf.py:1045), which would disable latent_video entirely.

# 16-dim packed Trossen action: [Lj0..5, Lgrip, Rj0..5, Rgrip, linv, angv]
DIM_NAMES = ["Lj0", "Lj1", "Lj2", "Lj3", "Lj4", "Lj5", "Lgrip",
             "Rj0", "Rj1", "Rj2", "Rj3", "Rj4", "Rj5", "Rgrip", "linv", "angv"]


# ---------------------------------------------------------------------------
# Distributed / device-mesh init (single GPU, standalone torchrun)
# ---------------------------------------------------------------------------
def init_mesh() -> DeviceMesh:
    """Replicates socket_test_optimized_AR.init_mesh / build_trt_engine._init_single_gpu_mesh.

    GrootSimPolicy needs a device_mesh and an initialised process group; torchrun
    (``--standalone --nproc_per_node=1``) provides RANK/WORLD_SIZE/MASTER_*.
    """
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )
    return mesh


# ---------------------------------------------------------------------------
# Dataset loading (mirrors eval/diag_action_vs_gt.py)
# ---------------------------------------------------------------------------
def episode_for_task(task_substr: str):
    """First episode whose task string matches (meta/tasks.jsonl + per-episode task_index)."""
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{DATA_ROOT}/meta/tasks.jsonl") if l.strip()}
    want = [i for i, t in tasks.items() if task_substr.lower() in t.lower()]
    for pq in sorted(glob.glob(f"{DATA_ROOT}/data/chunk-000/episode_*.parquet")):
        df = pd.read_parquet(pq, columns=["task_index"])
        if int(df["task_index"].iloc[0]) in want:
            ep = int(pq.split("episode_")[-1].split(".")[0])
            return ep, tasks[int(df["task_index"].iloc[0])]
    return 0, tasks.get(0, "?")


def load_frames(ep: int):
    """{cam: [np.uint8 (H,W,3), ...]} for the 3 Trossen cameras."""
    import imageio.v2 as imageio
    out = {}
    for cam in CAMS:
        f = f"{DATA_ROOT}/videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
        rd = imageio.get_reader(f, "ffmpeg")
        out[cam] = [np.asarray(fr)[:, :, :3].astype(np.uint8) for fr in rd]
    return out


# ---------------------------------------------------------------------------
# AR state helpers
# ---------------------------------------------------------------------------
def reset_ar_state(ah) -> None:
    """Reset the action head's autoregressive state between the two modes.

    Mirrors build_trt_engine._make_dataset_forward_loop (lines 144-148); also clears
    ``language`` so the very first call of the next pass takes the fresh-start path
    (wan_flow_matching_action_tf.py:1037-1040)."""
    ah.current_start_frame = 0
    ah.kv_cache1 = None
    ah.kv_cache_neg = None
    ah.crossattn_cache = None
    ah.crossattn_cache_neg = None
    ah.language = None
    ah.clip_feas = None
    ah.ys = None


def build_obs(frame_buffers: dict, num_frames: int, state_vec: np.ndarray, prompt: str) -> dict:
    """Build a raw observation in the exact format the trossen server feeds the policy
    (eval/server/trossen_policy_server.py:_convert_observation): per-camera video window,
    a (1, 16) packed state, and the language annotation. lazy_joint_forward_causal runs
    self.apply() (the eval transform) on this internally."""
    obs: dict = {}
    for cam in CAMS:
        buf = frame_buffers[cam]
        if len(buf) >= num_frames:
            frames_to_use = buf[-num_frames:]
        else:
            frames_to_use = buf.copy()
            while len(frames_to_use) < num_frames:
                frames_to_use.insert(0, buf[0])  # left-pad by repeating first frame
        obs[f"video.{cam}"] = np.stack(frames_to_use, axis=0).astype(np.uint8)  # (T, H, W, 3)
    obs["state.state"] = np.asarray(state_vec, dtype=np.float64).reshape(1, -1)   # (1, 16)
    obs["annotation.language.action_text"] = prompt
    return obs


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def extract_action(result_batch) -> np.ndarray:
    """Pull the UN-normalized action chunk out of result_batch.act.

    sim_policy.unapply sets batch.act to the un-normalized action dict; for trossen
    the single action key is ``action.action`` (base_48_wan_fine_aug_relative.yaml:367).
    Robust to dict / tianshou.Batch / attribute access and to split multi-key actions."""
    act = getattr(result_batch, "act", result_batch)
    items = {}
    if isinstance(act, dict):
        pairs = list(act.items())
    elif hasattr(act, "items"):
        pairs = list(act.items())
    elif hasattr(act, "keys"):
        pairs = [(k, act[k]) for k in act.keys()]
    else:
        pairs = [(k, getattr(act, k)) for k in dir(act) if not k.startswith("_")]
    for k, v in pairs:
        if isinstance(k, str) and k.startswith("action"):
            items[k] = v

    if "action.action" in items:
        arr = _to_np(items["action.action"])
    elif len(items) == 1:
        arr = _to_np(next(iter(items.values())))
    elif items:
        arr = np.concatenate([_to_np(items[k]) for k in sorted(items)], axis=-1)
    else:
        raise RuntimeError(f"Could not find an action.* entry on result_batch.act: {type(act)}")

    arr = arr.astype(np.float32)
    if arr.ndim == 3:      # (1, horizon, D)
        arr = arr[0]
    if arr.ndim == 1:      # (D,) -> single step
        arr = arr.reshape(1, -1)
    return arr              # (horizon, D)


# ---------------------------------------------------------------------------
# GT-conditioning: VAE-encode the ground-truth future block into latent_video
# ---------------------------------------------------------------------------
def encode_gt_latents(policy, ah, frames: dict, gt_state: np.ndarray, prompt: str,
                      t: int, num_frame_per_block: int) -> torch.Tensor:
    """Return latent_video = VAE-encoded GT *future* block, sliced to the last
    num_frame_per_block latent frames -- the GT analogue of the model's own
    ``video_pred[:, :, -num_frame_per_block:]`` (build_trt_engine.py:141).

    Pipeline (all steps mirror the model's own path so the latent lives in the same
    space as video_pred):
      1. Take a GT RGB window ENDING at the current step t (the block the model's
         dream would represent for this chunk). Window length = num_frame_per_block*4+1
         so the VAE (4x temporal downsample: n_lat = 1 + (T-1)//4) yields
         num_frame_per_block+1 latent frames; we drop the special first one.
      2. Run policy.eval_transform to get data["images"] -- the REAL 2x2 multi-camera
         tile (DreamTransform._prepare_video, dreamzero_cotrain.py:309-381; collate
         stacks it to [B, T, 2H, 2W, C] uint8).
      3. Normalize to [-1, 1] + resize to the model's target video resolution, exactly
         as wan_flow_matching_action_tf.py:1002-1035.
      4. ah.encode_video(...) (wan_flow_matching_action_tf.py:563) and slice.
    """
    T_total = len(frames[CAMS[0]])
    win = num_frame_per_block * 4 + 1
    start = t - (win - 1)                       # window ends AT t (the current obs time)
    idxs = [min(max(i, 0), T_total - 1) for i in range(start, t + 1)]  # clamp+left-pad

    gt_obs: dict = {}
    for cam in CAMS:
        gt_obs[f"video.{cam}"] = np.stack([frames[cam][i] for i in idxs], axis=0).astype(np.uint8)
    gt_obs["state.state"] = np.asarray(gt_state, dtype=np.float64).reshape(1, -1)
    gt_obs["annotation.language.action_text"] = prompt

    with torch.no_grad():
        # (2) real multi-camera tile via the model's own eval transform.
        # eval_transform expects the SAME batched obs that lazy_joint_forward_causal
        # feeds it: a leading batch dim on every modality (sim_policy.py:688-690).
        # Calling it on the RAW (unbatched) obs leaves state.state as (1, 16); inside
        # DreamTransform.apply_batch the per-element x[0] strip then collapses it to
        # (16,), and _prepare_state's np.pad(state, ((0,0),(0,K))) blows up with the
        # "(2,2) -> (1,2)" broadcast (pad_width is 2x2 but the array is now 1-D).
        # Mirror the deployed path so state.state becomes (1, 1, 16) -> (1, 16).
        if not policy._check_state_is_batched(gt_obs):
            gt_obs = unsqueeze_dict_values(gt_obs)
        transformed = policy.eval_transform(gt_obs)
        images = transformed["images"]                       # [B, T, 2H, 2W, C] uint8
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(np.asarray(images))
        images = images.to(device="cuda")
        if images.ndim == 4:                                 # [T, h, w, c] -> add batch
            images = images.unsqueeze(0)
        # The action head only normalizes uint8 images (wan_flow_matching_action_tf.py:1004);
        # guard against being handed already-float frames.
        assert images.dtype == torch.uint8 or float(images.max()) > 1.5, \
            f"expected uint8 / 0-255 images from eval_transform, got dtype={images.dtype}"

        # (3) normalize + resize -- mirrors wan_flow_matching_action_tf.py:1002-1035.
        videos = images.permute(0, 4, 1, 2, 3).contiguous()  # [b, t, h, w, c] -> [b, c, t, h, w]
        # float32 /255 then *2-1  ==  VideoNormalize(mean=std=0.5) (self.normalize_video, line 231/1010);
        # done in float32 (like the model) so the values are exactly [-1,1] before the bf16 cast.
        videos = videos.float() / 255.0
        videos = videos * 2.0 - 1.0
        assert videos.min() >= -1.001 and videos.max() <= 1.001, "gt frames must be in [-1,1]"
        videos = videos.to(dtype=torch.bfloat16)

        target_h = getattr(ah.config, "target_video_height", None)
        target_w = getattr(ah.config, "target_video_width", None)
        if target_h is None or target_w is None:
            # Fallback matches wan_flow_matching_action_tf.py:1021-1025.
            if getattr(ah.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            b, c, tt, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * tt, c, h, w),
                    size=(target_h, target_w), mode="bilinear", align_corners=False,
                ).reshape(b, c, tt, target_h, target_w)

        # (4) VAE-encode with the action head's own tile params, then slice.
        latents = ah.encode_video(
            videos,
            ah.tiled,
            (ah.tile_size_height, ah.tile_size_width),
            (ah.tile_stride_height, ah.tile_stride_width),
        )                                                    # [b, c_lat, t_lat, h_lat, w_lat]
        latent_video = latents[:, :, -num_frame_per_block:]
    return latent_video


# ---------------------------------------------------------------------------
# One full pass over the episode in a given mode
# ---------------------------------------------------------------------------
def run_pass(policy, ah, mode: str, frames: dict, gt_state: np.ndarray, gt_action: np.ndarray,
             prompt: str, step_starts, action_horizon: int, num_frame_per_block: int):
    """mode in {"normal", "gtcond"}. Returns (pred (N,D), gt (N,D)) accumulated from
    the 2nd chunk onward (the 1st chunk is self-conditioned in BOTH modes)."""
    assert mode in ("normal", "gtcond")
    reset_ar_state(ah)
    T_total = min(len(gt_action), len(frames[CAMS[0]]))
    frame_buffers = {cam: [] for cam in CAMS}
    prev_video_pred = None
    pred_rows, gt_rows = [], []

    for s, t in enumerate(step_starts):
        for cam in CAMS:
            frame_buffers[cam].append(frames[cam][t])

        num_frames = 1 if s == 0 else FRAMES_PER_CHUNK
        obs = build_obs(frame_buffers, num_frames, gt_state[t], prompt)

        # Decide the video context. On the very first chunk current_start_frame==0, so
        # latent_video is ignored regardless of mode (wan_flow_matching_action_tf.py:1082).
        if s == 0:
            latent_video = None
        elif mode == "normal":
            latent_video = prev_video_pred[:, :, -num_frame_per_block:] if prev_video_pred is not None else None
        else:  # gtcond
            latent_video = encode_gt_latents(policy, ah, frames, gt_state[t], prompt, t, num_frame_per_block)

        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = policy.lazy_joint_forward_causal(
                Batch(obs=obs), latent_video=latent_video
            )
        dist.barrier()
        prev_video_pred = video_pred

        act = extract_action(result_batch)  # (horizon, D)

        # Skip the first (self-conditioned, mode-identical) chunk; compare the rest.
        if s >= 1:
            n = min(action_horizon, act.shape[0], T_total - t)
            if n > 0:
                pred_rows.append(act[:n])
                gt_rows.append(gt_action[t:t + n])

        print(f"[{mode}] chunk {s:02d} t={t:04d} "
              f"latent_video={'None' if latent_video is None else tuple(latent_video.shape)} "
              f"pred_chunk={act.shape}", flush=True)

    pred = np.concatenate(pred_rows, 0) if pred_rows else np.zeros((0, gt_action.shape[1]), np.float32)
    gt = np.concatenate(gt_rows, 0) if gt_rows else np.zeros((0, gt_action.shape[1]), np.float32)
    return pred, gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch._dynamo.config.recompile_limit = 800  # AR model recompiles across shapes (server main())

    mesh = init_mesh()
    rank = dist.get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct EXACTLY like the trossen server (trossen_policy_server.py:200-205).
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("trossen"),
        model_path=MODEL_DIR,
        device=device,
        device_mesh=mesh,
    )
    ah = policy.trained_model.action_head
    num_frame_per_block = int(ah.num_frame_per_block)
    action_horizon = int(ah.action_horizon)

    if rank == 0:
        tw = getattr(ah.config, "target_video_height", None), getattr(ah.config, "target_video_width", None)
        print(f"[diag] model_path={MODEL_DIR}", flush=True)
        print(f"[diag] num_frame_per_block={num_frame_per_block} action_horizon={action_horizon} "
              f"tiled={ah.tiled} tile_size=({ah.tile_size_height},{ah.tile_size_width}) "
              f"tile_stride=({ah.tile_stride_height},{ah.tile_stride_width}) "
              f"target_video(hw)={tw} frame_seqlen={getattr(ah.model, 'frame_seqlen', None)}", flush=True)
        if action_horizon != num_frame_per_block * 4:
            print(f"[diag] NOTE: action_horizon ({action_horizon}) != num_frame_per_block*4 "
                  f"({num_frame_per_block * 4}); GT window still ends at the current obs time t.", flush=True)

    # Load one real coffee episode.
    ep, task = episode_for_task(TASK_SUBSTR)
    df = pd.read_parquet(f"{DATA_ROOT}/data/chunk-000/episode_{ep:06d}.parquet")
    gt_action = np.stack(df["action"].to_numpy()).astype(np.float32)          # (T, 16) GT targets
    gt_state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # (T, 16)
    frames = load_frames(ep)
    T_total = min(len(gt_action), len(frames[CAMS[0]]))

    step_starts = list(range(0, T_total - 1, action_horizon))[:MAX_CHUNKS]
    if rank == 0:
        print(f"[diag] episode {ep} task='{task}' | T={T_total} | "
              f"{len(step_starts)} chunks @ stride {action_horizon} (compare from chunk 1)", flush=True)

    # Two independent passes with an AR-state reset in between.
    normal_pred, normal_gt = run_pass(policy, ah, "normal", frames, gt_state, gt_action,
                                      task, step_starts, action_horizon, num_frame_per_block)
    gtcond_pred, gtcond_gt = run_pass(policy, ah, "gtcond", frames, gt_state, gt_action,
                                      task, step_starts, action_horizon, num_frame_per_block)

    if rank != 0:
        return

    # Align lengths (both passes cover identical step positions; guard off-by-one).
    K = min(len(normal_pred), len(gtcond_pred), len(normal_gt), len(gtcond_gt))
    if K == 0:
        print("[diag] ERROR: no comparable chunks (need >=2 chunks). Increase DIAG_MAX_CHUNKS "
              "or use a longer episode.", flush=True)
        return
    gt = normal_gt[:K]
    pn = normal_pred[:K]
    pg = gtcond_pred[:K]
    D = gt.shape[1]
    names = DIM_NAMES if D == len(DIM_NAMES) else [f"d{i}" for i in range(D)]

    print(f"\n[diag] compared {K} action rows per mode (2nd chunk onward)\n", flush=True)
    print(f"{'dim':6s} {'GT_std':>8s} | "
          f"{'Nrm_std':>8s} {'Nrm_rat':>8s} {'Nrm_cor':>8s} | "
          f"{'GTc_std':>8s} {'GTc_rat':>8s} {'GTc_cor':>8s} | flag", flush=True)

    def stats(g, p):
        gstd, pstd = float(g.std()), float(p.std())
        ratio = (pstd / gstd) if gstd > 1e-6 else float("nan")
        corr = float(np.corrcoef(g, p)[0, 1]) if gstd > 1e-6 and pstd > 1e-6 else float("nan")
        return gstd, pstd, ratio, corr

    for d in range(D):
        g = gt[:, d]
        gstd, pstd_n, ratio_n, corr_n = stats(g, pn[:, d])
        _, pstd_g, ratio_g, corr_g = stats(g, pg[:, d])
        # Flag joints that GT actually moves where GT-cond recovers amplitude that normal lacks.
        flag = ""
        if gstd > 0.1:
            if ratio_n < 0.4 and (ratio_g > 0.7):
                flag = "<-- GT-COND RECOVERS (context was the problem)"
            elif ratio_n < 0.4 and ratio_g < 0.4:
                flag = "<-- BOTH under-commit (head under-fit)"
        print(f"{names[d]:6s} {gstd:8.3f} | "
              f"{pstd_n:8.3f} {ratio_n:8.2f} {corr_n:+8.2f} | "
              f"{pstd_g:8.3f} {ratio_g:8.2f} {corr_g:+8.2f} | {flag}", flush=True)

    # Aggregate over the "reach" joints (arm joints, excluding grippers + base velocities).
    reach_idx = [i for i, nm in enumerate(names) if nm.startswith(("Lj", "Rj"))] or list(range(D))
    def agg_ratio(p):
        rs = []
        for d in reach_idx:
            gstd = float(gt[:, d].std())
            if gstd > 0.1:
                rs.append(float(p[:, d].std()) / gstd)
        return float(np.mean(rs)) if rs else float("nan")
    rn, rg = agg_ratio(pn), agg_ratio(pg)
    print(f"\n[diag] mean reach-joint std_ratio:  normal={rn:.2f}   gtcond={rg:.2f}", flush=True)

    print("\n[diag] READ: gtcond std_ratio >> normal and ~1 => coupling/AR-drift is the cause; "
          "gtcond still <<1 => action head under-fit.", flush=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    main()
