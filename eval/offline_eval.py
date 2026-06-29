#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Offline Weave eval on REAL held-out Trossen episodes (dreamzero-evals structure).

Instead of a sim rollout (whose rendered scene is out-of-distribution for a model
trained on real Trossen video), this replays the REAL Encord/Trossen LeRobot
episodes through the model and scores predicted vs ground-truth actions — so we
evaluate on exactly the training data distribution, with meaningful action error.

Same Weave structure as github.com/anu-wandb/dreamzero-evals:
  * weave.init inside an active wandb.run (lineage via use_artifact of the ckpt).
  * @weave.op run_episode(...) returns {episode_video, dream_video, side_by_side:
    VideoFileClip, action_mse, ...}; the episode_video is the REAL 3-cam strip, the
    dream_video is the WAM's prediction (server __debug__), side-by-side = real||dream.
  * one EvaluationLogger(model=<ckpt>, dataset=trossen-offline, name=...): per episode
    log_prediction(output=run_episode_out) -> log_score(action_mse, gripper_mae, ...)
    -> finish(); log_summary(...) at the end.

Runs as a lightweight client (no Isaac Sim / Arena) against the DreamZero Trossen
server. Env knobs: WAM_OFFLINE_EPISODES (n episodes), WAM_OFFLINE_MAX_STEPS,
WAM_OFFLINE_STRIDE, WAM_DATASET_DIR.
"""

from __future__ import annotations

import json
import os

import numpy as np
import weave
from openpi_client import image_tools, websocket_client_policy

DATASET = os.environ.get("WAM_DATASET_DIR", "/data/wam/datasets/encord_trossen")
ACTION_DIM = 16
GRIPPER_DIMS = (6, 13)  # left_joint_6, right_joint_6 (per the dataset state/action names)
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]

_CLIENT = None
_EPISODES = None  # list of (episode_index, prompt)


# ---------------------------- dataset reader ----------------------------

def _load_meta():
    info = json.load(open(os.path.join(DATASET, "meta", "info.json")))
    tasks = {}
    with open(os.path.join(DATASET, "meta", "tasks.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            tasks[d["task_index"]] = d["task"]
    eps = []
    with open(os.path.join(DATASET, "meta", "episodes.jsonl")) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            prompt = (d.get("tasks") or ["do the task"])[0]
            eps.append((i, prompt))
    return info, tasks, eps


def _parquet_path(info, ep): return os.path.join(DATASET, info["data_path"].format(episode_chunk=0, episode_index=ep))
def _video_path(info, ep, key):
    return os.path.join(DATASET, info["video_path"].format(episode_chunk=0, video_key=f"observation.images.{key}", episode_index=ep))


def _read_video(path, n_frames=None):
    """Return list of HWC uint8 frames via imageio+ffmpeg (handles AV1/h264)."""
    import imageio.v2 as imageio

    frames = []
    rd = imageio.get_reader(path)
    for i, fr in enumerate(rd):
        if n_frames is not None and i >= n_frames:
            break
        frames.append(np.asarray(fr, dtype=np.uint8))
    rd.close()
    return frames


def _write_video(path, frames, fps):
    """Write an mp4 from a list of HWC uint8 frames via imageio-ffmpeg."""
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(f, np.uint8) for f in frames], fps=fps, codec="libx264", macro_block_size=None)


def _read_episode(info, ep, max_frames):
    import pandas as pd

    df = pd.read_parquet(_parquet_path(info, ep))
    state = np.stack(df["observation.state"].to_numpy())[:max_frames].astype(np.float64)
    action = np.stack(df["action"].to_numpy())[:max_frames].astype(np.float32)
    vids = {k: _read_video(_video_path(info, ep, k), max_frames) for k in VIDEO_KEYS}
    n = min(len(state), len(action), *(len(v) for v in vids.values()))
    return state[:n], action[:n], {k: v[:n] for k, v in vids.items()}


# ---------------------------- video helpers ----------------------------

def _resize_w(a, w):
    a = np.asarray(a).astype(np.uint8)
    if a.shape[1] == w:
        return a
    from PIL import Image

    return np.asarray(Image.fromarray(a).resize((w, int(a.shape[0] * w / a.shape[1])))).astype(np.uint8)


def _clip(path):
    if not (path and os.path.exists(path)):
        return None
    try:  # moviepy 2.x
        from moviepy import VideoFileClip
    except Exception:  # moviepy 1.x
        from moviepy.editor import VideoFileClip
    return VideoFileClip(path, audio=False)


# ---------------------------- the eval op ----------------------------

@weave.op(name="run_episode")
def run_episode(episode_idx: int, prompt: str, max_steps: int, stride: int, open_loop_horizon: int) -> dict:
    """Replay one real episode through the model; score predicted vs GT actions and
    build real / dream / side-by-side videos."""
    info, _, _ = _load_meta()
    state, gt_action, vids = _read_episode(info, episode_idx, max_steps * stride)
    n = len(state)
    client = _CLIENT
    out_dir = os.environ.get("WAM_EVAL_VIDEO_DIR", "/data/wam/eval_videos_offline")
    os.makedirs(out_dir, exist_ok=True)

    sim_strip, dream_full, dream_sync = [], [], []
    pred_chunk, completed = None, 0
    err_sq, err_abs, grip_abs, n_scored = [], [], [], 0
    session = f"offline-ep{episode_idx}"

    for t in range(0, n, stride):
        ext = vids["exterior_image_1_left"][t]
        wl = vids["wrist_image_left"][t]
        wr = vids["wrist_image_right"][t]
        # Real 3-cam strip for the episode video.
        sim_strip.append(np.concatenate([_resize_w(ext, 256), _resize_w(wl, 256), _resize_w(wr, 256)], axis=1))

        if pred_chunk is None or completed >= open_loop_horizon:
            req = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(ext, 480, 640),
                "observation/wrist_image_left": image_tools.resize_with_pad(wl, 480, 640),
                "observation/wrist_image_right": image_tools.resize_with_pad(wr, 480, 640),
                "observation/state": np.asarray(state[t], dtype=np.float64).reshape(-1),
                "prompt": prompt,
                "session_id": session,
                "endpoint": "infer",
            }
            resp = client.infer(req)
            # dream video for this chunk
            dbg = resp.get("__debug__") if isinstance(resp, dict) else None
            ldf = []
            if isinstance(dbg, dict) and dbg.get("dream_video_mp4"):
                import io

                import imageio.v2 as imageio
                try:
                    for fr in imageio.get_reader(io.BytesIO(dbg["dream_video_mp4"]), format="mp4"):
                        ldf.append(np.flip(np.asarray(fr, np.uint8), axis=0).copy())
                except Exception:
                    pass
            dream_full.extend(ldf)
            pred_chunk = _parse_actions(resp)
            completed = 0
            _last_dream = ldf
        ci = completed
        completed += 1
        # synced dream frame for this step
        df = _last_dream[min(ci, len(_last_dream) - 1)] if _last_dream else None
        if df is not None:
            dream_sync.append(df)
        # score predicted action[ci] vs GT action at t
        if pred_chunk is not None and ci < pred_chunk.shape[0]:
            pa = pred_chunk[ci]
            ga = gt_action[t]
            err_sq.append(float(np.mean((pa - ga) ** 2)))
            err_abs.append(float(np.mean(np.abs(pa - ga))))
            grip_abs.append(float(np.mean(np.abs(pa[[GRIPPER_DIMS[0], GRIPPER_DIMS[1]]] - ga[[GRIPPER_DIMS[0], GRIPPER_DIMS[1]]]))))
            n_scored += 1

    # write videos
    ep = os.path.join(out_dir, f"episode_{episode_idx}.mp4")
    dr = os.path.join(out_dir, f"episode_{episode_idx}_dream.mp4")
    sb = os.path.join(out_dir, f"episode_{episode_idx}_sbs.mp4")
    paths = {"ep": None, "dr": None, "sb": None}
    try:
        _write_video(ep, sim_strip, fps=10); paths["ep"] = ep
    except Exception as e:
        print(f"[offline] real video write failed: {e}", flush=True)
    if dream_full:
        try:
            _write_video(dr, dream_full, fps=10); paths["dr"] = dr
        except Exception as e:
            print(f"[offline] dream write failed: {e}", flush=True)
    if dream_sync and sim_strip:
        try:
            w = max(np.asarray(dream_sync[0]).shape[1], sim_strip[0].shape[1])
            m = min(len(dream_sync), len(sim_strip))
            _write_video(sb, [np.concatenate([_resize_w(dream_sync[i], w), _resize_w(sim_strip[i], w)], axis=0) for i in range(m)], fps=10)
            paths["sb"] = sb
        except Exception as e:
            print(f"[offline] sbs write failed: {e}", flush=True)

    action_mse = float(np.mean(err_sq)) if err_sq else float("nan")
    print(f"[offline] episode {episode_idx} ('{prompt}'): scored {n_scored} steps, action_mse={action_mse:.4f}", flush=True)
    return {
        "episode_idx": episode_idx,
        "prompt": prompt,
        "n_scored": n_scored,
        "action_mse": action_mse,
        "action_mae": float(np.mean(err_abs)) if err_abs else float("nan"),
        "gripper_mae": float(np.mean(grip_abs)) if grip_abs else float("nan"),
        "episode_video": _clip(paths["ep"]),
        "dream_video": _clip(paths["dr"]),
        "side_by_side": _clip(paths["sb"]),
        "n_dream_frames": len(dream_full),
    }


def _parse_actions(resp) -> np.ndarray:
    if isinstance(resp, dict):
        for k in ("actions", "action", "action.action"):
            if k in resp:
                a = np.asarray(resp[k], dtype=np.float32)
                break
        else:
            a = np.asarray(next(iter(resp.values())), dtype=np.float32)
    else:
        a = np.asarray(resp, dtype=np.float32)
    return a.reshape(1, -1) if a.ndim == 1 else a


def main() -> None:
    global _CLIENT
    host = os.environ.get("DZ_HOST", "dreamzero-trossen-inference")
    port = int(os.environ.get("DZ_PORT", "8001"))
    n_eps = int(os.environ.get("WAM_OFFLINE_EPISODES", "5"))
    max_steps = int(os.environ.get("WAM_OFFLINE_MAX_STEPS", "120"))
    stride = int(os.environ.get("WAM_OFFLINE_STRIDE", "5"))
    horizon = int(os.environ.get("WAM_OFFLINE_HORIZON", "5"))

    from isaaclab_arena_dreamzero.weave_eval import finish_eval_tracing, init_eval_tracing  # reuse wandb+weave+lineage+tags

    run = init_eval_tracing()
    info, _, eps = _load_meta()
    sel = eps[:n_eps]
    print(f"[offline] {len(sel)} episodes, max_steps={max_steps}, stride={stride}, server={host}:{port}", flush=True)

    _CLIENT = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    from weave import EvaluationLogger

    model_label = os.environ.get("WEAVE_MODEL", "dreamzero-trossen-lora")
    art = os.environ.get("LORA_ARTIFACT", "")
    if ":" in art:
        model_label = f"dreamzero-trossen-lora:{art.rsplit(':', 1)[-1]}"
    el = EvaluationLogger(model=model_label, dataset="trossen-offline", name="trossen_offline")

    try:
        mses = []
        for ep_idx, prompt in sel:
            out = run_episode(episode_idx=ep_idx, prompt=prompt, max_steps=max_steps, stride=stride, open_loop_horizon=horizon)
            pred = el.log_prediction(inputs={"episode_idx": ep_idx, "prompt": prompt}, output=out)
            for s in ("action_mse", "action_mae", "gripper_mae", "n_scored"):
                v = out[s]
                if isinstance(v, (int, float)) and v == v:  # skip NaN
                    pred.log_score(scorer=s, score=v)
            pred.finish()
            if out["action_mse"] == out["action_mse"]:
                mses.append(out["action_mse"])
        el.log_summary({"num_episodes": len(sel), "mean_action_mse": float(np.mean(mses)) if mses else float("nan")})
        print(f"[offline] done. mean_action_mse={np.mean(mses) if mses else float('nan'):.4f}", flush=True)
    finally:
        finish_eval_tracing(run)


if __name__ == "__main__":
    main()
