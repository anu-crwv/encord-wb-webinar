#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""First-action-chunk probe: same rest-pose proprio, vary ONLY the input image/prompt,
compare the model's predicted action chunk. Pure server inference (no Isaac / no physics).

Decisive test for "dream right / action under-commits": if a REAL held-out frame yields a
committed reach (left-shoulder Lj1 -> ~2.3-2.4) while the SIM frame stays conservative
(~1.5), the failure is visual-domain->action generalization (fixable by domain matching);
if the REAL frame is ALSO weak, it's the action decoder/prior (a data/training problem).

Runs in the offline-eval CPU env (openpi_client + websockets). Talks to the DreamZero
Trossen server (DZ_HOST/DZ_PORT).
"""

from __future__ import annotations

import glob
import os

import numpy as np
from openpi_client import image_tools, websocket_client_policy

DATA_ROOT = os.environ.get("WAM_TROSSEN_DATA", "/data/wam/datasets/encord_trossen")
SIM_DEBUG = "/data/wam/eval_videos/debug"  # sim step-0 camera PNGs from a debug render
H, W = 480, 640  # adapter target (height, width); server applies the trained crop/resize
CAMS = ("exterior_image_1_left", "wrist_image_left", "wrist_image_right")
# The real Trossen rest pose (dataset episode first-frame state) -> what the sim seeds too.
REST_STATE = np.array([0.0, 1.047, 0.523, 0.628, 0, 0, 0, 0, 1.047, 0.523, 0.628, 0, 0, 0, 0, 0],
                      dtype=np.float64)


def _pack(images: dict, state: np.ndarray, prompt: str) -> dict:
    req = {"prompt": prompt, "endpoint": "infer", "session_id": f"probe-{os.getpid()}"}
    req["observation/state"] = np.asarray(state, dtype=np.float64).reshape(-1)
    for cam in CAMS:
        req[f"observation/{cam}"] = image_tools.resize_with_pad(images[cam], H, W)
    return req


def _stats(chunk: np.ndarray, label: str) -> None:
    # 16-dim order: [Lj0..5, Lgrip, Rj0..5, Rgrip, linv, angv]. Lj1 idx1, Rj1 idx8, grips 6/13.
    lj1, rj1 = chunk[:, 1], chunk[:, 8]
    print(f"\n===== {label} =====", flush=True)
    print(f"  chunk rows={chunk.shape[0]}", flush=True)
    print(f"  Lj1: mean={lj1.mean():+.3f} max={lj1.max():+.3f}  (real reach needs ~2.3-2.4)", flush=True)
    print(f"  Rj1: mean={rj1.mean():+.3f} max={rj1.max():+.3f}", flush=True)
    print(f"  Lgrip: [{chunk[:,6].min():+.3f},{chunk[:,6].max():+.3f}]  Rgrip: [{chunk[:,13].min():+.3f},{chunk[:,13].max():+.3f}]", flush=True)
    print(f"  Lj1 all rows={np.round(lj1, 3).tolist()}", flush=True)


def _real_first_frames() -> dict:
    import imageio.v2 as imageio
    out = {}
    for cam in CAMS:
        f = sorted(glob.glob(f"{DATA_ROOT}/videos/chunk-000/observation.images.{cam}/episode_000000.mp4"))[0]
        rd = imageio.get_reader(f, "ffmpeg")
        out[cam] = np.asarray(next(iter(rd)))[:, :, :3].astype(np.uint8)
    return out


def _sim_frames() -> dict | None:
    import imageio.v2 as imageio
    out = {}
    for cam in CAMS:
        p = os.path.join(SIM_DEBUG, f"{cam}_rgb.png")
        if not os.path.exists(p):
            print(f"[probe] missing sim frame {p}", flush=True)
            return None
        out[cam] = np.asarray(imageio.imread(p))[:, :, :3].astype(np.uint8)
    return out


def _query(client, images, state, prompt, action_dim=16) -> np.ndarray:
    resp = client.infer(_pack(images, state, prompt))
    act = resp.get("action", resp.get("actions"))
    arr = np.asarray(act, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, action_dim)
    return arr


def main() -> None:
    host = os.environ.get("DZ_HOST", "dreamzero-trossen-inference")
    port = int(os.environ.get("DZ_PORT", "8001"))
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"[probe] connected to {host}:{port}", flush=True)

    real = _real_first_frames()
    print(f"[probe] real frames: {[f'{k}{real[k].shape}' for k in CAMS]}", flush=True)
    sim = _sim_frames()

    # A) REAL frame + real state + real prompt (episode 0 = 'pour the coffee into the cup')
    _stats(_query(client, real, REST_STATE, "pour the coffee into the cup"),
           "A) REAL frame + real state + real prompt (coffee)")

    # B) SIM frame + rest state + sim prompt (the current eval setup)
    if sim is not None:
        _stats(_query(client, sim, REST_STATE, "Pick up the cube and place it in the bowl."),
               "B) SIM frame + rest state + sim prompt (cube)")
        # C) SIM frame + rest state + REAL prompt (isolate prompt vs image)
        _stats(_query(client, sim, REST_STATE, "pour the coffee into the cup"),
               "C) SIM frame + rest state + REAL prompt (coffee) [prompt-vs-image control]")

    print("\n[probe] done. Compare A (real) vs B/C (sim): if A commits (Lj1~2.4) and B/C stay ~1.5, "
          "the gap is visual-domain->action; if A is also weak, it's the action decoder.", flush=True)


if __name__ == "__main__":
    main()
