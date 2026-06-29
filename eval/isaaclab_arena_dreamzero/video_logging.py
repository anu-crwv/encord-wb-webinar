# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode video logging for the Trossen sim eval — sim rollout + WAM "dream"
video, stitched and synced side-by-side, logged to Weave as VideoFileClip media.

Mirrors the DreamZero DROID eval (deploy/cks/scripts/droid_weave_eval.py): three
videos per episode —
  * episode_N.mp4      — sim 3-camera strip (exterior | wrist_L | wrist_R)
  * episode_N_dream.mp4 — the WAM's predicted ("dream") video, concatenated per
                          fresh action chunk
  * episode_N_sbs.mp4  — dream stacked over sim, strict 1:1 temporal sync
and logged via a Weave EvaluationLogger (one prediction per episode), so they
render inline in the W&B/Weave workspace next to the rollout traces.

Driven from DreamZeroRemotePolicy: record_step() each sim step, flush_episode()
on policy.reset(). The dream stream requires the server to pack VAE-decoded frames
into response["__debug__"]["dream_video_mp4"] (DZ_DREAM_VIDEO=1 on the server).
"""

from __future__ import annotations

import io
import os

import numpy as np


def _resize_w(arr: np.ndarray, w: int) -> np.ndarray:
    a = np.asarray(arr).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[1] == w:
        return a
    from PIL import Image

    new_h = max(1, int(a.shape[0] * (w / a.shape[1])))
    return np.asarray(Image.fromarray(a).resize((w, new_h))).astype(np.uint8)


def decode_dream_mp4(mp4_bytes: bytes) -> list[np.ndarray]:
    """Decode the server's dream-video mp4 bytes to RGB frames, flipping the
    Y axis (the VAE decoder emits a bottom-origin convention vs the sim's
    top-origin — same fix the DROID eval applied)."""
    import imageio.v2 as imageio

    frames = []
    try:
        reader = imageio.get_reader(io.BytesIO(mp4_bytes), format="mp4")
        for fr in reader:
            frames.append(np.flip(np.asarray(fr, dtype=np.uint8), axis=0).copy())
    except Exception as e:  # noqa: BLE001
        print(f"[video_logging] dream decode failed: {e}", flush=True)
    return frames


class TrossenVideoLogger:
    """Accumulates per-step sim + dream frames and writes/logs them per episode."""

    def __init__(self, out_dir: str, model_label: str, fps: int = 15, cam_w: int = 256):
        self.enabled = os.environ.get("WAM_EVAL_VIDEO", "1") != "0"
        # Default to a PVC-backed dir so the mp4s survive the (ephemeral) eval pod.
        out_dir = os.environ.get("WAM_EVAL_VIDEO_DIR", out_dir)
        self.out_dir = out_dir
        self.model_label = model_label
        self.fps = fps
        self.cam_w = cam_w
        self.ep_idx = 0
        self._sim: list[np.ndarray] = []
        self._dream_full: list[np.ndarray] = []
        self._dream_sync: list[np.ndarray] = []
        self._el = None  # lazy weave EvaluationLogger
        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)

    def record_step(self, images: dict[str, np.ndarray], dream_frames, chunk_index: int) -> None:
        """images: {'exterior','wrist_left','wrist_right'} HWC uint8. dream_frames:
        decoded frames for the CURRENT chunk (or None). chunk_index: step within chunk."""
        if not self.enabled:
            return
        strip = np.concatenate(
            [_resize_w(images["exterior"], self.cam_w),
             _resize_w(images["wrist_left"], self.cam_w),
             _resize_w(images["wrist_right"], self.cam_w)],
            axis=1,
        )
        self._sim.append(strip)
        if dream_frames is not None and len(dream_frames) > 0:
            if chunk_index == 0:
                self._dream_full.extend(dream_frames)
            self._dream_sync.append(np.asarray(dream_frames[min(chunk_index, len(dream_frames) - 1)]))

    def build_episode_videos(self, episode_idx: int) -> dict:
        """Write the 3 MP4s for the just-finished episode and return their paths +
        counts. NO logging here — the eval driver (run_trossen_eval.run_episode)
        wraps these as VideoFileClip in the @weave.op output + EvaluationLogger
        prediction (the dreamzero-evals structure). Resets buffers after."""
        out = {"episode_video_path": None, "dream_video_path": None, "side_by_side_path": None,
               "n_steps": len(self._sim), "n_dream_frames": len(self._dream_full)}
        if not self.enabled or not self._sim:
            self.reset_episode()
            return out
        import mediapy

        n = episode_idx
        ep_path = os.path.join(self.out_dir, f"episode_{n}.mp4")
        dream_path = os.path.join(self.out_dir, f"episode_{n}_dream.mp4")
        sbs_path = os.path.join(self.out_dir, f"episode_{n}_sbs.mp4")

        try:
            mediapy.write_video(ep_path, self._sim, fps=self.fps)
            out["episode_video_path"] = ep_path
        except Exception as e:  # noqa: BLE001
            print(f"[video_logging] sim mp4 write failed: {e}", flush=True)

        if self._dream_full:
            try:
                mediapy.write_video(dream_path, [np.asarray(f, np.uint8) for f in self._dream_full], fps=self.fps)
                out["dream_video_path"] = dream_path
            except Exception as e:  # noqa: BLE001
                print(f"[video_logging] dream mp4 write failed: {e}", flush=True)

        # Side-by-side: dream stacked over sim, common width, strict 1:1 (dream_sync
        # has one frame per sim step by construction).
        if self._dream_sync and self._sim:
            try:
                tgt_w = max(np.asarray(self._dream_sync[0]).shape[1], self._sim[0].shape[1])
                m = min(len(self._dream_sync), len(self._sim))
                sbs = [np.concatenate([_resize_w(self._dream_sync[i], tgt_w), _resize_w(self._sim[i], tgt_w)], axis=0)
                       for i in range(m)]
                mediapy.write_video(sbs_path, sbs, fps=self.fps)
                out["side_by_side_path"] = sbs_path
            except Exception as e:  # noqa: BLE001
                print(f"[video_logging] side-by-side write failed: {e}", flush=True)

        print(f"[video_logging] wrote episode {n} mp4s "
              f"(sim={out['episode_video_path'] is not None}, dream={out['dream_video_path'] is not None}, "
              f"sbs={out['side_by_side_path'] is not None}, steps={out['n_steps']})", flush=True)
        self.reset_episode()
        return out

    def reset_episode(self) -> None:
        """Clear per-episode frame buffers (called at episode start by policy.reset)."""
        self._sim = []
        self._dream_full = []
        self._dream_sync = []
