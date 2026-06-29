# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Trossen embodiment adapter for the DreamZero remote policy.

Maps Arena's Trossen-embodiment observation dict onto the DreamZero server's
request, and declares the bimanual action dimension. This is the DreamZero
analogue of ``isaaclab_arena_openpi.policy.droid_adapter.Pi0DroidAdapter``.

The wire contract here MUST match the DreamZero server's Trossen input mapping
(the server-side companion task: generalize ``PolicyServerConfig`` to 3 cameras
+ a 16-dim packed action). The trained Trossen modality config
(``modality_config_trossen`` in
``groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml``) defines:

    video : video.exterior_image_1_left, video.wrist_image_left, video.wrist_image_right
    state : state.state                  (16-dim packed, q99-normalized server-side)
    action: action.action                (16-dim packed, q99-normalized server-side)
    lang  : annotation.language.language_instruction

so the request sends the three RGB views + the 16-dim state + the prompt, and the
server returns a 16-dim action chunk. The Arena-side camera/state observation
keys below are the ones the (to-be-authored) Trossen embodiment must expose.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from openpi_client import image_tools

from isaaclab_arena_dreamzero.policy.dreamzero_remote_policy import DreamZeroEmbodimentAdapter


@dataclass(frozen=True)
class TrossenObservation:
    """Per-env tensors needed to assemble a DreamZero Trossen request."""

    exterior_image_1_left: np.ndarray  # (H, W, 3) uint8
    wrist_image_left: np.ndarray  # (H, W, 3) uint8
    wrist_image_right: np.ndarray  # (H, W, 3) uint8
    state: np.ndarray  # (16,) float32 — packed bimanual joints + grippers + base


class DreamZeroTrossenAdapter(DreamZeroEmbodimentAdapter):
    """Wire format for the fine-tuned Trossen (bimanual mobile) DreamZero LoRA."""

    # Trossen AI mobile bimanual: two 6-DOF arms + 2 grippers + base, packed to 16.
    action_dim = 16

    # Image size sent to the server, (height, width). The Trossen eval transform
    # (VideoToTensor) validates the RAW input resolution against the training
    # camera resolution 640x480 and errors otherwise, so send 480x640 here; the
    # server then applies the trained crop/resize. resize_with_pad keeps aspect.
    # (Verified 2026-06-26 against dreamzero-trossen-lora:v3 — returns (24,16).)
    target_image_size = (480, 640)  # (height, width)

    # Arena observation groups. ``camera_obs`` is set by
    # isaaclab_arena.utils.cameras.make_camera_observation_cfg; ``policy`` is the
    # standard Isaac Lab ObservationsCfg group every Arena embodiment defines.
    arena_camera_obs_group = "camera_obs"
    arena_policy_obs_group = "policy"

    # Camera keys the Trossen embodiment exposes under ``camera_obs``. Arena's
    # make_camera_observation_cfg names each obs key "{camera_field}_{data_type}",
    # so the embodiment's camera fields are exterior_image_1_left / wrist_image_left /
    # wrist_image_right and the obs keys get a "_rgb" suffix.
    cam_exterior_key = "exterior_image_1_left_rgb"
    cam_wrist_left_key = "wrist_image_left_rgb"
    cam_wrist_right_key = "wrist_image_right_rgb"

    # The 16-dim packed proprio term the Trossen embodiment exposes under
    # ``policy``. Must be ordered to match the training ``state.state`` packing.
    state_key = "state"

    def extract(self, observation: dict[str, Any], env_id: int) -> TrossenObservation:
        cam = observation[self.arena_camera_obs_group]
        proprio = observation[self.arena_policy_obs_group]
        return TrossenObservation(
            exterior_image_1_left=cam[self.cam_exterior_key][env_id].detach().cpu().numpy(),
            wrist_image_left=cam[self.cam_wrist_left_key][env_id].detach().cpu().numpy(),
            wrist_image_right=cam[self.cam_wrist_right_key][env_id].detach().cpu().numpy(),
            state=proprio[self.state_key][env_id].detach().cpu().numpy(),
        )

    def pack_request(self, extracted: TrossenObservation, language_instruction: str) -> dict[str, Any]:
        h, w = self.target_image_size
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(extracted.exterior_image_1_left, h, w),
            "observation/wrist_image_left": image_tools.resize_with_pad(extracted.wrist_image_left, h, w),
            "observation/wrist_image_right": image_tools.resize_with_pad(extracted.wrist_image_right, h, w),
            "observation/state": np.asarray(extracted.state, dtype=np.float64).reshape(-1),
            "prompt": language_instruction,
        }
