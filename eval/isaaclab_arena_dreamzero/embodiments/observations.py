# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Custom observation terms for the Trossen Mobile-AI embodiment.

The DreamZero Trossen model consumes a single 16-dim packed ``state`` whose
order is fixed by the Encord dataset (meta/info.json ``observation.state.names``):

    [ left_joint_0..5 (6),  left_joint_6 = left gripper (1),
      right_joint_0..5 (6), right_joint_6 = right gripper (1),
      linear_vel, angular_vel ]                                   # mobile base

``pack_trossen_state_16d`` emits exactly that vector so it matches training, and
the DreamZeroTrossenAdapter reads it as ``policy["state"]``.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

try:  # robot.data.* may be warp arrays; convert defensively (mirrors DROID observations.py).
    import warp as wp
except Exception:  # pragma: no cover
    wp = None


def _to_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    if wp is not None:
        return wp.to_torch(x)
    return torch.as_tensor(x)

# Joint names verified by introspecting mobile_ai.usd (articulation root
# /mobile_ai/base_footprint). Arm joints are revolute; the gripper is the primary
# prismatic carriage on each side (the second carriage mimics it).
LEFT_ARM_JOINTS = [f"follower_left_joint_{i}" for i in range(6)]
RIGHT_ARM_JOINTS = [f"follower_right_joint_{i}" for i in range(6)]
LEFT_GRIPPER_JOINT = "follower_left_left_carriage_joint"
RIGHT_GRIPPER_JOINT = "follower_right_left_carriage_joint"


def _joint_index_tensor(robot, names: list[str], device) -> torch.Tensor:
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    return torch.tensor([name_to_idx[n] for n in names], device=device, dtype=torch.long)


def pack_trossen_state_16d(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the 16-dim packed Trossen state in the trained order (num_envs, 16)."""
    robot = env.scene[asset_cfg.name]
    jp = _to_torch(robot.data.joint_pos)  # (num_envs, n_joints) torch tensor

    # Use torch index tensors (not Python lists) — robot.data.* can be warp arrays
    # whose list-indexing raises "'<' not supported between 'list' and 'int'".
    left_arm = jp.index_select(1, _joint_index_tensor(robot, LEFT_ARM_JOINTS, jp.device))     # (E, 6)
    right_arm = jp.index_select(1, _joint_index_tensor(robot, RIGHT_ARM_JOINTS, jp.device))   # (E, 6)
    left_grip = jp.index_select(1, _joint_index_tensor(robot, [LEFT_GRIPPER_JOINT], jp.device))   # (E, 1)
    right_grip = jp.index_select(1, _joint_index_tensor(robot, [RIGHT_GRIPPER_JOINT], jp.device)) # (E, 1)

    # Mobile base planar velocities in the base frame: forward (x) + yaw (z).
    lin = _to_torch(robot.data.root_lin_vel_b)[:, 0:1]             # (E, 1)
    ang = _to_torch(robot.data.root_ang_vel_b)[:, 2:3]             # (E, 1)

    return torch.cat([left_arm, left_grip, right_arm, right_grip, lin, ang], dim=1)  # (E, 16)
