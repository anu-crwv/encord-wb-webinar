# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Trossen Mobile-AI bimanual embodiment for Isaac Lab-Arena.

Authored from the DROID embodiment template
(isaaclab_arena/embodiments/droid/droid.py) using contracts verified against
mobile_ai.usd + the Encord dataset meta/info.json:

  * articulation root /mobile_ai/base_footprint
  * arms  follower_left_joint_0..5 / follower_right_joint_0..5  (revolute)
  * grippers (prismatic carriages, second mimics first):
        follower_left_left_carriage_joint / follower_right_left_carriage_joint
  * base drive wheels left_wheel / right_wheel
  * camera mount links cam_high_link / follower_left_camera_link / follower_right_camera_link

The action term ORDER below matches the model's 16-dim output order exactly
(see observations.pack_trossen_state_16d):
  left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1) + base(2) = 16.

Items marked TODO(sim) need confirmation when first run in Isaac Sim (USD prim
hierarchy for camera prim_paths, gripper mimic behavior, camera extrinsics, and
whether mobile_ai.usd needs articulation-root / actuator tuning).
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

from isaaclab_arena_dreamzero.embodiments.observations import pack_trossen_state_16d

# Path to mobile_ai.usd; runner.sh fetches trossen_ai_isaac to /shared. Override via env.
TROSSEN_MOBILE_AI_USD = os.environ.get(
    "TROSSEN_MOBILE_AI_USD",
    "/shared/trossen_ai_isaac/assets/robots/mobile_ai/mobile_ai.usd",
)

_LEFT_ARM = [f"follower_left_joint_{i}" for i in range(6)]
_RIGHT_ARM = [f"follower_right_joint_{i}" for i in range(6)]
_LEFT_GRIP = "follower_left_left_carriage_joint"
_RIGHT_GRIP = "follower_right_left_carriage_joint"

# Real Trossen rest/start pose, read from the Encord LeRobot dataset (episode first
# frame; the first action just holds it). Both 6-DOF arms start BENT ("ready"):
# joint_0..5 = [0, 1.047, 0.523, 0.628, 0, 0]. The sim previously spawned at all-zeros
# (arms bolt-straight), which is out-of-distribution for the model -> it computed its
# first actions from a proprio state it never saw in training, corrupting the start of
# every rollout. Spawn at the real rest pose so the model starts in-distribution.
_REST_ARM = [0.0, 1.047, 0.523, 0.628, 0.0, 0.0]  # per arm, joint_0..5
_REST_JOINT_POS = {
    **{jn: _REST_ARM[i] for i, jn in enumerate(_LEFT_ARM)},
    **{jn: _REST_ARM[i] for i, jn in enumerate(_RIGHT_ARM)},
    _LEFT_GRIP: 0.0,
    _RIGHT_GRIP: 0.0,
}


@register_asset
class TrossenMobileAIEmbodiment(EmbodimentBase):
    """Trossen AI mobile bimanual robot, absolute joint-position control (16-dim)."""

    name = "trossen_mobile_ai"
    tags = ["embodiment", "bimanual", "mobile"]
    default_arm_mode = ArmMode.DUAL_ARM if hasattr(ArmMode, "DUAL_ARM") else None

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = TrossenSceneCfg()
        self.action_config = TrossenActionsCfg()
        self.camera_config = TrossenCameraCfg()
        self.observation_config = TrossenObservationsCfg()
        self.event_config = None
        # Seed the arms at the real Trossen rest pose via Arena's setter (it updates
        # scene_config.robot.init_state.joint_pos, which the reset actually honors).
        if initial_joint_pose is not None:
            self.set_joint_initial_pos(dict(zip(_LEFT_ARM + _RIGHT_ARM, initial_joint_pose)))
        else:
            self.set_joint_initial_pos(dict(_REST_JOINT_POS))


@configclass
class TrossenSceneCfg:
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TROSSEN_MOBILE_AI_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
                # Anchor the mobile base for this tabletop eval. The base action is already
                # frozen (scale=0), but the base is a FREE root, so arm reaction forces
                # (the domain-gapped policy flails the arms) were drifting/rotating the whole
                # robot -- the camera wandered off the table and knocked the cube off. Fixing
                # the root keeps the robot planted so the workspace stays framed.
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(0, 0, 0, 1),
            joint_pos=dict(_REST_JOINT_POS),
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            # TODO(sim): tune stiffness/damping; start with position-controlled implicit actuators.
            "left_arm": ImplicitActuatorCfg(joint_names_expr=["follower_left_joint_.*"], stiffness=400.0, damping=80.0),
            "right_arm": ImplicitActuatorCfg(joint_names_expr=["follower_right_joint_.*"], stiffness=400.0, damping=80.0),
            "grippers": ImplicitActuatorCfg(joint_names_expr=[".*_carriage_joint"], stiffness=200.0, damping=20.0),
            "wheels": ImplicitActuatorCfg(joint_names_expr=["left_wheel", "right_wheel"], stiffness=0.0, damping=10.0),
        },
    )


@configclass
class TrossenActionsCfg:
    """16-dim action, term order == model output order (see module docstring)."""

    # [0:6] left arm
    left_arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=list(_LEFT_ARM), preserve_order=True, use_default_offset=False
    )
    # [6] left gripper (primary carriage; the opposite carriage mimics in USD)
    left_gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[_LEFT_GRIP], preserve_order=True, use_default_offset=False
    )
    # [7:13] right arm
    right_arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=list(_RIGHT_ARM), preserve_order=True, use_default_offset=False
    )
    # [13] right gripper
    right_gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[_RIGHT_GRIP], preserve_order=True, use_default_offset=False
    )
    # [14:16] mobile base [linear_vel, angular_vel]. Consumed as 2 wheel-velocity
    # dims and FROZEN (scale=0) for the first tabletop milestone — the base isn't
    # expected to move; replace with a proper diff-drive (v,w)->wheels term for
    # full mobile evaluation. TODO(sim).
    base_action: ActionTermCfg = JointVelocityActionCfg(
        asset_name="robot", joint_names=["left_wheel", "right_wheel"], preserve_order=True, scale=0.0
    )


@configclass
class TrossenObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # The 16-dim packed state the adapter reads as policy["state"].
        state = ObsTerm(func=pack_trossen_state_16d)
        # Kept for parity/debugging; not consumed by the adapter.
        actions = ObsTerm(func=mdp_isaac_lab.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TrossenCameraCfg:
    """3 RGB cameras. Field names -> obs keys "{field}_rgb" consumed by the adapter
    (exterior_image_1_left_rgb / wrist_image_left_rgb / wrist_image_right_rgb).
    Mounted at 640x480 (native Trossen resolution the model expects).
    TODO(sim): verify camera prim_paths against the spawned USD hierarchy + set
    real extrinsics (offsets below are placeholders)."""

    exterior_image_1_left: CameraCfg | TiledCameraCfg = MISSING
    wrist_image_left: CameraCfg | TiledCameraCfg = MISSING
    wrist_image_right: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled = getattr(self, "_is_tiled_camera", True)
        Cam = TiledCameraCfg if is_tiled else CameraCfg
        Off = Cam.OffsetCfg
        common = dict(height=480, width=640, data_types=["rgb"],
                      spawn=sim_utils.PinholeCameraCfg(focal_length=2.1, focus_distance=28.0,
                                                       horizontal_aperture=5.376, vertical_aperture=4.032))
        # Reproduce the REAL Trossen camera frames from mobile_ai.usd (verified via
        # usd_probe.py). The massless *_color_optical_frame prims are pruned when Arena
        # spawns the articulation, so we parent at the real (spawned) camera LINKS and
        # bake the optical rotation into the offset: every *_color_optical_frame sits at
        # (-0.5,0.5,-0.5,0.5) relative to its link (ROS optical, +Z = optical axis), and
        # cam_high_link already carries the ~37deg downward pitch. convention="ros" so
        # the camera looks along the resulting +Z -> the exact real view; wrist cams then
        # track their arm links for free.
        _OPTICAL = (-0.5, 0.5, -0.5, 0.5)  # *_color_optical_frame local rot (w,x,y,z), all 3 cams
        self.exterior_image_1_left = Cam(
            prim_path="{ENV_REGEX_NS}/Robot/cam_high_link/exterior_cam",
            offset=Off(pos=(0.0, 0.0, 0.0), rot=_OPTICAL, convention="ros"), **common)
        self.wrist_image_left = Cam(
            prim_path="{ENV_REGEX_NS}/Robot/follower_left_camera_link/wrist_cam_left",
            offset=Off(pos=(0.0, 0.0, 0.0), rot=_OPTICAL, convention="ros"), **common)
        self.wrist_image_right = Cam(
            prim_path="{ENV_REGEX_NS}/Robot/follower_right_camera_link/wrist_cam_right",
            offset=Off(pos=(0.0, 0.0, 0.0), rot=_OPTICAL, convention="ros"), **common)
