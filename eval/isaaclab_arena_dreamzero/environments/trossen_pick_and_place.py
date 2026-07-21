# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Trossen bimanual pick-and-place environment for Isaac Lab-Arena.

Modeled on Arena's pick_and_place_maple_table_environment, but built for the TALL
Trossen mobile robot (arms at z~1.0-1.2 m): the table is RAISED into the arms'
reach so the task is physically possible (the stock Franka env's table sits at
ground level, ~1.2 m below the Trossen grippers — objects unreachable).

`TrossenWorkTable` = the maple-table USD lifted + placed in front of the robot via
a class-level initial_pose (the height is iterated against the debug render so the
surface lands in the gripper workspace). Objects are placed On() that surface.

Env knobs come through arena_env_args (embodiment, pick_up_object,
destination_location, hdr, light_intensity, + table_x/table_z overrides).
Tune the table pose with WAM_TABLE_X / WAM_TABLE_Z / WAM_TABLE_Y env vars while
iterating, so we don't rebuild the image for a number tweak.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Table pose (front of robot + raised into the arm workspace). Overridable via env
# while iterating with the debug render. Defaults: in front (x), centered (y),
# raised so the surface lands ~0.78 m (Trossen grippers reach ~0.5-0.9 m down/forward).
_TABLE_X = float(os.environ.get("WAM_TABLE_X", "0.20"))
_TABLE_Y = float(os.environ.get("WAM_TABLE_Y", "0.0"))
# Raised to the arm's actual working height (rest-pose gripper ~z1.32): the exterior
# camera then frames the table + objects like the real Encord dataset (vs a low table
# the camera barely catches). Verified against the real first-frame exterior image.
_TABLE_Z = float(os.environ.get("WAM_TABLE_Z", "0.96"))

# Explicit object placement (world coords) so the cube + bowl land CLOSE and centered
# in front of the robot -- within the arms' grasp and clearly in the exterior camera --
# instead of the On() solver scattering them across the whole table top. The table's
# usable top spans world x ~0.44..0.90 at WAM_TABLE_X=0.20; both objects MUST sit inside
# that (x>=~0.45) or they spawn past the front edge and drop to the floor. The cube sits
# right at the grippers' reach (x~0.50) and slightly left; the bowl to the right.
# Tunable via env for fast iteration.
_CUBE_X = float(os.environ.get("WAM_CUBE_X", "0.70"))
_CUBE_Y = float(os.environ.get("WAM_CUBE_Y", "0.10"))
_BOWL_X = float(os.environ.get("WAM_BOWL_X", "0.70"))
_BOWL_Y = float(os.environ.get("WAM_BOWL_Y", "-0.16"))
_OBJ_Z = float(os.environ.get("WAM_OBJ_Z", "1.06"))  # just above the raised table surface (~z1.0)


@register_asset
class TrossenWorkTable(LibraryBackground):
    """The Arena maple table, raised + placed in front of the Trossen so its surface
    is within the arms' reach (vs the stock ground-level placement)."""

    name = "trossen_work_table"
    tags = ["background", "robolab", "trossen"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/scenes/maple_table.usda"
    object_min_z = -0.05
    initial_pose = Pose(position_xyz=(_TABLE_X, _TABLE_Y, _TABLE_Z))


@register_environment
class TrossenPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "trossen_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> "IsaacLabArenaEnvironment":
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        # Raised work table. The cube + bowl are pinned to explicit, close, reachable
        # spots (set_initial_pose) directly in front of the robot -- NOT scattered by the
        # On() solver -- so the cube is always graspable and centered in the camera.
        background = self.asset_registry.get_asset_by_name("trossen_work_table")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.pick_up_object)()
        destination_location = self.asset_registry.get_asset_by_name(args_cli.destination_location)()

        pick_up_object.set_initial_pose(Pose(position_xyz=(_CUBE_X, _CUBE_Y, _OBJ_Z)))
        destination_location.set_initial_pose(Pose(position_xyz=(_BOWL_X, _BOWL_Y, _OBJ_Z)))

        # Domain-match refinements (grounded in the real Encord frames): (1) shrink the Arena
        # sorting bin (registered at scale 4x2) toward a shallow blue tray; (2) tint the pick
        # object bright yellow like the real Amazon-Basics batteries. Both env-gated + guarded so
        # a spawn-cfg quirk can't break the render.
        try:
            destination_location.scale = (
                float(os.environ.get("WAM_BIN_SX", "1.3")),
                float(os.environ.get("WAM_BIN_SY", "1.1")),
                float(os.environ.get("WAM_BIN_SZ", "0.6")),
            )
        except Exception as _e:  # noqa: BLE001
            print(f"[trossen_env] bin scale override skipped: {_e}", flush=True)
        if os.environ.get("WAM_OBJ_YELLOW", "1") == "1":
            try:
                pick_up_object.spawn_cfg_addon = {
                    "visual_material": sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.72, 0.05), roughness=0.5, metallic=0.0),
                    "visual_material_path": "material",
                }
            except Exception as _e:  # noqa: BLE001
                print(f"[trossen_env] object yellow tint skipped: {_e}", flush=True)

        # Table anchor kept so any extra clutter objects can still be placed via On().
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/trossen_work_table/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in args_cli.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Lighting. Domain-MATCH to the real Encord scene: warm, even indoor light (~4500K
        # fluorescent) rather than a bright neutral studio HDR — the real frames the model
        # trained on are warm-white. color_temperature is env-tunable while iterating on the
        # render. An HDR is only added if explicitly requested (domain-randomization path);
        # for the matched demo we leave it off so the fixed warm dome dominates.
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(
                intensity=args_cli.light_intensity,
                color_temperature=float(os.environ.get("WAM_LIGHT_TEMP", "4500")),
                enable_color_temperature=True,
            ),
        )
        if getattr(args_cli, "hdr", None):
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        scene = Scene(
            assets=[background, light, pick_up_object, destination_location, table_reference,
                    *additional_table_objects]
        )
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=float(getattr(args_cli, "episode_length", 20.0)),
        )

        # Third-person viewport looking at the raised workspace.
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.6, 0.0, 1.3), lookat=(_TABLE_X, 0.0, _TABLE_Z))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=embodiment, scene=scene, task=task, env_cfg_callback=_set_viewer_cfg
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="trossen_mobile_ai")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=500.0)
        parser.add_argument("--pick_up_object", type=str, default="rubiks_cube_hot3d_robolab")
        parser.add_argument("--destination_location", type=str, default="bowl_ycb_robolab")
        parser.add_argument("--episode_length", type=float, default=20.0)
        parser.add_argument("--additional_table_objects", nargs="*", type=str, default=[])
