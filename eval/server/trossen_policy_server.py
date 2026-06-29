"""DreamZero inference server for the fine-tuned Trossen (bimanual mobile) model.

The Trossen analogue of upstream ``socket_test_optimized_AR.py`` (the DROID
server). It reuses that module's websocket/distributed machinery and the
``ARDroidRoboarenaPolicy`` wrapper, overriding only the embodiment-specific
observation/action conversion for Trossen's contract:

    request  (from isaaclab_arena_dreamzero / DreamZeroTrossenAdapter):
        observation/exterior_image_1_left : (H, W, 3)
        observation/wrist_image_left      : (H, W, 3)
        observation/wrist_image_right     : (H, W, 3)
        observation/state                 : (16,)   packed bimanual state
        prompt                            : str
    model keys (modality_config_trossen):
        video.exterior_image_1_left / video.wrist_image_left / video.wrist_image_right : (T, H, W, 3)
        state.state                       : (1, 16)
        annotation.language.action_text   : str
    response:
        action                            : (N, 16) packed bimanual action

Run (single GH200 is enough; keep the RTX nodes for Isaac Sim):
    torchrun --nproc_per_node=1 eval/server/trossen_policy_server.py \
        --model_path /checkpoints/<trossen-ckpt> --port 8001

The upstream module must be importable (PYTHONPATH includes the dreamzero
clone); ``eval/runner.sh`` and the server manifest set that up.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import socket

import numpy as np
import torch
import torch.distributed as dist
import tyro

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

# Reuse the proven DROID server machinery; we only swap the converters + config.
from socket_test_optimized_AR import (
    Args,
    ARDroidRoboarenaPolicy,
    WebsocketPolicyServer,
    init_mesh,
)
from eval_utils.policy_server import PolicyServerConfig
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer

logger = logging.getLogger(__name__)

# Packed Trossen action/state width (two 6-DOF arms + 2 grippers + base).
TROSSEN_ACTION_DIM = 16

# request key -> model video key. Matches DreamZeroTrossenAdapter.pack_request.
TROSSEN_IMAGE_KEY_MAPPING = {
    "observation/exterior_image_1_left": "video.exterior_image_1_left",
    "observation/wrist_image_left": "video.wrist_image_left",
    "observation/wrist_image_right": "video.wrist_image_right",
}


class TrossenRoboarenaPolicy(ARDroidRoboarenaPolicy):
    """Trossen wrapper: 3 cameras + a single 16-dim packed state/action.

    Inherits frame-accumulation, session handling, distributed broadcast, and
    video saving from ``ARDroidRoboarenaPolicy``; overrides only the per-call
    observation/action format conversion.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Replace the DROID 3-buffer set with the Trossen camera views.
        self._frame_buffers = {dst: [] for dst in TROSSEN_IMAGE_KEY_MAPPING.values()}

    def _convert_observation(self, obs: dict) -> dict:
        converted: dict = {}

        # Accumulate frames per camera view (same windowing as the DROID wrapper:
        # 1 frame on the first call, then FRAMES_PER_CHUNK).
        for req_key, model_key in TROSSEN_IMAGE_KEY_MAPPING.items():
            if req_key in obs:
                data = obs[req_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[model_key].extend(list(data))
                    else:
                        self._frame_buffers[model_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK
        for model_key, buffer in self._frame_buffers.items():
            if len(buffer) == 0:
                continue
            if len(buffer) >= num_frames:
                frames_to_use = buffer[-num_frames:]
            else:
                frames_to_use = buffer.copy()
                while len(frames_to_use) < num_frames:
                    frames_to_use.insert(0, buffer[0])
            converted[model_key] = np.stack(frames_to_use, axis=0)  # (T, H, W, C)

        # Packed 16-dim state -> state.state (1, 16).
        if "observation/state" in obs:
            state = np.asarray(obs["observation/state"], dtype=np.float64).reshape(1, -1)
        else:
            state = np.zeros((1, TROSSEN_ACTION_DIM), dtype=np.float64)
        converted["state.state"] = state

        # Language. The DROID server feeds annotation.language.action_text and the
        # model consumes the text regardless of sub-key; mirror that here.
        converted["annotation.language.action_text"] = obs.get("prompt", "")
        return converted

    def infer(self, obs: dict) -> object:
        """Return the 16-dim action; when DZ_DREAM_VIDEO=1, also pack the WAM's
        decoded dream video (latest chunk) into __debug__ so the eval client can
        sync it side-by-side with the sim rollout (mirrors the DROID eval)."""
        action = super().infer(obs)
        if os.environ.get("DZ_DREAM_VIDEO") == "1" and getattr(self, "video_across_time", None):
            try:
                mp4 = self._decode_latest_dream_mp4()
                if mp4:
                    # Serialize the action as a plain nested list: an ndarray nested in
                    # a dict alongside the dream bytes does not round-trip cleanly through
                    # msgpack_numpy on the client (comes back as a raw nd-dict). A list is
                    # unambiguous; the client's _parse_action_response handles it.
                    return {"action": np.asarray(action, dtype=np.float32).tolist(),
                            "__debug__": {"dream_video_mp4": mp4}}
            except Exception as e:  # noqa: BLE001
                logger.warning("dream video pack failed: %s", e)
        return action

    def _decode_latest_dream_mp4(self) -> bytes | None:
        """VAE-decode the most recent predicted-video chunk to mp4 bytes."""
        import tempfile

        import imageio
        from einops import rearrange

        ah = self._policy.trained_model.action_head
        video_pred = self.video_across_time[-1]
        frames = ah.vae.decode(
            video_pred,
            tiled=ah.tiled,
            tile_size=(ah.tile_size_height, ah.tile_size_width),
            tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
        )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
            imageio.mimsave(tf.name, list(frames), fps=5, codec="libx264")
            with open(tf.name, "rb") as fh:
                return fh.read()

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Model returns a single packed ``action.action`` (N, 16). Return it as
        the roboarena ``action`` array (the client/adapter reads ``action``)."""
        packed = None
        for key, value in action_dict.items():
            if key.endswith("action.action") or key == "action.action" or key.endswith(".action"):
                packed = value
                break
        if packed is None and action_dict:
            # Fallback: first action.* entry.
            packed = next(iter(action_dict.values()))
        if packed is None:
            return np.zeros((1, TROSSEN_ACTION_DIM), dtype=np.float32)
        if isinstance(packed, torch.Tensor):
            packed = packed.cpu().numpy()
        packed = np.asarray(packed, dtype=np.float32)
        if packed.ndim == 1:
            packed = packed.reshape(1, -1)
        return packed


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "trossen"
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()
    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info("Rank %s initialized signal_group (gloo)", rank)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    hostname = socket.gethostname()
    if rank == 0:
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"trossen_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None

    wrapper_policy = TrossenRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
    )

    # Trossen camera contract: 1 exterior + 2 wrist views, 16-dim packed action.
    # The converter above is authoritative for the request keys; this config is
    # the documented client contract + image resize.
    server_config = PolicyServerConfig(
        # Trossen cameras are native 640x480; the eval transform validates raw input
        # resolution against it (VideoToTensor). (height, width).
        image_resolution=(480, 640),
        needs_wrist_camera=True,
        n_external_cameras=1,
        needs_stereo_camera=True,  # exposes wrist_image_right
        needs_session_id=True,
        action_space="joint_position",
    )

    if rank == 0:
        logging.info("Trossen server config: %s", server_config)
        RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        ).serve_forever()
    else:
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
