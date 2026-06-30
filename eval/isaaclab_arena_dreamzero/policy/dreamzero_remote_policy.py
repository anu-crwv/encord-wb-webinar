# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero remote closed-loop policy for Isaac Lab-Arena.

Connects Arena's policy runner to a DreamZero (groot) checkpoint served over the
openpi websocket protocol (the same ``DZ_HOST``/``DZ_PORT`` server the DROID eval
in ``deploy/cks/scripts/droid_weave_eval.py`` talks to). It is parameterized by a
``DreamZeroEmbodimentAdapter`` that owns the embodiment-specific obs extraction
and request packing, exactly like ``isaaclab_arena_openpi``'s
``Pi0EmbodimentAdapter`` — see ``trossen_adapter.py`` for the Trossen mapping.

Modeled on ``isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy``;
the differences from pi0 are:
  * the open-loop horizon is a config field (DreamZero has no per-variant table);
  * the response parser also accepts DreamZero's ``action`` / split
    ``action.*`` shapes, not just openpi's ``actions``.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import os

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any

import websockets.exceptions
from openpi_client import websocket_client_policy

# Optional Weave tracing: @_weave_op traces each server call as a rollout op when
# weave.init has run (see weave_eval.py); a no-op decorator otherwise.
try:
    import weave

    def _weave_op(*a, **k):
        return weave.op(*a, **k)
except Exception:  # noqa: BLE001
    def _weave_op(*a, **k):
        def _wrap(f):
            return f
        return _wrap(a[0]) if a and callable(a[0]) else _wrap

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_dreamzero.policy.dreamzero_remote_config import (
    MAX_RECONNECT_ATTEMPTS,
    DreamZeroRemotePolicyArgs,
)


class DreamZeroEmbodimentAdapter(ABC):
    """Translates between Arena's gym observation dict and the DreamZero wire
    format for a specific embodiment (Trossen, DROID, ...).

    Subclasses declare the embodiment-specific action dimension, the Arena
    observation keys they read, and how to pack the server request. The policy
    treats the :meth:`extract` return value as opaque and round-trips it through
    :meth:`pack_request`.
    """

    action_dim: int

    @abstractmethod
    def extract(self, observation: dict[str, Any], env_id: int) -> Any:
        """Pull a single env's tensors out of the Arena gym observation dict.

        ``env_id`` selects the per-env slice; the DreamZero server (like openpi)
        takes one observation per request, so the policy loops over envs.
        """

    @abstractmethod
    def pack_request(self, extracted: Any, language_instruction: str) -> dict[str, Any]:
        """Build the wire-format request payload the DreamZero server expects."""


class DreamZeroRemotePolicy(PolicyBase):
    """DreamZero remote closed-loop policy, parameterized by an embodiment adapter.

    Straight chunk replay: fetch one ``(open_loop_horizon, action_dim)`` chunk
    per env from the server and yield rows in order until exhausted.
    """

    name = "dreamzero_remote"
    config_class = DreamZeroRemotePolicyArgs

    def __init__(
        self,
        config: DreamZeroRemotePolicyArgs,
        embodiment_adapter: DreamZeroEmbodimentAdapter,
    ) -> None:
        super().__init__(config)
        self._adapter = embodiment_adapter
        self._open_loop_horizon = int(config.open_loop_horizon)
        assert self._open_loop_horizon > 0, "open_loop_horizon must be positive"
        self.device = config.policy_device

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port

        print(
            f"[DreamZeroRemotePolicy] Connecting to DreamZero server at "
            f"{self._remote_host}:{self._remote_port} (adapter={config.embodiment_adapter}, "
            f"action_dim={embodiment_adapter.action_dim}, horizon={self._open_loop_horizon}) ..."
        )
        self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._remote_host, port=self._remote_port
        )
        print("[DreamZeroRemotePolicy] Connected.")

        # Per-env action cache, lazy-allocated once num_envs is known.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self.task_description: str | None = None

        # Per-episode video (sim 3-cam strip + WAM dream, synced side-by-side) -> Weave.
        self._last_dream_frames: list = []
        self._video = None
        try:
            import os as _os

            from isaaclab_arena_dreamzero.video_logging import TrossenVideoLogger

            art = _os.environ.get("LORA_ARTIFACT", "")
            model_label = f"dreamzero-trossen-lora:{art.rsplit(':', 1)[-1]}" if ":" in art else "dreamzero-trossen-lora"
            out_dir = _os.environ.get("WAM_EVAL_VIDEO_DIR", "/eval/videos/trossen_rollouts")
            self._video = TrossenVideoLogger(out_dir=out_dir, model_label=model_label)
        except Exception as _e:  # noqa: BLE001
            print(f"[DreamZeroRemotePolicy] video logging disabled: {_e}", flush=True)

    # ---------------------- CLI / config plumbing -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "DreamZero Remote Policy",
            "Arguments for the DreamZero (groot) websocket client.",
        )
        group.add_argument(
            "--embodiment_adapter",
            type=str,
            default="trossen",
            choices=["trossen"],
            help="DreamZero-side embodiment adapter for obs/action wire format (default: trossen).",
        )
        group.add_argument("--policy_device", type=str, default="cuda", help="Torch device for action tensors.")
        group.add_argument("--remote_host", type=str, default="localhost", help="DreamZero server host.")
        group.add_argument("--remote_port", type=int, default=8001, help="DreamZero server port.")
        group.add_argument(
            "--open_loop_horizon",
            type=int,
            default=DreamZeroRemotePolicyArgs.open_loop_horizon,
            help="Replay this many action rows before refetching a chunk.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> DreamZeroRemotePolicy:
        adapter = _resolve_embodiment_adapter(args.embodiment_adapter)
        return DreamZeroRemotePolicy(
            DreamZeroRemotePolicyArgs(
                embodiment_adapter=args.embodiment_adapter,
                policy_device=args.policy_device,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                open_loop_horizon=args.open_loop_horizon,
            ),
            embodiment_adapter=adapter,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> DreamZeroRemotePolicy:
        """JSON-jobs-config path used by Arena's eval_runner.

        Overrides ``PolicyBase.from_dict`` because our ``__init__`` takes an
        adapter alongside the config dataclass (same shape as Pi0RemotePolicy).
        """
        config_dict = dict(config_dict)
        adapter_key = config_dict.pop("embodiment_adapter", "trossen")
        adapter = _resolve_embodiment_adapter(adapter_key)
        return cls(DreamZeroRemotePolicyArgs(embodiment_adapter=adapter_key, **config_dict), embodiment_adapter=adapter)

    # ---------------------- Policy interface -------------------

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "DreamZeroRemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        num_envs = env.unwrapped.num_envs
        self._maybe_init_per_env_state(num_envs)

        # The server takes one obs per request; loop over envs and refetch a
        # chunk for any env whose buffer is exhausted.
        actions = []
        for env_id in range(num_envs):
            chunk_exhausted = (
                self._cached_action_chunks[env_id] is None
                or self._next_chunk_steps[env_id] >= self._open_loop_horizon
            )
            if chunk_exhausted:
                self._cached_action_chunks[env_id] = self._fetch_action_chunk(observation, env_id)
                self._next_chunk_steps[env_id] = 0
            actions.append(self._cached_action_chunks[env_id][self._next_chunk_steps[env_id]])
            self._next_chunk_steps[env_id] += 1
            # Video capture for env 0: sim 3-cam strip + the synced dream frame for
            # this chunk step (chunk_index = the index we just consumed).
            if self._video is not None and env_id == 0:
                self._record_video_step(observation, chunk_index=self._next_chunk_steps[0] - 1)

        batch = np.stack(actions)  # (num_envs, action_dim)

        if os.environ.get("WAM_DEBUG_ACTIONS"):
            self._dbg_step = getattr(self, "_dbg_step", 0)
            if self._dbg_step == 0:
                # Lazy-load the real-data per-dim state distribution [min,p1,p99,max,mean].
                self._real_stats = None
                try:
                    self._real_stats = np.load(os.path.join(
                        os.environ.get("WAM_EVAL_SRC", "."), "real_state_stats.npy"))
                except Exception as e:  # noqa: BLE001
                    print(f"[ACTDBG] real_state_stats load failed: {e}", flush=True)
                self._labels = ["Lj0", "Lj1", "Lj2", "Lj3", "Lj4", "Lj5", "Lgrip",
                                "Rj0", "Rj1", "Rj2", "Rj3", "Rj4", "Rj5", "Rgrip", "linv", "angv"]
            if self._dbg_step % 10 == 0:
                ex = self._adapter.extract(observation, 0)
                st = np.asarray(ex.state, dtype=float)
                print(f"[ACTDBG s={self._dbg_step}] state_sent(16)={np.round(st, 3).tolist()}", flush=True)
                # Flag any state dim outside the real [p1, p99] band -> out-of-distribution.
                if getattr(self, "_real_stats", None) is not None and st.shape[0] == self._real_stats.shape[1]:
                    p1, p99 = self._real_stats[1], self._real_stats[2]
                    ood = [f"{self._labels[i]}={st[i]:+.3f}(real[{p1[i]:+.2f},{p99[i]:+.2f}])"
                           for i in range(len(st)) if st[i] < p1[i] - 0.05 or st[i] > p99[i] + 0.05]
                    print(f"[ACTDBG s={self._dbg_step}] OOD dims: {ood if ood else 'NONE (all in real range)'}", flush=True)
            self._dbg_step += 1

        return torch.from_numpy(batch).to(dtype=torch.float32, device=self.device)

    def _record_video_step(self, observation: dict[str, Any], chunk_index: int) -> None:
        try:
            tobs = self._adapter.extract(observation, 0)
            self._video.record_step(
                {
                    "exterior": tobs.exterior_image_1_left,
                    "wrist_left": tobs.wrist_image_left,
                    "wrist_right": tobs.wrist_image_right,
                },
                self._last_dream_frames,
                chunk_index,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[DreamZeroRemotePolicy] video record_step skipped: {e}", flush=True)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Episode boundary: clear the per-episode video frame buffers. The eval
        # driver (run_trossen_eval.run_episode) writes the videos after the rollout.
        if self._video is not None:
            self._video.reset_episode()
        self._last_dream_frames = []
        if self._cached_action_chunks is None:
            return
        ids = range(len(self._cached_action_chunks)) if env_ids is None else env_ids.reshape(-1).tolist()
        for env_id in ids:
            self._cached_action_chunks[env_id] = None
            self._next_chunk_steps[env_id] = 0

    def close(self) -> None:
        """Release the local websocket connection. Does NOT stop the server
        process, which runs in a separate container and outlives this client."""
        _close_websocket_best_effort(self._websocket_client)
        self._websocket_client = None

    # ---------------------- internals -------------------

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs, (
            f"DreamZeroRemotePolicy num_envs changed from {len(self._cached_action_chunks)}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )

    @_weave_op(name="trossen_fetch_action_chunk")
    def _fetch_action_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        extracted = self._adapter.extract(observation, env_id)
        request = self._adapter.pack_request(extracted, self.task_description)
        # The DreamZero server uses the roboarena WebsocketPolicyServer, which routes
        # by an "endpoint" key ("infer" vs "reset"); the openpi websocket client does
        # not add it, so set it here (a server-protocol concern, not embodiment-specific).
        request["endpoint"] = "infer"
        response = self._call_server_with_retry(request)

        # Stash the WAM "dream" video for this chunk (server packs it under
        # __debug__ when DZ_DREAM_VIDEO=1) so get_action can sync it per step.
        self._last_dream_frames = []
        if self._video is not None and isinstance(response, dict):
            dbg = response.get("__debug__")
            if isinstance(dbg, dict) and dbg.get("dream_video_mp4"):
                from isaaclab_arena_dreamzero.video_logging import decode_dream_mp4

                self._last_dream_frames = decode_dream_mp4(dbg["dream_video_mp4"])

        chunk = _parse_action_response(response, self._adapter.action_dim)
        assert chunk.shape[0] >= self._open_loop_horizon, (
            f"Server returned horizon {chunk.shape[0]} < configured open_loop_horizon {self._open_loop_horizon}"
        )
        return chunk[: self._open_loop_horizon].astype(np.float32, copy=True)

    def _call_server_with_retry(self, server_request: dict[str, Any]) -> dict[str, Any]:
        """Send the request, reconnecting up to ``MAX_RECONNECT_ATTEMPTS`` times.

        On reconnect, flush every env's cached chunk so the next ``get_action``
        re-queries with a fresh observation rather than replaying stale actions.
        """
        for attempt_index in range(MAX_RECONNECT_ATTEMPTS):
            try:
                return self._websocket_client.infer(server_request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as exc:
                is_last_attempt = (attempt_index + 1) >= MAX_RECONNECT_ATTEMPTS
                if is_last_attempt:
                    raise
                print(
                    f"[DreamZeroRemotePolicy] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                _close_websocket_best_effort(self._websocket_client)
                self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
                    host=self._remote_host, port=self._remote_port
                )
                if self._cached_action_chunks is not None:
                    for i in range(len(self._cached_action_chunks)):
                        self._cached_action_chunks[i] = None
                        self._next_chunk_steps[i] = 0
        raise RuntimeError("unreachable")


def _parse_action_response(response: Any, action_dim: int) -> np.ndarray:
    """Normalize the DreamZero server's reply into an ``(H, action_dim)`` array.

    Accepts the response shapes the DROID eval's client already handles
    (``deploy/cks/scripts/droid_weave_eval.py``):
      * ``{"actions": (H, D)}``            — openpi convention
      * ``{"action":  (H, D)}`` or ``(D,)``
      * ``{"action.joint_position", "action.gripper_position"}`` — split DROID form
      * a bare array
    """
    if isinstance(response, dict) and "action.joint_position" in response:
        jp = np.asarray(response["action.joint_position"], dtype=np.float32)
        gp = np.asarray(response["action.gripper_position"], dtype=np.float32).reshape(-1, 1)
        if jp.ndim == 1:
            jp = jp.reshape(1, -1)
        chunk = np.concatenate([jp, gp], axis=-1)
    elif isinstance(response, dict) and "actions" in response:
        chunk = np.asarray(response["actions"], dtype=np.float32)
    elif isinstance(response, dict) and "action" in response:
        chunk = np.asarray(response["action"], dtype=np.float32)
    else:
        chunk = np.asarray(response, dtype=np.float32)

    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    assert chunk.ndim == 2 and chunk.shape[1] == action_dim, (
        f"Expected actions of shape (H, {action_dim}); got {chunk.shape}"
    )
    return chunk


def _close_websocket_best_effort(client: websocket_client_policy.WebsocketClientPolicy | None) -> None:
    if client is None:
        return
    try:
        ws = getattr(client, "_ws", None)
        if ws is not None:
            ws.close()
    except (websockets.exceptions.ConnectionClosed, OSError):
        pass


def _resolve_embodiment_adapter(key: str) -> DreamZeroEmbodimentAdapter:
    """Instantiate the adapter registered under ``key`` (deferred import to
    avoid a circular import at module load)."""
    if key == "trossen":
        from isaaclab_arena_dreamzero.policy.trossen_adapter import DreamZeroTrossenAdapter

        return DreamZeroTrossenAdapter()
    raise ValueError(f"Unknown embodiment_adapter {key!r}; expected 'trossen'")
