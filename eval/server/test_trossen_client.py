"""Synthetic smoke test for the Trossen DreamZero server (no Isaac Sim needed).

Sends a few fake Trossen observations (3 random RGB views + a 16-dim state +
prompt) through the same openpi websocket client the Arena adapter uses, and
asserts the server returns an ``(N, 16)`` action chunk. This validates the
server + checkpoint + wire contract before the Isaac-Sim embodiment exists.

    python eval/server/test_trossen_client.py --host localhost --port 8001 --steps 3
"""

from __future__ import annotations

import argparse

import numpy as np
from openpi_client import websocket_client_policy

EXPECTED_ACTION_DIM = 16


def _fake_obs(h: int = 480, w: int = 640) -> dict:
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return {
        "observation/exterior_image_1_left": img(),
        "observation/wrist_image_left": img(),
        "observation/wrist_image_right": img(),
        "observation/state": rng.standard_normal(EXPECTED_ACTION_DIM).astype(np.float64),
        "prompt": "pick up the cube and place it in the bowl",
        "session_id": "trossen-smoke",
        # roboarena WebsocketPolicyServer routes by this key ("infer" vs "reset").
        "endpoint": "infer",
    }


def _parse_action(resp) -> np.ndarray:
    if isinstance(resp, dict):
        for k in ("actions", "action", "action.action"):
            if k in resp:
                arr = np.asarray(resp[k], dtype=np.float32)
                break
        else:
            arr = np.asarray(next(iter(resp.values())), dtype=np.float32)
    else:
        arr = np.asarray(resp, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[test] connected to {args.host}:{args.port}")

    for step in range(args.steps):
        resp = client.infer(_fake_obs())
        action = _parse_action(resp)
        print(f"[test] step {step}: action shape={action.shape} "
              f"min={action.min():.3f} max={action.max():.3f}")
        assert action.ndim == 2 and action.shape[1] == EXPECTED_ACTION_DIM, (
            f"expected (N, {EXPECTED_ACTION_DIM}); got {action.shape}"
        )

    print("[test] OK — server returns 16-dim Trossen action chunks.")


if __name__ == "__main__":
    main()
