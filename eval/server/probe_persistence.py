"""Decisive persistence probe: send a BYTE-IDENTICAL real observation to the live
Trossen server N times and measure the commanded action magnitude per call, under two
session lifecycles — WITHOUT stepping any simulator.

  * PERSISTENT: one session_id for all N calls. The server accumulates its frame
    window, advances current_start_frame, retains KV cache, and (if the path exists)
    feeds its own predicted video back. If magnitude SHRINKS across calls on identical
    input, the collapse is persistent session/temporal state — not closed-loop dynamics
    or visual OOD.
  * FRESH: a new session_id every call (server resets its per-session buffers). Each
    call is a fresh single-frame start. Magnitude should stay ~flat (~the 2x2 ~0.16).

This isolates "persistent conditioning" from "closed-loop covariate shift": the input
never changes, so any decay is attributable to the session lifecycle alone.

Run inside the server pod (openpi_client on PATH, /data mounted):
    python eval/server/probe_persistence.py --host localhost --port 8001 --n 12
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from openpi_client import websocket_client_policy

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # arm joints (skip grippers 6/13, base 14/15)


def _parse(resp) -> np.ndarray:
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


def _mag(a: np.ndarray, state: np.ndarray) -> tuple[float, float]:
    first = float(np.mean([abs(a[0, j] - state[j]) for j in ARM]))
    full = float(np.mean([np.mean(np.abs(a[:, j] - state[j])) for j in ARM]))
    return first, full


def _obs(frame3: np.ndarray, state: np.ndarray, instr: str, sid: str) -> dict:
    return {
        "observation/exterior_image_1_left": frame3[0],
        "observation/wrist_image_left": frame3[1],
        "observation/wrist_image_right": frame3[2],
        "observation/state": np.asarray(state, dtype=np.float64),
        "prompt": instr,
        "session_id": sid,
        "endpoint": "infer",
    }


def _run(client, frame3, state, instr, n, sid_fn, label):
    print(f"\n[{label}] identical input x{n}:", flush=True)
    firsts = []
    for i in range(n):
        a = _parse(client.infer(_obs(frame3, state, instr, sid_fn(i))))
        f, fu = _mag(a, state)
        firsts.append(f)
        print(f"  call {i:02d}: first={f:.4f} full={fu:.4f} shape={tuple(a.shape)}", flush=True)
    print(f"[{label}] first-action magnitude: call0={firsts[0]:.4f} "
          f"call{n-1}={firsts[-1]:.4f} ratio={firsts[-1]/max(firsts[0],1e-6):.2f}", flush=True)
    return firsts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--dir", default="/data/wam/diag2x2")
    ap.add_argument("--pose", type=int, default=0)
    ap.add_argument("--frames", choices=["real", "sim"], default="real",
                    help="which rendered source to feed (real_frames.npy vs sim_frames.npy)")
    ap.add_argument("--seq", action="store_true",
                    help="feed a CHANGING sequence (pose i per call) instead of one identical frame")
    args = ap.parse_args()

    allf = np.load(f"{args.dir}/{args.frames}_frames.npy")   # (K, 3, H, W, 3)
    alls = np.load(f"{args.dir}/real_states.npy")            # (K, 16) — poses are anchored to real states
    K = allf.shape[0]
    instr = os.environ.get("WAM_2X2_INSTR",
                           "pick up the two blue ethernet cables from the rack and plug each into a port on the network switch")

    def frame_of(i):
        return allf[i % K] if args.seq else allf[args.pose]

    def state_of(i):
        # In seq mode the state tracks the frame's pose; else fixed.
        return alls[i % K] if args.seq else alls[args.pose]

    mode = f"{args.frames}/{'SEQUENCE' if args.seq else 'identical'}"
    print(f"[probe] source={mode} K={K} pose={args.pose} frame={allf.shape[1:]}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    def run_regime(sid_fn, label):
        print(f"\n[{label}] {mode} x{args.n}:", flush=True)
        firsts = []
        for i in range(args.n):
            a = _parse(client.infer(_obs(frame_of(i), state_of(i), instr, sid_fn(i))))
            f, fu = _mag(a, state_of(i))
            firsts.append(f)
            print(f"  call {i:02d}: first={f:.4f} full={fu:.4f}", flush=True)
        print(f"[{label}] first mag: call0={firsts[0]:.4f} call{args.n-1}={firsts[-1]:.4f} "
              f"ratio={firsts[-1]/max(firsts[0],1e-6):.2f} min={min(firsts):.4f}", flush=True)
        return firsts

    persist = run_regime(lambda i: "probe-persist", "PERSISTENT")
    fresh = run_regime(lambda i: f"probe-fresh-{i}", "FRESH")

    print(f"\n[probe] ===== SUMMARY {mode} (first-action arm magnitude, rad) =====", flush=True)
    print(f"  PERSISTENT: {[round(x,3) for x in persist]}", flush=True)
    print(f"  FRESH:      {[round(x,3) for x in fresh]}", flush=True)
    pr = persist[-1] / max(persist[0], 1e-6)
    print(f"[probe] persist call0->callN ratio={pr:.2f}; persist_min={min(persist):.4f} "
          f"fresh_min={min(fresh):.4f}", flush=True)


if __name__ == "__main__":
    main()
