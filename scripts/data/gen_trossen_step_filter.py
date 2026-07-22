#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Generate meta/step_filter.jsonl for the v6 Trossen dataset (#3 targeted retrain).

The groot sharded loader (lerobot_sharded.py) samples window START anchors uniformly
from `step_filter[trajectory_id]` = all_indices \\ step_indices. Every v6 episode starts
with a long IDLE prefix (state motion-onset median ~70) whose action ~= "hold still";
those idle frames are equally-weighted anchors, so ~half the training windows teach the
model NOT to move. This writes, per episode, the deep-idle prefix indices to REMOVE as
anchors — keeping anchors from (onset - MARGIN) so windows still SPAN the static->motion
transition (oversampling reach-onset relative to idle). Metadata only: no parquet/video
change. MUST emit a line for EVERY episode (missing => KeyError in the loader); untrimmed
episodes get an empty list.

Run on a pod that mounts /data. Writes <ROOT>/meta/step_filter.jsonl."""

from __future__ import annotations
import json, os
import numpy as np, pandas as pd

ROOT = os.environ.get("V6_ROOT", "/data/wam/datasets/encord_trossen_v6")
CHUNK = 1000            # LeRobot chunks_size
MARGIN = 6              # keep this many pre-onset frames as anchors (span static->motion)
STATE_THR = 0.10        # arm rad moved from start => motion onset
H = 24                  # action horizon (max_delta) — leave >=H anchors


def main() -> None:
    eps = [json.loads(l) for l in open(f"{ROOT}/meta/episodes.jsonl") if l.strip()]
    onsets, n_trim, n_total = [], 0, 0
    out_path = f"{ROOT}/meta/step_filter.jsonl"
    with open(out_path, "w") as out:
        for e in eps:
            ei, L = int(e["episode_index"]), int(e["length"])
            pq = f"{ROOT}/data/chunk-{ei // CHUNK:03d}/episode_{ei:06d}.parquet"
            onset = 0
            try:
                s = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
                ds = np.abs(s[:, :6] - s[0, :6]).mean(1)
                idx = np.where(ds > STATE_THR)[0]
                onset = int(idx[0]) if len(idx) else 0
            except Exception as ex:  # noqa: BLE001
                print(f"[stepfilter] ep{ei} read failed ({ex}); no trim", flush=True)
            cut = max(0, onset - MARGIN)
            # Guard: never trim so hard that < H anchors remain (keep the episode usable).
            if cut >= L - H:
                cut = 0
            remove = list(range(0, cut))
            out.write(json.dumps({"episode_index": ei, "step_indices": remove}) + "\n")
            if remove:
                n_trim += 1
            onsets.append(onset)
            n_total += 1
    onsets = np.array(onsets)
    print(f"[stepfilter] wrote {out_path}: {n_total} episodes, {n_trim} trimmed "
          f"({100*n_trim/max(n_total,1):.0f}%); onset median={np.median(onsets):.0f} "
          f"mean={onsets.mean():.0f} p90={np.percentile(onsets,90):.0f}", flush=True)


if __name__ == "__main__":
    main()
