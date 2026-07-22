#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Episode-start audit for the v6 Trossen dataset (#2 of the cold-start investigation).

Quantifies, over pick-place episodes: the idle-prefix length (state motion-onset),
the ACTION "go"-onset (first commanded motion), the frequency of pure-idle training
windows, and whether trimming to the onset yields non-idle start anchors (alignment).
Read-only. Run on a pod that mounts /data (pandas + numpy)."""

from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

ROOT = os.environ.get("V6_ROOT", "/data/wam/datasets/encord_trossen_v6")
H = 24
STATE_THR = 0.10   # arm rad moved from start => state motion onset
ACT_THR = 0.05     # commanded |action-state| over arm => a real "go" command
IDLE_WIN_THR = 0.03


def main() -> None:
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{ROOT}/meta/tasks.jsonl") if l.strip()}
    want = [i for i, t in tasks.items()
            if ("yellow cylind" in t.lower() or ("pick up" in t.lower() and "place" in t.lower()))]
    files = sorted(glob.glob(f"{ROOT}/data/chunk-000/episode_*.parquet"))
    so_l, ao_l, iwf_l, trim_l = [], [], [], []
    n = 0
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["task_index", "action", "observation.state"])
        except Exception:
            continue
        if int(d["task_index"].iloc[0]) not in want:
            continue
        a = np.stack(d["action"].to_numpy()).astype(np.float32)
        s = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
        if len(a) < H + 20:
            continue
        ds = np.abs(s[:, :6] - s[0, :6]).mean(1)
        so = np.where(ds > STATE_THR)[0]
        so = int(so[0]) if len(so) else len(s)
        da = np.abs(a[:, :6] - s[:, :6]).mean(1)
        ao = np.where(da > ACT_THR)[0]
        ao = int(ao[0]) if len(ao) else len(a)
        so_l.append(so); ao_l.append(ao)
        starts = list(range(0, len(a) - H, 4))
        idle = sum(1 for st0 in starts if float(da[st0:st0 + H].max()) < IDLE_WIN_THR)
        iwf_l.append(idle / max(len(starts), 1))
        if so < len(a):
            trim_l.append(float(np.abs(a[so, :6] - s[so, :6]).mean()))
        n += 1
        if n >= 60:
            break
    so, ao, iwf = np.array(so_l), np.array(ao_l), np.array(iwf_l)
    print(f"DEEP EPISODE-START AUDIT over {n} pick-place episodes:")
    print(f"  STATE motion-onset idx : mean={so.mean():.0f} median={np.median(so):.0f} p90={np.percentile(so,90):.0f}")
    print(f"  ACTION go-onset idx    : mean={ao.mean():.0f} median={np.median(ao):.0f} p90={np.percentile(ao,90):.0f}")
    print(f"  action-onset - state-onset (frames): mean={ (ao-so).mean():.0f }")
    print(f"  PURE-IDLE window fraction (stride-4, {H}-len): mean={100*iwf.mean():.0f}% median={100*np.median(iwf):.0f}%")
    tv = np.mean(trim_l) if trim_l else float("nan")
    print(f"  trim-alignment: mean |action-state| at onset anchor = {tv:.3f} rad (>{ACT_THR} => trimming gives non-idle starts)")


if __name__ == "__main__":
    main()
