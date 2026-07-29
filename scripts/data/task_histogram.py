#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Task-distribution histogram for a LeRobot Trossen dataset: per-task-index episode
counts + activity-keyword grouping (to pick a dense, sim-buildable eval task).
Read-only. Run on a pod that mounts /data. Usage: task_histogram.py <dataset_root>"""

from __future__ import annotations
import glob, json, sys
from collections import Counter
import pandas as pd

# Activity keywords -> for grouping fine task_indices into buildable-vs-hard activities.
KEYWORDS = [
    "yellow cylind", "screw", "nut", "bolt", "coffee", "mug", "ethernet", "cable",
    "network switch", "rack", "tape", "safety glass", "glue", "towel", "batter",
    "refriger", "fridge", "soda", "can", "tray", "bin", "drawer", "organizer", "basket",
]


def main() -> None:
    root = sys.argv[1]
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{root}/meta/tasks.jsonl") if l.strip()}
    files = sorted(glob.glob(f"{root}/data/chunk-*/episode_*.parquet"))
    per_task = Counter()
    for f in files:
        try:
            per_task[int(pd.read_parquet(f, columns=["task_index"])["task_index"].iloc[0])] += 1
        except Exception:
            pass
    print(f"# {root}")
    print(f"# {len(files)} episodes, {len(tasks)} distinct task strings")
    print(f"# --- top 25 task strings by episode count ---")
    for ti, c in per_task.most_common(25):
        print(f"  {c:4d}  {tasks.get(ti, '?')[:95]}")
    # activity grouping
    act = Counter()
    for ti, c in per_task.items():
        name = tasks.get(ti, "").lower()
        hit = next((k for k in KEYWORDS if k in name), "OTHER")
        act[hit] += c
    print(f"# --- episodes grouped by activity keyword ---")
    for k, c in act.most_common():
        print(f"  {c:4d}  {k}")


if __name__ == "__main__":
    main()
