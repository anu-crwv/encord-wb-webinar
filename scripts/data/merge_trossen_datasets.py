#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Concatenate two LeRobot Trossen datasets into one merged trainable dataset.

Merges BASE (kept as episodes 0..N-1) + ADD (re-indexed N..N+M-1) into OUT:
  - hardlinks every parquet + video (same fs -> free; no 150GB duplication),
  - re-indexes ADD's episode_index globally and OFFSETS its task_index by len(BASE tasks)
    (rewriting ADD parquets' task_index column) so tasks.jsonl stays consistent,
  - merges meta/{episodes,tasks}.jsonl, rebuilds info.json totals,
  - re-runs the GEAR converter to recompute stats.json/modality.json/embodiment.json over
    the FULL merged set (norm stats must cover both halves), then re-patches modality's
    3 language keys and restores the merged tasks/episodes (the converter collapses them).

Default use: BASE=encord_trossen_v6 (source-data:v5 + captions:v6, 1363 eps),
ADD=encord_trossen_v4v2 (source-data:v4 + captions:v5, 531 eps) -> encord_trossen_v8 (~1894).
Run on a CPU node with the repo staged (needs the converter + pandas)."""

from __future__ import annotations
import glob, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

BASE = os.environ.get("MERGE_BASE", "/data/wam/datasets/encord_trossen_v6")
ADD = os.environ.get("MERGE_ADD", "/data/wam/datasets/encord_trossen_v4v2")
OUT = os.environ.get("MERGE_OUT", "/data/wam/datasets/encord_trossen_v8")
REPO = os.environ.get("WAM_REPO_ROOT", "/data/src/dreamzero-wam")
CONVERT = "scripts/data/convert_lerobot_to_gear.py"
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]
LANG_KEYS = ["annotation.language.language_instruction",
             "annotation.language.language_instruction_2",
             "annotation.language.language_instruction_3"]


def _hardlink(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    base, add, out = Path(BASE), Path(ADD), Path(OUT)
    info = json.loads((base / "meta/info.json").read_text())
    CHUNK = int(info.get("chunks_size") or 1000)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    base_eps = [json.loads(l) for l in open(base / "meta/episodes.jsonl") if l.strip()]
    base_tasks = [json.loads(l) for l in open(base / "meta/tasks.jsonl") if l.strip()]
    add_eps = [json.loads(l) for l in open(add / "meta/episodes.jsonl") if l.strip()]
    add_tasks = [json.loads(l) for l in open(add / "meta/tasks.jsonl") if l.strip()]
    n_base = len(base_eps)
    ntask_base = max(t["task_index"] for t in base_tasks) + 1
    print(f"[merge] BASE {n_base} eps / {len(base_tasks)} tasks  +  ADD {len(add_eps)} eps / {len(add_tasks)} tasks", flush=True)

    # 1. BASE: hardlink all parquet + videos verbatim (episodes 0..n_base-1).
    for f in glob.glob(str(base / "data/chunk-*/episode_*.parquet")):
        _hardlink(f, str(out / os.path.relpath(f, base)))
    for f in glob.glob(str(base / "videos/chunk-*/*/episode_*.mp4")):
        _hardlink(f, str(out / os.path.relpath(f, base)))

    # 2. ADD: re-index episode + offset task_index; rewrite parquet, hardlink videos.
    merged_eps = list(base_eps)
    for e in add_eps:
        oi = int(e["episode_index"]); ni = n_base + oi
        och, nch = f"chunk-{oi // CHUNK:03d}", f"chunk-{ni // CHUNK:03d}"
        df = pd.read_parquet(add / f"data/{och}/episode_{oi:06d}.parquet")
        if "task_index" in df.columns:
            df["task_index"] = df["task_index"] + ntask_base
        if "episode_index" in df.columns:
            df["episode_index"] = ni
        (out / f"data/{nch}").mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"data/{nch}/episode_{ni:06d}.parquet")
        for k in VIDEO_KEYS:
            _hardlink(str(add / f"videos/{och}/observation.images.{k}/episode_{oi:06d}.mp4"),
                      str(out / f"videos/{nch}/observation.images.{k}/episode_{ni:06d}.mp4"))
        ne = dict(e); ne["episode_index"] = ni
        merged_eps.append(ne)

    merged_tasks = list(base_tasks) + [{**t, "task_index": t["task_index"] + ntask_base} for t in add_tasks]
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in merged_tasks))
    info.update(total_episodes=len(merged_eps),
                total_frames=int(sum(e["length"] for e in merged_eps)),
                total_tasks=len(merged_tasks),
                total_videos=len(merged_eps) * len(VIDEO_KEYS),
                total_chunks=(len(merged_eps) - 1) // CHUNK + 1,
                splits={"train": f"0:{len(merged_eps)}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"[merge] concatenated -> {len(merged_eps)} eps, {info['total_frames']} frames, {len(merged_tasks)} tasks", flush=True)

    # 3. Recompute stats/modality/embodiment over the FULL merged set.
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    # 4. Re-patch modality annotation (converter handles one task key) + restore merged meta.
    mod_path = out / "meta/modality.json"
    mod = json.loads(mod_path.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    mod_path.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in merged_tasks))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))
    print(f"[merge] done -> {out} ({len(merged_eps)} eps). Re-run gen_trossen_step_filter.py next.", flush=True)


if __name__ == "__main__":
    main()
