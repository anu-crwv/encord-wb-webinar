#!/usr/bin/env python3
"""Merge the two Encord artifacts into one trainable LeRobot v2.0 dataset for the `trossen` embodiment.

The Encord export is split across two W&B artifacts:
  - encord-labels:v1       -> data/chunk-000/episode_*.parquet (action[16] + observation.state[16] +
                              task_index + annotation.language.language_instruction{,_2,_3}) + meta
  - encord-source-data:v1  -> videos/chunk-000/<cam>/episode_*.mp4   (the actual camera streams)

Neither alone is trainable. This script downloads both, merges them into one LeRobot dataset dir
(parquet from labels + videos from source-data), writes a self-consistent meta/info.json, then runs
scripts/data/convert_lerobot_to_gear.py to produce modality.json / embodiment.json / stats.json, and
finally patches modality.json's language section to mirror oxe_droid's 3 language keys.

Camera-key reconciliation: the source-data mp4 keys are kept as canonical
  observation.images.exterior_image_1_left  (≙ Trossen cam_high / overhead)
  observation.images.wrist_image_left        (≙ cam_left_wrist)
  observation.images.wrist_image_right        (≙ cam_right_wrist)

Standalone (stdlib + wandb; pandas only used by the converter it shells out to). Run on a CPU node.

Usage:
  python build_encord_dataset.py --out /data/wam/datasets/encord_trossen
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_LABELS = "encord-wb-physical-ai/wam-finetune-webinar/encord-labels:v1"
DEFAULT_SOURCE = "encord-wb-physical-ai/wam-finetune-webinar/encord-source-data:v1"
DEFAULT_OUT = os.path.join(os.environ.get("WAM_DATA_ROOT", "/data/wam"), "datasets", "encord_trossen")
# Canonical camera keys = the source-data mp4 keys (kept as-is; no file renaming).
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]
# Language keys mirror oxe_droid (the Encord captioning agent emitted this exact schema).
LANG_KEYS = [
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
]


def _ep_index(path: str):
    m = re.search(r"episode_(\d+)\.(parquet|mp4)$", path)
    return int(m.group(1)) if m else None


def fetch(artifact, dest: str, max_eps: int | None):
    """Download an artifact to dest. If max_eps is set, fetch only meta/* + the first max_eps
    episodes' files (per-file via get_path) — avoids pulling the full (34GB) source-data for a smoke."""
    if max_eps is None:
        return artifact.download(root=dest)
    n = 0
    for path in artifact.manifest.entries:
        i = _ep_index(path)
        if "/meta/" in path or (i is not None and i < max_eps):
            artifact.get_path(path).download(root=dest)
            n += 1
    print(f"[encord]   selectively fetched {n} files (<= {max_eps} episodes) from {artifact.name}")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-artifact", default=os.environ.get("WAM_LABELS_ARTIFACT", DEFAULT_LABELS))
    p.add_argument("--source-artifact", default=os.environ.get("WAM_SOURCE_ARTIFACT", DEFAULT_SOURCE))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embodiment-tag", default="trossen")
    p.add_argument("--max-episodes", type=int, default=int(os.environ["WAM_MAX_EPISODES"]) if os.environ.get("WAM_MAX_EPISODES") else None,
                   help="Only merge the first N episodes (selective download). Default: all.")
    p.add_argument("--convert-script", default="scripts/data/convert_lerobot_to_gear.py")
    p.add_argument("--skip-convert", action="store_true", help="merge only; don't run the GEAR converter")
    args = p.parse_args()
    N = args.max_episodes

    import wandb
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)
    for k in VIDEO_KEYS:
        (out / "videos" / "chunk-000" / f"observation.images.{k}").mkdir(parents=True)

    run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                     project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                     job_type="preprocess", name="build-encord-trossen")
    print(f"[encord] downloading artifacts (records lineage); max_episodes={N}...")
    dl = os.path.join(os.environ.get("TMPDIR", "/tmp"), "encord_dl")
    lab = Path(fetch(run.use_artifact(args.labels_artifact), os.path.join(dl, "lab"), N))
    src = Path(fetch(run.use_artifact(args.source_artifact), os.path.join(dl, "src"), N))
    lab_ds, src_ds = lab / "dataset", src / "dataset"

    # Select episodes (first N, or all) from the labels episodes.jsonl.
    all_eps = [json.loads(l) for l in open(lab_ds / "meta" / "episodes.jsonl") if l.strip()]
    episodes = all_eps[:N] if N else all_eps

    # 1. Copy per-episode parquet (labels) + 3 videos (source-data) for the selected episodes only.
    for e in episodes:
        i = e["episode_index"]
        shutil.copy2(lab_ds / f"data/chunk-000/episode_{i:06d}.parquet",
                     out / f"data/chunk-000/episode_{i:06d}.parquet")
        for k in VIDEO_KEYS:
            shutil.copy2(src_ds / f"videos/chunk-000/observation.images.{k}/episode_{i:06d}.mp4",
                         out / f"videos/chunk-000/observation.images.{k}/episode_{i:06d}.mp4")
    # meta: episodes (selected) + tasks (full, for task_index lookups)
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    shutil.copy2(lab_ds / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")

    # 3. Build a self-consistent info.json: state/action features (from labels) + the 3 video keys,
    #    correct fps/totals, DROID-style path templates.
    lab_info = json.loads((lab_ds / "meta" / "info.json").read_text())
    feats = lab_info.get("features", {})
    feats = lab_info.get("features", {})
    info = {
        "codebase_version": "v2.0",
        "robot_type": args.embodiment_tag,
        "total_episodes": len(episodes),
        "total_frames": int(sum(e["length"] for e in episodes)),
        "total_tasks": len({json.loads(l)["task_index"] for l in open(out / "meta" / "tasks.jsonl") if l.strip()}),
        "total_videos": len(episodes) * len(VIDEO_KEYS),
        "total_chunks": 1,
        "chunks_size": lab_info.get("chunks_size", 1000),
        "fps": lab_info.get("fps", 30),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": feats.get("action", {"dtype": "float32", "shape": [16], "names": None}),
            "observation.state": feats.get("observation.state", {"dtype": "float32", "shape": [16], "names": None}),
            # height/width are informational (the transform resizes decoded frames); the loader
            # requires names incl. "channel" + a video_info.video.fps block (lerobot.py:1037-1046).
            **{f"observation.images.{k}": {"dtype": "video", "shape": [480, 640, 3],
                                            "names": ["height", "width", "channel"],
                                            "video_info": {"video.fps": float(lab_info.get("fps", 30) or 30)}}
               for k in VIDEO_KEYS},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # 4. Validate: every episode has its parquet + 3 videos.
    miss = []
    for e in episodes:
        i = e["episode_index"]
        if not (out / f"data/chunk-000/episode_{i:06d}.parquet").exists():
            miss.append(f"parquet {i}")
        for k in VIDEO_KEYS:
            if not (out / f"videos/chunk-000/observation.images.{k}/episode_{i:06d}.mp4").exists():
                miss.append(f"video {k}/{i}")
    if miss:
        sys.exit(f"[encord] missing files: {miss[:6]}")
    print(f"[encord] merged {len(episodes)} episodes, {info['total_frames']} frames -> {out}")

    if args.skip_convert:
        run.finish(); return

    # 5. Run the GEAR converter (writes modality.json, embodiment.json, stats.json).
    cmd = [sys.executable, args.convert_script, "--dataset-path", str(out),
           "--embodiment-tag", args.embodiment_tag, "--force"]
    print("[encord] running converter:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.environ.get("WAM_REPO_ROOT", "."))

    # 6. Patch modality.json language section to the 3 oxe_droid-style keys (the converter only
    #    handles a single task key). State/action/video sections from the converter are correct.
    mod_path = out / "meta" / "modality.json"
    mod = json.loads(mod_path.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    mod_path.write_text(json.dumps(mod, indent=4))
    print("[encord] patched modality.json annotation ->", list(mod["annotation"].keys()))

    # 7. Restore the real tasks.jsonl (full) + the SELECTED episodes.jsonl: the converter (run with
    #    --force) regenerated them from the (string-less) parquet, collapsing tasks.jsonl to a single
    #    empty task. The parquet's task_index references the full tasks table, so we keep all tasks.
    shutil.copy2(lab_ds / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    ntasks = sum(1 for l in open(out / "meta" / "tasks.jsonl") if l.strip())
    print(f"[encord] restored tasks.jsonl ({ntasks} tasks) + {len(episodes)} episodes")
    print("[encord] modality summary: state=%s action=%s video=%s annotation=%s" % (
        list(mod["state"]), list(mod["action"]), list(mod["video"]), list(mod["annotation"])))
    run.finish()
    print("[encord] done.")


if __name__ == "__main__":
    main()
