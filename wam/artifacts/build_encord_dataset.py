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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-artifact", default=DEFAULT_LABELS)
    p.add_argument("--source-artifact", default=DEFAULT_SOURCE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embodiment-tag", default="trossen")
    p.add_argument("--convert-script", default="scripts/data/convert_lerobot_to_gear.py")
    p.add_argument("--skip-convert", action="store_true", help="merge only; don't run the GEAR converter")
    args = p.parse_args()

    import wandb
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "videos").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)

    run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                     project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                     job_type="preprocess", name="build-encord-trossen")
    print("[encord] downloading artifacts (records lineage)...")
    lab = Path(run.use_artifact(args.labels_artifact).download())
    src = Path(run.use_artifact(args.source_artifact).download())
    lab_ds, src_ds = lab / "dataset", src / "dataset"

    # 1. data/ + meta episodes/tasks come from the LABELS artifact.
    shutil.copytree(lab_ds / "data", out / "data", dirs_exist_ok=True)
    shutil.copy2(lab_ds / "meta" / "episodes.jsonl", out / "meta" / "episodes.jsonl")
    shutil.copy2(lab_ds / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")

    # 2. videos/ come from the SOURCE-DATA artifact (keep its cam keys as canonical).
    shutil.copytree(src_ds / "videos", out / "videos", dirs_exist_ok=True)

    # 3. Build a self-consistent info.json: state/action features (from labels) + the 3 video keys,
    #    correct fps/totals, DROID-style path templates.
    lab_info = json.loads((lab_ds / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in open(out / "meta" / "episodes.jsonl") if l.strip()]
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

    # 7. Restore the real tasks.jsonl + episodes.jsonl from the labels artifact: the converter (run
    #    with --force) regenerated them from the (string-less) parquet, collapsing tasks.jsonl to a
    #    single empty task. The parquet's task_index references 0..N-1, so we need the original map.
    shutil.copy2(lab_ds / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")
    shutil.copy2(lab_ds / "meta" / "episodes.jsonl", out / "meta" / "episodes.jsonl")
    ntasks = sum(1 for l in open(out / "meta" / "tasks.jsonl") if l.strip())
    print(f"[encord] restored tasks.jsonl ({ntasks} tasks) + episodes.jsonl from labels artifact")
    print("[encord] modality summary: state=%s action=%s video=%s annotation=%s" % (
        list(mod["state"]), list(mod["action"]), list(mod["video"]), list(mod["annotation"])))
    run.finish()
    print("[encord] done.")


if __name__ == "__main__":
    main()
