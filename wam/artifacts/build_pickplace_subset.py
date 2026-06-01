#!/usr/bin/env python3
"""Build a pick-place subsample of the DreamZero DROID dataset and register it as a W&B dataset artifact.

This is the "preprocessing" node of the lineage graph
    raw DROID (HF)  ->  [this script]  ->  droid-pickplace dataset artifact  ->  training run  ->  checkpoint

It downloads ONLY the episodes whose language annotation looks like a pick-and-place task
(and, by default, only successful episodes), keeping the original LeRobot episode indices and
chunk layout so the upstream `groot` loader reads it unchanged. The result is a small, valid,
standalone LeRobot v2.0 dataset that is logged as a managed W&B artifact (portable + versioned)
and linked into the dataset Registry as the BASELINE variant of the curation experiment.

Standalone by design (only stdlib + huggingface_hub + wandb) so it can run inside a slim
Kubernetes Job via a mounted configmap. Config is read from env vars / CLI; defaults mirror
wam/config.py.

Usage:
    python build_pickplace_subset.py --n 150
    python build_pickplace_subset.py --n 5 --no-wandb        # cheap structural smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("Install huggingface_hub: pip install huggingface_hub")

# --- defaults (mirror wam/config.py) ------------------------------------------
DEFAULT_REPO = os.environ.get("WAM_DROID_REPO", "GEAR-Dreams/DreamZero-DROID-Data")
DEFAULT_OUT = os.path.join(os.environ.get("WAM_DATA_ROOT", "/data/wam"), "datasets", "droid_pickplace_v0")
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "wam-finetune-webinar")
DATASET_REGISTRY = os.environ.get("WAM_DATASET_REGISTRY", "wandb-registry-dataset")
VIDEO_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]
# Known meta files (avoids listing the 124k-file repo tree, which triggers HF 429s).
META_FILES = [
    "meta/episodes.jsonl", "meta/info.json", "meta/tasks.jsonl", "meta/modality.json",
    "meta/stats.json", "meta/relative_stats.json", "meta/relative_stats_dreamzero.json",
    "meta/relative_horizon_stats_dreamzero.json",
]


def hf_get(repo_id: str, filename: str, local_dir: str, retries: int = 6, required: bool = True):
    """Download ONE file by exact path (no repo-tree listing → avoids 429s), with backoff."""
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename,
                                   local_dir=local_dir)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "404" in msg or "entrynotfound" in msg:
                if required:
                    raise
                print(f"[subset] optional file missing, skipping: {filename}", file=sys.stderr)
                return None
            wait = min(2 ** attempt, 60)
            print(f"[subset] {filename}: {type(e).__name__} (attempt {attempt}/{retries}); retry in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {filename} after {retries} attempts")


def select_episodes(episodes_path: Path, keywords, require_success, n, shuffle, seed):
    """Return the chosen episode metadata dicts from meta/episodes.jsonl."""
    rows = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            tasks = ep.get("tasks") or []
            text = " ".join(tasks).lower()
            if not any(k in text for k in keywords):
                continue
            if require_success and not ep.get("success", False):
                continue
            rows.append(ep)

    print(f"[subset] {len(rows)} episodes match keywords={keywords} require_success={require_success}")
    if shuffle:
        import random

        random.Random(seed).shuffle(rows)
    else:
        rows.sort(key=lambda e: e["episode_index"])
    return rows[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--out", default=DEFAULT_OUT, help="Output dataset directory (on the data PVC)")
    p.add_argument("--n", type=int, default=150, help="Number of episodes to keep")
    p.add_argument("--keywords", default="pick,place", help="Comma list; episode kept if any task contains any")
    p.add_argument("--allow-failed", action="store_true", help="Keep failed episodes too (default: successful only)")
    p.add_argument("--shuffle", action="store_true", help="Random sample instead of first-N (deterministic w/ --seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-name", default="droid-pickplace")
    p.add_argument("--collection", default="droid-pickplace", help="Registry collection name")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--no-wandb", action="store_true", help="Build the subset only; skip artifact logging")
    p.add_argument("--no-link", action="store_true", help="Log the artifact but skip linking to the Registry")
    args = p.parse_args()

    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Stage downloads in a raw dir (original names/chunks), then re-index into `out`.
    raw = out.parent / (out.name + "_raw")
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)

    # 1. Grab meta/ by exact filenames (no repo-tree listing → no 429).
    print(f"[subset] downloading meta/ from {args.repo_id} ...")
    for mf in META_FILES:
        hf_get(args.repo_id, mf, str(raw), required=mf in ("meta/episodes.jsonl", "meta/info.json"))

    info = json.loads((raw / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    data_tmpl = info["data_path"]      # data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    video_tmpl = info["video_path"]    # videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4

    # 2. Select episodes.
    chosen = select_episodes(raw / "meta" / "episodes.jsonl", keywords, not args.allow_failed,
                             args.n, args.shuffle, args.seed)
    if not chosen:
        sys.exit("[subset] no episodes matched — check --keywords / --allow-failed")
    print(f"[subset] selected {len(chosen)} episodes")

    # 3. Download exactly the parquet + 3 videos for each selected episode (one request per file).
    files = []
    for ep in chosen:
        idx = ep["episode_index"]
        chunk = idx // chunks_size
        files.append(data_tmpl.format(episode_chunk=chunk, episode_index=idx))
        for vk in VIDEO_KEYS:
            files.append(video_tmpl.format(episode_chunk=chunk, video_key=vk, episode_index=idx))
    print(f"[subset] downloading {len(files)} files for {len(chosen)} episodes ...")
    for i, fn in enumerate(files, 1):
        hf_get(args.repo_id, fn, str(raw))
        if i % 100 == 0:
            print(f"[subset]   {i}/{len(files)} files")

    # 4. RE-INDEX into a clean LeRobot dataset with CONTIGUOUS episode indices 0..N-1.
    #    The upstream loader treats episode_index as a positional index into its per-episode
    #    arrays (e.g. trajectory_lengths[episode_index]), so sparse original indices break it.
    #    The SHARDED loader also filters its cached dataframe by the parquet's INTERNAL
    #    episode_index column, so we must rewrite that column too (not just the filenames).
    #    We rename files into chunk-000 with new indices, rewrite the internal episode_index
    #    column, and rewrite episodes.jsonl. tasks.jsonl / stats / modality / relative_stats are
    #    copied unchanged (task_index references the full tasks table; frame_index is per-episode).
    import pyarrow as pa
    import pyarrow.parquet as pq

    if (out / "meta").exists():
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)
    for vk in VIDEO_KEYS:
        (out / "videos" / "chunk-000" / vk).mkdir(parents=True)

    new_episodes, index_map = [], []
    for new_idx, ep in enumerate(chosen):
        orig = ep["episode_index"]
        oc = orig // chunks_size
        dst_parquet = out / data_tmpl.format(episode_chunk=0, episode_index=new_idx)
        shutil.move(str(raw / data_tmpl.format(episode_chunk=oc, episode_index=orig)), str(dst_parquet))
        # rewrite the internal episode_index column to the new contiguous index
        t = pq.read_table(dst_parquet)
        if "episode_index" in t.column_names:
            i = t.schema.get_field_index("episode_index")
            t = t.set_column(i, "episode_index",
                             pa.array([new_idx] * t.num_rows, type=t.schema.field("episode_index").type))
            pq.write_table(t, dst_parquet)
        for vk in VIDEO_KEYS:
            shutil.move(str(raw / video_tmpl.format(episode_chunk=oc, video_key=vk, episode_index=orig)),
                        str(out / video_tmpl.format(episode_chunk=0, video_key=vk, episode_index=new_idx)))
        new_episodes.append({"episode_index": new_idx, "tasks": ep.get("tasks", []),
                             "length": ep["length"], "success": ep.get("success", True)})
        index_map.append({"new_index": new_idx, "orig_index": orig})

    # copy all meta files unchanged, then overwrite episodes.jsonl + info.json
    for f in (raw / "meta").glob("*"):
        if f.name not in ("episodes.jsonl", "info.json"):
            shutil.copy2(f, out / "meta" / f.name)
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = int(sum(ep["length"] for ep in new_episodes))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    shutil.rmtree(raw)

    # 5. Structural validation: every re-indexed file must exist on disk.
    missing = []
    for ep in new_episodes:
        idx = ep["episode_index"]
        files = [out / data_tmpl.format(episode_chunk=0, episode_index=idx)]
        files += [out / video_tmpl.format(episode_chunk=0, video_key=vk, episode_index=idx) for vk in VIDEO_KEYS]
        missing += [str(fp) for fp in files if not fp.exists()]
    if missing:
        print(f"[subset] WARNING: {len(missing)} expected files missing, e.g. {missing[:3]}", file=sys.stderr)
        sys.exit(1)

    # 6. Write a provenance manifest next to the data.
    manifest = {
        "source_repo": args.repo_id,
        "filter": {"keywords": keywords, "require_success": not args.allow_failed,
                   "shuffle": args.shuffle, "seed": args.seed, "n_requested": args.n},
        "n_episodes": len(new_episodes),
        "total_frames": info["total_frames"],
        "fps": info.get("fps"),
        "reindexed": True,
        "index_map": index_map,                       # new_index -> original DROID episode_index
        "tasks_sample": [ep.get("tasks", [])[:1] for ep in new_episodes[:10]],
    }
    (out / "wam_subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    size_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"[subset] built {len(chosen)}-episode dataset at {out} ({size_gb:.2f} GB)")

    if args.no_wandb:
        print("[subset] --no-wandb set; skipping artifact logging.")
        return

    # 7. Log as a managed W&B dataset artifact + link into the dataset Registry.
    import wandb

    run = wandb.init(entity=args.entity, project=args.project, job_type="preprocess",
                     name=f"build-{args.artifact_name}", config=manifest)
    art = wandb.Artifact(name=args.artifact_name, type="dataset", metadata=manifest,
                         description=f"DROID pick-place subsample ({len(chosen)} successful episodes) in LeRobot v2.0 format.")
    art.add_dir(str(out))
    logged = run.log_artifact(art)
    logged.wait()
    print(f"[subset] logged artifact {args.entity}/{args.project}/{args.artifact_name}:{logged.version}")
    if not args.no_link:
        target = f"{DATASET_REGISTRY}/{args.collection}"
        run.link_artifact(logged, target_path=target)
        print(f"[subset] linked to Registry: {target}")
    run.finish()


if __name__ == "__main__":
    main()
