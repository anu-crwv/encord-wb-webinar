#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Assemble DAgger round-0 dataset v10 = real(v8) + CLEAN scripted-expert sim demos, and log the
demos as a W&B artifact.

Unlike the flawed v9 (open-loop real actions on fixed-object sim renders), these demos are the
scripted IK expert's own successful pick-place rollouts: sim pixels + the expert's 16-dim actions +
CORRECT object grounding (the arm actually reaches the object that is rendered). This is proper sim
BC data — the thing v9 was trying to be.

Demos live under WAM_DEMO_ROOT/<group>/shard-*/ (data + videos + demos.json), written by
scripted_expert_trossen.py. Each demo parquet stores its caption as a STRING in
annotation.language.language_instruction; v8 uses INTEGER task indices, so we remap each demo's
caption -> a task_index (appending new tasks to tasks.jsonl) and rewrite the demo's task_index +
all annotation.language.* columns as that integer, matching the loader's expected format.

Demos are duplicated WAM_DEMO_DUP times (few unique trajectories) and appended to a hardlink-copy of
v8, re-indexed after v8. Then recompute stats + regenerate step_filter. Run on a CPU node with the
repo staged + wandb-api-key."""

from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

V8 = os.environ.get("WAM_V8", "/data/wam/datasets/encord_trossen_v8")
DEMO_ROOT = os.environ.get("WAM_DEMO_ROOT", "/data/wam/datasets/dagger_demos")
GROUPS = os.environ.get("WAM_DEMO_GROUPS", "batt,cyl").split(",")
OUT = os.environ.get("WAM_V10", "/data/wam/datasets/encord_trossen_v10")
DUP = int(os.environ.get("WAM_DEMO_DUP", "8"))
REPO = os.environ.get("WAM_REPO_ROOT", "/data/src/dreamzero-wam")
CONVERT = "scripts/data/convert_lerobot_to_gear.py"
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]
LANG_KEYS = ["annotation.language.language_instruction",
             "annotation.language.language_instruction_2",
             "annotation.language.language_instruction_3"]
LANG0 = "annotation.language.language_instruction"


def _hardlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _collect_demos():
    """Return [(parquet_path, {cam: mp4_path})] for every successful demo episode."""
    out = []
    for g in GROUPS:
        for pq in sorted(glob.glob(f"{DEMO_ROOT}/{g}/shard-*/data/chunk-*/episode_*.parquet")):
            m = re.search(r"episode_(\d+)\.parquet$", pq)
            i = int(m.group(1))
            base = pq.rsplit("/data/", 1)[0]
            ch = f"chunk-{i // 1000:03d}"
            vids = {k: f"{base}/videos/{ch}/observation.images.{k}/episode_{i:06d}.mp4" for k in VIDEO_KEYS}
            if all(os.path.exists(v) for v in vids.values()):
                out.append((pq, vids))
    return out


def main() -> None:
    v8, out = Path(V8), Path(OUT)
    info = json.loads((v8 / "meta/info.json").read_text())
    CHUNK = int(info.get("chunks_size") or 1000)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    v8_eps = [json.loads(l) for l in open(v8 / "meta/episodes.jsonl") if l.strip()]
    tasks_list = [json.loads(l) for l in open(v8 / "meta/tasks.jsonl") if l.strip()]
    cap_to_idx = {t["task"]: int(t["task_index"]) for t in tasks_list}
    next_idx = (max(cap_to_idx.values()) + 1) if cap_to_idx else 0
    n_base = len(v8_eps)

    # 1. hardlink all of v8 (real episodes 0..n_base-1)
    for f in glob.glob(str(v8 / "data/chunk-*/episode_*.parquet")):
        _hardlink(f, str(out / os.path.relpath(f, v8)))
    for f in glob.glob(str(v8 / "videos/chunk-*/*/episode_*.mp4")):
        _hardlink(f, str(out / os.path.relpath(f, v8)))

    # 2. append demos (x DUP), re-indexed after v8, with caption -> integer task_index remap
    demos = _collect_demos()
    print(f"[v10] {len(demos)} unique demo episodes x{DUP} dup; base real={n_base}", flush=True)
    if not demos:
        raise SystemExit("[v10] FATAL: no demo episodes found under " + DEMO_ROOT)
    merged_eps = list(v8_eps)
    ni = n_base
    for _dup in range(DUP):
        for pq, vids in demos:
            df = pd.read_parquet(pq)
            cap = str(df[LANG0].iloc[0])
            if cap not in cap_to_idx:            # new caption -> append a task
                cap_to_idx[cap] = next_idx
                tasks_list.append({"task_index": next_idx, "task": cap})
                next_idx += 1
            idx = cap_to_idx[cap]
            L = len(df)
            df["episode_index"] = ni
            df["task_index"] = idx
            for k in LANG_KEYS:                   # integer annotations (v8 loader convention)
                df[k] = idx
            ch = f"chunk-{ni // CHUNK:03d}"
            (out / f"data/{ch}").mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / f"data/{ch}/episode_{ni:06d}.parquet")
            for k in VIDEO_KEYS:
                _hardlink(vids[k], str(out / f"videos/{ch}/observation.images.{k}/episode_{ni:06d}.mp4"))
            merged_eps.append({"episode_index": ni, "tasks": [cap], "length": int(L)})
            ni += 1
    n_demo = ni - n_base
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    info.update(total_episodes=len(merged_eps),
                total_frames=int(sum(e["length"] for e in merged_eps)),
                total_tasks=len(tasks_list),
                total_videos=len(merged_eps) * len(VIDEO_KEYS),
                total_chunks=(len(merged_eps) - 1) // CHUNK + 1,
                splits={"train": f"0:{len(merged_eps)}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"[v10] {len(merged_eps)} eps ({n_base} real + {n_demo} demo), {info['total_frames']} frames, "
          f"{len(tasks_list)} tasks", flush=True)

    # 3. recompute stats/modality over v10 + re-patch annotation + restore meta
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    modp = out / "meta/modality.json"
    mod = json.loads(modp.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    modp.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))

    # 4. step_filter over v10 (idle-prefix trim) — reuse the generator
    subprocess.run([sys.executable, "scripts/data/gen_trossen_step_filter.py"],
                   check=True, cwd=REPO, env={**os.environ, "V6_ROOT": str(out)})

    # 5. log the DAgger demos as a W&B artifact
    if os.environ.get("WAM_LOG_ARTIFACT", "1") == "1":
        try:
            import wandb
            run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                             project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                             job_type="preprocess", name="dagger-v10")
            art = wandb.Artifact("trossen-dagger-demos", type="dataset",
                                 metadata={"unique_demos": len(demos), "dup": DUP, "groups": GROUPS,
                                           "source": "scripted_expert_trossen (successful IK pick-place rollouts)",
                                           "v10_total_eps": len(merged_eps), "real_eps": n_base, "demo_eps": n_demo})
            for g in GROUPS:
                d = f"{DEMO_ROOT}/{g}"
                if os.path.isdir(d):
                    art.add_dir(d, name=g)
            run.log_artifact(art)
            run.finish()
            print("[v10] logged W&B artifact trossen-dagger-demos", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[v10] W&B artifact log skipped: {e}", flush=True)
    print(f"[v10] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
