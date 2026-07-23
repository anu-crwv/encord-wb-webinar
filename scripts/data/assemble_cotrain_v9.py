#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Assemble the co-training dataset v9 = real(v8) + SIM-rendered episodes, and log the sim
data as a W&B artifact.

The sim episodes (rendered by render_sim_cotraining.py under WAM_SIM_ROOT/<group>/shard-*/)
carry SIM videos + the REAL parquet labels. They're duplicated WAM_SIM_DUP times and
appended to a hardlink-copy of v8 (re-indexed after v8's episodes) so the real:sim ratio is
meaningful (sim is only ~144 unique trajectories). Then recompute stats (converter over v9)
+ regenerate step_filter. Sim episodes reuse v8's task_index/tasks.jsonl (same trajectories),
and language comes from the parquet annotation columns, so no task remap is needed.

Run on a CPU node with the repo staged + wandb-api-key (for the artifact log)."""

from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

V8 = os.environ.get("WAM_V8", "/data/wam/datasets/encord_trossen_v8")
SIM_ROOT = os.environ.get("WAM_SIM_ROOT", "/data/wam/datasets/sim_cotrain")
GROUPS = os.environ.get("WAM_SIM_GROUPS", "cyl,batt").split(",")
OUT = os.environ.get("WAM_V9", "/data/wam/datasets/encord_trossen_v9")
DUP = int(os.environ.get("WAM_SIM_DUP", "6"))
REPO = os.environ.get("WAM_REPO_ROOT", "/data/src/dreamzero-wam")
CONVERT = "scripts/data/convert_lerobot_to_gear.py"
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]
LANG_KEYS = ["annotation.language.language_instruction",
             "annotation.language.language_instruction_2",
             "annotation.language.language_instruction_3"]


def _hardlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _collect_sim():
    """Return [(parquet_path, {cam: mp4_path})] for every rendered sim episode."""
    out = []
    for g in GROUPS:
        for pq in sorted(glob.glob(f"{SIM_ROOT}/{g}/shard-*/data/chunk-*/episode_*.parquet")):
            m = re.search(r"episode_(\d+)\.parquet$", pq)
            i = int(m.group(1))
            base = pq.split("/data/")[0]
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
    v8_tasks = [json.loads(l) for l in open(v8 / "meta/tasks.jsonl") if l.strip()]
    tasks_by_idx = {t["task_index"]: t["task"] for t in v8_tasks}
    n_base = len(v8_eps)

    # 1. hardlink all of v8 (real episodes 0..n_base-1)
    for f in glob.glob(str(v8 / "data/chunk-*/episode_*.parquet")):
        _hardlink(f, str(out / os.path.relpath(f, v8)))
    for f in glob.glob(str(v8 / "videos/chunk-*/*/episode_*.mp4")):
        _hardlink(f, str(out / os.path.relpath(f, v8)))

    # 2. append sim episodes, duplicated DUP times, re-indexed after v8
    sim = _collect_sim()
    print(f"[v9] {len(sim)} unique sim episodes x{DUP} dup; base real={n_base}", flush=True)
    merged_eps = list(v8_eps)
    ni = n_base
    for _dup in range(DUP):
        for pq, vids in sim:
            df = pd.read_parquet(pq)
            L = len(df)
            df["episode_index"] = ni
            ti = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
            ch = f"chunk-{ni // CHUNK:03d}"
            (out / f"data/{ch}").mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / f"data/{ch}/episode_{ni:06d}.parquet")
            for k in VIDEO_KEYS:
                _hardlink(vids[k], str(out / f"videos/{ch}/observation.images.{k}/episode_{ni:06d}.mp4"))
            merged_eps.append({"episode_index": ni, "tasks": [tasks_by_idx.get(ti, "")], "length": int(L)})
            ni += 1
    n_sim = ni - n_base
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in v8_tasks))
    info.update(total_episodes=len(merged_eps),
                total_frames=int(sum(e["length"] for e in merged_eps)),
                total_tasks=len(v8_tasks),
                total_videos=len(merged_eps) * len(VIDEO_KEYS),
                total_chunks=(len(merged_eps) - 1) // CHUNK + 1,
                splits={"train": f"0:{len(merged_eps)}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"[v9] {len(merged_eps)} eps ({n_base} real + {n_sim} sim), {info['total_frames']} frames", flush=True)

    # 3. recompute stats/modality over v9 + re-patch annotation + restore meta
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    modp = out / "meta/modality.json"
    mod = json.loads(modp.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    modp.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in v8_tasks))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))

    # 4. step_filter over v9 (idle-prefix trim) — reuse the generator
    subprocess.run([sys.executable, "scripts/data/gen_trossen_step_filter.py"],
                   check=True, cwd=REPO, env={**os.environ, "V6_ROOT": str(out)})

    # 5. log the SIM co-training data as a W&B artifact (per instruction; sim mp4s are small)
    if os.environ.get("WAM_LOG_ARTIFACT", "1") == "1":
        try:
            import wandb
            run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                             project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                             job_type="preprocess", name="sim-cotrain-v9")
            art = wandb.Artifact("trossen-sim-cotrain", type="dataset",
                                 metadata={"unique_sim_eps": len(sim), "dup": DUP, "groups": GROUPS,
                                           "source": "render_sim_cotraining (real v8 traj replayed in sim)",
                                           "v9_total_eps": len(merged_eps), "real_eps": n_base, "sim_eps": n_sim})
            for g in GROUPS:
                d = f"{SIM_ROOT}/{g}"
                if os.path.isdir(d):
                    art.add_dir(d, name=g)
            run.log_artifact(art)
            run.finish()
            print("[v9] logged W&B artifact trossen-sim-cotrain", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[v9] W&B artifact log skipped: {e}", flush=True)
    print(f"[v9] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
