#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Assemble DAgger round-1 dataset v14 = v13 (all real Encord v4/v5/v7 + community) + ON-POLICY CORRECTIVE
demos rolled out from the v13 checkpoint-2000 policy (the offline lone-peak that reaches to ~23cm but stalls).

Round-1 is the DAgger on-policy step: we rolled out v13@2000 in sim at beta=0.3 (~70% policy-driven -> it
visits its OWN near-object stall states) and labelled each visited state with the scripted expert's corrective
action (keep descending -> close -> lift). Those (policy-state -> expert-action) demos live under
WAM_R1_ROOT/<group>/ written by dagger_rollout_relabel.py. This gives the head exactly the recovery data it
lacks: how to finish the last 23cm from where its own rollout stalls.

Same as assemble_dagger_v11 (hardlink base + append round-1 x DUP + recompute stats/step_filter) with TWO
changes for the v13 base: (1) v13's v7 videos are SYMLINKS to the checkpoints pvc (data pvc is full), so the
linker RE-SYMLINKS them rather than copying 130GB; (2) round-1 demo parquets get index=arange(L) defensively
(the sharded loader asserts per-trajectory index contiguity)."""

from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = os.environ.get("WAM_V13_BASE", "/data/wam/datasets/encord_trossen_v13")
R1_ROOT = os.environ.get("WAM_R1_ROOT", "/data/wam/datasets/dagger_r1")
GROUPS = os.environ.get("WAM_R1_GROUPS", "v13cyl,v13batt").split(",")
OUT = os.environ.get("WAM_V14", "/data/wam/datasets/encord_trossen_v14")
DUP = int(os.environ.get("WAM_R1_DUP", "30"))
REPO = os.environ.get("WAM_REPO_ROOT", "/data/src/dreamzero-wam")
CONVERT = "scripts/data/convert_lerobot_to_gear.py"
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]
LANG_KEYS = ["annotation.language.language_instruction",
             "annotation.language.language_instruction_2",
             "annotation.language.language_instruction_3"]
LANG0 = "annotation.language.language_instruction"


def _link(src, dst):
    """Hardlink within a filesystem; RE-SYMLINK if src is a symlink (v13's v7 videos point to /checkpoints);
    symlink (never copy) on cross-device. Reads/decodes follow symlinks transparently."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if os.path.islink(src):
        os.symlink(os.path.realpath(src), dst); return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def _collect_demos():
    """[(parquet, {cam: mp4})] for every round-1 corrective episode. Layout <root>/<group>/data/chunk-*/..."""
    out = []
    for g in GROUPS:
        for pq in sorted(glob.glob(f"{R1_ROOT}/{g}/data/chunk-*/episode_*.parquet")):
            i = int(re.search(r"episode_(\d+)\.parquet$", pq).group(1))
            base = pq.rsplit("/data/", 1)[0]
            ch = f"chunk-{i // 1000:03d}"
            vids = {k: f"{base}/videos/{ch}/observation.images.{k}/episode_{i:06d}.mp4" for k in VIDEO_KEYS}
            if all(os.path.exists(v) for v in vids.values()):
                out.append((pq, vids))
            else:
                print(f"[v14] skip {pq}: missing video(s)", flush=True)
    return out


def main() -> None:
    base_ds, out = Path(BASE), Path(OUT)
    info = json.loads((base_ds / "meta/info.json").read_text())
    CHUNK = int(info.get("chunks_size") or 1000)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    base_eps = [json.loads(l) for l in open(base_ds / "meta/episodes.jsonl") if l.strip()]
    tasks_list = [json.loads(l) for l in open(base_ds / "meta/tasks.jsonl") if l.strip()]
    cap_to_idx = {t["task"]: int(t["task_index"]) for t in tasks_list}
    next_idx = (max(cap_to_idx.values()) + 1) if cap_to_idx else 0
    n_base = len(base_eps)

    # 1. bring in all of v13 (real Encord + community). Parquets hardlink (same fs); videos hardlink for
    #    v8/community real files and RE-SYMLINK for the v7 symlinks (cross-pvc).
    for f in glob.glob(str(base_ds / "data/chunk-*/episode_*.parquet")):
        _link(f, str(out / os.path.relpath(f, base_ds)))
    n_sym = 0
    for f in glob.glob(str(base_ds / "videos/chunk-*/*/episode_*.mp4")):
        if os.path.islink(f):
            n_sym += 1
        _link(f, str(out / os.path.relpath(f, base_ds)))
    print(f"[v14] base v13: {n_base} eps ({n_sym} v7 videos re-symlinked)", flush=True)

    # 2. append round-1 corrective demos (x DUP), re-indexed after v13
    demos = _collect_demos()
    print(f"[v14] {len(demos)} unique round-1 demo episodes x{DUP} dup; base v13 eps={n_base}", flush=True)
    if not demos:
        raise SystemExit("[v14] FATAL: no round-1 demo episodes found under " + R1_ROOT)
    merged_eps = list(base_eps)
    ni = n_base
    for _dup in range(DUP):
        for pq, vids in demos:
            df = pd.read_parquet(pq)
            cap = str(df[LANG0].iloc[0]) if LANG0 in df.columns else "pick up the object and place it in the bin."
            if cap not in cap_to_idx:
                cap_to_idx[cap] = next_idx
                tasks_list.append({"task_index": next_idx, "task": cap})
                next_idx += 1
            idx = cap_to_idx[cap]
            L = len(df)
            df["episode_index"] = ni
            df["task_index"] = idx
            df["index"] = np.arange(L, dtype=np.int64)   # per-trajectory contiguous (loader asserts this)
            for k in LANG_KEYS:
                df[k] = idx
            ch = f"chunk-{ni // CHUNK:03d}"
            (out / f"data/{ch}").mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / f"data/{ch}/episode_{ni:06d}.parquet")
            for k in VIDEO_KEYS:
                _link(vids[k], str(out / f"videos/{ch}/observation.images.{k}/episode_{ni:06d}.mp4"))
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
    r1_frames = sum(e["length"] for e in merged_eps[n_base:])
    print(f"[v14] {len(merged_eps)} eps ({n_base} v13 + {n_demo} round-1), {info['total_frames']} frames, "
          f"{len(tasks_list)} tasks; round-1 = {r1_frames} frames ({100*r1_frames/info['total_frames']:.1f}%)", flush=True)

    # 3. recompute stats/modality over v14 + re-patch annotation + restore meta
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    modp = out / "meta/modality.json"
    mod = json.loads(modp.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    modp.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))

    # 4. step_filter over v14
    subprocess.run([sys.executable, "scripts/data/gen_trossen_step_filter.py"],
                   check=True, cwd=REPO, env={**os.environ, "V6_ROOT": str(out)})

    # 5. log the round-1 DAgger demos as a W&B artifact
    if os.environ.get("WAM_LOG_ARTIFACT", "1") == "1":
        try:
            import wandb
            run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                             project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                             job_type="preprocess", name="dagger-v14")
            art = wandb.Artifact("trossen-dagger-r1-v13-demos", type="dataset",
                                 metadata={"unique_demos": len(demos), "dup": DUP, "groups": GROUPS,
                                           "source": "dagger_rollout_relabel (v13 ckpt-2000 policy @ beta=0.3, expert-relabelled on-policy states)",
                                           "beta": 0.3, "v14_total_eps": len(merged_eps),
                                           "v13_eps": n_base, "round1_eps": n_demo})
            for g in GROUPS:
                d = f"{R1_ROOT}/{g}"
                if os.path.isdir(d):
                    art.add_dir(d, name=g)
            run.log_artifact(art)
            run.finish()
            print("[v14] logged W&B artifact trossen-dagger-r1-v13-demos", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[v14] W&B artifact log skipped: {e}", flush=True)
    print(f"[v14] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
