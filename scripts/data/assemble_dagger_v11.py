#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Assemble DAgger round-1 dataset v11 = v10 (real v8 + clean scripted round-0 demos) + ON-POLICY
CORRECTIVE demos, and log the round-1 demos as a W&B artifact.

Round-1 is the DAgger on-policy step: we rolled out the v10 ckpt-2000 policy in sim at beta=0.3
(so ~70% of steps are the policy's OWN actions -> it visits its own drifted/hovering states) and
labelled each visited state with the scripted expert's corrective action (descend toward the object
/ close / lift). Those (policy-state -> expert-action) demos live under WAM_R1_ROOT/<group>/ written
by dagger_rollout_relabel.py in the SAME LeRobot schema scripted_expert_trossen.py produces.

This is the classic DAgger aggregate D = D_0 (v10) u D_1 (round-1): we hardlink all of v10 and append
the round-1 demos, re-indexed after v10, upweighted WAM_R1_DUP times so the on-policy corrections carry
real gradient weight against the ~400-real + round-0-demo bulk. Caption -> integer task_index remap +
stats/step_filter recompute are identical to the v10 assembler. Run on a CPU node with the repo staged
+ wandb-api-key.

Diff vs assemble_dagger_v10.py: (1) base = v10 (not v8); (2) round-1 demos live at
<root>/<group>/data/chunk-*/... (NO shard-* level, unlike round-0 dagger_demos); (3) separate
WAM_R1_DUP; (4) artifact name trossen-dagger-r1-demos."""

from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

BASE = os.environ.get("WAM_V10_BASE", "/data/wam/datasets/encord_trossen_v10")
R1_ROOT = os.environ.get("WAM_R1_ROOT", "/data/wam/datasets/dagger_r1")
GROUPS = os.environ.get("WAM_R1_GROUPS", "cyl,batt").split(",")
OUT = os.environ.get("WAM_V11", "/data/wam/datasets/encord_trossen_v11")
DUP = int(os.environ.get("WAM_R1_DUP", "2"))
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
    """Return [(parquet_path, {cam: mp4_path})] for every round-1 corrective episode.
    Round-1 layout is <root>/<group>/data/chunk-*/episode_*.parquet (no shard-* level)."""
    out = []
    for g in GROUPS:
        for pq in sorted(glob.glob(f"{R1_ROOT}/{g}/data/chunk-*/episode_*.parquet")):
            m = re.search(r"episode_(\d+)\.parquet$", pq)
            i = int(m.group(1))
            base = pq.rsplit("/data/", 1)[0]                       # -> <root>/<group>
            ch = f"chunk-{i // 1000:03d}"
            vids = {k: f"{base}/videos/{ch}/observation.images.{k}/episode_{i:06d}.mp4" for k in VIDEO_KEYS}
            if all(os.path.exists(v) for v in vids.values()):
                out.append((pq, vids))
            else:
                print(f"[v11] skip {pq}: missing video(s)", flush=True)
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

    # 1. hardlink all of v10 (episodes 0..n_base-1) — real v8 + clean round-0 demos
    for f in glob.glob(str(base_ds / "data/chunk-*/episode_*.parquet")):
        _hardlink(f, str(out / os.path.relpath(f, base_ds)))
    for f in glob.glob(str(base_ds / "videos/chunk-*/*/episode_*.mp4")):
        _hardlink(f, str(out / os.path.relpath(f, base_ds)))

    # 2. append round-1 corrective demos (x DUP), re-indexed after v10, caption -> integer task_index
    demos = _collect_demos()
    print(f"[v11] {len(demos)} unique round-1 demo episodes x{DUP} dup; base v10 eps={n_base}", flush=True)
    if not demos:
        raise SystemExit("[v11] FATAL: no round-1 demo episodes found under " + R1_ROOT)
    merged_eps = list(base_eps)
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
    print(f"[v11] {len(merged_eps)} eps ({n_base} v10 + {n_demo} round-1), {info['total_frames']} frames, "
          f"{len(tasks_list)} tasks", flush=True)

    # 3. recompute stats/modality over v11 + re-patch annotation + restore meta
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    modp = out / "meta/modality.json"
    mod = json.loads(modp.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    modp.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))

    # 4. step_filter over v11 (idle-prefix trim) — reuse the generator
    subprocess.run([sys.executable, "scripts/data/gen_trossen_step_filter.py"],
                   check=True, cwd=REPO, env={**os.environ, "V6_ROOT": str(out)})

    # 5. log the round-1 DAgger demos as a W&B artifact
    if os.environ.get("WAM_LOG_ARTIFACT", "1") == "1":
        try:
            import wandb
            run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                             project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                             job_type="preprocess", name="dagger-v11")
            art = wandb.Artifact("trossen-dagger-r1-demos", type="dataset",
                                 metadata={"unique_demos": len(demos), "dup": DUP, "groups": GROUPS,
                                           "source": "dagger_rollout_relabel (v10 ckpt-2000 policy @ beta=0.3, expert-relabelled on-policy states)",
                                           "beta": 0.3, "v11_total_eps": len(merged_eps),
                                           "v10_eps": n_base, "round1_eps": n_demo})
            for g in GROUPS:
                d = f"{R1_ROOT}/{g}"
                if os.path.isdir(d):
                    art.add_dir(d, name=g)
            run.log_artifact(art)
            run.finish()
            print("[v11] logged W&B artifact trossen-dagger-r1-demos", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[v11] W&B artifact log skipped: {e}", flush=True)
    print(f"[v11] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
