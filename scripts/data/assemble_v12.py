#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Assemble v12 = v8 (1894 real captioned eps) + compatible Trossen COMMUNITY datasets, normalized to our
16-dim schema. The community data adds real WidowX-AI-arm manipulation DIVERSITY to strengthen the shared
action head (extends the v4->v6 diversity gain that fixed the offline under-shoot).

SELF-SELECTING: only ingests community datasets that are (a) LeRobot codebase v2.1 (per-episode parquet +
per-episode mp4 layout == our v8; v3.0 packs episodes/videos into shared files -> deferred to a converter),
(b) action names map UNAMBIGUOUSLY to our 16-dim order by JOINT NAME (handles base-first vs base-last, `.pos`
suffix, `*_carriage_joint`==gripper; rejects O1 datasets whose joints have no left/right label), and (c) have
the three RGB cameras we need (cam_high/cam_head->exterior, cam_left_wrist->wrist_left, cam_right_wrist->
wrist_right; rejects depth-only or ambiguously-named cams). Everything else is SKIPPED with a logged reason.

Videos are av1 640x480 == our v8 (training already decodes av1, verified), so mp4s are HARDLINKED as-is (no
transcode). Only the 16-dim action/observation.state is remapped by name and padded with base=0 for the 14-dim
stationary sets. Captions come from each source dataset's meta/tasks.jsonl. Base of v8 is hardlinked, community
appended + reindexed after it; then stats + step_filter recompute + W&B artifact, exactly like assemble_dagger_v11."""

from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

V8 = os.environ.get("WAM_V8", "/data/wam/datasets/encord_trossen_v8")
COMM = os.environ.get("WAM_COMMUNITY_ROOT", "/data/wam/datasets/community")
OUT = os.environ.get("WAM_V12", "/data/wam/datasets/encord_trossen_v12")
REPO = os.environ.get("WAM_REPO_ROOT", "/data/src/dreamzero-wam")
CONVERT = "scripts/data/convert_lerobot_to_gear.py"
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left", "wrist_image_right"]
LANG_KEYS = ["annotation.language.language_instruction",
             "annotation.language.language_instruction_2",
             "annotation.language.language_instruction_3"]
LANG0 = "annotation.language.language_instruction"

# our 16-dim canonical order (== v8 action/state names)
TGT = [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)] + ["linear_vel", "angular_vel"]
TGT_IDX = {n: i for i, n in enumerate(TGT)}
# camera semantic map: our key -> ordered source-name candidates (strict; unambiguous only)
CAM_SRC = {"exterior_image_1_left": ["cam_high", "cam_head"],
           "wrist_image_left": ["cam_left_wrist"],
           "wrist_image_right": ["cam_right_wrist"]}


def _hardlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _norm_joint(raw: str) -> str:
    s = raw[:-4] if raw.endswith(".pos") else raw
    s = s.replace("left_left_carriage_joint", "left_joint_6").replace("left_right_carriage_joint", "left_joint_6")
    s = s.replace("right_left_carriage_joint", "right_joint_6").replace("right_right_carriage_joint", "right_joint_6")
    if s in ("x.vel", "x_vel", "x", "x.pos"):
        s = "linear_vel"
    if s in ("theta.vel", "theta_vel", "theta", "theta.pos"):
        s = "angular_vel"
    return s


def _action_names(info: dict) -> list:
    a = info["features"]["action"].get("names")
    if isinstance(a, dict):
        a = a.get("motors") or a.get("axes") or sum(([v] if isinstance(v, str) else list(v) for v in a.values()), [])
    return list(a) if a else []


def _build_remap(names: list):
    """(src_idx -> tgt_idx) pairs mapping the source action/state vector into our 16-dim order.
    Returns None if any name is unknown OR two source dims map to the same target (ambiguous, e.g. O1)."""
    pairs, seen = [], set()
    for si, raw in enumerate(names):
        n = _norm_joint(raw)
        if n not in TGT_IDX:
            return None, f"unknown joint name {raw!r}"
        ti = TGT_IDX[n]
        if ti in seen:
            return None, f"duplicate target for {raw!r} (ambiguous arm labels)"
        seen.add(ti)
        pairs.append((si, ti))
    return pairs, None


def _cam_map(info: dict):
    have = {k.split(".")[-1]: k for k in info["features"] if k.startswith("observation.images")}
    depth = {c for c, k in have.items() if (info["features"][k].get("info") or {}).get("video.is_depth_map")}
    out = {}
    for tgt, cands in CAM_SRC.items():
        pick = next((c for c in cands if c in have and c not in depth), None)
        if pick is None:
            return None, f"no RGB source for {tgt} (have={sorted(have)}, depth={sorted(depth)})"
        out[tgt] = pick
    return out, None


def _remap_vec(df, col, pairs, L):
    src = np.stack(df[col].to_numpy())
    out = np.zeros((L, 16), dtype=np.float32)
    for si, ti in pairs:
        out[:, ti] = src[:, si]
    return list(out)


def _read_tasks(srcdir: str) -> dict:
    """Robustly read source task strings -> {task_index: caption}. Handles tasks.jsonl (line-delimited),
    tasks.json (array or index->str map), and malformed lines (skipped)."""
    tasks = {}
    p_jsonl = os.path.join(srcdir, "meta/tasks.jsonl")
    p_json = os.path.join(srcdir, "meta/tasks.json")
    if os.path.exists(p_jsonl):
        for line in open(p_jsonl):
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line); tasks[int(t["task_index"])] = t["task"]
            except Exception:  # noqa: BLE001
                continue
    elif os.path.exists(p_json):
        try:
            obj = json.loads(open(p_json).read())
            if isinstance(obj, list):
                for i, t in enumerate(obj):
                    tasks[i] = t if isinstance(t, str) else t.get("task", str(t))
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        tasks[int(k)] = v if isinstance(v, str) else v.get("task", str(v))
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
    return tasks


def _collect_community():
    """Yield (name, srcdir, info, remap_pairs, cam_map, tasks). Any per-dataset error -> SKIP (logged),
    never fatal to the whole assembly."""
    for srcdir in sorted(glob.glob(f"{COMM}/*/")):
        name = os.path.basename(srcdir.rstrip("/"))
        try:
            info_p = os.path.join(srcdir, "meta/info.json")
            if not os.path.exists(info_p):
                print(f"[v12] SKIP {name}: no info.json", flush=True); continue
            info = json.loads(open(info_p).read())
            cb = str(info.get("codebase_version"))
            if cb != "v2.1":
                print(f"[v12] SKIP {name}: codebase {cb} (v3.0 packed layout -> deferred to converter)", flush=True); continue
            pairs, err = _build_remap(_action_names(info))
            if pairs is None:
                print(f"[v12] SKIP {name}: action remap failed ({err})", flush=True); continue
            cmap, cerr = _cam_map(info)
            if cmap is None:
                print(f"[v12] SKIP {name}: cameras ({cerr})", flush=True); continue
            tasks = _read_tasks(srcdir)
            print(f"[v12] USE  {name}: cams={cmap} tasks={list(tasks.values())[:3]}", flush=True)
            yield name, srcdir, info, pairs, cmap, tasks
        except Exception as e:  # noqa: BLE001
            print(f"[v12] SKIP {name}: meta read error ({type(e).__name__}: {e})", flush=True); continue


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

    # 1. hardlink all of v8 (real captioned episodes 0..n_base-1)
    for f in glob.glob(str(v8 / "data/chunk-*/episode_*.parquet")):
        _hardlink(f, str(out / os.path.relpath(f, v8)))
    for f in glob.glob(str(v8 / "videos/chunk-*/*/episode_*.mp4")):
        _hardlink(f, str(out / os.path.relpath(f, v8)))
    print(f"[v12] hardlinked v8 base: {n_base} eps", flush=True)

    # 2. append normalized community episodes, reindexed after v8
    merged_eps = list(v8_eps)
    ni = n_base
    per_ds = {}
    for name, srcdir, info_src, pairs, cmap, tasks in _collect_community():
        src_chunk = int(info_src.get("chunks_size") or 1000)
        n_ds = 0
        for pq in sorted(glob.glob(f"{srcdir}/data/chunk-*/episode_*.parquet")):
            try:
                si = int(re.search(r"episode_(\d+)\.parquet$", pq).group(1))
                df = pd.read_parquet(pq)
                L = len(df)
                # caption: source per-episode task_index -> its string (fallback: dataset name)
                cap = tasks.get(int(df["task_index"].iloc[0]), name.replace("_", " ")) if "task_index" in df.columns else name.replace("_", " ")
                if cap not in cap_to_idx:
                    cap_to_idx[cap] = next_idx
                    tasks_list.append({"task_index": next_idx, "task": cap})
                    next_idx += 1
                idx = cap_to_idx[cap]
                # map the 3 RGB videos FIRST (skip episode if any missing)
                sch = f"chunk-{si // src_chunk:03d}"
                srcvids = {tgt: f"{srcdir}/videos/{sch}/observation.images.{cmap[tgt]}/episode_{si:06d}.mp4" for tgt in VIDEO_KEYS}
                if not all(os.path.exists(v) for v in srcvids.values()):
                    print(f"[v12]   {name} ep{si}: missing video(s) -> skip", flush=True); continue
                # build the v12 frame table (remap 16-dim action/state, v8-style annotation ints)
                new = pd.DataFrame({
                    "action": _remap_vec(df, "action", pairs, L),
                    "observation.state": _remap_vec(df, "observation.state", pairs, L),
                    "timestamp": df["timestamp"].to_numpy() if "timestamp" in df else np.arange(L, dtype=np.float32) / 30.0,
                    "frame_index": np.arange(L, dtype=np.int64),
                    "episode_index": np.full(L, ni, dtype=np.int64),
                    "index": np.arange(L, dtype=np.int64),  # per-trajectory contiguous (loader asserts this)
                    "task_index": np.full(L, idx, dtype=np.int64),
                })
                for k in LANG_KEYS:
                    new[k] = idx
                och = f"chunk-{ni // CHUNK:03d}"
                (out / f"data/{och}").mkdir(parents=True, exist_ok=True)
                new.to_parquet(out / f"data/{och}/episode_{ni:06d}.parquet")
                for tgt in VIDEO_KEYS:
                    _hardlink(srcvids[tgt], str(out / f"videos/{och}/observation.images.{tgt}/episode_{ni:06d}.mp4"))
                merged_eps.append({"episode_index": ni, "tasks": [cap], "length": int(L)})
                ni += 1; n_ds += 1
            except Exception as e:  # noqa: BLE001
                print(f"[v12]   {name} {os.path.basename(pq)}: ERROR {e} -> skip", flush=True)
        per_ds[name] = n_ds
        print(f"[v12] +{n_ds} eps from {name}", flush=True)

    n_comm = ni - n_base
    if n_comm == 0:
        raise SystemExit("[v12] FATAL: no community episodes ingested")
    # global contiguous 'index'
    gi = 0
    # (index recomputed lazily is expensive; leave per-parquet index as 0 — convert step regenerates ordering)
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    info.update(total_episodes=len(merged_eps),
                total_frames=int(sum(e["length"] for e in merged_eps)),
                total_tasks=len(tasks_list),
                total_videos=len(merged_eps) * len(VIDEO_KEYS),
                total_chunks=(len(merged_eps) - 1) // CHUNK + 1,
                splits={"train": f"0:{len(merged_eps)}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"[v12] {len(merged_eps)} eps ({n_base} v8 + {n_comm} community from {sum(1 for v in per_ds.values() if v)} datasets), "
          f"{info['total_frames']} frames, {len(tasks_list)} tasks", flush=True)
    print(f"[v12] per-dataset: {per_ds}", flush=True)

    # 3. recompute stats/modality over v12 + re-patch annotation + restore meta
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    modp = out / "meta/modality.json"
    mod = json.loads(modp.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    modp.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))

    # 4. step_filter (idle-prefix trim) over v12
    subprocess.run([sys.executable, "scripts/data/gen_trossen_step_filter.py"],
                   check=True, cwd=REPO, env={**os.environ, "V6_ROOT": str(out)})

    # 5. log the normalized community demos as a W&B artifact (lineage)
    if os.environ.get("WAM_LOG_ARTIFACT", "1") == "1":
        try:
            import wandb
            run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                             project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                             job_type="preprocess", name="assemble-v12")
            art = wandb.Artifact("trossen-community-demos", type="dataset",
                                 metadata={"source": "TrossenRoboticsCommunity (LeRobot v2.1, WXAI arms)",
                                           "v12_total_eps": len(merged_eps), "v8_eps": n_base,
                                           "community_eps": n_comm, "per_dataset": per_ds})
            run.log_artifact(art)
            run.finish()
            print("[v12] logged W&B artifact trossen-community-demos", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[v12] W&B artifact log skipped: {e}", flush=True)
    print(f"[v12] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
