#!/usr/bin/env python3
# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0
"""Assemble v13 = v12 (v8 1894 real captioned + ~350 community WXAI eps) + encord-source-data:v7 (668 NEW
real captioned Trossen episodes, ~1.34M frames, 7 new *real in-domain* tasks: coil wire, fold microfiber
towels, clear kitchen counter, open/close drawers, distribute poker supplies). v7 is the highest-value
addition to the shared action head yet -- real Trossen data (not cross-embodiment), and pure task DIVERSITY
(the v4->v6 lever). v7 is LeRobot v2.1 (per-episode parquet+mp4, same as v8), 14-dim joints (base-vel padded
to 0 like the stationary community sets) and cam_high/cam_left_wrist/cam_right_wrist -> our 3 RGB keys.

v13 reuses v12 wholesale (hardlinked, no re-ingest of community) and only appends v7. v7 raw lives on the
CHECKPOINTS pvc (the DATA pvc is full); its videos are cross-filesystem, so they are SYMLINKED into v13
(hardlink falls back to symlink on EXDEV; the loader/stats/step-filter all follow symlinks). v7's embedded
caption is a useless generic "Stationary profile", so the instruction is DERIVED from the task folder name."""

from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

# reuse the proven, tested normalizers from the v12 assembler (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_v12 import _build_remap, _cam_map, _remap_vec, VIDEO_KEYS, LANG_KEYS  # noqa: E402

V8 = os.environ.get("WAM_V8", "/data/wam/datasets/encord_trossen_v8")
V12 = os.environ.get("WAM_V12", "/data/wam/datasets/encord_trossen_v12")
V7RAW = os.environ.get("WAM_V7RAW", "/checkpoints/wam/v7raw")
OUT = os.environ.get("WAM_V13", "/data/wam/datasets/encord_trossen_v13")
REPO = os.environ.get("WAM_REPO_ROOT", "/data/src/dreamzero-wam")
CONVERT = "scripts/data/convert_lerobot_to_gear.py"

# v7 task-folder -> clean imperative instruction (the 3 "coil wire" variants collapse to one task caption)
V7_CAPTIONS = {
    "Clear kitchen counter (stationary)": "Clear the kitchen counter.",
    "Coil wire": "Coil the wire.",
    "Coil wire 2": "Coil the wire.",
    "Coil wire - stationary": "Coil the wire.",
    "Distribute tx hold em supplies (stationary)": "Distribute the Texas hold 'em poker supplies.",
    "Fold microfiber towels - Stationary": "Fold the microfiber towels.",
    "Open - Close Drawers (Stationary)": "Open and close the drawers.",
}


def _feat_names(info: dict, key: str) -> list:
    a = (info["features"].get(key) or {}).get("names")
    if isinstance(a, dict):
        a = a.get("motors") or a.get("axes") or sum(([v] if isinstance(v, str) else list(v) for v in a.values()), [])
    return list(a) if a else []


def _link(src: str, dst: str) -> None:
    """Hardlink within a filesystem; on cross-fs (EXDEV) SYMLINK -- never copy (v7 videos are 130GB on
    another pvc). Reads/decodes follow symlinks transparently."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def _v7_caption(task_top: str) -> str:
    if task_top in V7_CAPTIONS:
        return V7_CAPTIONS[task_top]
    s = re.sub(r"\s*\(.*?\)", "", task_top)
    s = re.sub(r"[-_]\s*stationary\s*$", "", s, flags=re.I).strip()
    return (s[0].upper() + s[1:] + ".") if s else task_top


def _collect_v7():
    """Yield (epdir, info, pairs_action, pairs_state, cam_map, caption) for each self-contained v7 episode
    dataset under V7RAW/<Task>/<Loc>/<Person>/<Date>/episode_*/. Any per-episode error -> SKIP (logged)."""
    infos = sorted(glob.glob(f"{V7RAW}/*/*/*/*/*/meta/info.json"))
    if not infos:  # be depth-tolerant if the export nesting differs
        infos = sorted(glob.glob(f"{V7RAW}/**/meta/info.json", recursive=True))
    n_seen = n_use = 0
    skips = {}
    for info_p in infos:
        n_seen += 1
        epdir = os.path.dirname(os.path.dirname(info_p))
        task_top = os.path.relpath(epdir, V7RAW).split(os.sep)[0]
        try:
            info = json.loads(open(info_p).read())
            if str(info.get("codebase_version")) != "v2.1":
                skips["not v2.1"] = skips.get("not v2.1", 0) + 1; continue
            pairs_a, err = _build_remap(_feat_names(info, "action"))
            if pairs_a is None:
                skips[f"action:{err}"] = skips.get(f"action:{err}", 0) + 1; continue
            s_names = _feat_names(info, "observation.state")
            pairs_s = _build_remap(s_names)[0] if s_names else pairs_a
            if pairs_s is None:
                pairs_s = pairs_a
            cmap, cerr = _cam_map(info)
            if cmap is None:
                skips[f"cam:{cerr}"] = skips.get(f"cam:{cerr}", 0) + 1; continue
            n_use += 1
            yield epdir, info, pairs_a, pairs_s, cmap, _v7_caption(task_top)
        except Exception as e:  # noqa: BLE001
            skips[f"{type(e).__name__}"] = skips.get(f"{type(e).__name__}", 0) + 1
    print(f"[v13] v7 scan: {n_seen} episode dirs, {n_use} usable; skips={skips}", flush=True)


def main() -> None:
    v12, out = Path(V12), Path(OUT)
    assert (v12 / "meta/info.json").exists(), f"v12 base not found at {V12}"
    info = json.loads((v12 / "meta/info.json").read_text())
    CHUNK = int(info.get("chunks_size") or 1000)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    base_eps = [json.loads(l) for l in open(v12 / "meta/episodes.jsonl") if l.strip()]
    tasks_list = [json.loads(l) for l in open(v12 / "meta/tasks.jsonl") if l.strip()]
    cap_to_idx = {t["task"]: int(t["task_index"]) for t in tasks_list}
    next_idx = (max(cap_to_idx.values()) + 1) if cap_to_idx else 0
    n_base = len(base_eps)

    # 1. bring in the v12 base (v8 + community). Videos hardlink as-is (index-agnostic). Parquets: v8's
    #    per-frame `index` is already contiguous (real export), so hardlink; the community parquets were
    #    written with index=0 (a bug -> fails the loader's per-trajectory contiguity assert), so REWRITE
    #    their `index` to arange(L). v8 episodes are the first V8_N indices.
    v8_n = int(json.loads((Path(V8) / "meta/info.json").read_text())["total_episodes"])
    for f in glob.glob(str(v12 / "videos/chunk-*/*/episode_*.mp4")):
        _link(f, str(out / os.path.relpath(f, v12)))
    fixed_comm = 0
    for f in glob.glob(str(v12 / "data/chunk-*/episode_*.parquet")):
        ei = int(re.search(r"episode_(\d+)\.parquet$", f).group(1))
        dst = str(out / os.path.relpath(f, v12))
        if ei < v8_n:
            _link(f, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            d = pd.read_parquet(f)
            d["index"] = np.arange(len(d), dtype=np.int64)
            d.to_parquet(dst)
            fixed_comm += 1
    print(f"[v13] v12 base: {n_base} eps (v8 {v8_n} hardlinked, {fixed_comm} community index-fixed)", flush=True)

    # 2. append v7 (real captioned), reindexed after v12; videos symlinked cross-pvc
    merged_eps = list(base_eps)
    ni = n_base
    per_cap = {}
    for epdir, info_src, pairs_a, pairs_s, cmap, cap in _collect_v7():
        try:
            pqs = glob.glob(f"{epdir}/data/chunk-*/episode_*.parquet")
            if not pqs:
                print(f"[v13]   {epdir}: no parquet -> skip", flush=True); continue
            df = pd.read_parquet(pqs[0])
            L = len(df)
            srcvids = {}
            ok = True
            for tgt in VIDEO_KEYS:
                cand = glob.glob(f"{epdir}/videos/chunk-*/observation.images.{cmap[tgt]}/episode_*.mp4")
                if not cand:
                    ok = False; break
                srcvids[tgt] = cand[0]
            if not ok:
                print(f"[v13]   {epdir}: missing video ({tgt}) -> skip", flush=True); continue
            if cap not in cap_to_idx:
                cap_to_idx[cap] = next_idx
                tasks_list.append({"task_index": next_idx, "task": cap})
                next_idx += 1
            idx = cap_to_idx[cap]
            new = pd.DataFrame({
                "action": _remap_vec(df, "action", pairs_a, L),
                "observation.state": _remap_vec(df, "observation.state", pairs_s, L),
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
                _link(srcvids[tgt], str(out / f"videos/{och}/observation.images.{tgt}/episode_{ni:06d}.mp4"))
            merged_eps.append({"episode_index": ni, "tasks": [cap], "length": int(L)})
            ni += 1
            per_cap[cap] = per_cap.get(cap, 0) + 1
        except Exception as e:  # noqa: BLE001
            print(f"[v13]   {epdir}: ERROR {e} -> skip", flush=True)

    n_v7 = ni - n_base
    if n_v7 == 0:
        raise SystemExit("[v13] FATAL: no v7 episodes ingested")
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    info.update(total_episodes=len(merged_eps),
                total_frames=int(sum(e["length"] for e in merged_eps)),
                total_tasks=len(tasks_list),
                total_videos=len(merged_eps) * len(VIDEO_KEYS),
                total_chunks=(len(merged_eps) - 1) // CHUNK + 1,
                splits={"train": f"0:{len(merged_eps)}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"[v13] {len(merged_eps)} eps ({n_base} v12 + {n_v7} v7), {info['total_frames']} frames, "
          f"{len(tasks_list)} tasks", flush=True)
    print(f"[v13] v7 per-caption: {per_cap}", flush=True)

    # 3. recompute stats/modality over v13 + re-patch annotation keys + restore meta
    subprocess.run([sys.executable, CONVERT, "--dataset-path", str(out),
                    "--embodiment-tag", "trossen", "--force"], check=True, cwd=REPO)
    modp = out / "meta/modality.json"
    mod = json.loads(modp.read_text())
    mod["annotation"] = {k.replace("annotation.", ""): {"original_key": k} for k in LANG_KEYS}
    modp.write_text(json.dumps(mod, indent=4))
    (out / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks_list))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in merged_eps))

    # 4. step_filter (idle-prefix trim) over v13
    subprocess.run([sys.executable, "scripts/data/gen_trossen_step_filter.py"],
                   check=True, cwd=REPO, env={**os.environ, "V6_ROOT": str(out)})

    # 5. log lineage artifact
    if os.environ.get("WAM_LOG_ARTIFACT", "1") == "1":
        try:
            import wandb
            run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "encord-wb-physical-ai"),
                             project=os.environ.get("WANDB_PROJECT", "wam-finetune-webinar"),
                             job_type="preprocess", name="assemble-v13")
            art = wandb.Artifact("trossen-v13-alldata", type="dataset",
                                 metadata={"base": "v12 (v8 + community)", "added": "encord-source-data:v7",
                                           "v13_total_eps": len(merged_eps), "v12_eps": n_base,
                                           "v7_eps": n_v7, "v7_per_caption": per_cap,
                                           "v13_total_frames": info["total_frames"]})
            run.log_artifact(art)
            run.finish()
            print("[v13] logged W&B artifact trossen-v13-alldata", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[v13] W&B artifact log skipped: {e}", flush=True)
    print(f"[v13] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
