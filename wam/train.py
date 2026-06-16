#!/usr/bin/env python3
"""Artifact-driven, W&B-tracked, multi-node training entrypoint for DreamZero DROID fine-tuning.

One W&B run id is shared across the launcher and all ranks, so lineage + metrics + the output
checkpoint all land on a single run:

    rank0: wandb.init(id=RUN_ID) -> use_artifact(4 Registry artifacts) -> stage dataset on PVC
           -> finish() and signal ready   (hand the run id off)
    all  : torchrun (multi-node) -m wam._ds_launch  (runpy's the unmodified upstream
           experiment.py; the global-rank0 trainer RESUMES the same run id for metric logging) ->
           trains, sharded across the GPUs with DeepSpeed ZeRO-3
    rank0: reopen the run -> log the fine-tuned checkpoint as a model artifact + link to Registry

The run is created in shared mode (x_primary) and handed off serially via WANDB_RUN_ID + resume,
because HF Trainer's WandbCallback can't be forced into a shared-mode attach from env. `groot/` is
upstream and untouched — we only feed resolved artifact paths into its existing Hydra overrides.
All knobs are env vars.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import wandb

from wam import config as C
from wam.wandb_utils import log_checkpoint_artifact

# --- W&B target + Registry artifacts to consume (overridable via env) ---------
ENTITY = os.environ.get("WANDB_ENTITY", C.WANDB_ENTITY)
PROJECT = os.environ.get("WANDB_PROJECT", C.WANDB_PROJECT)
REG_ORG = os.environ.get("WAM_REGISTRY_ORG", "encord-wb-webinar")
ART = {
    "wan2-1-i2v-14b-480p": os.environ.get("WAM_WAN_ARTIFACT", f"{REG_ORG}/wandb-registry-model/wan2-1-i2v-14b-480p:v0"),
    "umt5-xxl": os.environ.get("WAM_UMT5_ARTIFACT", f"{REG_ORG}/wandb-registry-model/umt5-xxl:v0"),
    "dreamzero-agibot": os.environ.get("WAM_AGIBOT_ARTIFACT", f"{REG_ORG}/wandb-registry-model/dreamzero-agibot:v0"),
}
DATA_ART = os.environ.get("WAM_DATASET_ARTIFACT", f"{REG_ORG}/wandb-registry-dataset/droid-pickplace:latest")

# --- distributed topology -----------------------------------------------------
NODE_RANK = int(os.environ.get("NODE_RANK") or os.environ.get("JOB_COMPLETION_INDEX") or 0)
NNODES = int(os.environ.get("NNODES", "1"))
NUM_GPUS = os.environ.get("NUM_GPUS", "1")          # GPUs per node
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
IS_PRIMARY = NODE_RANK == 0

# --- training knobs (smoke defaults; scale by overriding env) -----------------
MAX_STEPS = os.environ.get("MAX_STEPS", "300")               # "small 300-sample" smoke @ bs=1
ARCH = os.environ.get("TRAIN_ARCHITECTURE", "lora")
PER_DEV_BS = os.environ.get("PER_DEVICE_BATCH_SIZE", "1")
SAVE_STEPS = os.environ.get("SAVE_STEPS", "100")
LEARNING_RATE = os.environ.get("LEARNING_RATE", "1e-4")
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "true")
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "bfloat16")
VIDEO_BACKEND = os.environ.get("VIDEO_BACKEND", "")
# DeepSpeed config — ZeRO-3 shards the 14B params across GPUs so the model fits 2x GH200.
DEEPSPEED = os.environ.get("DEEPSPEED", "groot/vla/configs/deepspeed/zero3.json" if NNODES > 1 else "")
# Init from DreamZero-AgiBot (new-embodiment transfer). Off by default (loads +45GB and overwrites
# the action-head config); agibot still recorded as a lineage input via use_artifact.
USE_AGIBOT_INIT = os.environ.get("USE_AGIBOT_INIT", "0") == "1"

REPO_ROOT = os.environ.get("WAM_REPO_ROOT", str(Path(__file__).resolve().parents[1]))
DATA_CACHE = os.path.join(os.environ.get("WAM_DATA_CACHE", "/data/wam/artifact_cache"), "droid-pickplace")
CKPT_NAME = os.environ.get("WAM_CKPT_ARTIFACT_NAME", "dreamzero-droid-pickplace-lora")
# Embodiment / data-config knobs (defaults = DROID). For a new embodiment, set the Hydra data config
# + its data-root key, e.g. DATA_CONFIG=dreamzero/trossen_relative DATA_ROOT_KEY=trossen_data_root.
DATA_CONFIG = os.environ.get("DATA_CONFIG", "dreamzero/droid_relative")
DATA_ROOT_KEY = os.environ.get("DATA_ROOT_KEY", "droid_data_root")
# If set, train from this local PVC dataset dir and SKIP the dataset Registry artifact (used for an
# embodiment whose dataset isn't registered yet — base models still recorded as lineage inputs).
DATASET_LOCAL_DIR = os.environ.get("WAM_DATASET_LOCAL_DIR", "")
# Shared run id + output dir are DETERMINISTIC from WANDB_RUN_ID (same env on every node).
RUN_ID = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"/checkpoints/wam/runs/droid_pickplace_{ARCH}_{RUN_ID}")
READY_FLAG = os.path.join("/checkpoints/wam/rendezvous", f"{RUN_ID}.ready")


def model_dir(name: str) -> str:
    """Deterministic on-PVC path for a base model (matches the reference-artifact target)."""
    return C.model_local_dir(name)


def build_overrides(wan_dir, umt5_dir, agibot_dir, data_dir) -> list[str]:
    ov = [
        "report_to=wandb", f"wandb_project={PROJECT}",
        f"data={DATA_CONFIG}", f"{DATA_ROOT_KEY}={data_dir}",
        f"train_architecture={ARCH}",
        "num_frames=33", "action_horizon=24", "num_views=3",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "num_frame_per_block=2", "num_action_per_block=24", "num_state_per_block=1",
        "seed=42",
        f"training_args.learning_rate={LEARNING_RATE}", "training_args.warmup_ratio=0.05",
        f"output_dir={OUTPUT_DIR}",
        f"per_device_train_batch_size={PER_DEV_BS}", f"max_steps={MAX_STEPS}",
        "weight_decay=1e-5", "save_total_limit=5",   # groot asserts >=5; checkpoints are ~39MB each w/ save_only_model
        # Save the model (LoRA adapter) only — skip the optimizer/scheduler state. For a fine-tune we
        # don't need to resume, and under ZeRO-3 the offloaded optimizer state is a ~36GB single-file
        # torch.save that fails the zip integrity check on the VAST PVC (>32GB). This was the only
        # failure of an otherwise-complete 300-step run.
        "++training_args.save_only_model=true",
        f"gradient_checkpointing={GRADIENT_CHECKPOINTING}",
        "upload_checkpoints=false", "bf16=true", "tf32=true", "eval_bf16=true",
        "dataloader_pin_memory=false", "dataloader_num_workers=1",
        "image_resolution_width=320", "image_resolution_height=176",
        f"save_lora_only={'true' if ARCH == 'lora' else 'false'}",
        "max_chunk_size=4", "frame_seqlen=880",
        f"++action_head_cfg.config.model_dtype={MODEL_DTYPE}",
        "save_strategy=steps", f"save_steps={SAVE_STEPS}",
        f"dit_version={wan_dir}",
        f"text_encoder_pretrained_path={wan_dir}/models_t5_umt5-xxl-enc-bf16.pth",
        f"image_encoder_pretrained_path={wan_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"vae_pretrained_path={wan_dir}/Wan2.1_VAE.pth",
        f"tokenizer_path={umt5_dir}",
    ]
    if DEEPSPEED:
        ov.append(f"training_args.deepspeed={DEEPSPEED}")
    if USE_AGIBOT_INIT:
        ov += [f"pretrained_model_path={agibot_dir}",
               "++action_head_cfg.config.skip_component_loading=true",
               "++action_head_cfg.config.defer_lora_injection=true"]
    if VIDEO_BACKEND:
        ov.append(f"++train_dataset.dataset_kwargs.video_backend={VIDEO_BACKEND}")
    return ov


def primary_prepare():
    """rank0: open the shared run, record artifact lineage, and stage the dataset on the PVC."""
    os.makedirs(os.path.dirname(READY_FLAG), exist_ok=True)
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)
    run = wandb.init(
        entity=ENTITY, project=PROJECT, id=RUN_ID, job_type="train", name=Path(OUTPUT_DIR).name,
        settings=wandb.Settings(mode="shared", x_primary=True, x_label="launcher-node0"),
        config=dict(arch=ARCH, max_steps=int(MAX_STEPS), nnodes=NNODES, num_gpus=int(NUM_GPUS),
                    per_device_batch_size=int(PER_DEV_BS), learning_rate=LEARNING_RATE,
                    deepspeed=DEEPSPEED, use_agibot_init=USE_AGIBOT_INIT, data_config=DATA_CONFIG,
                    artifacts=dict(**ART, dataset=(DATASET_LOCAL_DIR or DATA_ART))),
    )
    print("[train] rank0 recording artifact lineage (base models)...")
    for name, art in ART.items():
        run.use_artifact(art)          # input-lineage edge; weights already on the PVC
    if DATASET_LOCAL_DIR:
        # Train from a local (not-yet-registered) dataset dir on the PVC; no dataset artifact edge.
        data_dir = DATASET_LOCAL_DIR
        print(f"[train] using local dataset dir {data_dir} (no dataset artifact)")
    else:
        # dataset is a managed artifact -> download to a CLEAN dir (digest-based download won't delete
        # files absent from the new manifest, so wipe first to avoid cross-version contamination).
        shutil.rmtree(DATA_CACHE, ignore_errors=True)
        run.use_artifact(DATA_ART).download(root=DATA_CACHE)
        data_dir = DATA_CACHE
    # Release the run id so the trainer subprocess can RESUME it for metric logging (HF Trainer's
    # WandbCallback can't be forced into shared-mode attach via env, so we hand the run off serially).
    run.finish()
    Path(READY_FLAG).write_text(json.dumps({"run_id": RUN_ID, "data_dir": data_dir}))
    print(f"[train] rank0 ready; dataset at {data_dir}; run {RUN_ID} handed off")
    return None


def wait_ready(timeout_s: int = 3600):
    """non-primary nodes: block until rank0 has staged the dataset + signalled ready."""
    print(f"[train] node{NODE_RANK} waiting for rank0 ready flag {READY_FLAG} ...")
    start = 0
    while not os.path.exists(READY_FLAG):
        time.sleep(5)
        start += 5
        if start > timeout_s:
            sys.exit(f"[train] node{NODE_RANK} timed out waiting for rank0 ready flag")
    print(f"[train] node{NODE_RANK} saw ready flag")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if IS_PRIMARY:
        primary_prepare()
    else:
        wait_ready()

    wan_dir, umt5_dir, agibot_dir = (model_dir("wan2-1-i2v-14b-480p"),
                                     model_dir("umt5-xxl"), model_dir("dreamzero-agibot"))
    data_dir = DATASET_LOCAL_DIR or DATA_CACHE

    # 2. Launch the upstream trainer across all nodes. The global-rank0 trainer process RESUMES the
    #    handed-off run (WANDB_RUN_ID + resume) so HF Trainer metrics land on rank0's run; other ranks
    #    don't log to W&B (HF WandbCallback only runs on world_process_zero).
    env = os.environ.copy()
    env.update(
        WANDB_ENTITY=ENTITY, WANDB_PROJECT=PROJECT, WANDB_RUN_ID=RUN_ID, WANDB_RESUME="allow",
        HYDRA_FULL_ERROR="1", PYTHONPATH=f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}",
    )
    launch = ["torchrun", f"--nproc_per_node={NUM_GPUS}"]
    if NNODES > 1:
        launch += [f"--nnodes={NNODES}", f"--node_rank={NODE_RANK}",
                   f"--master_addr={MASTER_ADDR}", f"--master_port={MASTER_PORT}"]
    else:
        launch += ["--standalone"]
    # Launch through wam._ds_launch (applies the DeepSpeed ZeRO-3 leaf-module patch, then runpy's the
    # unmodified upstream experiment.py). `-m` keeps it reliable regardless of site/sitecustomize.
    cmd = launch + ["-m", "wam._ds_launch"] + build_overrides(wan_dir, umt5_dir, agibot_dir, data_dir)
    print("[train] launching:\n  " + " \\\n  ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)

    # 3. rank0 reopens the run to record outcome + output artifact (output-lineage edge). The trainer
    #    subprocess just released the run; retry until W&B no longer reports it "in use".
    if not IS_PRIMARY:
        sys.exit(proc.returncode)
    run = None
    for attempt in range(12):
        try:
            run = wandb.init(entity=ENTITY, project=PROJECT, id=RUN_ID, resume="allow",
                             settings=wandb.Settings(mode="shared", x_primary=True, x_label="launcher-node0"))
            break
        except Exception as e:  # noqa: BLE001  (ServerResponseError: run in use, until trainer releases)
            print(f"[train] reopen attempt {attempt + 1} failed ({e}); retrying in 15s")
            time.sleep(15)
    if run is None:
        sys.exit("[train] could not reopen run to log checkpoint artifact")
    if proc.returncode != 0:
        run.summary["train_returncode"] = proc.returncode
        run.finish(exit_code=1)
        sys.exit(f"[train] trainer exited with code {proc.returncode}")
    ckpts = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    upload_dir = str(ckpts[-1]) if ckpts else OUTPUT_DIR
    log_checkpoint_artifact(
        run, name=CKPT_NAME, ckpt_dir=upload_dir,
        metadata=dict(arch=ARCH, max_steps=int(MAX_STEPS), learning_rate=LEARNING_RATE, nnodes=NNODES,
                      data_config=DATA_CONFIG,
                      base_wan=ART["wan2-1-i2v-14b-480p"], tokenizer=ART["umt5-xxl"],
                      pretrained=ART["dreamzero-agibot"], dataset=(DATASET_LOCAL_DIR or DATA_ART),
                      source_run=RUN_ID, checkpoint=Path(upload_dir).name),
        registry="wandb-registry-model", aliases=["smoke", "latest"],
    )
    run.finish()
    print("[train] done.")


if __name__ == "__main__":
    main()
