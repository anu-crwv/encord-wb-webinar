# dreamzero-wam — W&B-native fine-tuning harness for DreamZero (DROID world-action model)

A clean, **Weights & Biases–native** training harness forked from
[`dreamzero0/dreamzero`](https://github.com/dreamzero0/dreamzero). It fine-tunes the DreamZero DROID
world-action model on a versioned dataset and tracks **everything** — base models, datasets, training
runs, and output checkpoints — as W&B **Artifacts** with full **lineage**, so you can systematically
compare how data-curation choices affect policy performance.

> **New here? Read [Quickstart](#quickstart) → [The workflow](#the-workflow) → [Reading results in W&B](#reading-results-in-wb).**
> You don't need deep W&B knowledge — the three steps below are just `kubectl apply`, and everything
> shows up in one W&B project.

The upstream model code lives untouched in [`groot/`](groot/) (so it keeps tracking upstream); all the
W&B + orchestration logic is the additive [`wam/`](wam/) package and the [`deploy/cks/`](deploy/cks/)
Kubernetes manifests.

---

## What you get

```
 base models (HF)                     raw DROID (HF)
      │  bootstrap_models.py                │  build_pickplace_subset.py   ← "preprocessing" node
      ▼                                     ▼
 W&B model artifacts            W&B dataset artifact (droid-pickplace, re-indexed LeRobot v2.0)
 (reference → PVC)                          │
      └───────────────┬────────────────────┘
                      ▼   use_artifact()  (input lineage)
              ┌──────────────────┐
              │  wam/train.py    │  multi-node DeepSpeed ZeRO-3 over 2× GH200
              │  (groot trainer) │
              └──────────────────┘
                      │   log_artifact()  (output lineage)
                      ▼
        W&B model artifact: dreamzero-droid-pickplace-lora  → linked to the Registry
```

Every training run's **lineage graph** in W&B shows the 4 inputs (3 base models + 1 dataset) and the 1
output checkpoint. Swap in a different dataset version (e.g. curated vs random) and the runs line up
side-by-side for comparison.

---

## Prerequisites

- **Cluster access**: the CoreWeave kubeconfig `CWKubeconfig_anyscale-wb-webinar` (kept out of git).
  ```bash
  export KUBECONFIG="/path/to/CWKubeconfig_anyscale-wb-webinar"
  kubectl -n dreamzero get nodes
  ```
- **W&B**: account on entity **`encord-wb-physical-ai`**, project **`wam-finetune-webinar`**, with access
  to the **`encord-wb-webinar`** org Registry. A `wandb-api-key` secret must exist in the `dreamzero`
  namespace (see [deploy/cks/infra/wandb-secret-setup.md](deploy/cks/infra/wandb-secret-setup.md) if it
  doesn't):
  ```bash
  kubectl -n dreamzero get secret wandb-api-key   # should exist
  ```
- **Cluster resources (already provisioned)**: namespace `dreamzero`; RWX PVCs `dreamzero-data` (datasets,
  caches) and `dreamzero-checkpoints` (model weights, run outputs); 2× `NVIDIA-GH200-480GB` GPU nodes
  (arm64) + amd64 CPU nodes.

---

## Quickstart

Three steps, run once in order (steps 1–2 only need to be re-run when you change the base models or the
dataset). Set `KUBECONFIG` first (above).

```bash
cd deploy/cks

# 0. Stage the bootstrap scripts as a configmap (re-run after editing wam/artifacts/*.py)
kubectl -n dreamzero create configmap wam-bootstrap-scripts \
  --from-file=build_pickplace_subset.py=../../wam/artifacts/build_pickplace_subset.py \
  --from-file=bootstrap_models.py=../../wam/artifacts/bootstrap_models.py \
  -o yaml --dry-run=client | kubectl apply -f -

# 1. Download base models -> PVC -> register as W&B reference artifacts + Registry  (CPU node, ~75GB)
kubectl apply -f bootstrap/00-models-download.yaml

# 2. Build the pick-place DROID subset -> W&B dataset artifact + Registry            (CPU node)
kubectl apply -f bootstrap/01-pickplace-subset.yaml

# 3. Train: stage the repo on the PVC, then launch the multi-node job (see "Run training" below)
```

Watch any job:
```bash
kubectl -n dreamzero get pods -l app.kubernetes.io/part-of=dreamzero
kubectl -n dreamzero logs -l job-name=wam-models-download -f      # or wam-pickplace-subset
```

---

## The workflow

### Step 1 — Register base models  ·  [`bootstrap/00-models-download.yaml`](deploy/cks/bootstrap/00-models-download.yaml)
Downloads the three base models to the checkpoints PVC and logs each as a W&B **reference artifact**
(a pointer to the PVC path — versioned + lineage-tracked, but no 70GB re-upload), linked into the
`model` Registry:

| Artifact | HF source | Role |
|---|---|---|
| `wan2-1-i2v-14b-480p` | `Wan-AI/Wan2.1-I2V-14B-480P` | DiT backbone + VAE + CLIP + T5 |
| `umt5-xxl` | `google/umt5-xxl` | tokenizer |
| `dreamzero-agibot` | `GEAR-Dreams/DreamZero-AgiBot` | pretrained policy (new-embodiment transfer init) |

### Step 2 — Build the dataset (the "preprocessing" lineage node)  ·  [`bootstrap/01-pickplace-subset.yaml`](deploy/cks/bootstrap/01-pickplace-subset.yaml)
[`wam/artifacts/build_pickplace_subset.py`](wam/artifacts/build_pickplace_subset.py) downloads only the
episodes whose language annotation is a pick-and-place task (successful episodes only), **re-indexes** them
to a clean, contiguous LeRobot v2.0 dataset (the upstream sharded loader requires contiguous episode
indices in the filenames, `episodes.jsonl`, *and* the internal parquet `episode_index` column), and logs it
as the managed dataset artifact **`droid-pickplace`** (aliased `baseline`), linked into the `dataset`
Registry. Default: 150 episodes (`SUBSET_N`).

This is the variant you change for the curation experiment — see
[Adding a dataset variant](#adding-a-dataset-variant-curation-experiment).

### Step 3 — Run training
[`wam/train.py`](wam/train.py) opens one W&B run, `use_artifact`s the 4 Registry artifacts (input
lineage), runs the upstream trainer, and logs the fine-tuned checkpoint artifact (output lineage). The
14B model doesn't fit one 96GB GH200 at full precision, so DeepSpeed ZeRO-3 is used either way. **Two
manifests:**

- **`train/droid-pickplace-train-single.yaml`** — ✅ **validated / recommended.** Single GH200, ZeRO-3
  with **CPU param offload** to the Grace 480GB host (NVLink-C2C). A full 300-step LoRA run completes in
  **~45 min at ~9 s/step**. No cross-node networking.
- **`train/droid-pickplace-train.yaml`** — multi-node across both GH200s (Indexed 2-pod Job + headless
  Service). Faster in theory but currently unreliable on this cluster's plain-TCP fabric (see
  [known issue](#how-the-gh200-fit--zero-3-works-and-known-issues)); use only after RDMA is set up.

```bash
# Stage this repo onto the data PVC (the training pods run from /data/src/dreamzero-wam).
kubectl -n dreamzero apply -f infra/stager.yaml          # a sleep pod mounting dreamzero-data
tar czf - --exclude .git --exclude __pycache__ -C .. dreamzero-wam \
  | kubectl -n dreamzero exec -i wam-stager -- sh -c 'rm -rf /data/src/dreamzero-wam && tar xzf - -C /data/src'

# Launch the validated single-GH200 run. WANDB_RUN_ID must be unique per launch — inject at apply:
RID="wamrun$(date +%m%d%H%M)"
sed "s/WAMRUNIDPLACEHOLDER/$RID/" train/droid-pickplace-train-single.yaml | kubectl apply -f -

# Watch
kubectl -n dreamzero get pods -l app=wam-train
kubectl -n dreamzero logs -l app=wam-train --prefix -f | grep -vE "DEPRECATION|not on PATH"
```

The run appears at `https://wandb.ai/encord-wb-physical-ai/wam-finetune-webinar`. On success it logs
`dreamzero-droid-pickplace-lora` and links it to the `model` Registry.

> **Smoke vs full run.** Defaults are a LoRA run of `MAX_STEPS=300` (~45 min single-GH200). For a fast
> wiring check set `MAX_STEPS=2`. `save_only_model=true` keeps each checkpoint to ~39 MB (the LoRA
> adapter); the optimizer state is not saved (no resume needed for a fine-tune). Scale via the env knobs below.

---

## Reading results in W&B

- **Project** `encord-wb-physical-ai/wam-finetune-webinar` — every run (job types `register-model`,
  `preprocess`, `train`). Open a `train` run → **Overview → Lineage** to see the 4 input artifacts and the
  output checkpoint as a graph.
- **Registry** (`encord-wb-webinar`) — `wandb-registry-model` (base weights + trained checkpoints) and
  `wandb-registry-dataset` (DROID variants). Each collection's versions are what you compare across
  experiments.
- **Artifacts** carry metadata: model artifacts record the HF repo + PVC path; the dataset records the
  filter rule, episode count, and original→new index map; checkpoints record the run + base artifacts.

---

## Adding a dataset variant (curation experiment)

The whole point: measure whether better data curation improves the policy. Each variant is just a **new
version of the `droid-pickplace` dataset collection**:

1. Produce a new LeRobot v2.0 dataset (e.g. captioned / QC'd / curated) on the data PVC.
2. Log it as a new version of the same artifact + link it (mirror
   [`build_pickplace_subset.py`](wam/artifacts/build_pickplace_subset.py)).
3. Train against it by pointing `WAM_DATASET_ARTIFACT` at the new version (or `:latest`).

The runs share the same base models and differ only in the dataset → directly comparable in W&B, with
lineage proving which data produced which checkpoint.

---

## Configuration (env vars)

Set on the training Job (see the manifest `env:` block). All have sensible defaults.

| Var | Default | Meaning |
|---|---|---|
| `WANDB_ENTITY` / `WANDB_PROJECT` | `encord-wb-physical-ai` / `wam-finetune-webinar` | where runs + artifacts land |
| `WAM_REGISTRY_ORG` | `encord-wb-webinar` | org that owns the Registry |
| `WAM_DATASET_ARTIFACT` | `…/wandb-registry-dataset/droid-pickplace:latest` | dataset to train on (swap for variants) |
| `MAX_STEPS` | `300` | training steps (`2` = fast wiring check) |
| `SAVE_STEPS` | `100` | checkpoint interval |
| `TRAIN_ARCHITECTURE` | `lora` | `lora` or `full` |
| `PER_DEVICE_BATCH_SIZE` | `1` | per-GPU batch |
| `DEEPSPEED` | `configs/deepspeed/zero3_offload.json` (single-GH200 manifest) | DeepSpeed config; empty = no DeepSpeed |
| `MODEL_DTYPE` | `bfloat16` | resident model dtype |
| `USE_AGIBOT_INIT` | `0` | `1` = initialize the policy from DreamZero-AgiBot (new-embodiment transfer; needs >96GB / more GPUs) |
| `SUBSET_N` (step 2) | `150` | episodes in the dataset subset |

---

## Repo layout

```
dreamzero-wam/
├── groot/                      # upstream DreamZero model + trainer — VERBATIM, untouched
├── wam/                        # the W&B layer (the deliverable)
│   ├── config.py               # entity/project/registry + PVC paths
│   ├── artifacts/              # bootstrap_models.py, build_pickplace_subset.py
│   ├── wandb_utils.py          # use_artifact resolve + log_checkpoint_artifact helpers
│   ├── train.py                # multi-node, artifact-driven training entrypoint
│   ├── _ds_launch.py           # launch shim: applies DeepSpeed patches, runpy's groot trainer
│   └── _ds_zero3_leaf.py       # ZeRO-3 compatibility patches (VAE leaf modules + ckpt routing)
├── scripts/
│   ├── train/droid_pickplace.sh
│   └── data/                   # upstream DROID conversion utilities (kept)
├── deploy/cks/
│   ├── infra/                  # wandb-secret-setup.md, stager pod, PVC notes
│   ├── bootstrap/              # 00-models-download.yaml, 01-pickplace-subset.yaml  (CPU nodes)
│   └── train/                  # droid-pickplace-train-single.yaml (1× GH200, validated) + ...-train.yaml (2× GH200)
├── configs/deepspeed/          # zero3_offload.json (single-GH200 CPU offload) — kept out of groot/
├── docs/                       # upstream DROID/embodiment/backbone guides
├── eval/                       # placeholder — see "Evaluation"
└── requirements-train.txt      # additive training deps on top of the nvcr pytorch image
```

---

## How the GH200 fit / ZeRO-3 works (and known issues)

The 14B video-diffusion model + the ~29k-token (33-frame) sequence does **not** fit a single 96GB GH200
at full precision, so DeepSpeed ZeRO-3 is used. The **validated** configuration is **single GH200 with
ZeRO-3 CPU param offload** ([`configs/deepspeed/zero3_offload.json`](configs/deepspeed/zero3_offload.json)):
the params live on the Grace 480GB host and stream to the GPU over NVLink-C2C — a full 300-step LoRA run
completes in ~45 min at ~9 s/step. Making the upstream model work under ZeRO-3 needed a few
**non-architectural** shims (all in `wam/_ds_*`, applied via the launch shim — `groot/` stays verbatim):

- **nvtx**: the nvcr image's `nvtx` lacks the API pip-DeepSpeed calls → disabled in the boot script.
- **VAE / encoders**: marked as ZeRO-3 *leaf modules* (their `isinstance`-driven conv-cache breaks if
  ZeRO-3 hooks their internals).
- **activation checkpointing**: routed the DiT's `torch.utils.checkpoint` → DeepSpeed's ZeRO-3-aware
  checkpoint, wrapped in `autocast(bf16)`.
- **W&B run handoff**: the launcher finishes the run so the trainer can resume it for metrics, then
  reopens it to log the checkpoint.
- **`save_only_model=true`**: under ZeRO-3 the offloaded optimizer state is a single ~36GB `torch.save`
  that fails on the VAST PVC; a fine-tune doesn't need it, so only the ~39MB LoRA checkpoint is saved.

**Known issue — multi-node runs hit an NCCL stall on this cluster.** `train/droid-pickplace-train.yaml`
(2× GH200) gets `NET/Socket message truncated` (mitigated by `NCCL_PROTO=simple` etc.) and then an
intermittent ZeRO-3 all-gather hang → 30-min watchdog timeout, because the GH200 pods have **no
RDMA/InfiniBand** (NCCL falls back to plain TCP over `eth0`). The single-GH200 offload path above avoids
cross-node NCCL entirely and is what we run. To make multi-node reliable, the fix is RDMA/IB: request the
CoreWeave RDMA NICs into the pods (+ IMEX + NCCL IB env), not TCP-socket tuning.

---

## Evaluation

Deferred. An eval harness (compare training variants under the same test scenarios) will plug in here; it
consumes a checkpoint via `use_artifact("…/wandb-registry-model/dreamzero-droid-pickplace-lora:<ver>")` and
logs eval results linked to that artifact so each variant is comparable. Working eval code will be added by
the W&B/Encord team.

---

## Encord instructions

<!-- TODO: Encord-specific setup and dataset-export instructions go here. -->

_(To be completed.)_
