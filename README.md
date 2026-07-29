# Fine-tuning a World Action Model for a new Embodient 

> *The video looked right. The robot did not move.*

That one sentence is the whole project. We took **DreamZero** — a 14-billion-parameter **World Action
Model** (WAM): a Wan 2.1 image-to-video diffusion backbone with a bimanual action head — and taught it
to drive a robot it had never seen: the **Trossen AI mobile bimanual platform**. On held-out real
demonstrations the fine-tuned model predicted credible actions and its generated "dream" video showed the
reach we wanted. Turning that understanding into reliable *autonomous* motion is the hard part, and chasing
it end-to-end is what produced everything in this repo.

This repository is the **full-stack, Weights & Biases–native pipeline** with **Encord integration** for data pipeline behind that effort:

1. **Data** — convert and curate real robot data from **Encord** into versioned, trainable datasets.
2. **Training** — a W&B-native harness that fine-tunes the 14B WAM on a **new 16-dim action space**, at
   scale, with every input and output tracked as a W&B **Artifact** with full **lineage**.
3. **Evaluation** — a Trossen **sim + offline eval suite authored from scratch** on NVIDIA Isaac Lab–Arena,
   with results, videos, and leaderboards in **W&B Weave**.

Everything lands in one place: your **`<YOUR_WANDB_PROJECT>`** W&B project (entity
`<YOUR_WANDB_ENTITY>`), with a shared **`<YOUR_WANDB_ORG>`** org Registry.

> **The engineering story** behind this — the "credible video, motionless robot" puzzle and the full-stack
> investigation it kicked off — is written up in
> **[*Finetuning a World Action Model (WAM)*](reports/)** and the diagnostics report
> [`reports/dreamzero_trossen_sim_demo_investigation.md`](reports/dreamzero_trossen_sim_demo_investigation.md).

---

## Built to be reused

The three pieces below were built to **outlive this one robot**. Point them at a different embodiment,
a different dataset, or a different checkpoint and they carry over — that reuse is the deliverable, not
just the Trossen model.

| Reusable building block | Where | What a future project reuses |
|---|---|---|
| **W&B training harness** | [`wam/`](wam/) | Artifact-driven, embodiment-agnostic fine-tuning of the 14B WAM. Swap the dataset artifact / data-config and the same harness (ZeRO-3 fit, lineage, Registry linking, checkpoint upload) trains a *different* robot. |
| **Data pipeline** | [`scripts/encord/`](scripts/encord/) · [`scripts/data/`](scripts/data/) · [`wam/artifacts/`](wam/artifacts/) | Encord → S3 → **versioned W&B LeRobot v2.0 datasets** with source + label lineage. The registration/export/caption/assemble steps generalize to any Encord-hosted embodiment dataset. |
| **Evaluation environments** | [`eval/isaaclab_arena_dreamzero/`](eval/isaaclab_arena_dreamzero/) | A **new bimanual embodiment, scene, task, policy adapter, inference server, and Weave layer** for Isaac Lab–Arena — none of which shipped upstream. Reusable for any DreamZero-family policy; the embodiment module is a template for the next robot. |

See **[Reusing this for a new embodiment](#reusing-this-for-a-new-embodiment)** for the concrete recipe.

The upstream model + trainer live **verbatim** in [`groot/`](groot/) so they keep tracking upstream; all
of our logic is additive: the [`wam/`](wam/) package, the [`eval/`](eval/) suite, the
[`scripts/`](scripts/) data tooling, and the [`deploy/cks/`](deploy/cks/) Kubernetes manifests.

---

## What you get

```
   Encord (S3 video + labels)                    base models (HF)
        │  scripts/encord/*                            │  wam/artifacts/bootstrap_models.py
        │  (register → export → caption)               ▼
        ▼                                        W&B model artifacts  (Wan2.1-I2V-14B, umt5-xxl,
   W&B dataset artifacts                         DreamZero-AgiBot)  → reference → PVC + Registry
   (encord-source-data + encord-labels/captions)      │
        │  scripts/data/assemble_*.py                  │
        ▼  → versioned LeRobot v2.0 dataset            │
   W&B dataset artifact (trossen)  ──────────┐         │
                                             ▼         ▼   use_artifact()  (input lineage)
                                     ┌───────────────────────────┐
                                     │  wam/train.py             │  DeepSpeed ZeRO-3 on GH200
                                     │  (groot trainer, LoRA)    │  new 16-dim Trossen action space
                                     └───────────────────────────┘
                                             │  log_artifact()  (output lineage) + Registry link
                                             ▼
                              W&B model artifact: dreamzero-trossen-lora:<ver>
                                             │  use_artifact()
                                             ▼
                       eval/  →  Isaac Lab–Arena sim + offline real-data eval
                                             │
                                             ▼
                       W&B Weave: per-episode scores, leaderboards, sim‖dream video
```

Every training run's **lineage graph** in W&B shows its inputs (base models + the exact dataset version)
and its output checkpoint; every eval run `use_artifact`s the checkpoint it tested, so **dataset →
training run → checkpoint → eval** is one connected graph. Change the dataset version and the runs line
up side-by-side to answer *"did better data actually make a better policy?"*

---

## Prerequisites

- **Cluster access** — your Kubernetes cluster's kubeconfig (kept out of git):
  ```bash
  export KUBECONFIG="/path/to/your-kubeconfig"
  kubectl -n <YOUR_NAMESPACE> get nodes
  ```
- **W&B** — account on entity **`<YOUR_WANDB_ENTITY>`**, project **`<YOUR_WANDB_PROJECT>`**, with access
  to the **`<YOUR_WANDB_ORG>`** org Registry. A `wandb-api-key` secret must exist in your
  namespace (see [deploy/cks/infra/wandb-secret-setup.md](deploy/cks/infra/wandb-secret-setup.md)):
  ```bash
  kubectl -n <YOUR_NAMESPACE> get secret wandb-api-key   # should exist
  ```
- **Encord** (data pipeline only) — an Encord SSH key + AWS SSO for your S3 profile:
  ```bash
  export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
  aws sso login --profile <YOUR_AWS_PROFILE>
  ```
- **Cluster resources (already provisioned)** — your namespace; RWX PVCs `dreamzero-data` (datasets,
  caches) and `dreamzero-checkpoints` (weights, run outputs); 2× `NVIDIA-GH200-480GB` GPU nodes (arm64) for
  training + the policy server; `NVIDIA-RTX-PRO-6000` (amd64) nodes for Isaac Sim rendering; amd64 CPU nodes
  for data jobs.

---

## The W&B layer — training + tracking

This is the heart of the harness: [`wam/train.py`](wam/train.py) opens **one W&B run**, `use_artifact`s
the base models + the dataset (recording **input lineage**), runs the upstream `groot` trainer, and logs
the fine-tuned checkpoint as an artifact linked into the Registry (**output lineage**). Nothing about the
model code changes — the W&B layer wraps it.

### Step 1 — Register base models · [`bootstrap/00-models-download.yaml`](deploy/cks/bootstrap/00-models-download.yaml)

Downloads the three base models to the checkpoints PVC and logs each as a W&B **reference artifact** (a
pointer to the PVC path — versioned + lineage-tracked, no 70 GB re-upload), linked into the `model` Registry:

| Artifact | HF source | Role |
|---|---|---|
| `wan2-1-i2v-14b-480p` | `Wan-AI/Wan2.1-I2V-14B-480P` | DiT backbone + VAE + CLIP + T5 |
| `umt5-xxl` | `google/umt5-xxl` | tokenizer |
| `dreamzero-agibot` | `GEAR-Dreams/DreamZero-AgiBot` | pretrained policy (new-embodiment transfer init) |

```bash
export KUBECONFIG="/path/to/your-kubeconfig"
cd deploy/cks

# Stage the bootstrap scripts as a configmap (re-run after editing wam/artifacts/*.py)
kubectl -n <YOUR_NAMESPACE> create configmap wam-bootstrap-scripts \
  --from-file=bootstrap_models.py=../../wam/artifacts/bootstrap_models.py \
  --from-file=build_encord_dataset.py=../../wam/artifacts/build_encord_dataset.py \
  -o yaml --dry-run=client | kubectl apply -f -

kubectl apply -f bootstrap/00-models-download.yaml     # CPU node, ~75 GB
```

### Step 2 — Assemble the training dataset

See **[The Encord data pipeline](#the-encord-data-pipeline)** below — it produces the versioned
`trossen` LeRobot dataset artifact this step consumes.

### Step 3 — Run training

The 14B model + long video sequence does **not** fit one 96 GB GH200 at full precision, so DeepSpeed ZeRO-3
is used either way. Two paths:

- **[`train/encord-trossen-train-single.yaml`](deploy/cks/train/encord-trossen-train-single.yaml)** — ✅
  **validated / recommended.** Single GH200, ZeRO-3 with **CPU param offload** to the Grace 480 GB host over
  NVLink-C2C. No cross-node networking.
- **[`train/encord-trossen-train-full.yaml`](deploy/cks/train/encord-trossen-train-full.yaml)** — multi-node
  across both GH200s (proven to run; needs RDMA to be reliable — see [known issues](#how-the-gh200-fit--zero-3-works-and-known-issues)).

The versioned manifests in [`deploy/cks/train/`](deploy/cks/train/) (`encord-trossen-lora-v12-community.yaml`,
`…-v13-alldata.yaml`, `…-v13-regen2k.yaml`, …) are **data-curation experiments** — same recipe, different
dataset version — and are the template for your own variant.

```bash
# Stage this repo onto the data PVC (training pods run from /data/src/dreamzero-wam).
kubectl -n <YOUR_NAMESPACE> apply -f deploy/cks/infra/stager.yaml
tar czf - --exclude .git --exclude __pycache__ -C .. dreamzero-wam \
  | kubectl -n <YOUR_NAMESPACE> exec -i wam-stager -- sh -c 'rm -rf /data/src/dreamzero-wam && tar xzf - -C /data/src'

# Launch the validated single-GH200 run (WANDB_RUN_ID must be unique per launch):
RID="trossen$(date +%m%d%H%M)"
sed "s/WAMRUNIDPLACEHOLDER/$RID/" deploy/cks/train/encord-trossen-train-single.yaml | kubectl apply -f -

# Watch
kubectl -n <YOUR_NAMESPACE> get pods -l app=wam-train
kubectl -n <YOUR_NAMESPACE> logs -l app=wam-train --prefix -f | grep -vE "DEPRECATION|not on PATH"
```

The run appears at `https://wandb.ai/<YOUR_WANDB_ENTITY>/<YOUR_WANDB_PROJECT>`. On success it logs
`dreamzero-trossen-lora:<ver>` and links it to the `model` Registry, with lineage back to the dataset and
base models.

> **Smoke vs full run.** Set `MAX_STEPS=2` for a fast wiring check. LoRA checkpoints are small (~39 MB
> adapter); `save_only_model=true` skips the optimizer state (not needed for a fine-tune). Scale via the
> [env knobs](#configuration-env-vars). `save_total_limit` + `upload_checkpoints` are raised by default so
> peak mid-training checkpoints are tracked, not pruned.

### Reading results in W&B / Weave

- **Project** `<YOUR_WANDB_ENTITY>/<YOUR_WANDB_PROJECT>` — every run (`register-model`, `preprocess`,
  `train`, `eval`). Open a `train` run → **Overview → Lineage** to see input artifacts + output checkpoint.
- **Registry** (`<YOUR_WANDB_ORG>`) — `wandb-registry-model` (base weights + trained checkpoints) and
  `wandb-registry-dataset` (dataset variants). Each collection's versions are what you compare.
- **Weave** — eval rollouts trace under the run (`weave.init` inside `wandb.init`, same project). Each
  episode is an `EvaluationLogger` prediction with per-episode scores and **sim‖dream side-by-side video**;
  a stable `dataset` + varying `model` gives a leaderboard for comparing checkpoints side-by-side.

---

## The Encord data pipeline

Real Trossen data lives in **Encord**, backed by S3. This pipeline registers it, exports it, captions it,
and **assembles it into versioned, trainable W&B LeRobot v2.0 datasets** — with full source + label
lineage so W&B is the source of truth for *which data trained which model*. Full details:
[`scripts/encord/README.md`](scripts/encord/README.md).

The export is split across two artifact families that overlay into one dataset:

- **`encord-source-data:vN`** — the camera streams (3 cams: `exterior_image_1_left`, `wrist_image_left`,
  `wrist_image_right`), written to LeRobot/DROID-style paths.
- **`encord-labels:vN` / `encord-captions:vN`** — per-episode `state[16]` / `action[16]` / `task_index` +
  language instructions, rewritten from Encord captions.

```bash
export ENCORD_SSH_KEY_FILE=/path/to/encord_ssh_private_key
aws sso login --profile <YOUR_AWS_PROFILE>

# 1. Register S3 data into Encord (Cloud Synced Folders + metadata). See scripts/encord/README.md.
cd scripts/encord/data-registration
uv run --script update_cloud_synced_folder_metadata.py registration.json --dry-run
uv run --script update_cloud_synced_folder_metadata.py registration.json

# 2. Export the 3-camera data groups → W&B video artifact (encord-source-data:vN)
AWS_PROFILE=<YOUR_AWS_PROFILE> uv run --script scripts/encord/dataset-export/export_encord_dataset_to_wandb.py \
  --dataset-hash <encord_dataset_hash>            # add --limit 3 for a smoke

# 3. Export single-view captions/labels → W&B label overlay (encord-labels / encord-captions:vN)
#    (--metadata-yaml defaults to scripts/encord/label-export/label_export_config.yaml)
uv run --script scripts/encord/label-export/export_single_view_labels_to_wandb.py \
  --source-artifact-ref encord-source-data:vN
```

Then **assemble** the source + label artifacts into one trainable dataset. On the cluster this runs as a
CPU job that pulls the artifacts, merges parquet + videos, writes a self-consistent `meta/info.json`, and
runs [`convert_lerobot_to_gear.py`](scripts/data/convert_lerobot_to_gear.py) to emit
`modality.json` / `embodiment.json` / `stats.json`:

```bash
# Assemble → versioned trainable dataset artifact (e.g. the all-data v13 build)
kubectl -n <YOUR_NAMESPACE> apply -f deploy/cks/data/wam-encord-v13-assemble.yaml
```

The assemblers in [`scripts/data/`](scripts/data/) (`assemble_v12.py`, `assemble_v13.py`,
`assemble_dagger_v14.py`, …) are the **curation experiments**: each mixes real Encord Trossen data
(task-diverse captioned episodes) with community data and/or corrective demos, and logs the result as a new
version of the `trossen` dataset collection. `build_encord_dataset.py` is the base 2-artifact merge; the
`assemble_*` scripts extend it with mixing, re-indexing, and caption→task-index remapping.

> **The 16-dim Trossen action space.** Packing is
> `[left_joint_0..5, left gripper, right_joint_0..5, right gripper, linear_vel, angular_vel]`, q99-normalized,
> registered under the `trossen` embodiment tag (projector `trossen:32`, data config `trossen_relative`).
> Registering a genuinely new action space is one of the reusable outcomes here.

---

## The evaluation suite (authored from scratch)

**Nothing existed to evaluate this model.** The reference eval (`arhanjain/sim-evals` and the internal
`dreamzero-evals` built on it) is hard-wired to the **DROID** embodiment — Franka 7-DOF, 8-dim actions, 2
cameras. Our model is **16-dim bimanual with 3 cameras**, so the entire rollout/eval stack in
[`eval/isaaclab_arena_dreamzero/`](eval/isaaclab_arena_dreamzero/) was written from scratch on **NVIDIA
Isaac Lab–Arena**, keeping only the *conventions* of the Weave layer so results look familiar. Full details:
[`eval/README.md`](eval/README.md).

Two complementary evals share the same Weave structure:

### 1. Offline real-data eval — [`eval/offline_eval.py`](eval/offline_eval.py) · ✅ primary metric
Replays **real held-out Trossen episodes** through the model and scores **predicted vs ground-truth
actions** (`action_mse` / `action_mae` / `gripper_mae`). Evaluated on the exact training distribution →
**no sim-to-real gap**, so it is the trustworthy action-quality number (representative run: mean
`action_mse ≈ 0.117`). Pure CPU client against the policy server — no Isaac Sim. Per episode it logs the
**real** 3-cam footage, the model's **dream** video, and the two **side-by-side** in Weave.

```bash
# Stand up the inference server (loads the checkpoint), then run the CPU eval job:
kubectl -n <YOUR_NAMESPACE> apply -f eval/deploy/trossen-inference.yaml        # RTX/GH200 node; scale to 1
kubectl -n <YOUR_NAMESPACE> apply -f eval/deploy/trossen-offline-eval-job.yaml # CPU node
```

### 2. Closed-loop sim eval — [`eval/run_trossen_eval.py`](eval/run_trossen_eval.py) · Isaac Lab–Arena
The robot **acts in physics**: Isaac Sim renders the 3 cameras → our policy → 16-dim actions → env steps →
success metric. This is the closed-loop behavior check. We authored the whole stack:

- [`embodiments/trossen.py`](eval/isaaclab_arena_dreamzero/embodiments/trossen.py) — the **Trossen
  embodiment**: `mobile_ai.usd` articulation, 16-dim joint-position action in the trained order, 3 cameras
  at the training keys, and a real-rest spawn pose so the model starts in-distribution.
- [`environments/trossen_pick_and_place.py`](eval/isaaclab_arena_dreamzero/environments/trossen_pick_and_place.py)
  — a **custom scene** with the work table raised into the tall robot's reach, objects, lighting, and a
  `PickAndPlaceTask` success predicate.
- [`policy/`](eval/isaaclab_arena_dreamzero/policy/) — `DreamZeroRemotePolicy` + the 3-cam / 16-dim
  `DreamZeroTrossenAdapter`, modeled on Arena's `isaaclab_arena_openpi`.
- [`server/`](eval/server/) — the **DreamZero Trossen inference server** (3 cams / 16-dim packed action,
  packs the dream video into responses).
- [`weave_eval.py`](eval/isaaclab_arena_dreamzero/weave_eval.py) + [`video_logging.py`](eval/isaaclab_arena_dreamzero/video_logging.py)
  — Weave tracing, the `EvaluationLogger` leaderboard, and the sim / dream / side-by-side mp4 builder.

```bash
kubectl -n <YOUR_NAMESPACE> apply -f eval/deploy/trossen-inference.yaml     # policy server (scale to 1)
kubectl -n <YOUR_NAMESPACE> apply -f eval/deploy/trossen-eval-job.yaml      # Isaac Sim image → runner.sh → run_trossen_eval.py
```

---

## Reusing this for a new embodiment

The point of the repo. To fine-tune + evaluate a WAM on a *different* robot:

1. **Data** — export it through the Encord pipeline (or supply any LeRobot v2.0 dataset), then log it as a
   W&B dataset artifact. Mirror [`build_encord_dataset.py`](wam/artifacts/build_encord_dataset.py) /
   `assemble_*.py`: register the new packed action layout + camera keys and emit
   `modality.json` / `embodiment.json` / `stats.json`.
2. **Train** — copy a `deploy/cks/train/encord-trossen-*.yaml` manifest, point `WAM_DATASET_LOCAL_DIR` /
   `WAM_DATASET_SOURCE_ARTIFACTS` at the new dataset, set `DATA_CONFIG` / `DATA_ROOT_KEY` for the new
   embodiment. The harness (`wam/train.py`, ZeRO-3 fit, lineage, Registry linking) is unchanged.
3. **Evaluate** — for offline eval, only the modality contract changes. For sim, copy
   [`embodiments/trossen.py`](eval/isaaclab_arena_dreamzero/embodiments/trossen.py) as a template for the
   new articulation + cameras and add a scene/task; the policy adapter, server, and Weave layer carry over.
   To evaluate a *new checkpoint of the same robot*, just set the server's `MODEL_DIR` and the eval's
   `WEAVE_MODEL` — no code changes.

---

## Configuration (env vars)

Set on the training Job (see the manifest `env:` block). All have sensible defaults.

| Var | Default | Meaning |
|---|---|---|
| `WANDB_ENTITY` / `WANDB_PROJECT` | `<YOUR_WANDB_ENTITY>` / `<YOUR_WANDB_PROJECT>` | where runs + artifacts land |
| `WAM_REGISTRY_ORG` | `<YOUR_WANDB_ORG>` | org that owns the Registry |
| `DATA_CONFIG` | `dreamzero/trossen_relative` | embodiment data config (action space + modality) |
| `WAM_DATASET_LOCAL_DIR` | — | assembled LeRobot dataset dir on the PVC to train on |
| `WAM_DATASET_SOURCE_ARTIFACTS` | — | comma-sep artifact refs recorded as input lineage |
| `MAX_STEPS` | `300` | training steps (`2` = fast wiring check) |
| `SAVE_STEPS` | `100` | checkpoint interval |
| `TRAIN_ARCHITECTURE` | `lora` | `lora` or `full` |
| `PER_DEVICE_BATCH_SIZE` | `1` | per-GPU batch |
| `DEEPSPEED` | `configs/deepspeed/zero3_offload.json` (single-GH200) | DeepSpeed config; empty = none |
| `MODEL_DTYPE` | `bfloat16` | resident model dtype |
| `USE_AGIBOT_INIT` | `0` | `1` = init policy from DreamZero-AgiBot (new-embodiment transfer; needs more GPU) |
| `WAM_CKPT_ARTIFACT_NAME` | `dreamzero-trossen-lora` | output checkpoint artifact name |

---

## Repo layout

```
dreamzero-wam/
├── groot/                      # upstream DreamZero model + trainer — VERBATIM, untouched
├── wam/                        # the W&B training layer
│   ├── config.py               # entity/project/registry + PVC paths
│   ├── artifacts/              # bootstrap_models.py, build_encord_dataset.py, build_pickplace_subset.py
│   ├── wandb_utils.py          # use_artifact resolve + log_checkpoint_artifact helpers
│   ├── train.py                # artifact-driven training entrypoint
│   └── _ds_*.py                # ZeRO-3 compatibility shims (VAE leaf modules, ckpt routing, launch)
├── scripts/
│   ├── encord/                 # Encord register → export → caption pipeline (see its README)
│   ├── data/                   # assemble_*.py (dataset curation) + LeRobot↔GEAR converters
│   └── train/                  # encord_trossen.sh, droid_pickplace.sh
├── eval/                       # the Trossen eval suite (authored from scratch — see eval/README.md)
│   ├── isaaclab_arena_dreamzero/ # embodiment + scene/task + policy adapter + Weave layer
│   ├── server/                 # DreamZero Trossen inference server (3 cam / 16-dim)
│   ├── offline_eval.py         # offline real-data action eval (primary metric)
│   ├── run_trossen_eval.py     # closed-loop Isaac Lab–Arena sim eval
│   └── deploy/                 # eval + inference k8s manifests
├── deploy/cks/
│   ├── infra/                  # wandb-secret-setup.md, stager pod, PVC notes
│   ├── bootstrap/              # 00-models-download.yaml, 0N-encord-dataset*.yaml
│   ├── data/                   # wam-encord-vNN-assemble.yaml (dataset assembly jobs)
│   └── train/                  # encord-trossen-train-{single,full}.yaml + versioned experiments
├── configs/deepspeed/          # zero3_offload.json (single-GH200 CPU offload)
├── reports/                    # the WAM fine-tuning write-up + sim investigation
└── requirements-train.txt      # additive training deps on top of the nvcr pytorch image
```

---

## How the GH200 fit / ZeRO-3 works (and known issues)

The 14B video-diffusion model + the ~29k-token (33-frame) sequence does **not** fit a single 96 GB GH200 at
full precision, so DeepSpeed ZeRO-3 is used. The **validated** path is **single GH200 with ZeRO-3 CPU param
offload** ([`configs/deepspeed/zero3_offload.json`](configs/deepspeed/zero3_offload.json)): params live on
the Grace 480 GB host and stream to the GPU over NVLink-C2C. Making the upstream model work under ZeRO-3
needed a few **non-architectural** shims (all in `wam/_ds_*`, applied via a launch shim so `groot/` stays
verbatim):

- **nvtx** — the nvcr image's `nvtx` lacks the API pip-DeepSpeed calls → disabled in the boot script.
- **VAE / encoders** — marked ZeRO-3 *leaf modules* (their `isinstance`-driven conv-cache breaks if ZeRO-3
  hooks their internals).
- **activation checkpointing** — routed the DiT's `torch.utils.checkpoint` → DeepSpeed's ZeRO-3-aware
  checkpoint, wrapped in `autocast(bf16)`.
- **W&B run handoff** — the launcher finishes the run so the trainer can resume it for metrics, then reopens
  it to log the checkpoint.
- **`save_only_model=true`** — under ZeRO-3 the offloaded optimizer state is a single ~36 GB `torch.save`
  that fails on the VAST PVC; a fine-tune doesn't need it, so only the ~39 MB LoRA checkpoint is saved.

**Known issue — multi-node runs need RDMA.** The 2× GH200 manifest gets `NET/Socket message truncated` and
then an intermittent ZeRO-3 all-gather hang → watchdog timeout, because the GH200 pods have **no
RDMA/InfiniBand** (NCCL falls back to plain TCP over `eth0`). The single-GH200 offload path avoids
cross-node NCCL entirely and is what we run. The fix for reliable multi-node is RDMA/IB (CoreWeave RDMA NICs
into the pods + IMEX + NCCL IB env), not TCP-socket tuning.

**Known limiter — sim-to-real visual gap.** The model trained on *real* Trossen frames; Isaac renders
synthetic assets, so closed-loop *success* can read low until the rendered scene resembles training. Treat
sim success as a behavior check; the offline eval is the trustworthy action-quality number.

---

## License

Apache-2.0 (see [LICENSE](LICENSE) / [COPYRIGHT](COPYRIGHT)). Upstream DreamZero code in [`groot/`](groot/)
retains its original license.
