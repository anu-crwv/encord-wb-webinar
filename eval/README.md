# Evaluation — Trossen DreamZero sim eval (Isaac Lab-Arena)

A simulation eval for the fine-tuned **Trossen** DreamZero model (`dreamzero-trossen-lora`,
bimanual mobile, 16-dim action, 3 RGB cameras), built on **NVIDIA Isaac Lab-Arena**
(`github.com/isaac-sim/IsaacLab-Arena`) — the general, embodiment-swappable successor to the
DROID-only `arhanjain/sim-evals`. Arena runs the sim + camera render on an RTX GPU and calls our
model over a websocket policy server, exactly like the bundled `isaaclab_arena_openpi` (pi0)
adapter.

> Scoping/decision rationale lives in the plan doc; this README is the implementation reference.

## Architecture

```
Arena policy_runner (RTX node, Isaac Sim 6.0)            DreamZero server (GH200 or RTX)
  ├─ Trossen embodiment + scene + task   ── obs ──▶  DreamZeroRemotePolicy  ── ws ──▶  GrootSimPolicy
  │   (mobile_ai.usd, 3 TiledCameras)                 (this repo, eval/)        :8001    (LoRA + trossen
  └─ success predicate + metrics  ◀── 16-dim action ◀──  + Trossen adapter              transform)
```

The runner instantiates the policy by **dotted import path** (no Arena registry edit needed):
`isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy`.

## What's in this repo (done — `eval/`)

- **`isaaclab_arena_dreamzero/`** — the DreamZero policy adapter for Arena, modeled on
  `isaaclab_arena_openpi`:
  - `policy/dreamzero_remote_policy.py` — `DreamZeroRemotePolicy(PolicyBase)`: connects to the
    DreamZero server with `openpi_client`'s `WebsocketClientPolicy`, does per-env open-loop chunk
    replay, and tolerates the server's `actions` / `action` / split `action.*` response shapes.
    Plus the `DreamZeroEmbodimentAdapter` ABC (`extract` + `pack_request`).
  - `policy/trossen_adapter.py` — `DreamZeroTrossenAdapter`: the Trossen wire mapping
    (`action_dim=16`; 3 RGB views + 16-dim packed `state` + prompt). Keys mirror the trained
    `modality_config_trossen`.
  - `policy/dreamzero_remote_config.py` — connection/runtime config dataclass.
- **`jobs_configs/trossen_pnp_dreamzero_jobs_config.json`** — eval jobs config; `policy_*` fields are
  complete, `environment`/`embodiment` are `TODO_` until the embodiment is authored.

### Verified interface contracts (so the remaining pieces line up)

| Concern | Contract |
| --- | --- |
| Policy selection | `policy_runner.get_policy_cls()` accepts a dotted import path → no registry change |
| Wire client | `openpi_client.websocket_client_policy.WebsocketClientPolicy` (same as the DROID eval) |
| Server response | `(H, 16)` under `actions` / `action`; `H >= open_loop_horizon` (default 5) |
| Trossen video keys | `video.exterior_image_1_left`, `video.wrist_image_left`, `video.wrist_image_right` |
| Trossen state/action | packed `state.state` (16), `action.action` (16), q99-normalized server-side |
| Arena obs groups | cameras under `camera_obs`, proprio under `policy` |

## Remaining work (needs the cluster RTX-PRO-6000 nodes + the DreamZero server repo)

1. **DreamZero server — Trossen input mapping** (server-side; lives in upstream DreamZero, not here).
   Generalize `PolicyServerConfig` from the DROID defaults (2 cams / 8-dim) to **3 cams / 16-dim**, and
   map the request keys this adapter sends (`observation/exterior_image_1_left`,
   `observation/wrist_image_left`, `observation/wrist_image_right`, `observation/state`, `prompt`) onto
   the trossen modality keys. The checkpoint loads via
   `GrootSimPolicy(embodiment_tag=EmbodimentTag.TROSSEN, model_path=…)` — tag, projector index
   (`trossen: 32`), and `trossen_relative` data config are already registered;
   `load_lora`+`merge_and_unload` already handle the LoRA. **If the server's request key names differ,
   adjust the constants in `trossen_adapter.py`** (they are the single source of truth on our side).
2. **Trossen embodiment in Arena** — register a new Arena embodiment from `trossen_ai_isaac`'s
   `mobile_ai.usd`: dual-arm articulation with a `JointPositionAction` (`joint_[0-5]`×2 + 2 grippers +
   base ≈ 16-dim), a 16-dim `state` proprio term under `policy` (ordered to match training), and **3
   `TiledCamera`s** under `camera_obs` keyed `exterior_image_1_left` / `wrist_image_left` /
   `wrist_image_right`, posed/FoV-matched to the real cameras.
3. **Scene + task + success predicate** — one simple bimanual pick-place (cube in bowl) with a
   programmatic success check (grasped + placed within XY/Z thresholds), mirroring how
   `deploy/cks/scripts/droid_weave_eval.py` scores DROID.
4. **Metrics → Weave** — Arena ships its own metrics/video/report. Either add a thin Weave sink
   (reuse `weave.init` / `EvaluationLogger` / `VideoFileClip` from `droid_weave_eval.py`, leaderboard
   `dataset=trossen-sim-<task>` stable, `model=dreamzero-trossen-lora:<ver>` varying) or post-process
   Arena's report into Weave.
5. **Orchestration** — `eval/runner.sh` (clone Arena + trossen_ai_isaac → install deps → put
   `eval/isaaclab_arena_dreamzero` on `PYTHONPATH` → TCP-probe the server → run policy_runner with the
   jobs config) + k8s Job (RTX node, Isaac Sim) and the DreamZero server as a Service, on an
   **Isaac Sim 6.0.0 + Arena (amd64)** image.

## Primary caveat
Our world-action model trained on **real** Trossen frames; Isaac renders differ → sim success can read
~0 regardless of model quality. The first end-to-end milestone proves the **pipeline is green**, not a
quality verdict. Mitigations: camera-pose/FoV matching, lighting/texture domain randomization, and
reporting action-space agreement alongside task success.

## Reference clones (not committed)
`/Users/anu/Projects/IsaacLab-Arena` and `/Users/anu/Projects/trossen_ai_isaac` were cloned for API
reference. Templates: `isaaclab_arena_openpi/policy/{pi0_remote_policy,droid_adapter}.py` (wire format)
and `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py` (groot-family obs/action semantics).
