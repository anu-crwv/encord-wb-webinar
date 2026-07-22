# DreamZero Trossen — sim closed-loop failure investigation & fix results

**Goal:** a working Isaac-Sim demo of the fine-tuned Trossen DreamZero (WAM) model doing
autonomous pick-place. **Status: not yet achievable with the current model.** The blocker
is now precisely localized, every cheap/medium fix has been tested, and the remaining path
is scoped. This report covers the full investigation and the 5-task fix program.

## 1. Executive conclusion

The sim hover is **closed-loop covariate shift**, not the hypotheses we (and the earlier
notes) chased. Ruled out *with evidence*: action normalization, abs/delta, stale stats,
joint mapping, controller plumbing, chunk truncation, static visual OOD, sim-state
semantics, session/KV-cache state, dream-video coupling, and — newly — the idle-prefix
training bias as the *dominant* cause.

What holds up: the action head is **well-directed and full-amplitude given an
in-distribution, motion-bearing observation context** (offline, teacher-forced real
trajectories: amplitude ~0.9–1.1, direction corr 0.73–0.88). It **fails only in
self-generated closed-loop sim rollout**: from rest it commits ~0.04 rad directionless
jitter, the gripper never approaches the object (min distance pinned at the ~0.32 m start),
and it drifts away. This is the classic imitation-learning compounding-error problem: the
policy is fine on the expert manifold but cannot recover once its own (visually-OOD) sim
rollout leaves it.

**Confidence: high.** Two independent fix attempts confirm it:
- an **inference-time motion kickstart** does not fix it (0/40 rollouts reach; no
  object-homing after handover, across seed length, hold, and direction);
- a **targeted retrain** that removes the idle-prefix bias retains offline quality
  (amp 0.93) but leaves closed-loop **unchanged** (0/8 reach) — so initiation was not the
  bottleneck.

## 2. Evidence chain (all committed on `encord-v6-data-weave-eval`)

| test | result | rules out / shows |
|---|---|---|
| Norm/abs-delta/stats audit | baked q99 match v6 0.96–1.06×; `relative_action:false`; only `trossen` key | not normalization/units/stale |
| Chunk-shape + GT replay | chunk flat/front-loaded (79% by step 4); GT actions → sim reaches (0.386 m descent) | not truncation; not controller |
| 2×2 modality swap | all 4 cells commit ~0.16 rad; sim image keeps 101% of real | **not static visual OOD**; not state |
| Persistence probe | persistent ≡ fresh on identical input (real & sim) | not KV-cache/session state |
| AR-ladder (teacher-forced moving sim) | direction corr **0.73** (sim) ≈ 0.75 (real); dreamed ≈ observed | head is directed **given motion context**; not dream coupling |
| Closed-loop trajectory | born directionless ~0.04 jitter from rest, no ramp into reach | the failure is self-generated closed-loop |
| Dataset start-window audit | idle prefix median ~65 frames; first-actions ~0.0006 rad ("hold still") | a **real** data defect (idle-prefix bias) |
| **#1 inference kickstart sweep** | 0/40 reach; post-handover ≈ 0; longer seeds/holds worse | inference seed **insufficient** |
| **#3 step-filter retrain (2000 steps)** | offline amp **0.93**, direction 0.44–0.90 (not degraded) | retrain kept quality |
| **#4 A/B (retrained vs baseline)** | retrained no-seed **0/8 reach, −0.058 m** ≈ baseline v6 | idle-prefix fix **did not** fix closed loop |

## 3. What works vs what fails

**Works (model is genuinely good at):**
- Offline action prediction on real held-out data — amplitude ~0.9–1.1, direction 0.86.
- World-model "dream" (plausible future video) — rides the pretrained WAN backbone.
- Directed action **when its context already contains in-distribution motion**.
- The sim embodiment/controller executes correct actions faithfully (GT replay reaches).

**Fails:**
- Autonomous closed-loop reaching in sim from rest — hovers/drifts, 0 reach, 0 grasp.
- Not rescued by: more data volume (v6), CFG, chunk horizon, domain-matched rendering,
  motion kickstart, or idle-prefix removal.

## 4. Recommendation

The remaining blocker (closed-loop covariate shift + sim visual/dynamics OOD) is a
**training-distribution** problem, not an inference trick. In priority order:

1. **Sim co-training with action-labeled sim rollouts (principled fix).** Render the sim
   along known-good trajectories (scripted/IK/teleop or replayed real joint paths), pair
   each sim observation with its *exact* executable action, and co-train the head on
   real + action-aligned sim (balanced, keep the clean-video objective). This teaches the
   head to act on its own (sim) observation distribution — directly addresses covariate
   shift and sim OOD. Estimated multi-week; the highest-confidence route to an autonomous
   sim demo.
2. **Real closed-loop / recovery data (DAgger-style)** if the demo must be real-robot:
   collect states the policy actually visits + correct actions (randomized starts,
   overshoot/missed-grasp recovery). A small pilot first.
3. **A longer/higher-rank retrain is NOT the lever.** Confirmed on the final artifact
   `dreamzero-trossen-lora:v7` (checkpoint-3000, train_loss 0.098): no-seed closed-loop
   still 0-reach, drifts away (post-handover −0.02 to −0.07, min-dist ~0.32) — identical
   to ckpt-2000 and to the fully-cooked v6@8000. Training length/quality does not move the
   closed-loop result. The step-filter remains a legitimate data fix to keep in the recipe,
   but it is not the fix for the demo.

**Realistic demo reframe (if a demo is needed before the above lands):** demo what the
model *does* do well — its **real-data action prediction + world-model dream** (offline,
on held-out real Trossen episodes: amplitude/direction + the dream video), rather than
autonomous sim closed-loop. This is honest and strong. A fully-scripted sim pick-place
with the model "advising" is possible but would not be a faithful model demo.

## 5. Deliverables / reproducibility

- Diagnostics (branch `encord-v6-data-weave-eval`): `eval/diag_chunk_shape.py`,
  `diag_2x2_modality.py` (+`diag_render_at_poses.py`), `server/probe_persistence.py`,
  `diag_ar_ladder.py`, `run_trossen_kickstart.py`, `scripts/data/{audit_trossen_starts,
  gen_trossen_step_filter}.py`, and the `eval/deploy/*.yaml` job manifests.
- Data fix: `meta/step_filter.jsonl` on the v6 dataset (idle-prefix anchors removed,
  metadata-only). Retrain: `encord_trossen_lora_v6trim0721` (fresh LoRA @3000 on
  step-filtered v6). Served for A/B as `dreamzero-trossen-inference-rt` (checkpoint-2000).
- All raw per-rollout numbers are in the job logs; success gate (post-handover progress →
  pre-grasp reach → grasp/place) applied throughout — **no config passed the first gate.**
