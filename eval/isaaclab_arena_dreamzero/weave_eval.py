# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Weave + W&B tracing for the Trossen sim eval (mirrors the DreamZero DROID eval).

Implements the weave-in-workspaces pattern so eval rollouts map to the W&B
experiment in the workspace (https://docs.wandb.ai/weave/guides/tools/weave-in-workspaces):
a W&B run is started first, then ``weave.init`` with the SAME project — Weave then
auto-associates ``@weave.op`` traces with the active ``wandb.run``. Everything lands
in the ``wam-finetune-webinar`` project.

``init_eval_tracing()`` is called by ``run_trossen_eval.py`` before Arena's
``eval_runner.main()``. It also monkeypatches Arena's ``MetricsLogger`` so each
job's aggregate metrics (success_rate, object_moved_rate, num_episodes) are pushed
to the W&B run summary AND logged as a Weave ``EvaluationLogger`` leaderboard entry
(stable ``dataset`` per task, varying ``model`` = the checkpoint version) — the same
side-by-side comparison convention the DROID eval used.
"""

from __future__ import annotations

import os


def _model_label() -> str:
    """e.g. 'dreamzero-trossen-lora:v3' from LORA_ARTIFACT, else WEAVE_MODEL."""
    model = os.environ.get("WEAVE_MODEL", "dreamzero-trossen-lora")
    art = os.environ.get("LORA_ARTIFACT", "")
    if ":" in art:
        return f"{model}:{art.rsplit(':', 1)[-1]}"
    return model


def _eval_tags() -> list[str]:
    """W&B run tags for easy navigation. Auto-derives the checkpoint version +
    embodiment; extra tags (e.g. 'full-scale', '3000-steps', 'smoke') come from the
    comma-separated WAM_EVAL_TAGS env."""
    tags = ["eval", "sim-eval", "trossen", "isaac-lab-arena"]
    art = os.environ.get("LORA_ARTIFACT", "")
    if ":" in art:
        tags.append(art.rsplit(":", 1)[-1])  # e.g. v4
    tags += [t.strip() for t in os.environ.get("WAM_EVAL_TAGS", "").split(",") if t.strip()]
    return sorted(set(tags))


def init_eval_tracing():
    """Start the W&B run + Weave (same project) and install the metrics hook.

    Returns the wandb run, or None if disabled (WAM_EVAL_NO_WEAVE=1) or unavailable.
    """
    if os.environ.get("WAM_EVAL_NO_WEAVE") == "1":
        print("[weave_eval] disabled via WAM_EVAL_NO_WEAVE=1", flush=True)
        return None
    project = os.environ.get("WEAVE_PROJECT", "wam-finetune-webinar")
    entity = os.environ.get("WANDB_ENTITY") or None
    try:
        import wandb
        import weave
    except Exception as e:  # noqa: BLE001
        print(f"[weave_eval] wandb/weave unavailable ({e}); continuing without tracing", flush=True)
        return None

    # W&B run FIRST, then weave.init() with the same project -> traces auto-associate
    # with the active run (the "inside wandb.init()" mapping the workspace doc describes).
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="eval",
        name=os.environ.get("WAM_EVAL_RUN_NAME") or None,
        tags=_eval_tags(),
        config={
            "checkpoint": os.environ.get("LORA_ARTIFACT", ""),
            "model": _model_label(),
            "embodiment": "trossen_mobile_ai",
            "jobs_config": os.environ.get("EVAL_JOBS_CONFIG", ""),
        },
    )
    # Lineage: record the fine-tuned model artifact as an INPUT to this eval run, so
    # W&B shows dataset -> training run -> checkpoint artifact -> eval run.
    art = os.environ.get("LORA_ARTIFACT", "").strip()
    if art:
        try:
            run.use_artifact(art, type="model")
            print(f"[weave_eval] use_artifact({art}) recorded for lineage", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[weave_eval] use_artifact({art}) failed: {e}", flush=True)

    weave.init(project)
    print(f"[weave_eval] wandb run={run.id} + weave.init({project!r}); rollouts trace under this run", flush=True)
    return run


def finish_eval_tracing(run) -> None:
    if run is None:
        return
    try:
        import wandb

        wandb.finish()
    except Exception as e:  # noqa: BLE001
        print(f"[weave_eval] wandb.finish error: {e}", flush=True)
