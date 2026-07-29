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
import re

def _video_media(path):
    """Wrap an mp4 PATH as a weave.Content(mimetype='video/mp4') — weave's native, documented
    renderable-file type, which shows a video PLAYER on the trace. We deliberately do NOT use a
    moviepy VideoFileClip: weave's auto moviepy handler is broken in the installed 0.53.x (its
    custom-type serializer op fails to load — "Op loading exception ... name 'true' is not defined"),
    so clips fell back to the "<VideoFileClip object at 0x..>" repr instead of a video. The Content
    API sidesteps that handler entirely and is what the weave docs recommend for video."""
    try:
        import os as _os
        if not (path and _os.path.exists(path)):
            return None
        import weave as _w
        try:
            return _w.Content.from_path(path, mimetype="video/mp4")
        except TypeError:  # older/newer from_path signature
            return _w.Content.from_path(path)
    except Exception as e:  # noqa: BLE001
        print(f"[weave_eval] Content.from_path failed for {path}: {e}", flush=True)
        return None


# A @weave.op that builds weave.Content video objects from mp4 PATHS and RETURNS them as top-level
# output keys, so episode_video / dream_video / side_by_side render as players on the run_episode
# trace. Defined at import (guarded); traces once weave.init() has run.
try:
    import weave as _weave

    @_weave.op(name="run_episode")
    def _rollout_trace(inputs: dict, metrics: dict, video_paths: dict) -> dict:
        vp = video_paths or {}
        return {
            **(metrics or {}),
            "episode_video": _video_media(vp.get("episode_video_path")),
            "dream_video": _video_media(vp.get("dream_video_path")),
            "side_by_side": _video_media(vp.get("side_by_side_path")),
        }
except Exception:  # weave not importable in this context -> no tracing (guarded everywhere)
    _rollout_trace = None


def sanitize_label(s: str, fallback: str = "model") -> str:
    """Weave's EvaluationLogger (>=0.51) validates model/dataset/scorer names as identifiers:
    must start with a letter/underscore and contain only [A-Za-z0-9_]. Checkpoint refs like
    'dreamzero-trossen-lora-v6:ckpt-8000' have '-'/':' and fail — sanitize to underscores."""
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(s or "").strip())
    if not s:
        s = fallback
    if not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s


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
    # WAM_EVAL_RESUME_RUN_ID: resume INTO an existing run (e.g. a training run) so the eval's
    # weave traces + summary attach to that experiment, instead of minting a fresh eval run.
    resume_id = os.environ.get("WAM_EVAL_RESUME_RUN_ID", "").strip()
    run = wandb.init(
        entity=entity,
        project=project,
        id=resume_id or None,
        resume=("allow" if resume_id else None),
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
    if resume_id:
        print(f"[weave_eval] resumed INTO run id={resume_id}", flush=True)
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


class WeaveEvalLogger:
    """Shared results logger for the sim harnesses (scripted expert + kickstart eval).

    Reuses init_eval_tracing() (W&B run + weave.init, same project) + a weave
    EvaluationLogger leaderboard (stable dataset=trossen_sim_<task>, varying model) +
    VideoFileClip rollout media — the SAME convention run_trossen_eval.py uses — so every
    sim result renders side-by-side in the wam-finetune-webinar Weave workspace. Fully
    guarded: if weave/wandb are unavailable or disabled, all calls no-op (never breaks a run).
    """

    def __init__(self, model_label: str, task_name: str):
        self.run = init_eval_tracing()
        self.el = None
        self._n = 0
        if self.run is not None:
            try:
                from weave import EvaluationLogger
                self.el = EvaluationLogger(
                    model=sanitize_label(model_label, "model"),
                    dataset=sanitize_label(f"trossen_sim_{task_name}", "trossen_sim"),
                    name=sanitize_label(task_name, "task"),
                )
                print(f"[weave_eval] EvaluationLogger model={model_label} dataset=trossen_sim_{task_name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[weave_eval] EvaluationLogger init failed: {e}", flush=True)

    @staticmethod
    def _clip(path):
        try:
            import os
            from moviepy.editor import VideoFileClip
            return VideoFileClip(path, audio=False) if path and os.path.exists(path) else None
        except Exception as e:  # noqa: BLE001
            print(f"[weave_eval] VideoFileClip wrap failed: {e}", flush=True)
            return None

    def log_episode(self, inputs: dict, output: dict, scores: dict,
                    video_paths: dict | None = None, video_path: str | None = None) -> None:
        """One leaderboard prediction via the PROVEN run_episode pattern: a @weave.op builds the
        clips from mp4 PATHS and RETURNS them, so episode_video / dream_video / side_by_side render
        inline on the run_episode call (fixes the repr-only bug). ``video_paths`` is the dict from
        TrossenVideoLogger.build_episode_videos(); ``video_path`` is a single-mp4 shorthand. A
        wandb.Video of the sim mp4 is also logged to the run media panel as a belt-and-braces render."""
        if self.el is None:
            return
        try:
            vp = dict(video_paths or {})
            if video_path and "episode_video_path" not in vp:
                vp["episode_video_path"] = video_path
            # Build clips INSIDE the op (from paths) + return them -> weave renders as inline video.
            if _rollout_trace is not None:
                out = _rollout_trace(inputs=inputs, metrics=output, video_paths=vp)
            else:
                out = dict(output)
            pred = self.el.log_prediction(inputs=inputs, output=out)
            for k, v in scores.items():
                try:
                    sv = float(v)  # bool/int -> float so Weave AGGREGATES (success_rate etc.); raw bools render empty
                except (TypeError, ValueError):
                    sv = v
                pred.log_score(scorer=sanitize_label(k, "score"), score=sv)
            pred.finish()
            # belt-and-braces: sim mp4 to the W&B run media panel too
            sim_mp4 = vp.get("episode_video_path")
            if sim_mp4 and self.run is not None:
                try:
                    import os as _os
                    import wandb
                    if _os.path.exists(sim_mp4):
                        self.run.log({f"rollout/{sanitize_label(str(inputs.get('task', 'ep')))}":
                                      wandb.Video(sim_mp4, fps=15, format="mp4")})
                except Exception as e:  # noqa: BLE001
                    print(f"[weave_eval] wandb.Video log failed: {e}", flush=True)
            self._n += 1
        except Exception as e:  # noqa: BLE001
            print(f"[weave_eval] log_episode failed: {e}", flush=True)

    def log_summary(self, d: dict) -> None:
        if self.el is None:
            return
        try:
            self.el.log_summary(d)
        except Exception as e:  # noqa: BLE001
            print(f"[weave_eval] log_summary failed: {e}", flush=True)

    def finish(self) -> None:
        print(f"[weave_eval] logged {self._n} episodes to Weave", flush=True)
        finish_eval_tracing(self.run)
