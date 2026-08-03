# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "numpy",
#     "pyyaml",
# ]
# ///
"""Load Trossen caption templates and infer active arms from joint motion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CLASSIFICATION_TITLES = (
    "Language Instruction 1",
    "Language Instruction 2",
    "Language Instruction 3",
)
DEFAULT_TASK_CAPTIONS_PATH = Path(__file__).resolve().parents[1] / "task_captions.yaml"
SOURCE_PARQUET_COLUMNS = ("observation.state", "action")
LEFT_ARM_SLICE = slice(0, 7)
RIGHT_ARM_SLICE = slice(7, 14)
DEFAULT_ACTIVITY_THRESHOLD = 1e-4
DEFAULT_DOMINANCE_RATIO = 2.5


@dataclass(frozen=True)
class TaskCaptionTemplate:
    canonical: str
    paraphrase: str
    arm_template: str


def required_text(value: Any, *, task_name: str, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Task {task_name!r} has an empty {field!r} value.")
    return text


def load_task_captions(path: Path = DEFAULT_TASK_CAPTIONS_PATH) -> dict[str, TaskCaptionTemplate]:
    if not path.is_file():
        raise ValueError(f"Caption map does not exist: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        raise ValueError("Caption map must contain a top-level 'tasks' mapping.")

    templates: dict[str, TaskCaptionTemplate] = {}
    for raw_name, raw_template in payload["tasks"].items():
        task_name = str(raw_name).strip()
        if not task_name:
            raise ValueError("Caption map contains an empty task name.")
        if not isinstance(raw_template, dict):
            raise ValueError(f"Task {task_name!r} must map to an object.")
        unexpected = set(raw_template) - {"canonical", "paraphrase", "arm_template"}
        if unexpected:
            raise ValueError(
                f"Task {task_name!r} has unsupported fields: {sorted(unexpected)}"
            )
        arm_template = required_text(
            raw_template.get("arm_template"),
            task_name=task_name,
            field="arm_template",
        )
        if arm_template.count("{arm_phrase}") != 1:
            raise ValueError(
                f"Task {task_name!r} arm_template must contain '{{arm_phrase}}' exactly once."
            )
        templates[task_name] = TaskCaptionTemplate(
            canonical=required_text(
                raw_template.get("canonical"),
                task_name=task_name,
                field="canonical",
            ),
            paraphrase=required_text(
                raw_template.get("paraphrase"),
                task_name=task_name,
                field="paraphrase",
            ),
            arm_template=arm_template,
        )
    if not templates:
        raise ValueError("Caption map contains no tasks.")
    return templates


TASK_CAPTIONS = load_task_captions()


def caption_variants_for_task(
    task_name: str,
    arm_phrase: str = "the robot arm",
    task_captions: dict[str, TaskCaptionTemplate] | None = None,
) -> tuple[str, str, str]:
    template = (task_captions or TASK_CAPTIONS)[task_name]
    return (
        template.canonical,
        template.paraphrase,
        template.arm_template.format(arm_phrase=arm_phrase),
    )


def vectors_from_table(table: Any, column: str) -> Any:
    import numpy as np

    if column not in table.column_names:
        raise ValueError(f"Source Parquet is missing required column: {column}")
    vectors = np.asarray(table[column].to_pylist(), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] < 14:
        raise ValueError(f"{column} must contain vectors with at least 14 values")
    return vectors


def robust_motion_score(values: Any) -> float:
    import numpy as np

    if values.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return float(np.percentile(norms, 95)) if norms.size else 0.0


def infer_arm_phrase_from_arrays(
    observation_state: Any,
    action: Any,
    *,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    dominance_ratio: float = DEFAULT_DOMINANCE_RATIO,
) -> str:
    left_score = robust_motion_score(observation_state[:, LEFT_ARM_SLICE]) + (
        0.25 * robust_motion_score(action[:, LEFT_ARM_SLICE])
    )
    right_score = robust_motion_score(observation_state[:, RIGHT_ARM_SLICE]) + (
        0.25 * robust_motion_score(action[:, RIGHT_ARM_SLICE])
    )

    if max(left_score, right_score) < activity_threshold:
        return "the robot arm"
    if left_score > right_score * dominance_ratio:
        return "the left arm"
    if right_score > left_score * dominance_ratio:
        return "the right arm"
    return "both robot arms"


def infer_arm_phrase_from_table(table: Any) -> str:
    return infer_arm_phrase_from_arrays(
        vectors_from_table(table, "observation.state"),
        vectors_from_table(table, "action"),
    )
