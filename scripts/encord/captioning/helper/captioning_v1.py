"""Deterministic V1 caption variants for Trossen Encord datasets."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np


DEFAULT_ONTOLOGY_HASH = "a6aca77e-6283-4530-93ab-35a8681af0a0"
CLASSIFICATION_TITLES = (
    "Language Instruction 1",
    "Language Instruction 2",
    "Language Instruction 3",
)
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


TASK_CAPTIONS: dict[str, TaskCaptionTemplate] = {
    "Pour nuts & bolts": TaskCaptionTemplate(
        canonical="pour the nuts and bolts into the tray",
        paraphrase="empty the nuts and bolts into the tray",
        arm_template="use {arm_phrase} to pour the nuts and bolts into the tray",
    ),
    "Batteries": TaskCaptionTemplate(
        canonical="place the batteries into the tray",
        paraphrase="put the batteries in the tray",
        arm_template="use {arm_phrase} to place the batteries into the tray",
    ),
    "Pour Coffee 2": TaskCaptionTemplate(
        canonical="pour the coffee into the cup",
        paraphrase="pour coffee into the cup",
        arm_template="use {arm_phrase} to pour the coffee into the cup",
    ),
    "Sort glue by type": TaskCaptionTemplate(
        canonical="sort the glue bottles by type",
        paraphrase="group the glue bottles by type",
        arm_template="use {arm_phrase} to sort the glue bottles by type",
    ),
    "Sort tape & safety glasses (2)": TaskCaptionTemplate(
        canonical="sort the tape and safety glasses",
        paraphrase="separate the tape and safety glasses",
        arm_template="use {arm_phrase} to sort the tape and safety glasses",
    ),
    "Microfiber towels": TaskCaptionTemplate(
        canonical="fold the microfiber towels",
        paraphrase="fold the towels",
        arm_template="use {arm_phrase} to fold the microfiber towels",
    ),
    "Coil wire": TaskCaptionTemplate(
        canonical="coil the wire",
        paraphrase="wind the wire into a coil",
        arm_template="use {arm_phrase} to coil the wire",
    ),
    "Plug ethernet cable into network device": TaskCaptionTemplate(
        canonical="plug the ethernet cable into the network switch",
        paraphrase="connect the ethernet cable to the network switch",
        arm_template="use {arm_phrase} to plug the ethernet cable into the network switch",
    ),
    "Plug ethernet cable into network device 2": TaskCaptionTemplate(
        canonical="plug the ethernet cable into the network switch",
        paraphrase="connect the ethernet cable to the network switch",
        arm_template="use {arm_phrase} to plug the ethernet cable into the network switch",
    ),
    "Plug ethernet cable into network switch 3": TaskCaptionTemplate(
        canonical="plug the ethernet cable into the network switch",
        paraphrase="connect the ethernet cable to the network switch",
        arm_template="use {arm_phrase} to plug the ethernet cable into the network switch",
    ),
}


def caption_variants_for_task(task_name: str, arm_phrase: str) -> tuple[str, str, str]:
    template = TASK_CAPTIONS[task_name]
    return (
        template.canonical,
        template.paraphrase,
        template.arm_template.format(arm_phrase=arm_phrase),
    )


def vectors_from_table(table: Any, column: str) -> np.ndarray:
    if column not in table.column_names:
        raise ValueError(f"Source parquet missing required column: {column}")
    vectors = np.asarray(table[column].to_pylist(), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] < 14:
        raise ValueError(f"{column} must be a 2D vector column with at least 14 values")
    return vectors


def robust_motion_score(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    deltas = np.diff(values, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    if norms.size == 0:
        return 0.0
    return float(np.percentile(norms, 95))


def infer_arm_phrase_from_arrays(
    observation_state: np.ndarray,
    action: np.ndarray,
    *,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    dominance_ratio: float = DEFAULT_DOMINANCE_RATIO,
) -> str:
    left_score = (
        robust_motion_score(observation_state[:, LEFT_ARM_SLICE])
        + 0.25 * robust_motion_score(action[:, LEFT_ARM_SLICE])
    )
    right_score = (
        robust_motion_score(observation_state[:, RIGHT_ARM_SLICE])
        + 0.25 * robust_motion_score(action[:, RIGHT_ARM_SLICE])
    )

    if max(left_score, right_score) < activity_threshold:
        return "the robot arms"
    if left_score > right_score * dominance_ratio:
        return "the left arm"
    if right_score > left_score * dominance_ratio:
        return "the right arm"
    return "both arms"


def infer_arm_phrase_from_table(table: Any) -> str:
    return infer_arm_phrase_from_arrays(
        vectors_from_table(table, "observation.state"),
        vectors_from_table(table, "action"),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        return bucket, unquote(parsed.path.lstrip("/"))
    raise ValueError(f"Unsupported S3 URI format: {uri}")


def cache_path_for_s3_uri(cache_root: Path, uri: str) -> Path:
    bucket, key = parse_s3_uri(uri)
    key_parts = [part for part in key.split("/") if part]
    if any(part == ".." for part in key_parts):
        raise ValueError(f"Unsafe S3 key for local cache path: {key}")
    return cache_root / bucket / Path(*key_parts)


def download_s3_to_cache(client_s3: Any, uri: str, cache_root: Path) -> Path:
    cached = cache_path_for_s3_uri(cache_root, uri)
    if cached.exists():
        return cached

    bucket, key = parse_s3_uri(uri)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cached.with_name(f"{cached.name}.tmp-{os.getpid()}")
    try:
        client_s3.download_file(bucket, key, str(tmp_path))
        tmp_path.replace(cached)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return cached


def read_cached_parquet_table(
    client_s3: Any,
    uri: str,
    cache_root: Path,
    *,
    columns: tuple[str, ...] | None = None,
) -> Any:
    import pyarrow.parquet as pq

    parquet_columns = list(columns) if columns is not None else None
    return pq.read_table(download_s3_to_cache(client_s3, uri, cache_root), columns=parquet_columns)
