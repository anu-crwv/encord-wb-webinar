# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "numpy",
#     "pyarrow",
#     "typer",
# ]
# ///
"""Build and validate the webinar's train-ready LeRobot metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from encord_source import CAMERA_TO_FEATURE

CHUNK_SIZE = 1000
LANGUAGE_KEYS = (
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
)
REQUIRED_PARQUET_COLUMNS = ("action", "observation.state", "timestamp", "frame_index")
TROSSEN_SPLITS = (
    ("left_joint_pos", 0, 7),
    ("right_joint_pos", 7, 14),
    ("base_velocity", 14, 16),
)
RELATIVE_STATS_KEYS = ("left_joint_pos", "right_joint_pos")
RELATIVE_STATS_ACTION_HORIZON = 24
TROSSEN_STATE_ACTION_NAMES = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "linear_vel",
    "angular_vel",
)
TROSSEN_VECTOR_DIM = len(TROSSEN_STATE_ACTION_NAMES)
DEFAULT_EMBODIMENT_TAG = "trossen_ai_mobile"


def normalize_video_features(source_info: dict[str, Any]) -> dict[str, Any]:
    info = dict(source_info)
    features = dict(info.get("features") or {})
    for camera, output_name in CAMERA_TO_FEATURE.items():
        source_key = f"observation.images.{camera}"
        output_key = f"observation.images.{output_name}"
        if source_key in features and output_key not in features:
            features[output_key] = features.pop(source_key)
    info["features"] = features
    return info


def validate_source_info(uri: str, raw_info: dict[str, Any]) -> tuple[dict[str, Any], float]:
    info = normalize_video_features(raw_info)
    fps_value = info.get("fps")
    if fps_value is None:
        raise typer.BadParameter(f"Source metadata has no FPS: {uri}")
    try:
        fps = round(float(fps_value), 6)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid FPS in {uri}: {fps_value!r}") from exc

    missing = [
        feature
        for feature in CAMERA_TO_FEATURE.values()
        if f"observation.images.{feature}" not in (info.get("features") or {})
    ]
    if missing:
        raise typer.BadParameter(f"Source metadata is missing required camera features: {missing}")
    return info, fps


def set_column(table: Any, name: str, array: Any) -> Any:
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def normalize_vector_column(table: Any, column: str, episode_index: int) -> Any:
    import pyarrow as pa

    field = table.schema.field(column)
    actual_dim = getattr(field.type, "list_size", None)
    if actual_dim == TROSSEN_VECTOR_DIM:
        return table

    normalized = []
    for row_index, value in enumerate(table[column].to_pylist()):
        if value is None:
            raise typer.BadParameter(
                f"{column} has a null vector at episode {episode_index}, row {row_index}"
            )
        vector = list(value)
        if len(vector) < TROSSEN_VECTOR_DIM:
            raise typer.BadParameter(
                f"{column} has dim {len(vector)} at episode {episode_index}, row {row_index}; "
                f"expected at least {TROSSEN_VECTOR_DIM}"
            )
        normalized.append(vector[:TROSSEN_VECTOR_DIM])

    value_type = getattr(field.type, "value_type", pa.float32())
    return set_column(
        table,
        column,
        pa.array(normalized, type=pa.list_(value_type, list_size=TROSSEN_VECTOR_DIM)),
    )


def rewrite_episode_table(table: Any, episode_index: int, global_start: int, task_id: int) -> Any:
    import pyarrow as pa

    missing = [column for column in REQUIRED_PARQUET_COLUMNS if column not in table.column_names]
    if missing:
        raise typer.BadParameter(f"Source parquet missing required columns: {missing}")

    table = normalize_vector_column(table, "action", episode_index)
    table = normalize_vector_column(table, "observation.state", episode_index)
    rows = table.num_rows
    if rows == 0:
        raise typer.BadParameter(f"Episode {episode_index} has no frames")

    table = set_column(table, "episode_index", pa.array([episode_index] * rows, type=pa.int64()))
    table = set_column(table, "frame_index", pa.array(range(rows), type=pa.int64()))
    table = set_column(
        table,
        "index",
        pa.array(range(global_start, global_start + rows), type=pa.int64()),
    )
    table = set_column(table, "task_index", pa.array([task_id] * rows, type=pa.int64()))
    for key in LANGUAGE_KEYS:
        table = set_column(table, key, pa.array([task_id] * rows, type=pa.int64()))
    return table


def infer_features(table: Any) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in table.column_names:
        field = table.schema.field(name)
        if name in {"action", "observation.state"}:
            features[name] = {
                "dtype": "float32",
                "shape": [TROSSEN_VECTOR_DIM],
                "names": list(TROSSEN_STATE_ACTION_NAMES),
            }
        elif name in {"episode_index", "frame_index", "index", "task_index", *LANGUAGE_KEYS}:
            features[name] = {"dtype": "int64", "shape": [1], "names": None}
        elif name == "timestamp":
            features[name] = {"dtype": "float32", "shape": [1], "names": None}
        else:
            features[name] = {"dtype": str(field.type), "shape": [1], "names": None}
    return features


def build_info(
    source_info: dict[str, Any],
    first_table: Any,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    info = dict(source_info)
    features = dict(info.get("features") or {})
    features.update(infer_features(first_table))
    for key in LANGUAGE_KEYS:
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    video_feature_count = sum(
        1 for feature in features.values() if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_episodes * video_feature_count,
            "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
    )
    return info


def table_column_as_numpy(table: Any, column: str) -> Any:
    import numpy as np

    values = table[column].to_pylist()
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    return data.reshape(-1, 1) if data.ndim == 1 else data


def array_stats(data: Any) -> dict[str, Any]:
    import numpy as np

    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def stats_columns(info: dict[str, Any], available_columns: set[str]) -> list[str]:
    return [
        key
        for key, feature in (info.get("features") or {}).items()
        if key in available_columns and "float" in str((feature or {}).get("dtype") or "")
    ]


def compute_stats(parquet_paths: list[Path], columns: list[str], info: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    if not columns:
        raise typer.BadParameter("No floating-point columns are available for stats.json")
    all_data: dict[str, list[Any]] = {column: [] for column in columns}
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=columns)
        for column in columns:
            data = table_column_as_numpy(table, column)
            if data is not None:
                expected = int(((info.get("features") or {}).get(column) or {}).get("shape", [1])[0])
                if data.shape[1] != expected:
                    raise typer.BadParameter(
                        f"Cannot build stats.json: {column} has dim {data.shape[1]}; expected {expected}"
                    )
                all_data[column].append(data)
    result = {}
    for column, arrays in all_data.items():
        if not arrays:
            raise typer.BadParameter(f"Cannot build stats.json: no data found for {column}")
        result[column] = array_stats(np.concatenate(arrays, axis=0))
    return result


def build_modality(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features") or {}
    modality: dict[str, Any] = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for original_key, section in (("observation.state", "state"), ("action", "action")):
        feature = features.get(original_key) or {}
        if feature.get("names") != list(TROSSEN_STATE_ACTION_NAMES):
            raise typer.BadParameter(f"{original_key} does not match the 16-value Trossen layout")
        for name, start, end in TROSSEN_SPLITS:
            modality[section][name] = {
                "original_key": original_key,
                "start": start,
                "end": end,
                "rotation_type": None,
                "absolute": True,
                "dtype": feature.get("dtype", "float32"),
                "range": None,
            }
    for key, feature in sorted(features.items()):
        if feature.get("dtype") == "video" and key.startswith("observation.images."):
            modality["video"][key.removeprefix("observation.images.")] = {"original_key": key}
        elif key.startswith("annotation."):
            modality["annotation"][key.removeprefix("annotation.")] = {"original_key": key}
    return modality


def compute_relative_stats(parquet_paths: list[Path], modality: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    result: dict[str, Any] = {}
    for key in RELATIVE_STATS_KEYS:
        action_meta = modality["action"][key]
        state_meta = modality["state"][key]
        relative_chunks = []
        columns = list(dict.fromkeys([action_meta["original_key"], state_meta["original_key"]]))
        for parquet_path in parquet_paths:
            table = pq.read_table(parquet_path, columns=columns)
            action_data = table_column_as_numpy(table, action_meta["original_key"])
            state_data = table_column_as_numpy(table, state_meta["original_key"])
            action_slice = action_data[:, action_meta["start"] : action_meta["end"]]
            state_slice = state_data[:, state_meta["start"] : state_meta["end"]]
            usable = max(action_slice.shape[0] - RELATIVE_STATS_ACTION_HORIZON + 1, 0)
            for frame_index in range(usable):
                reference = state_slice[frame_index]
                relative_chunks.append(
                    action_slice[frame_index : frame_index + RELATIVE_STATS_ACTION_HORIZON] - reference
                )
        if not relative_chunks:
            raise typer.BadParameter(
                f"Cannot build relative stats for {key}: every episode is shorter than "
                f"{RELATIVE_STATS_ACTION_HORIZON} frames"
            )
        result[key] = array_stats(np.concatenate(relative_chunks, axis=0))
    return result


def build_embodiment(info: dict[str, Any]) -> dict[str, str]:
    tag = str(info.get("robot_type") or info.get("embodiment_tag") or DEFAULT_EMBODIMENT_TAG)
    return {"robot_type": tag, "embodiment_tag": tag}
