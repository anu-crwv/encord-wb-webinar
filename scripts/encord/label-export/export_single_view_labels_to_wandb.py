# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "numpy",
#     "pyarrow",
#     "pyyaml",
#     "typer",
#     "wandb>=0.18.0",
# ]
# ///
"""Export Encord single-view captions as a W&B labels overlay artifact."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import typer
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WANDB_CONFIG = SCRIPT_DIR.parent / "wandb_config.yaml"
DEFAULT_YAML_CONFIG = SCRIPT_DIR / "label_export_config.yaml"
EXPORT_ROOT = REPO_ROOT / "exports/encord-label-export"
S3_CACHE_ROOT = REPO_ROOT / "exports/encord-dataset-export" / "_cache" / "s3"
CHUNK_SIZE = 1000
LANG_KEYS = [
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
]
LANGUAGE_INSTRUCTION_PATTERN = re.compile(r"^language instruction(?:\s*([123]))?$", re.IGNORECASE)
LANGUAGE_INSTRUCTION_VALUE_PATTERN = re.compile(r"^language_instruction(?:_([123]))?$", re.IGNORECASE)
REQUIRED_PARQUET_COLUMNS = ["action", "observation.state", "timestamp", "frame_index"]
TROSSEN_STATE_ACTION_SPLITS = [
    ("left_joint_pos", 0, 7),
    ("right_joint_pos", 7, 14),
    ("base_velocity", 14, 16),
]
RELATIVE_STATS_KEYS = ["left_joint_pos", "right_joint_pos"]
RELATIVE_STATS_ACTION_HORIZON = 24
DEFAULT_TROSSEN_EMBODIMENT_TAG = "trossen_ai_mobile"
TROSSEN_STATE_ACTION_NAMES = [
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
]
TROSSEN_VECTOR_DIM = len(TROSSEN_STATE_ACTION_NAMES)
EPISODE_DIR_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")
EPISODE_BASE_RE = re.compile(r"^(episode_\d+)(?:_[A-Za-z0-9]+)?$")


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"{label} does not exist: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"{label} must contain a YAML object")
    return loaded


def required(config: dict[str, Any], key: str, label: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise typer.BadParameter(f"{label} is missing required key: {key}")
    return value


def configured_tags(config: dict[str, Any]) -> list[str]:
    tags = config.get("tags") or []
    if isinstance(tags, str):
        return [tags]
    if not isinstance(tags, list):
        raise typer.BadParameter("Label export config tags must be a list or string.")
    return [str(tag) for tag in tags]


def create_client():
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        return bucket, unquote(parsed.path.lstrip("/"))
    raise ValueError(f"Unsupported S3 URI format: {uri}")


def s3_client(unsigned: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if unsigned:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def s3_cache_path(bucket: str, key: str) -> Path:
    key_parts = key.split("/")
    if not bucket or not key or any(part == ".." for part in key_parts):
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    cache_parts = [part for part in key_parts if part not in {"", "."}]
    if not cache_parts:
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    return S3_CACHE_ROOT / bucket / Path(*cache_parts)


def read_s3_cached_bytes(client_s3: Any, uri: str) -> tuple[bytes, bool]:
    bucket, key = parse_s3_uri(uri)
    cache_path = s3_cache_path(bucket, key)
    downloaded = False

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            client_s3.download_file(bucket, key, str(tmp_path))
            os.replace(tmp_path, cache_path)
            downloaded = True
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return cache_path.read_bytes(), downloaded


def normalize_source_path(value: Any) -> str:
    path = str(value or "")
    if path.startswith("s3://"):
        return unquote(urlparse(path).path.lstrip("/"))
    if path.startswith("http://") or path.startswith("https://"):
        return unquote(urlparse(path).path.lstrip("/"))
    return path.lstrip("/")


def episode_path_from_source(value: Any) -> str | None:
    parts = [part for part in normalize_source_path(value).split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_DIR_RE.match(part):
            return "/".join(parts[: index + 1]) + "/"
    return None


def normalized_episode_path(episode_path: str | None) -> str | None:
    if not episode_path:
        return None
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    if not parts:
        return None
    match = EPISODE_BASE_RE.match(parts[-1])
    if match:
        parts[-1] = match.group(1)
    return "/".join(parts) + "/"


def episode_id_from_path(episode_path: str | None) -> str | None:
    if not episode_path:
        return None
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    if not parts:
        return None
    match = EPISODE_BASE_RE.match(parts[-1])
    return match.group(1) if match else parts[-1]


def episode_path_from_metadata(metadata: dict[str, Any], fallback_name: Any = None) -> str | None:
    if metadata.get("episode_path"):
        return str(metadata["episode_path"])
    for key in ["source_key", "source_uri", "s3_uri", "source_s3_uri", "objectUrl", "object_url"]:
        derived = episode_path_from_source(metadata.get(key))
        if derived:
            return derived
    return episode_path_from_source(fallback_name)


def episode_match_key(episode_path: str | None) -> str | None:
    return normalized_episode_path(episode_path)


def get_single_project_dataset(client: Any, project_hash: str):
    typer.echo(f"Loading Encord project {project_hash}...")
    project = client.get_project(project_hash)
    typer.echo("Finding attached dataset...")
    datasets = list(project.list_datasets())
    if not datasets:
        raise typer.BadParameter(f"Project {project_hash} has no attached datasets.")
    if len(datasets) > 1:
        details = ", ".join(f"{item.title} ({item.dataset_hash})" for item in datasets)
        raise typer.BadParameter(
            f"Project {project_hash} has multiple attached datasets: {details}. "
            "This exporter supports exactly one dataset for now."
        )
    typer.echo(f"Using dataset {datasets[0].dataset_hash} ({datasets[0].title}).")
    return project, datasets[0]


def export_labels(project: Any) -> list[dict[str, Any]]:
    typer.echo("Listing label rows...")
    label_rows = list(project.list_label_rows_v2())
    typer.echo(f"Found {len(label_rows)} label rows.")
    if label_rows:
        typer.echo("Initializing labels...")
        progress_interval = max(1, min(100, len(label_rows) // 10 or 1))
        with project.create_bundle(bundle_size=min(100, len(label_rows))) as bundle:
            for index, label_row in enumerate(label_rows, start=1):
                label_row.initialise_labels(bundle=bundle)
                if index % progress_interval == 0:
                    typer.echo(f"Initialized {index}/{len(label_rows)} label rows.")
        if len(label_rows) % progress_interval:
            typer.echo(f"Initialized {len(label_rows)} label rows.")

    typer.echo("Serializing labels...")
    labels = []
    for label_row in label_rows:
        row = label_row.to_encord_dict()
        if isinstance(row, dict):
            row.setdefault("data_hash", getattr(label_row, "data_hash", None))
            row.setdefault("label_hash", getattr(label_row, "label_hash", None))
            row.setdefault("data_title", getattr(label_row, "data_title", None))
        labels.append(row)
    return labels


def read_dataset_metadata(client: Any, dataset_hash: str) -> dict[str, dict[str, Any]]:
    typer.echo("Loading dataset metadata...")
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    typer.echo(f"Found {len(data_rows)} data rows.")
    backing_ids = [row.backing_item_uuid for row in data_rows if getattr(row, "backing_item_uuid", None)]
    typer.echo(f"Fetching client metadata for {len(backing_ids)} storage items...")
    storage_items = {str(item.uuid): item for item in client.get_storage_items(backing_ids)} if backing_ids else {}

    metadata_by_hash: dict[str, dict[str, Any]] = {}
    for row in data_rows:
        item = storage_items.get(str(getattr(row, "backing_item_uuid", "")))
        metadata_by_hash[str(row.uid)] = {
            "data_hash": row.uid,
            "data_title": row.title,
            "data_type": str(getattr(row, "data_type", "")),
            "encord_storage_item_uuid": str(getattr(row, "backing_item_uuid", "")),
            "client_metadata": getattr(item, "client_metadata", None) or {},
        }
    typer.echo(f"Collected metadata for {len(metadata_by_hash)} data rows.")
    return metadata_by_hash


def source_s3_uri(client_meta: dict[str, Any]) -> Any:
    return (
        client_meta.get("source_uri")
        or client_meta.get("s3_uri")
        or client_meta.get("source_s3_uri")
        or client_meta.get("objectUrl")
        or client_meta.get("object_url")
    )


def source_bucket(client_meta: dict[str, Any]) -> str | None:
    uri = source_s3_uri(client_meta)
    if not uri:
        return None
    try:
        bucket, _ = parse_s3_uri(str(uri))
    except ValueError:
        return None
    return bucket


def source_parquet_uri(client_meta: dict[str, Any], fallback_title: Any = None) -> str | None:
    for key in ["parquet_uri", "source_parquet_uri"]:
        if client_meta.get(key):
            return str(client_meta[key])
    episode_path = episode_path_from_metadata(client_meta, fallback_title)
    episode_id = str(client_meta.get("episode_id") or episode_id_from_path(episode_path) or "")
    bucket = source_bucket(client_meta)
    if not episode_path or not episode_id or not bucket:
        return None
    return f"s3://{bucket}/{episode_path.rstrip('/')}/data/chunk-000/{episode_id}.parquet"


def source_info_uri(source_uri: str | None, episode_path: str | None) -> str | None:
    if not source_uri or not episode_path:
        return None
    try:
        bucket, _ = parse_s3_uri(str(source_uri))
    except ValueError:
        return None
    return f"s3://{bucket}/{episode_path.rstrip('/')}/meta/info.json"


def source_dataset_items(metadata_by_hash: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for data_hash, row in sorted(metadata_by_hash.items()):
        client_meta = row.get("client_metadata") or {}
        items.append({
            "data_hash": data_hash,
            "data_title": row.get("data_title"),
            "data_type": row.get("data_type"),
            "encord_storage_item_uuid": row.get("encord_storage_item_uuid"),
            "source_s3_uri": source_s3_uri(client_meta),
            "source_parquet_uri": source_parquet_uri(client_meta, row.get("data_title")),
            "client_metadata": client_meta,
        })
    return items


def language_instruction_index(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    for pattern in (LANGUAGE_INSTRUCTION_PATTERN, LANGUAGE_INSTRUCTION_VALUE_PATTERN):
        match = pattern.fullmatch(normalized)
        if match:
            return int(match.group(1) or 1)
    return None


def language_instruction_candidates(label: Any) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    if isinstance(label, dict):
        instruction_index = language_instruction_index(label.get("name"))
        if instruction_index is None:
            instruction_index = language_instruction_index(label.get("value"))
        if instruction_index is not None and "answers" in label:
            candidates.extend((instruction_index, text) for text in strings_from(label.get("answers")))

        for key, value in label.items():
            instruction_index = language_instruction_index(key)
            if instruction_index is not None:
                candidates.extend((instruction_index, text) for text in strings_from(value))
            candidates.extend(language_instruction_candidates(value))
    elif isinstance(label, list):
        for value in label:
            candidates.extend(language_instruction_candidates(value))
    return candidates


def strings_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(strings_from(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(strings_from(item))
        return strings
    return []


def caption_text(label: dict[str, Any]) -> str | None:
    candidates = language_instruction_candidates(label)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def preview_rows(labels: list[dict[str, Any]], metadata_by_hash: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        data_hash = str(label.get("data_hash") or "")
        data_meta = metadata_by_hash.get(data_hash, {})
        client_meta = data_meta.get("client_metadata") or {}
        data_title = label.get("data_title") or data_meta.get("data_title")
        episode_path = episode_path_from_metadata(client_meta, data_title)
        rows.append({
            "data_hash": data_hash,
            "data_title": data_title,
            "label_hash": label.get("label_hash"),
            "language_instruction": caption_text(label),
            "episode_id": client_meta.get("episode_id") or episode_id_from_path(episode_path),
            "episode_path": episode_path,
            "camera_name": client_meta.get("camera_name"),
            "source_s3_uri": source_s3_uri(client_meta),
            "source_parquet_uri": source_parquet_uri(client_meta, data_title),
        })
    return rows


def qualified_artifact_ref(wandb_config: dict[str, Any], artifact_ref: str) -> str:
    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    if "/" not in artifact_ref.split(":", 1)[0]:
        return f"{entity}/{project}/{artifact_ref}"
    return artifact_ref


def artifact_aliases(artifact: Any) -> list[str]:
    aliases = getattr(artifact, "aliases", None) or []
    return [str(alias) for alias in aliases]


def artifact_attr(artifact: Any, name: str) -> Any:
    value = getattr(artifact, name, None)
    return value() if callable(value) else value


def artifact_version(artifact: Any) -> str:
    version = artifact_attr(artifact, "version")
    if version not in (None, ""):
        return str(version)

    name = str(artifact_attr(artifact, "name") or "")
    if ":" in name:
        return name.rsplit(":", 1)[1]

    raise ValueError("Could not resolve source dataset artifact to an immutable W&B version.")


def source_artifact_fields(source_artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset_artifact": source_artifact["resolved_ref"],
        "source_dataset_artifact_requested": source_artifact["requested_ref"],
        "source_dataset_artifact_version": source_artifact["version"],
        "source_dataset_artifact_digest": source_artifact.get("digest"),
        "source_dataset_artifact_url": source_artifact.get("url"),
        "source_dataset_artifact_aliases": source_artifact.get("aliases", []),
    }


def load_source_artifact_metadata(
    *,
    wandb_config: dict[str, Any],
    source_artifact_ref: str,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    import wandb

    artifact_ref = qualified_artifact_ref(wandb_config, source_artifact_ref)

    typer.echo(f"Loading source dataset artifact metadata from {artifact_ref}...")
    artifact = wandb.Api().artifact(artifact_ref)
    version = artifact_version(artifact)
    resolved_ref = f"{artifact_ref.split(':', 1)[0]}:{version}"
    source_artifact = {
        "requested_ref": source_artifact_ref,
        "qualified_requested_ref": artifact_ref,
        "resolved_ref": resolved_ref,
        "version": version,
        "digest": artifact_attr(artifact, "digest"),
        "url": artifact_attr(artifact, "url"),
        "aliases": artifact_aliases(artifact),
    }
    if resolved_ref != artifact_ref:
        typer.echo(f"Resolved source dataset artifact to {resolved_ref}.")

    artifact_dir = output_dir / "source_artifact_metadata"
    manifest_file = Path(
        artifact.get_entry("dataset/meta/source_dataset_manifest.json").download(root=str(artifact_dir))
    )
    items_file = Path(
        artifact.get_entry("dataset/meta/source_dataset_items.json").download(root=str(artifact_dir))
    )
    return json.loads(manifest_file.read_text()), json.loads(items_file.read_text()), source_artifact


def source_episode_order(source_items: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_episode_index: dict[int, dict[str, Any]] = {}
    ambiguous_keys: set[str] = set()
    for item in source_items:
        client_meta = item.get("client_metadata") or {}
        episode_index = item.get("episode_index")
        if episode_index is None:
            continue
        entry = by_episode_index.setdefault(int(episode_index), {
            "episode_index": int(episode_index),
            "encord_data_hash": item.get("data_hash"),
            "encord_data_group_uuid": item.get("data_group_uuid"),
            "source_video_items": [],
        })
        entry["source_video_items"].append(item)
        episode_path = episode_path_from_metadata(client_meta, item.get("artifact_path") or item.get("source_uri"))
        key = episode_match_key(episode_path)
        if not key or key in ambiguous_keys:
            continue
        existing = by_key.get(key)
        if existing is not None and int(existing["episode_index"]) != int(episode_index):
            by_key.pop(key, None)
            ambiguous_keys.add(key)
            continue
        by_key[key] = entry
    if ambiguous_keys:
        typer.echo(f"Skipped {len(ambiguous_keys)} ambiguous source artifact episode paths.")
    return by_key, ambiguous_keys


def row_match_key(row: dict[str, Any]) -> str | None:
    return episode_match_key(row.get("episode_path"))


def read_s3_json(client_s3: Any, uri: str) -> dict[str, Any] | None:
    try:
        body, _ = read_s3_cached_bytes(client_s3, uri)
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        typer.echo(f"Warning: could not read {uri}: {exc}", err=True)
        return None


def download_parquet_table(client_s3: Any, uri: str):
    import pyarrow.parquet as pq

    body, downloaded = read_s3_cached_bytes(client_s3, uri)
    return pq.read_table(BytesIO(body)), downloaded


def set_column(table: Any, name: str, array: Any) -> Any:
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def normalize_vector_column(table: Any, column: str, expected_dim: int, episode_index: int) -> Any:
    import pyarrow as pa

    field = table.schema.field(column)
    actual_dim = getattr(field.type, "list_size", None)
    if actual_dim == expected_dim:
        return table

    normalized = []
    for row_index, value in enumerate(table[column].to_pylist()):
        if value is None:
            raise ValueError(f"{column} has null vector at episode {episode_index}, row {row_index}")
        vector = list(value)
        if len(vector) < expected_dim:
            raise ValueError(
                f"{column} has dim {len(vector)} at episode {episode_index}, row {row_index}; "
                f"expected at least {expected_dim}"
            )
        normalized.append(vector[:expected_dim])

    value_type = getattr(field.type, "value_type", pa.float32())
    typer.echo(f"Normalizing {column} dim {actual_dim or 'variable'} to {expected_dim} for episode {episode_index}.")
    return set_column(table, column, pa.array(normalized, type=pa.list_(value_type, list_size=expected_dim)))


def rewrite_label_table(table: Any, episode_index: int, global_start: int, task_id: int) -> Any:
    import pyarrow as pa

    missing = [column for column in REQUIRED_PARQUET_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"Source parquet missing required columns: {missing}")

    table = normalize_vector_column(table, "action", TROSSEN_VECTOR_DIM, episode_index)
    table = normalize_vector_column(table, "observation.state", TROSSEN_VECTOR_DIM, episode_index)

    rows = table.num_rows
    table = set_column(table, "episode_index", pa.array([episode_index] * rows, type=pa.int64()))
    table = set_column(table, "frame_index", pa.array(list(range(rows)), type=pa.int64()))
    table = set_column(table, "index", pa.array(list(range(global_start, global_start + rows)), type=pa.int64()))
    table = set_column(table, "task_index", pa.array([task_id] * rows, type=pa.int64()))
    for key in LANG_KEYS:
        table = set_column(table, key, pa.array([task_id] * rows, type=pa.int64()))
    return table


def write_parquet(path: Path, table: Any) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def infer_features_from_table(table: Any) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in table.column_names:
        field = table.schema.field(name)
        dtype = str(field.type)
        if name in {"action", "observation.state"} and hasattr(field.type, "list_size"):
            features[name] = {
                "dtype": "float32",
                "shape": [field.type.list_size],
                "names": TROSSEN_STATE_ACTION_NAMES,
            }
        elif name in {"episode_index", "frame_index", "index", "task_index", *LANG_KEYS}:
            features[name] = {"dtype": "int64", "shape": [1], "names": None}
        elif name == "timestamp":
            features[name] = {"dtype": "float32", "shape": [1], "names": None}
        else:
            features[name] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def stats_columns(info: dict[str, Any], available_columns: set[str]) -> list[str]:
    columns = []
    for key, feature in (info.get("features") or {}).items():
        dtype = str((feature or {}).get("dtype") or "")
        if key in available_columns and "float" in dtype:
            columns.append(key)
    return columns


def table_column_as_numpy(table: Any, column: str) -> Any:
    import numpy as np

    values = table[column].to_pylist()
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


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


def feature_dim(info: dict[str, Any], column: str) -> int | None:
    shape = ((info.get("features") or {}).get(column) or {}).get("shape") or []
    if not shape:
        return None
    return int(shape[0])


def align_stats_data(data: Any, column: str, expected_dim: int | None) -> Any:
    if expected_dim is None or data.shape[1] == expected_dim:
        return data
    if data.shape[1] < expected_dim:
        raise ValueError(f"Cannot build stats.json: {column} has dim {data.shape[1]}; expected {expected_dim}")
    return data[:, :expected_dim]


def compute_stats(parquet_paths: list[Path], columns: list[str], info: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    if not columns:
        raise ValueError("Cannot build stats.json: no floating-point columns found in exported parquet files")

    all_data: dict[str, list[Any]] = {column: [] for column in columns}
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=columns)
        for column in columns:
            if column not in table.column_names:
                continue
            data = table_column_as_numpy(table, column)
            if data is not None:
                all_data[column].append(align_stats_data(data, column, feature_dim(info, column)))

    stats = {}
    for column, arrays in all_data.items():
        if not arrays:
            raise ValueError(f"Cannot build stats.json: no data found for column {column}")
        data = np.concatenate(arrays, axis=0)
        stats[column] = array_stats(data)
    return stats


def validate_stats_json(info: dict[str, Any], stats: dict[str, Any]) -> None:
    required_stats = {"mean", "std", "min", "max", "q01", "q99"}
    features = info.get("features") or {}
    for key, values in stats.items():
        missing = required_stats - set(values)
        if missing:
            raise ValueError(f"stats.json {key} is missing stats: {sorted(missing)}")
        expected_dim = int((features.get(key) or {}).get("shape", [1])[0])
        for stat_name in required_stats:
            actual_dim = len(values[stat_name])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"stats.json {key}.{stat_name} has length {actual_dim}; expected {expected_dim}"
                )


def compute_relative_stats(
    parquet_paths: list[Path],
    modality: dict[str, Any],
    relative_keys: list[str],
    action_horizon: int,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    if action_horizon <= 0:
        raise ValueError(f"Action horizon must be positive, got {action_horizon}")

    stats: dict[str, Any] = {}
    for key in relative_keys:
        action_meta = (modality.get("action") or {}).get(key)
        state_meta = (modality.get("state") or {}).get(key)
        if not action_meta:
            raise ValueError(f"Cannot build relative_stats_dreamzero.json: missing action.{key} modality")
        if not state_meta:
            raise ValueError(f"Cannot build relative_stats_dreamzero.json: missing state.{key} modality")

        action_col = action_meta["original_key"]
        state_col = state_meta["original_key"]
        columns = list(dict.fromkeys([action_col, state_col]))
        relative_chunks = []

        for parquet_path in parquet_paths:
            table = pq.read_table(parquet_path, columns=columns)
            action_data = table_column_as_numpy(table, action_col)
            state_data = table_column_as_numpy(table, state_col)
            if action_data is None or state_data is None:
                continue

            action_slice = action_data[:, int(action_meta["start"]):int(action_meta["end"])]
            state_slice = state_data[:, int(state_meta["start"]):int(state_meta["end"])]
            if action_slice.shape[1] != state_slice.shape[1]:
                raise ValueError(
                    f"Cannot build relative stats for {key}: action dim {action_slice.shape[1]} "
                    f"does not match state dim {state_slice.shape[1]}"
                )

            # Match DreamZero action delta indices [0, ..., action_horizon - 1].
            usable = max(action_slice.shape[0] - action_horizon + 1, 0)
            for frame_index in range(usable):
                ref_state = state_slice[frame_index]
                actions = action_slice[frame_index:frame_index + action_horizon]
                relative_chunks.append(actions - ref_state)

        if not relative_chunks:
            raise ValueError(f"Cannot build relative stats for {key}: no usable action windows found")

        stats[key] = array_stats(np.concatenate(relative_chunks, axis=0))

    return stats


def validate_relative_stats_json(
    modality: dict[str, Any],
    stats: dict[str, Any],
    relative_keys: list[str],
) -> None:
    required_stats = {"mean", "std", "min", "max", "q01", "q99"}
    if set(stats) != set(relative_keys):
        raise ValueError(
            "relative_stats_dreamzero.json keys do not match expected keys: "
            f"expected {relative_keys}, got {sorted(stats)}"
        )

    for key in relative_keys:
        action_meta = (modality.get("action") or {}).get(key) or {}
        expected_dim = int(action_meta["end"]) - int(action_meta["start"])
        missing = required_stats - set(stats[key])
        if missing:
            raise ValueError(f"relative_stats_dreamzero.json {key} is missing stats: {sorted(missing)}")
        for stat_name in required_stats:
            actual_dim = len(stats[key][stat_name])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"relative_stats_dreamzero.json {key}.{stat_name} has length {actual_dim}; "
                    f"expected {expected_dim}"
                )


def merge_features(source_features: dict[str, Any], inferred_features: dict[str, Any]) -> dict[str, Any]:
    features = dict(source_features)
    for name, inferred in inferred_features.items():
        merged = dict(features.get(name) or {})
        merged.update(inferred)
        source_names = (features.get(name) or {}).get("names")
        if source_names is not None and merged.get("shape") == (features.get(name) or {}).get("shape"):
            merged["names"] = source_names
        features[name] = merged
    return features


def build_info(
    *,
    source_info: dict[str, Any] | None,
    first_table: Any,
    total_episodes: int,
    max_episode_index: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    info = dict(source_info or {})
    source_features = dict((source_info or {}).get("features") or {})
    features = merge_features(source_features, infer_features_from_table(first_table))
    for key in LANG_KEYS:
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    video_feature_count = sum(
        1 for feature in features.values() if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": video_feature_count,
        "total_chunks": ((max_episode_index + 1) // CHUNK_SIZE) + (1 if (max_episode_index + 1) % CHUNK_SIZE else 0),
        "chunks_size": CHUNK_SIZE,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        ),
        "features": features,
    })
    return info


def vector_dim(info: dict[str, Any], key: str) -> int:
    feature = (info.get("features") or {}).get(key)
    if not feature:
        raise ValueError(f"Cannot build modality.json: {key} is missing from info.json features")
    shape = feature.get("shape") or []
    if not shape:
        raise ValueError(f"Cannot build modality.json: {key} has no shape in info.json")
    return int(shape[0])


def validate_trossen_state_action_names(info: dict[str, Any], key: str) -> None:
    feature = (info.get("features") or {}).get(key) or {}
    names = feature.get("names")
    if names != TROSSEN_STATE_ACTION_NAMES:
        raise ValueError(
            f"Cannot build modality.json: {key} names do not match expected Trossen layout. "
            f"Expected {TROSSEN_STATE_ACTION_NAMES}, got {names}"
        )


def state_action_modality_entry(info: dict[str, Any], original_key: str, start: int, end: int) -> dict[str, Any]:
    feature = info["features"][original_key]
    return {
        "original_key": original_key,
        "start": start,
        "end": end,
        "rotation_type": None,
        "absolute": True,
        "dtype": feature.get("dtype", "float32"),
        "range": None,
    }


def build_modality_json(info: dict[str, Any]) -> dict[str, Any]:
    state_dim = vector_dim(info, "observation.state")
    action_dim = vector_dim(info, "action")
    required_dim = max(end for _, _, end in TROSSEN_STATE_ACTION_SPLITS)
    if state_dim < required_dim:
        raise ValueError(f"observation.state has dim {state_dim}; expected at least {required_dim}")
    if action_dim < required_dim:
        raise ValueError(f"action has dim {action_dim}; expected at least {required_dim}")
    validate_trossen_state_action_names(info, "observation.state")
    validate_trossen_state_action_names(info, "action")

    features = info.get("features") or {}
    modality: dict[str, Any] = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for name, start, end in TROSSEN_STATE_ACTION_SPLITS:
        modality["state"][name] = state_action_modality_entry(info, "observation.state", start, end)
        modality["action"][name] = state_action_modality_entry(info, "action", start, end)

    for key, feature in sorted(features.items()):
        if feature.get("dtype") == "video" and key.startswith("observation.images."):
            modality["video"][key.replace("observation.images.", "")] = {"original_key": key}
        elif key.startswith("annotation."):
            modality["annotation"][key.replace("annotation.", "")] = {"original_key": key}

    return modality


def validate_modality_json(info: dict[str, Any], modality: dict[str, Any]) -> None:
    features = info.get("features") or {}
    for section in ["state", "action", "video", "annotation"]:
        for key, meta in (modality.get(section) or {}).items():
            original_key = meta.get("original_key")
            if original_key not in features:
                raise ValueError(f"modality.json {section}.{key} references missing feature {original_key}")
            if section in {"state", "action"}:
                dim = vector_dim(info, original_key)
                start = int(meta["start"])
                end = int(meta["end"])
                if start < 0 or end <= start or end > dim:
                    raise ValueError(
                        f"modality.json {section}.{key} range [{start}, {end}) does not fit {original_key} dim {dim}"
                    )


def build_embodiment_json(info: dict[str, Any]) -> dict[str, str]:
    tag = str(info.get("robot_type") or info.get("embodiment_tag") or DEFAULT_TROSSEN_EMBODIMENT_TAG)
    return {"robot_type": tag, "embodiment_tag": tag}


def validate_embodiment_json(embodiment: dict[str, Any]) -> None:
    robot_type = embodiment.get("robot_type")
    embodiment_tag = embodiment.get("embodiment_tag")
    if not robot_type:
        raise ValueError("embodiment.json is missing robot_type")
    if not embodiment_tag:
        raise ValueError("embodiment.json is missing embodiment_tag")
    if robot_type != embodiment_tag:
        raise ValueError(
            f"embodiment.json robot_type and embodiment_tag must match: {robot_type!r} != {embodiment_tag!r}"
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))


def make_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = EXPORT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def skipped_label(row: dict[str, Any], reason: str, episode_path_key: str | None = None) -> dict[str, Any]:
    skipped = {
        "data_hash": row.get("data_hash"),
        "data_title": row.get("data_title"),
        "reason": reason,
    }
    if row.get("episode_path"):
        skipped["episode_path"] = row.get("episode_path")
    if episode_path_key:
        skipped["episode_path_key"] = episode_path_key
    return skipped


def duplicate_label_signature(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("language_instruction") or ""), str(row.get("source_parquet_uri") or "")


def export_label_overlay(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    source_artifact: dict[str, Any],
    source_items: list[dict[str, Any]],
    source_manifest: dict[str, Any] | None,
    limit: int | None,
) -> dict[str, Any]:
    client_s3 = s3_client(unsigned=False)
    source_by_key, ambiguous_source_keys = source_episode_order(source_items)
    selected_by_key: dict[str, tuple[int, dict[str, Any], dict[str, Any]]] = {}
    selected_signatures: dict[str, tuple[str, str]] = {}
    seen_parquets: dict[str, str] = {}
    ambiguous_label_keys: set[str] = set()
    skipped: list[dict[str, Any]] = []

    for row in rows:
        caption = row.get("language_instruction")
        parquet_uri = row.get("source_parquet_uri")
        if not caption:
            skipped.append(skipped_label(row, "missing_language_instruction"))
            continue
        if not parquet_uri:
            skipped.append(skipped_label(row, "missing_source_parquet_uri"))
            continue
        match_key = row_match_key(row)
        if not match_key:
            skipped.append(skipped_label(row, "missing_episode_path"))
            continue
        if match_key in ambiguous_source_keys:
            skipped.append(skipped_label(row, "ambiguous_source_episode_path", match_key))
            continue
        if match_key in ambiguous_label_keys:
            skipped.append(skipped_label(row, "ambiguous_duplicate_label", match_key))
            continue

        source_episode = source_by_key.get(match_key)
        if source_episode is None:
            skipped.append(skipped_label(row, "missing_source_episode_match", match_key))
            continue

        signature = duplicate_label_signature(row)
        existing = selected_by_key.get(match_key)
        if existing is not None:
            if selected_signatures.get(match_key) == signature:
                skipped.append(skipped_label(row, "duplicate_episode_path_same_label", match_key))
                continue
            existing_row = existing[1]
            existing_parquet = str(existing_row.get("source_parquet_uri") or "")
            if existing_parquet:
                seen_parquets.pop(existing_parquet, None)
            selected_by_key.pop(match_key, None)
            selected_signatures.pop(match_key, None)
            ambiguous_label_keys.add(match_key)
            skipped.append(skipped_label(existing_row, "ambiguous_duplicate_label", match_key))
            skipped.append(skipped_label(row, "ambiguous_duplicate_label", match_key))
            continue

        parquet_key = str(parquet_uri)
        if parquet_key in seen_parquets:
            skipped.append(skipped_label(row, "duplicate_episode_parquet", match_key))
            continue
        seen_parquets[parquet_key] = match_key

        episode_index = int(source_episode["episode_index"])
        selected_by_key[match_key] = (episode_index, row, source_episode)
        selected_signatures[match_key] = signature

    selected = sorted(selected_by_key.values(), key=lambda item: item[0])
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise typer.BadParameter("No caption label rows matched exportable episodes.")
    task_to_id: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    parquet_paths: list[Path] = []
    first_table = None
    first_source_info = None
    total_frames = 0
    parquet_cache_hits = 0
    parquet_cache_downloads = 0

    typer.echo(f"Using shared S3 object cache at {S3_CACHE_ROOT}")
    for index, (episode_index, row, source_episode) in enumerate(selected, start=1):
        caption = str(row["language_instruction"])
        task_id = task_to_id.setdefault(caption, len(task_to_id))
        if len(task_rows) < len(task_to_id):
            task_rows.append({"task_index": task_id, "task": caption})

        parquet_uri = str(row["source_parquet_uri"])
        table, downloaded = download_parquet_table(client_s3, parquet_uri)
        if downloaded:
            parquet_cache_downloads += 1
            cache_status = "downloaded"
        else:
            parquet_cache_hits += 1
            cache_status = "cached"
        typer.echo(f"[{index}/{len(selected)}] {cache_status} {parquet_uri}")
        table = rewrite_label_table(
            table,
            episode_index=episode_index,
            global_start=total_frames,
            task_id=task_id,
        )
        if first_table is None:
            first_table = table
            info_uri = source_info_uri(row.get("source_s3_uri"), row.get("episode_path"))
            first_source_info = read_s3_json(client_s3, info_uri) if info_uri else None

        output_path = (
            output_dir
            / "dataset"
            / "data"
            / f"chunk-{episode_index // CHUNK_SIZE:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        write_parquet(output_path, table)
        parquet_paths.append(output_path)
        length = table.num_rows
        total_frames += length

        episode_row = {
            "episode_index": episode_index,
            "tasks": [caption],
            "length": length,
            "encord_label_data_hash": row.get("data_hash"),
            "encord_label_hash": row.get("label_hash"),
            "encord_label_data_title": row.get("data_title"),
            "encord_label_camera_name": row.get("camera_name"),
            "episode_id": row.get("episode_id"),
            "episode_path": row.get("episode_path"),
            "source_parquet_uri": parquet_uri,
        }
        episode_row.update({
            "source_dataset_episode_index": source_episode.get("episode_index"),
            "source_dataset_data_hash": source_episode.get("encord_data_hash"),
            "source_dataset_data_group_uuid": source_episode.get("encord_data_group_uuid"),
        })
        episode_rows.append(episode_row)
        manifest_rows.append({
            **episode_row,
            "artifact_path": output_path.relative_to(output_dir).as_posix(),
        })

    assert first_table is not None
    dataset_meta = output_dir / "dataset" / "meta"
    write_jsonl(dataset_meta / "tasks.jsonl", sorted(task_rows, key=lambda item: item["task_index"]))
    write_jsonl(dataset_meta / "episodes.jsonl", sorted(episode_rows, key=lambda item: item["episode_index"]))
    info = build_info(
        source_info=first_source_info,
        first_table=first_table,
        total_episodes=len(episode_rows),
        max_episode_index=max(row["episode_index"] for row in episode_rows),
        total_frames=total_frames,
        total_tasks=len(task_rows),
    )
    modality = build_modality_json(info)
    validate_modality_json(info, modality)
    stats_cols = stats_columns(info, set(first_table.column_names))
    stats = compute_stats(parquet_paths, stats_cols, info)
    validate_stats_json(info, stats)
    relative_stats = compute_relative_stats(
        parquet_paths=parquet_paths,
        modality=modality,
        relative_keys=RELATIVE_STATS_KEYS,
        action_horizon=RELATIVE_STATS_ACTION_HORIZON,
    )
    validate_relative_stats_json(modality, relative_stats, RELATIVE_STATS_KEYS)
    embodiment = build_embodiment_json(info)
    validate_embodiment_json(embodiment)
    write_json(dataset_meta / "info.json", info)
    write_json(dataset_meta / "embodiment.json", embodiment)
    write_json(dataset_meta / "modality.json", modality)
    write_json(dataset_meta / "stats.json", stats)
    write_json(dataset_meta / "relative_stats_dreamzero.json", relative_stats)

    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        **source_artifact_fields(source_artifact),
        "source_dataset_manifest": source_manifest,
        "label_episode_count": len(episode_rows),
        "label_task_count": len(task_rows),
        "label_frame_count": total_frames,
        "embodiment_tag": embodiment["embodiment_tag"],
        "stats_columns": stats_cols,
        "relative_stats_keys": RELATIVE_STATS_KEYS,
        "relative_stats_action_horizon": RELATIVE_STATS_ACTION_HORIZON,
        "source_match_key_count": len(source_by_key),
        "ambiguous_source_episode_path_count": len(ambiguous_source_keys),
        "ambiguous_label_episode_path_count": len(ambiguous_label_keys),
        "skipped_label_count": len(skipped),
        "s3_cache_root": str(S3_CACHE_ROOT),
        "parquet_cache_hit_count": parquet_cache_hits,
        "parquet_cache_download_count": parquet_cache_downloads,
        "episodes": sorted(manifest_rows, key=lambda item: item["episode_index"]),
        "skipped_labels": skipped,
    }
    write_json(output_dir / "label_export_manifest.json", summary)
    return summary


def log_to_wandb(
    *,
    wandb_config: dict[str, Any],
    metadata: dict[str, Any],
    source_artifact: dict[str, Any],
    output_dir: Path,
    tags: list[str],
) -> dict[str, Any]:
    import wandb

    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    label_name = required(wandb_config, "label_artifact_name", "W&B config")
    table_name = required(wandb_config, "table_name", "W&B config")

    labels_path = output_dir / "encord_labels.json"
    preview_path = output_dir / "label_preview_rows.json"
    manifest_path = output_dir / "label_export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run_name = (
        f"encord-label-overlay-{label_name}-"
        f"{str(manifest.get('encord_project_hash') or 'unknown')[:8]}-"
        f"{manifest.get('label_episode_count', 0)}eps"
    )

    with wandb.init(entity=entity, project=project, job_type="encord-label-export", name=run_name) as run:
        typer.echo(f"Logging to W&B run {run.url}...")
        typer.echo(f"Using source dataset artifact {source_artifact['resolved_ref']}.")
        run.use_artifact(source_artifact["resolved_ref"])

        typer.echo(f"Logging label overlay artifact {label_name}...")
        label_artifact = wandb.Artifact(
            label_name,
            type="dataset",
            metadata={
                "encord_project_hash": manifest.get("encord_project_hash"),
                "encord_dataset_hash": manifest.get("encord_dataset_hash"),
                **source_artifact_fields(source_artifact),
                "label_version_note": metadata.get("label_version_note"),
                "captioning_method": metadata.get("captioning_method"),
                "qc_status": metadata.get("qc_status"),
                "label_episode_count": manifest.get("label_episode_count"),
                "label_task_count": manifest.get("label_task_count"),
                "label_frame_count": manifest.get("label_frame_count"),
            },
            description=str(metadata.get("label_version_note", "")),
        )
        label_artifact.add_dir(str(output_dir / "dataset"), name="dataset")
        label_artifact.add_file(str(labels_path), name="encord_labels.json")
        label_artifact.add_file(str(preview_path), name="label_preview_rows.json")
        label_artifact.add_file(str(manifest_path), name="label_export_manifest.json")
        logged_labels = run.log_artifact(label_artifact, aliases=["latest"], tags=tags)
        logged_labels.wait()
        labels_ref = f"{label_name}:{logged_labels.version}"
        typer.echo(f"Logged label overlay artifact {labels_ref}.")

        typer.echo("Logging preview table...")
        table = wandb.Table(columns=[
            "data_hash",
            "data_title",
            "label_hash",
            "language_instruction",
            "episode_id",
            "episode_path",
            "camera_name",
            "source_s3_uri",
            "source_parquet_uri",
        ])
        for row in json.loads(preview_path.read_text()):
            table.add_data(
                row.get("data_hash"),
                row.get("data_title"),
                row.get("label_hash"),
                row.get("language_instruction"),
                row.get("episode_id"),
                row.get("episode_path"),
                row.get("camera_name"),
                row.get("source_s3_uri"),
                row.get("source_parquet_uri"),
            )
        run.log({table_name: table})
        typer.echo("Logged preview table.")

        return {**source_artifact_fields(source_artifact), "labels_artifact": labels_ref, "run_url": run.url}


def main(
    source_artifact_ref: Annotated[
        str,
        typer.Option(help="Required W&B dataset artifact this label overlay materializes with."),
    ],
    metadata_yaml: Annotated[Path, typer.Option(help="Required YAML notes for this dataset/label version.")] = DEFAULT_YAML_CONFIG,
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
    limit: Annotated[int | None, typer.Option(help="Optional max number of caption episodes to export.")] = None,
) -> None:
    typer.echo("Loading config...")
    metadata = load_yaml(metadata_yaml, "metadata YAML")
    metadata_notes = {
        key: value for key, value in metadata.items()
        if key not in {"tags", "source_artifact_ref"}
    }
    wandb_settings = load_yaml(wandb_config, "W&B config")
    project_hash = str(required(metadata, "encord_project_hash", "metadata YAML"))

    output_dir = make_output_dir()
    source_manifest, source_items, source_artifact = load_source_artifact_metadata(
        wandb_config=wandb_settings,
        source_artifact_ref=source_artifact_ref,
        output_dir=output_dir,
    )

    client = create_client()
    project, project_dataset = get_single_project_dataset(client, project_hash)
    dataset_hash = str(project_dataset.dataset_hash)

    typer.echo("Exporting labels and source metadata...")
    labels = export_labels(project)
    data_metadata = read_dataset_metadata(client, dataset_hash)
    dataset_items = source_dataset_items(data_metadata)
    rows = preview_rows(labels, data_metadata)

    source_manifest_local = {
        "encord_project_hash": project_hash,
        "encord_project_title": project.title,
        "encord_dataset_hash": dataset_hash,
        "encord_dataset_title": project_dataset.title,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_item_count": len(dataset_items),
        "label_row_count": len(labels),
        **metadata_notes,
        "source_artifact_ref": source_artifact["resolved_ref"],
        "source_artifact_ref_requested": source_artifact["requested_ref"],
        **source_artifact_fields(source_artifact),
    }

    label_summary = export_label_overlay(
        rows=rows,
        output_dir=output_dir,
        source_artifact=source_artifact,
        source_items=source_items,
        source_manifest=source_manifest,
        limit=limit,
    )
    label_summary.update({
        "encord_project_hash": project_hash,
        "encord_project_title": project.title,
        "encord_dataset_hash": dataset_hash,
        "encord_dataset_title": project_dataset.title,
        **source_artifact_fields(source_artifact),
        "label_version_note": metadata_notes.get("label_version_note"),
        "captioning_method": metadata_notes.get("captioning_method"),
        "qc_status": metadata_notes.get("qc_status"),
    })
    write_json(output_dir / "label_export_manifest.json", label_summary)

    typer.echo(f"Writing local export files to {output_dir}...")
    write_json(output_dir / "source_dataset_manifest.json", source_manifest_local)
    write_json(output_dir / "source_dataset_items.json", dataset_items)
    write_json(output_dir / "encord_labels.json", {"export_info": source_manifest_local, "label_rows": labels})
    write_json(output_dir / "encord_data_metadata.json", data_metadata)
    write_json(output_dir / "label_preview_rows.json", rows)
    typer.echo("Wrote local export files.")

    lineage = log_to_wandb(
        wandb_config=wandb_settings,
        metadata=metadata_notes,
        source_artifact=source_artifact,
        output_dir=output_dir,
        tags=configured_tags(metadata),
    )
    write_json(output_dir / "wandb_lineage.json", lineage)

    typer.echo(f"exported {label_summary['label_episode_count']} label episodes")
    typer.echo(f"dataset: {dataset_hash} ({project_dataset.title})")
    typer.echo(f"source artifact: {lineage['source_dataset_artifact']}")
    typer.echo(f"label overlay artifact: {lineage['labels_artifact']}")
    typer.echo(f"local files: {output_dir}")
    typer.echo(f"run: {lineage['run_url']}")


if __name__ == "__main__":
    typer.run(main)
