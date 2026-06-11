# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
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

import typer
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WANDB_CONFIG = SCRIPT_DIR.parent / "wandb_config.yaml"
EXPORT_ROOT = REPO_ROOT / "exports/encord-label-export"
CHUNK_SIZE = 1000
LANG_KEYS = [
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
]
REQUIRED_PARQUET_COLUMNS = ["action", "observation.state", "timestamp", "frame_index"]
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
        return episode_path
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


def episode_keys(episode_path: str | None, episode_id: Any = None) -> list[str]:
    keys = []
    if episode_path:
        keys.append(episode_path)
        normal = normalized_episode_path(episode_path)
        if normal:
            keys.append(normal)
        path_id = episode_id_from_path(episode_path)
        if path_id:
            keys.append(path_id)
    if episode_id not in (None, ""):
        keys.append(str(episode_id))
    return list(dict.fromkeys(keys))


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


def language_instruction(label: Any) -> Any:
    if isinstance(label, dict):
        is_instruction = label.get("value") == "language_instruction" or label.get("name") == "Language Instruction"
        if is_instruction and "answers" in label:
            return label.get("answers")
        for value in label.values():
            found = language_instruction(value)
            if found not in (None, ""):
                return found
    elif isinstance(label, list):
        for value in label:
            found = language_instruction(value)
            if found not in (None, ""):
                return found
    return None


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
    strings = strings_from(language_instruction(label))
    return strings[0] if strings else None


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


def load_source_artifact_metadata(
    *,
    wandb_config: dict[str, Any],
    source_artifact_ref: str,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    import wandb

    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    artifact_ref = source_artifact_ref
    if "/" not in artifact_ref.split(":", 1)[0]:
        artifact_ref = f"{entity}/{project}/{artifact_ref}"

    typer.echo(f"Loading source dataset artifact metadata from {artifact_ref}...")
    artifact = wandb.Api().artifact(artifact_ref)
    artifact_dir = output_dir / "source_artifact_metadata"
    manifest_file = Path(
        artifact.get_path("dataset/meta/source_dataset_manifest.json").download(root=str(artifact_dir))
    )
    items_file = Path(
        artifact.get_path("dataset/meta/source_dataset_items.json").download(root=str(artifact_dir))
    )
    return json.loads(manifest_file.read_text()), json.loads(items_file.read_text())


def source_episode_order(source_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_episode_index: dict[int, dict[str, Any]] = {}
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
        for key in episode_keys(episode_path, client_meta.get("episode_id")):
            by_key[key] = entry
    return by_key


def row_match_keys(row: dict[str, Any]) -> list[str]:
    return episode_keys(row.get("episode_path"), row.get("episode_id"))


def read_s3_json(client_s3: Any, uri: str) -> dict[str, Any] | None:
    try:
        bucket, key = parse_s3_uri(uri)
        body = client_s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        typer.echo(f"Warning: could not read {uri}: {exc}", err=True)
        return None


def download_parquet_table(client_s3: Any, uri: str):
    import pyarrow.parquet as pq

    bucket, key = parse_s3_uri(uri)
    body = client_s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pq.read_table(BytesIO(body))


def set_column(table: Any, name: str, array: Any) -> Any:
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def rewrite_label_table(table: Any, episode_index: int, global_start: int, task_id: int) -> Any:
    import pyarrow as pa

    missing = [column for column in REQUIRED_PARQUET_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"Source parquet missing required columns: {missing}")

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
            features[name] = {"dtype": "float32", "shape": [field.type.list_size], "names": None}
        elif name in {"episode_index", "frame_index", "index", "task_index", *LANG_KEYS}:
            features[name] = {"dtype": "int64", "shape": [1], "names": None}
        elif name == "timestamp":
            features[name] = {"dtype": "float32", "shape": [1], "names": None}
        else:
            features[name] = {"dtype": dtype, "shape": [1], "names": None}
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
    features = dict((source_info or {}).get("features") or {})
    features.update(infer_features_from_table(first_table))
    for key in LANG_KEYS:
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
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


def export_label_overlay(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    source_artifact_ref: str,
    source_items: list[dict[str, Any]],
    source_manifest: dict[str, Any] | None,
    limit: int | None,
) -> dict[str, Any]:
    client_s3 = s3_client(unsigned=False)
    source_by_key = source_episode_order(source_items)
    selected: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
    skipped: list[dict[str, Any]] = []
    seen_parquets: set[str] = set()
    fallback_index = 0

    for row in rows:
        caption = row.get("language_instruction")
        parquet_uri = row.get("source_parquet_uri")
        if not caption:
            skipped.append({"data_hash": row.get("data_hash"), "reason": "missing language_instruction"})
            continue
        if not parquet_uri:
            skipped.append({"data_hash": row.get("data_hash"), "reason": "missing source_parquet_uri"})
            continue
        if parquet_uri in seen_parquets:
            skipped.append({"data_hash": row.get("data_hash"), "reason": "duplicate episode parquet"})
            continue
        seen_parquets.add(parquet_uri)

        source_episode = next((source_by_key[key] for key in row_match_keys(row) if key in source_by_key), None)
        if source_episode is not None:
            episode_index = int(source_episode["episode_index"])
        else:
            episode_index = fallback_index
            fallback_index += 1
        selected.append((episode_index, row, source_episode))
        if limit is not None and len(selected) >= limit:
            break

    if not selected:
        raise typer.BadParameter("No caption label rows matched exportable episodes.")

    selected.sort(key=lambda item: item[0])
    task_to_id: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    first_table = None
    first_source_info = None
    total_frames = 0

    for index, (episode_index, row, source_episode) in enumerate(selected, start=1):
        caption = str(row["language_instruction"])
        task_id = task_to_id.setdefault(caption, len(task_to_id))
        if len(task_rows) < len(task_to_id):
            task_rows.append({"task_index": task_id, "task": caption})

        parquet_uri = str(row["source_parquet_uri"])
        typer.echo(f"[{index}/{len(selected)}] downloading {parquet_uri}")
        table = rewrite_label_table(
            download_parquet_table(client_s3, parquet_uri),
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
        if source_episode is not None:
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
    write_json(dataset_meta / "info.json", build_info(
        source_info=first_source_info,
        first_table=first_table,
        total_episodes=len(episode_rows),
        max_episode_index=max(row["episode_index"] for row in episode_rows),
        total_frames=total_frames,
        total_tasks=len(task_rows),
    ))

    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_artifact": source_artifact_ref,
        "source_dataset_manifest": source_manifest,
        "label_episode_count": len(episode_rows),
        "label_task_count": len(task_rows),
        "label_frame_count": total_frames,
        "skipped_label_count": len(skipped),
        "episodes": sorted(manifest_rows, key=lambda item: item["episode_index"]),
        "skipped_labels": skipped,
    }
    write_json(output_dir / "label_export_manifest.json", summary)
    return summary


def log_to_wandb(
    *,
    wandb_config: dict[str, Any],
    metadata: dict[str, Any],
    source_artifact_ref: str,
    output_dir: Path,
) -> dict[str, str]:
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
        f"encord-labels-{label_name}-"
        f"{str(manifest.get('encord_project_hash') or 'unknown')[:8]}-"
        f"{manifest.get('label_episode_count', 0)}eps"
    )

    with wandb.init(entity=entity, project=project, job_type="encord-label-export", name=run_name) as run:
        typer.echo(f"Logging to W&B run {run.url}...")
        typer.echo(f"Using source dataset artifact {source_artifact_ref}.")
        run.use_artifact(source_artifact_ref)

        typer.echo(f"Logging labels artifact {label_name}...")
        label_artifact = wandb.Artifact(
            label_name,
            type="labels",
            metadata={
                "encord_project_hash": manifest.get("encord_project_hash"),
                "encord_dataset_hash": manifest.get("encord_dataset_hash"),
                "source_dataset_artifact": source_artifact_ref,
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
        logged_labels = run.log_artifact(label_artifact, aliases=["latest", "single-view"])
        logged_labels.wait()
        labels_ref = f"{label_name}:{logged_labels.version}"
        typer.echo(f"Logged labels artifact {labels_ref}.")

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

        return {"source_dataset_artifact": source_artifact_ref, "labels_artifact": labels_ref, "run_url": run.url}


def main(
    metadata_yaml: Annotated[Path, typer.Option(help="Required YAML notes for this dataset/label version.")],
    source_artifact_ref: Annotated[
        str,
        typer.Option(help="Required W&B dataset artifact this labels artifact overlays."),
    ],
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
    limit: Annotated[int | None, typer.Option(help="Optional max number of caption episodes to export.")] = None,
) -> None:
    typer.echo("Loading config...")
    metadata = load_yaml(metadata_yaml, "metadata YAML")
    metadata_notes = {key: value for key, value in metadata.items() if key != "source_artifact_ref"}
    wandb_settings = load_yaml(wandb_config, "W&B config")
    project_hash = str(required(metadata, "encord_project_hash", "metadata YAML"))

    output_dir = make_output_dir()
    source_manifest, source_items = load_source_artifact_metadata(
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
        "source_artifact_ref": source_artifact_ref,
    }

    label_summary = export_label_overlay(
        rows=rows,
        output_dir=output_dir,
        source_artifact_ref=source_artifact_ref,
        source_items=source_items,
        source_manifest=source_manifest,
        limit=limit,
    )
    label_summary.update({
        "encord_project_hash": project_hash,
        "encord_project_title": project.title,
        "encord_dataset_hash": dataset_hash,
        "encord_dataset_title": project_dataset.title,
        "source_dataset_artifact": source_artifact_ref,
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
        source_artifact_ref=source_artifact_ref,
        output_dir=output_dir,
    )
    write_json(output_dir / "wandb_lineage.json", lineage)

    typer.echo(f"exported {label_summary['label_episode_count']} label episodes")
    typer.echo(f"dataset: {dataset_hash} ({project_dataset.title})")
    typer.echo(f"source artifact: {lineage['source_dataset_artifact']}")
    typer.echo(f"labels artifact: {lineage['labels_artifact']}")
    typer.echo(f"local files: {output_dir}")
    typer.echo(f"run: {lineage['run_url']}")


if __name__ == "__main__":
    typer.run(main)
