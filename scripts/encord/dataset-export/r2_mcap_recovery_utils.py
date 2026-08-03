# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "mcap-protobuf-support",
#     "tqdm",
#     "typer",
# ]
# ///
"""Utilities for recovering Encord data groups from R2 MCAP objects."""

from __future__ import annotations

import json
import multiprocessing
import queue
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from download_r2_prefix_to_cache import (
    R2Object,
    cache_one,
    r2_cache_path,
    transfer_config,
)
from tqdm import tqdm

CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_TOPICS = {
    "cam_high": "/observation.images.cam_high",
    "cam_left_wrist": "/observation.images.cam_left_wrist",
    "cam_right_wrist": "/observation.images.cam_right_wrist",
}
METADATA_ROLES = ("info", "tasks", "episodes", "episodes_stats")
METADATA_PATHS = {
    "info": Path("meta/info.json"),
    "tasks": Path("meta/tasks.jsonl"),
    "episodes": Path("meta/episodes.jsonl"),
    "episodes_stats": Path("meta/episodes_stats.jsonl"),
}
MCAP_PREFIX_MARKER = "raw-feed/trossen-data/"
RECOVERY_MARKER = ".encord_r2_recovery_complete.json"
FILE_MAP_FILENAME = "file_map.json"
PROJECT_MANIFEST_FILENAME = "project_recovery_manifest.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceItem:
    uuid: str
    title: str
    client_metadata: dict[str, Any]
    role: str


@dataclass(frozen=True)
class RecoveryEpisode:
    project_hash: str
    dataset_hash: str
    data_hash: str
    data_title: str
    group_uuid: str
    group_name: str
    group_client_metadata: dict[str, Any]
    episode_path: str
    episode_id: str
    episode_index: int
    task_name: str
    r2_bucket: str
    r2_key: str
    r2_size: int
    fps: float
    videos: dict[str, SourceItem]
    metadata_items: dict[str, SourceItem]


@dataclass(frozen=True)
class ExtractionJob:
    episode: RecoveryEpisode
    mcap_path: Path
    output_root: Path
    ffmpeg_bin: str
    ffprobe_bin: str
    overwrite: bool


@dataclass(frozen=True)
class ExtractionResult:
    data_hash: str
    status: str
    episode_dir: str
    error: str | None = None
    details: dict[str, Any] | None = None


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    if hasattr(value, "value"):
        return json_safe(value.value)
    return str(value)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(json_safe(value), indent=2) + "\n")
    temp_path.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp_path.write_text(value)
    temp_path.replace(path)


def batched(values: list[str], size: int = 500) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def get_storage_items_batched(client: Any, item_ids: list[str]) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for batch in batched(list(dict.fromkeys(item_ids))):
        for item in client.get_storage_items(batch):
            items[str(item.uuid)] = item
    return items


def metadata_role(item: Any) -> str | None:
    metadata = getattr(item, "client_metadata", None) or {}
    role = str(metadata.get("metadata_file_role") or "")
    if role in METADATA_ROLES:
        return role
    name = PurePosixPath(str(getattr(item, "name", ""))).name.lower()
    by_name = {
        "info.json": "info",
        "tasks.jsonl": "tasks",
        "episodes.jsonl": "episodes",
        "episodes_stats.jsonl": "episodes_stats",
    }
    return by_name.get(name)


def camera_role(item: Any) -> tuple[str | None, bool]:
    metadata = getattr(item, "client_metadata", None) or {}
    declared = str(metadata.get("camera_name") or "")
    if declared in CAMERA_ORDER:
        return declared, False

    sensor_key = str(metadata.get("sensor_key") or "")
    if sensor_key.startswith("observation.images."):
        inferred = sensor_key.removeprefix("observation.images.")
        if inferred in CAMERA_ORDER:
            return inferred, True

    for part in PurePosixPath(str(getattr(item, "name", ""))).parts:
        if not part.startswith("observation.images."):
            continue
        inferred = part.removeprefix("observation.images.")
        if inferred in CAMERA_ORDER:
            return inferred, True
    return None, False


def source_item_score(
    item: SourceItem,
    episode_path: str,
) -> tuple[int, int, int, int, str]:
    metadata = item.client_metadata
    return (
        int(item.title.lstrip("/").startswith(episode_path.lstrip("/"))),
        int(str(metadata.get("source_uri") or "").startswith("s3://")),
        int(bool(metadata.get("source_key"))),
        len(metadata),
        item.uuid,
    )


def episode_fields(episode_path: str) -> tuple[str, int, str]:
    parts = PurePosixPath(episode_path.strip("/")).parts
    if len(parts) < 7 or "/".join(parts[:2]) != "raw-feed/trossen-data":
        raise ValueError(f"Unsupported source episode path: {episode_path}")
    episode_id = parts[-1]
    if not episode_id.startswith("episode_"):
        raise ValueError(f"Episode path does not end in episode_*: {episode_path}")
    index_text = episode_id.removeprefix("episode_").split("_", 1)[0]
    return episode_id, int(index_text), parts[2]


def r2_key_for_episode(episode_path: str, r2_prefix: str) -> str:
    normalized = episode_path.strip("/")
    if not normalized.startswith(MCAP_PREFIX_MARKER):
        raise ValueError(
            f"Episode path does not start with {MCAP_PREFIX_MARKER!r}: {episode_path}"
        )
    relative = normalized.removeprefix(MCAP_PREFIX_MARKER)
    return f"{r2_prefix.strip('/')}/{relative}.mcap"


def create_encord_client(ssh_key_file: Path, domain: str) -> Any:
    from encord.user_client import EncordUserClient

    key_path = ssh_key_file.expanduser().resolve()
    if not key_path.is_file():
        raise ValueError(f"Encord SSH key file does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(
        ssh_private_key_path=key_path,
        domain=domain.rstrip("/"),
    )


def discover_project_episodes(
    *,
    client: Any,
    project_hash: str,
    dataset_hash: str | None,
    r2_bucket: str,
    r2_prefix: str,
    episode_path_contains: str | None,
    limit: int | None,
) -> tuple[dict[str, Any], list[RecoveryEpisode]]:
    source_warnings: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()

    def add_warning(
        kind: str,
        *,
        message: str,
        data_hash: str = "",
        group_uuid: str = "",
        episode_path: str = "",
        **details: Any,
    ) -> None:
        warning_counts[kind] += 1
        source_warnings.append(
            {
                "kind": kind,
                "message": message,
                "data_hash": data_hash,
                "group_uuid": group_uuid,
                "episode_path": episode_path,
                **details,
            }
        )

    project = client.get_project(project_hash)
    attached_datasets = list(project.list_datasets())
    if dataset_hash is None:
        if len(attached_datasets) != 1:
            details = ", ".join(
                f"{dataset.title} ({dataset.dataset_hash})"
                for dataset in attached_datasets
            )
            raise ValueError(
                f"Project {project_hash} must have exactly one dataset unless --dataset-hash is passed; "
                f"found: {details or 'none'}"
            )
        selected_dataset_hash = str(attached_datasets[0].dataset_hash)
        selected_dataset_title = str(attached_datasets[0].title)
    else:
        selected_dataset_hash = dataset_hash
        selected_dataset_title = next(
            (
                str(dataset.title)
                for dataset in attached_datasets
                if str(dataset.dataset_hash) == dataset_hash
            ),
            "",
        )

    dataset = client.get_dataset(selected_dataset_hash)
    rows = list(dataset.data_rows)
    backing_ids = [str(row.backing_item_uuid) for row in rows if row.backing_item_uuid]
    group_items = get_storage_items_batched(client, backing_ids)

    selected: list[tuple[Any, Any, str]] = []
    for row in rows:
        group_item = group_items.get(str(row.backing_item_uuid))
        if group_item is None:
            add_warning(
                "missing_backing_group",
                message=f"Could not resolve backing group {row.backing_item_uuid}; skipped.",
                data_hash=str(row.uid),
                group_uuid=str(row.backing_item_uuid or ""),
            )
            continue
        group_metadata = getattr(group_item, "client_metadata", None) or {}
        episode_path = str(group_metadata.get("episode_path") or "")
        if not episode_path:
            add_warning(
                "missing_episode_path",
                message="Data group has no client_metadata.episode_path; skipped.",
                data_hash=str(row.uid),
                group_uuid=str(group_item.uuid),
            )
            continue
        if episode_path_contains and episode_path_contains not in episode_path:
            continue
        selected.append((row, group_item, episode_path))

    selected.sort(key=lambda value: (value[2], str(value[0].uid)))
    if limit is not None:
        selected = selected[:limit]

    child_ids: list[str] = []
    for _row, group_item, _episode_path in selected:
        group_metadata = getattr(group_item, "client_metadata", None) or {}
        child_ids.extend(
            str(value) for value in group_metadata.get("video_uuids") or []
        )
        child_ids.extend(str(value) for value in group_metadata.get("json_uuids") or [])
    child_items = get_storage_items_batched(client, child_ids)

    episodes: list[RecoveryEpisode] = []
    for row, group_item, episode_path in selected:
        group_metadata = dict(getattr(group_item, "client_metadata", None) or {})
        video_ids = [str(value) for value in group_metadata.get("video_uuids") or []]
        json_ids = [str(value) for value in group_metadata.get("json_uuids") or []]
        try:
            episode_id, episode_index, task_name = episode_fields(episode_path)
        except (TypeError, ValueError) as exc:
            add_warning(
                "invalid_episode_path",
                message=f"{exc}; skipped.",
                data_hash=str(row.uid),
                group_uuid=str(group_item.uuid),
                episode_path=episode_path,
            )
            continue

        video_candidates: dict[str, list[SourceItem]] = {
            camera: [] for camera in CAMERA_ORDER
        }
        for item_id in video_ids:
            item = child_items.get(item_id)
            if item is None:
                add_warning(
                    "missing_video_item",
                    message=f"Could not resolve referenced video item {item_id}.",
                    data_hash=str(row.uid),
                    group_uuid=str(group_item.uuid),
                    episode_path=episode_path,
                    storage_item_uuid=item_id,
                )
                continue
            item_metadata = dict(getattr(item, "client_metadata", None) or {})
            camera_name, inferred = camera_role(item)
            if camera_name is None:
                add_warning(
                    "unknown_camera_role",
                    message=f"Could not identify camera role for {item.name}; ignored.",
                    data_hash=str(row.uid),
                    group_uuid=str(group_item.uuid),
                    episode_path=episode_path,
                    storage_item_uuid=str(item.uuid),
                )
                continue
            if inferred:
                add_warning(
                    "inferred_camera_role",
                    message=f"Inferred {camera_name} from the canonical item path.",
                    data_hash=str(row.uid),
                    group_uuid=str(group_item.uuid),
                    episode_path=episode_path,
                    storage_item_uuid=str(item.uuid),
                )
            item_metadata.setdefault("camera_name", camera_name)
            item_metadata.setdefault("sensor_key", f"observation.images.{camera_name}")
            video_candidates[camera_name].append(
                SourceItem(
                    uuid=str(item.uuid),
                    title=str(item.name),
                    client_metadata=item_metadata,
                    role=camera_name,
                )
            )

        videos: dict[str, SourceItem] = {}
        for camera_name, candidates in video_candidates.items():
            if not candidates:
                title = (
                    episode_path
                    + "videos/chunk-000/"
                    + f"observation.images.{camera_name}/{episode_id}.mp4"
                )
                videos[camera_name] = SourceItem(
                    uuid="",
                    title=title,
                    client_metadata={
                        "camera_name": camera_name,
                        "sensor_key": f"observation.images.{camera_name}",
                    },
                    role=camera_name,
                )
                add_warning(
                    "synthesized_camera_item",
                    message=(
                        f"No source item identified for {camera_name}; synthesized its "
                        "canonical title and will require the MCAP topic to validate."
                    ),
                    data_hash=str(row.uid),
                    group_uuid=str(group_item.uuid),
                    episode_path=episode_path,
                    camera_name=camera_name,
                )
                continue
            selected_item = max(
                candidates,
                key=lambda candidate: source_item_score(candidate, episode_path),
            )
            videos[camera_name] = selected_item
            if len(candidates) > 1:
                add_warning(
                    "duplicate_camera_role",
                    message=(
                        f"Found {len(candidates)} references for {camera_name}; selected "
                        f"{selected_item.uuid} using episode path and metadata completeness."
                    ),
                    data_hash=str(row.uid),
                    group_uuid=str(group_item.uuid),
                    episode_path=episode_path,
                    camera_name=camera_name,
                    candidate_storage_item_uuids=[
                        candidate.uuid for candidate in candidates
                    ],
                    selected_storage_item_uuid=selected_item.uuid,
                )

        metadata_items: dict[str, SourceItem] = {}
        for item_id in json_ids:
            item = child_items.get(item_id)
            if item is None:
                continue
            role = metadata_role(item)
            if role is None:
                continue
            metadata_items[role] = SourceItem(
                uuid=str(item.uuid),
                title=str(item.name),
                client_metadata=dict(getattr(item, "client_metadata", None) or {}),
                role=role,
            )

        fps_values = [
            float(item.client_metadata.get("collection_fps") or 0)
            for item in videos.values()
            if item.client_metadata.get("collection_fps")
        ]
        fps = fps_values[0] if fps_values else 30.0
        episodes.append(
            RecoveryEpisode(
                project_hash=project_hash,
                dataset_hash=selected_dataset_hash,
                data_hash=str(row.uid),
                data_title=str(row.title),
                group_uuid=str(group_item.uuid),
                group_name=str(getattr(group_item, "name", None) or row.title),
                group_client_metadata=group_metadata,
                episode_path=episode_path,
                episode_id=episode_id,
                episode_index=episode_index,
                task_name=task_name,
                r2_bucket=r2_bucket,
                r2_key=r2_key_for_episode(episode_path, r2_prefix),
                r2_size=0,
                fps=fps,
                videos=videos,
                metadata_items=metadata_items,
            )
        )

    project_info = {
        "project_hash": project_hash,
        "project_title": str(project.title),
        "dataset_hash": selected_dataset_hash,
        "dataset_title": selected_dataset_title or str(getattr(dataset, "title", "")),
        "project_data_row_count": len(rows),
        "selected_episode_count": len(episodes),
        "source_warning_count": len(source_warnings),
        "source_warning_summary": dict(warning_counts),
        "source_warnings": source_warnings,
    }
    return project_info, episodes


def list_r2_objects(client_r2: Any, bucket: str, prefix: str) -> dict[str, int]:
    objects: dict[str, int] = {}
    paginator = client_r2.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.strip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            objects[key] = int(item["Size"])
    return objects


def bind_r2_sizes(
    episodes: list[RecoveryEpisode],
    objects: dict[str, int],
) -> tuple[list[RecoveryEpisode], list[RecoveryEpisode]]:
    found: list[RecoveryEpisode] = []
    missing: list[RecoveryEpisode] = []
    for episode in episodes:
        size = objects.get(episode.r2_key)
        if size is None:
            missing.append(episode)
        else:
            found.append(replace(episode, r2_size=size))
    return found, missing


def episode_output_dir(output_root: Path, episode: RecoveryEpisode) -> Path:
    parts = PurePosixPath(episode.episode_path.strip("/")).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe output episode path: {episode.episode_path}")
    return output_root.joinpath(*parts)


def relative_child_path(episode: RecoveryEpisode, item: SourceItem) -> Path:
    normalized_title = item.title.lstrip("/")
    episode_prefix = episode.episode_path.lstrip("/")
    if normalized_title.startswith(episode_prefix):
        relative = normalized_title[len(episode_prefix) :]
        parts = PurePosixPath(relative).parts
        if parts and not any(part in {"", ".", ".."} for part in parts):
            return Path(*parts)
    if item.role in CAMERA_ORDER:
        return (
            Path("videos")
            / "chunk-000"
            / f"observation.images.{item.role}"
            / f"{episode.episode_id}.mp4"
        )
    return METADATA_PATHS[item.role]


def recovered_client_metadata(
    episode: RecoveryEpisode,
    item: SourceItem,
) -> dict[str, Any]:
    metadata = dict(item.client_metadata)
    metadata.setdefault("source_key", item.title)
    if not metadata.get("source_uri"):
        source_uri = next(
            (
                str(video.client_metadata.get("source_uri"))
                for video in episode.videos.values()
                if str(video.client_metadata.get("source_uri") or "").startswith(
                    "s3://"
                )
            ),
            "",
        )
        if source_uri.startswith("s3://"):
            source_bucket = source_uri.removeprefix("s3://").split("/", 1)[0]
            metadata["source_uri"] = f"s3://{source_bucket}/{item.title}"
        else:
            metadata["source_uri"] = f"r2://{episode.r2_bucket}/{episode.r2_key}"
    metadata.setdefault("file_ext", Path(item.title).suffix)
    if item.role in METADATA_ROLES:
        metadata.setdefault("metadata_file_role", item.role)
    metadata.update(
        {
            "recovered_from_r2_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
            "recovered_from_mcap": True,
        }
    )
    recovery_context = {
        "recovery_source_project_hash": episode.project_hash,
        "recovery_source_dataset_hash": episode.dataset_hash,
        "recovery_source_data_hash": episode.data_hash,
        "recovery_source_storage_item_uuid": item.uuid,
    }
    metadata.update({key: value for key, value in recovery_context.items() if value})
    metadata.setdefault("episode_path", episode.episode_path)
    metadata.setdefault("episode_id", episode.episode_id)
    metadata.setdefault("episode_index", episode.episode_index)
    metadata.setdefault("task_name", episode.task_name)
    return metadata


def default_metadata_item(episode: RecoveryEpisode, role: str) -> SourceItem:
    relative_path = METADATA_PATHS[role].as_posix()
    title = episode.episode_path + relative_path
    return SourceItem(
        uuid="",
        title=title,
        client_metadata={
            "source_key": title,
            "source_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
            "file_ext": Path(relative_path).suffix,
            "metadata_file_role": role,
            "episode_path": episode.episode_path,
            "episode_id": episode.episode_id,
            "episode_index": episode.episode_index,
            "task_name": episode.task_name,
        },
        role=role,
    )


def output_specs(episode: RecoveryEpisode, output_root: Path) -> list[dict[str, Any]]:
    episode_dir = episode_output_dir(output_root, episode)
    specs: list[dict[str, Any]] = []
    for camera_name in CAMERA_ORDER:
        item = episode.videos[camera_name]
        output_path = episode_dir / relative_child_path(episode, item)
        specs.append(
            {
                "relative_path": output_path.relative_to(output_root).as_posix(),
                "title": item.title,
                "data_type": "video",
                "role": camera_name,
                "client_metadata": recovered_client_metadata(episode, item),
            }
        )
    for role in METADATA_ROLES:
        item = episode.metadata_items.get(role) or default_metadata_item(episode, role)
        output_path = episode_dir / METADATA_PATHS[role]
        specs.append(
            {
                "relative_path": output_path.relative_to(output_root).as_posix(),
                "title": item.title,
                "data_type": "text",
                "role": role,
                "client_metadata": recovered_client_metadata(episode, item),
            }
        )
    return specs


def valid_completion_marker(job: ExtractionJob) -> dict[str, Any] | None:
    episode_dir = episode_output_dir(job.output_root, job.episode)
    marker_path = episode_dir / RECOVERY_MARKER
    if job.overwrite or not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("schema_version") != SCHEMA_VERSION:
        return None
    if (
        marker.get("r2_key") != job.episode.r2_key
        or marker.get("r2_size") != job.episode.r2_size
    ):
        return None
    for spec in output_specs(job.episode, job.output_root):
        output_path = job.output_root / spec["relative_path"]
        if not output_path.is_file() or output_path.stat().st_size == 0:
            return None
    return marker


def ffmpeg_demuxer(codec_format: str) -> str:
    normalized = codec_format.lower()
    mapping = {
        "av1": "obu",
        "h264": "h264",
        "avc": "h264",
        "h265": "hevc",
        "hevc": "hevc",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported MCAP compressed video format: {codec_format!r}")
    return mapping[normalized]


def start_ffmpeg(
    *,
    ffmpeg_bin: str,
    codec_format: str,
    fps: float,
    output_path: Path,
) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        ffmpeg_demuxer(codec_format),
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def probe_video(ffprobe_bin: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams or int((payload.get("format") or {}).get("size") or 0) <= 0:
        raise RuntimeError(f"ffprobe found no valid video stream in {path}")
    return payload


def metadata_documents(
    episode: RecoveryEpisode,
    extraction_details: dict[str, Any],
) -> dict[str, str]:
    frame_counts = extraction_details["frame_counts"]
    frame_count = min(int(frame_counts[camera]) for camera in CAMERA_ORDER)
    first_video_metadata = episode.videos[CAMERA_ORDER[0]].client_metadata
    info = {
        "codebase_version": first_video_metadata.get("codebase_version", "v2.1"),
        "robot_type": first_video_metadata.get("robot_type", "trossen_ai_mobile"),
        "fps": episode.fps,
        "total_episodes": 1,
        "total_frames": frame_count,
        "total_videos": len(CAMERA_ORDER),
        "features": {
            f"observation.images.{camera}": {
                "dtype": "video",
                "video_info": {"video.fps": episode.fps},
            }
            for camera in CAMERA_ORDER
        },
        "recovery": {
            "source_project_hash": episode.project_hash,
            "source_dataset_hash": episode.dataset_hash,
            "source_data_hash": episode.data_hash,
            "source_r2_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
            "source_format": "mcap",
        },
    }
    task = {"task_index": 0, "task": episode.task_name}
    episode_row = {
        "episode_index": episode.episode_index,
        "tasks": [episode.task_name],
        "length": frame_count,
    }
    stats = {
        "episode_index": episode.episode_index,
        "stats": {},
        "recovered_from_mcap": True,
        "source_r2_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
    }
    return {
        "info": json.dumps(info, indent=2) + "\n",
        "tasks": json.dumps(task) + "\n",
        "episodes": json.dumps(episode_row) + "\n",
        "episodes_stats": json.dumps(stats) + "\n",
    }


def extract_episode(job: ExtractionJob) -> ExtractionResult:
    episode = job.episode
    episode_dir = episode_output_dir(job.output_root, episode)
    marker = valid_completion_marker(job)
    if marker is not None:
        return ExtractionResult(
            data_hash=episode.data_hash,
            status="cached",
            episode_dir=str(episode_dir),
            details=marker,
        )

    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    episode_dir.mkdir(parents=True, exist_ok=True)
    video_paths = {
        camera: episode_dir / relative_child_path(episode, episode.videos[camera])
        for camera in CAMERA_ORDER
    }
    temp_paths = {
        camera: path.with_name(f".{path.name}.{uuid4().hex}.tmp.mp4")
        for camera, path in video_paths.items()
    }
    for path in video_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    topic_to_camera = {topic: camera for camera, topic in CAMERA_TOPICS.items()}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    frame_counts: Counter[str] = Counter()
    codec_formats: dict[str, str] = {}

    try:
        with job.mcap_path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            statistics = summary.statistics if summary is not None else None
            for _schema, channel, _message, decoded in reader.iter_decoded_messages(
                topics=list(CAMERA_TOPICS.values())
            ):
                camera = topic_to_camera.get(channel.topic)
                if camera is None:
                    continue
                codec_format = str(getattr(decoded, "format", "") or "")
                payload = bytes(getattr(decoded, "data", b""))
                if not codec_format or not payload:
                    raise RuntimeError(
                        f"Empty compressed video message on {channel.topic}"
                    )
                if camera not in processes:
                    codec_formats[camera] = codec_format
                    processes[camera] = start_ffmpeg(
                        ffmpeg_bin=job.ffmpeg_bin,
                        codec_format=codec_format,
                        fps=episode.fps,
                        output_path=temp_paths[camera],
                    )
                process = processes[camera]
                if process.stdin is None:
                    raise RuntimeError(f"ffmpeg stdin is unavailable for {camera}")
                process.stdin.write(payload)
                frame_counts[camera] += 1

        missing = [camera for camera in CAMERA_ORDER if camera not in processes]
        if missing:
            raise RuntimeError(
                f"MCAP is missing video topics for: {', '.join(missing)}"
            )

        ffmpeg_errors: list[str] = []
        for camera in CAMERA_ORDER:
            process = processes[camera]
            if process.stdin is not None:
                process.stdin.close()
            stderr = (
                process.stderr.read().decode("utf-8", errors="replace")
                if process.stderr
                else ""
            )
            return_code = process.wait()
            if return_code != 0:
                ffmpeg_errors.append(
                    f"{camera}: {stderr.strip() or f'exit {return_code}'}"
                )
        if ffmpeg_errors:
            raise RuntimeError("ffmpeg extraction failed: " + "; ".join(ffmpeg_errors))

        probes = {
            camera: probe_video(job.ffprobe_bin, temp_paths[camera])
            for camera in CAMERA_ORDER
        }
        for camera in CAMERA_ORDER:
            temp_paths[camera].replace(video_paths[camera])

        duration_ns = 0
        message_count = None
        if statistics is not None:
            duration_ns = max(
                0, statistics.message_end_time - statistics.message_start_time
            )
            message_count = statistics.message_count
        details = {
            "schema_version": SCHEMA_VERSION,
            "r2_bucket": episode.r2_bucket,
            "r2_key": episode.r2_key,
            "r2_size": episode.r2_size,
            "mcap_cache_path": str(job.mcap_path),
            "mcap_message_count": message_count,
            "mcap_duration_seconds": duration_ns / 1_000_000_000,
            "fps": episode.fps,
            "frame_counts": dict(frame_counts),
            "codec_formats": codec_formats,
            "video_probes": probes,
        }

        documents = metadata_documents(episode, details)
        for role, content in documents.items():
            write_text_atomic(episode_dir / METADATA_PATHS[role], content)
        write_json_atomic(episode_dir / RECOVERY_MARKER, details)
        return ExtractionResult(
            data_hash=episode.data_hash,
            status="extracted",
            episode_dir=str(episode_dir),
            details=details,
        )
    except Exception as exc:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        return ExtractionResult(
            data_hash=episode.data_hash,
            status="failed",
            episode_dir=str(episode_dir),
            error=f"{type(exc).__name__}: {exc}",
        )


def extraction_worker(
    job_queue: Any,
    result_queue: Any,
) -> None:
    while True:
        job = job_queue.get()
        if job is None:
            return
        result_queue.put(extract_episode(job))


def run_download_extract_pipeline(
    *,
    episodes: list[RecoveryEpisode],
    client_r2: Any,
    cache_root: Path,
    output_root: Path,
    download_workers: int,
    extract_workers: int,
    multipart_concurrency: int,
    multipart_threshold_mb: int,
    multipart_chunksize_mb: int,
    overwrite_downloads: bool,
    overwrite_extracted: bool,
    byte_progress: bool,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> tuple[dict[str, Any], dict[str, ExtractionResult]]:
    transfer = transfer_config(
        multipart_concurrency=multipart_concurrency,
        multipart_threshold_mb=multipart_threshold_mb,
        multipart_chunksize_mb=multipart_chunksize_mb,
    )
    context = multiprocessing.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=extraction_worker,
            args=(job_queue, result_queue),
            name=f"mcap-extractor-{index + 1}",
        )
        for index in range(extract_workers)
    ]
    for worker in workers:
        worker.start()

    total_bytes = sum(episode.r2_size for episode in episodes)
    download_results: dict[str, Any] = {}
    extraction_results: dict[str, ExtractionResult] = {}
    queued_extractions = 0

    def drain_extraction_results(progress: tqdm[Any]) -> None:
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                return
            extraction_results[result.data_hash] = result
            progress.update(1)
            progress.set_postfix(
                Counter(item.status for item in extraction_results.values())
            )

    try:
        with (
            tqdm(
                total=total_bytes,
                desc="Downloading MCAPs",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                mininterval=2.0,
                position=0,
            ) as download_progress,
            tqdm(
                total=len(episodes),
                desc="Extracting episodes",
                unit="episode",
                dynamic_ncols=True,
                mininterval=2.0,
                position=1,
            ) as extraction_progress,
            ThreadPoolExecutor(max_workers=download_workers) as executor,
        ):
            callback = download_progress.update if byte_progress else None
            future_to_episode = {}
            for episode in episodes:
                obj = R2Object(episode.r2_bucket, episode.r2_key, episode.r2_size)
                future = executor.submit(
                    cache_one,
                    client_r2,
                    obj,
                    cache_root,
                    transfer,
                    False,
                    overwrite_downloads,
                    callback,
                )
                future_to_episode[future] = episode

            for future in as_completed(future_to_episode):
                episode = future_to_episode[future]
                download = future.result()
                download_results[episode.data_hash] = asdict(download)
                if download.action not in {"failed", "size_conflict"} and (
                    download.action not in {"downloaded", "overwritten"}
                    or not byte_progress
                ):
                    download_progress.update(download.size)
                download_progress.set_postfix(
                    Counter(result["action"] for result in download_results.values())
                )
                if download.action not in {"failed", "size_conflict"}:
                    job_queue.put(
                        ExtractionJob(
                            episode=episode,
                            mcap_path=Path(download.cache_path),
                            output_root=output_root,
                            ffmpeg_bin=ffmpeg_bin,
                            ffprobe_bin=ffprobe_bin,
                            overwrite=overwrite_extracted,
                        )
                    )
                    queued_extractions += 1
                drain_extraction_results(extraction_progress)

            for _worker in workers:
                job_queue.put(None)
            while len(extraction_results) < queued_extractions:
                result = result_queue.get()
                extraction_results[result.data_hash] = result
                extraction_progress.update(1)
                extraction_progress.set_postfix(
                    Counter(item.status for item in extraction_results.values())
                )
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        raise
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        job_queue.close()
        result_queue.close()

    return download_results, extraction_results


def build_file_map(
    *,
    episodes: list[RecoveryEpisode],
    extraction_results: dict[str, ExtractionResult],
    output_root: Path,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    data_groups: dict[str, Any] = {}
    for episode in episodes:
        result = extraction_results.get(episode.data_hash)
        if result is None or result.status not in {"cached", "extracted"}:
            continue
        specs = output_specs(episode, output_root)
        for spec in specs:
            files[spec["relative_path"]] = {
                key: value for key, value in spec.items() if key != "relative_path"
            }

        video_titles = {
            spec["role"]: spec["title"]
            for spec in specs
            if spec["data_type"] == "video"
        }
        metadata_titles = {
            spec["role"]: spec["title"] for spec in specs if spec["data_type"] == "text"
        }
        ordered_items = [video_titles[camera] for camera in CAMERA_ORDER] + [
            metadata_titles[role] for role in METADATA_ROLES
        ]
        group_metadata = dict(episode.group_client_metadata)
        group_metadata.update(
            {
                "recovered_from_r2_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
                "recovery_source_project_hash": episode.project_hash,
                "recovery_source_dataset_hash": episode.dataset_hash,
                "recovery_source_data_hash": episode.data_hash,
                "recovery_source_group_uuid": episode.group_uuid,
            }
        )
        data_groups[episode.group_name] = {
            "items": ordered_items,
            "layout": "trossen-three-camera-metadata",
            "roles": {
                "videos": video_titles,
                "metadata": metadata_titles,
            },
            "client_metadata": group_metadata,
            "source_data_hash": episode.data_hash,
            "source_group_uuid": episode.group_uuid,
            "episode_path": episode.episode_path,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "data_groups": data_groups,
        "image_groups": {},
    }


def build_project_manifest(
    *,
    project_info: dict[str, Any],
    episodes: list[RecoveryEpisode],
    missing: list[RecoveryEpisode],
    cache_root: Path,
    output_root: Path,
    download_results: dict[str, Any] | None = None,
    extraction_results: dict[str, ExtractionResult] | None = None,
) -> dict[str, Any]:
    download_results = download_results or {}
    extraction_results = extraction_results or {}
    rows = []
    for episode in episodes:
        extraction = extraction_results.get(episode.data_hash)
        rows.append(
            {
                "data_hash": episode.data_hash,
                "data_title": episode.data_title,
                "group_uuid": episode.group_uuid,
                "group_name": episode.group_name,
                "episode_path": episode.episode_path,
                "r2_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
                "r2_size": episode.r2_size,
                "mcap_cache_path": str(
                    r2_cache_path(cache_root, episode.r2_bucket, episode.r2_key)
                ),
                "output_episode_dir": str(episode_output_dir(output_root, episode)),
                "download": download_results.get(episode.data_hash),
                "extraction": asdict(extraction) if extraction is not None else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        **project_info,
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "matched_episode_count": len(episodes),
        "missing_episode_count": len(missing),
        "missing_episodes": [
            {
                "data_hash": episode.data_hash,
                "episode_path": episode.episode_path,
                "candidate_r2_uri": f"r2://{episode.r2_bucket}/{episode.r2_key}",
            }
            for episode in missing
        ],
        "episodes": rows,
    }
