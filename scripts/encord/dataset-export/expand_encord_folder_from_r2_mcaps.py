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
"""Expand an Encord video folder from new three-camera R2 MCAP episodes.

The command is audit-only unless ``--apply`` is passed. In apply mode it runs
four bounded stages: R2 download, MCAP extraction, Encord video upload, and
video-only data-group creation.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import queue
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from fractions import Fraction
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Annotated, Any
from uuid import UUID, uuid4

import typer
from download_r2_prefix_to_cache import (
    DEFAULT_MULTIPART_CHUNKSIZE_MB,
    DEFAULT_MULTIPART_THRESHOLD_MB,
    R2_CACHE_ROOT,
    R2Object,
    cache_one,
    r2_client,
    r2_endpoint_url,
    transfer_config,
)
from r2_mcap_recovery_utils import (
    CAMERA_ORDER,
    ExtractionJob,
    RecoveryEpisode,
    SourceItem,
    create_encord_client,
    episode_fields,
    extract_episode,
    list_r2_objects,
    output_specs,
    write_json_atomic,
)
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_REGISTRATION_DIR = SCRIPT_DIR.parent / "data-registration"
RECOVERY_ROOT = REPO_ROOT / "exports" / "encord-dataset-export" / "recovered" / "r2"
DEFAULT_REPORT_JSON = DATA_REGISTRATION_DIR / "encord_r2_folder_expansion_report.json"
DEFAULT_STATE_JSON = DATA_REGISTRATION_DIR / "encord_r2_folder_expansion_state.json"
DEFAULT_VIDEO_FOLDER_HASH = "019fa354-8eb5-7ea1-badd-87c7f31db011"
DEFAULT_GROUP_FOLDER_HASH = "019fb038-8536-7284-b931-fa1f25ba7282"
DEFAULT_BUCKET = "trossen-robotics-data"
DEFAULT_R2_PREFIX = "trossen-data-mobile"
DEFAULT_EPISODE_LIMIT = 2250
DEFAULT_DOWNLOAD_WORKERS = 12
DEFAULT_EXTRACT_WORKERS = 4
DEFAULT_UPLOAD_WORKERS = 10
DEFAULT_GROUP_BATCH_SIZE = 50
DEFAULT_QUEUE_DEPTH = 24
DEFAULT_MULTIPART_CONCURRENCY = 2
STATE_SCHEMA_VERSION = 1
STATE_KIND = "r2-encord-folder-expansion-v1"
INGESTION_PROBE = "r2-expanded-video-only"
EPISODE_NAME_RE = re.compile(r"^episode_(\d+)(?:_[0-9A-Za-z][0-9A-Za-z._-]*)?$")


class SelectionMode(StrEnum):
    BALANCED = "balanced"
    LEXICAL = "lexical"


@dataclass(frozen=True)
class VideoInventory:
    title_to_uuid: dict[str, str]
    camera_uuids_by_episode: dict[str, dict[str, str]]
    ingestion_ids_by_episode: dict[str, set[str]]
    duplicate_slots: list[dict[str, str]]
    unclassified_video_count: int

    @property
    def complete_episodes(self) -> set[str]:
        required = set(CAMERA_ORDER)
        return {
            episode_path
            for episode_path, cameras in self.camera_uuids_by_episode.items()
            if set(cameras) == required
        }


@dataclass(frozen=True)
class GroupInventory:
    group_uuids_by_ingestion_episode: dict[tuple[str, str], str]
    group_counts_by_ingestion_episode: dict[tuple[str, str], int]
    camera_maps_by_group_uuid: dict[str, dict[str, str]]
    validation_failures_by_group_uuid: dict[str, list[str]]
    episode_counts: dict[str, int]


@dataclass(frozen=True)
class DownloadedEpisode:
    episode: RecoveryEpisode
    cache_path: str
    download_action: str


@dataclass(frozen=True)
class ExtractedEpisode:
    episode: RecoveryEpisode
    cache_path: str
    download_action: str
    episode_dir: str
    details: dict[str, Any]


@dataclass(frozen=True)
class UploadedEpisode:
    episode: RecoveryEpisode
    cache_path: str
    download_action: str
    episode_dir: str
    camera_uuid_map: dict[str, str]
    uploaded_uuids: list[str]


@dataclass(frozen=True)
class TerminalEvent:
    episode_path: str
    status: str
    stage: str
    error: str | None = None
    group_uuid: str | None = None
    camera_uuid_map: dict[str, str] | None = None
    r2_key: str | None = None
    details: dict[str, Any] | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    for scheme in ("s3://", "r2://"):
        if text.startswith(scheme):
            text = text.removeprefix(scheme).split("/", 1)[-1]
    return text.lstrip("/")


def canonical_episode_path(value: Any) -> str | None:
    parts = PurePosixPath(normalize_path(value)).parts
    for index, part in enumerate(parts):
        if not EPISODE_NAME_RE.fullmatch(part):
            continue
        start = next(
            (
                candidate
                for candidate in range(max(0, index - 8), index)
                if parts[candidate : candidate + 2] == ("raw-feed", "trossen-data")
            ),
            None,
        )
        if start is None:
            continue
        selected = parts[start : index + 1]
        if len(selected) >= 7:
            return "/".join(selected) + "/"
    return None


def camera_from_values(metadata: dict[str, Any], title: str) -> str | None:
    declared = str(metadata.get("camera_name") or "")
    if declared in CAMERA_ORDER:
        return declared
    sensor_key = str(metadata.get("sensor_key") or "")
    for camera in CAMERA_ORDER:
        if sensor_key.endswith(camera):
            return camera
    for part in PurePosixPath(title).parts:
        if not part.startswith("observation.images."):
            continue
        camera = part.removeprefix("observation.images.")
        if camera in CAMERA_ORDER:
            return camera
    return None


def item_metadata(item: Any) -> dict[str, Any]:
    return dict(getattr(item, "client_metadata", None) or {})


def inventory_video_folder(folder: Any) -> VideoInventory:
    from encord.orm.storage import StorageItemType

    title_to_uuid: dict[str, str] = {}
    camera_uuids_by_episode: dict[str, dict[str, str]] = defaultdict(dict)
    ingestion_ids_by_episode: dict[str, set[str]] = defaultdict(set)
    duplicate_slots: list[dict[str, str]] = []
    unclassified = 0

    for item in folder.list_items(
        page_size=1000,
        item_types=[StorageItemType.VIDEO],
    ):
        title = str(item.name)
        item_uuid = str(item.uuid)
        metadata = item_metadata(item)
        title_to_uuid.setdefault(title, item_uuid)
        episode_path = (
            canonical_episode_path(metadata.get("episode_path"))
            or canonical_episode_path(metadata.get("source_key"))
            or canonical_episode_path(title)
        )
        camera = camera_from_values(metadata, title)
        if episode_path is None or camera is None:
            unclassified += 1
            continue
        current = camera_uuids_by_episode[episode_path].get(camera)
        if current and current != item_uuid:
            duplicate_slots.append(
                {
                    "episode_path": episode_path,
                    "camera": camera,
                    "first_uuid": current,
                    "duplicate_uuid": item_uuid,
                }
            )
            continue
        camera_uuids_by_episode[episode_path][camera] = item_uuid
        ingestion_id = str(metadata.get("r2_expansion_ingestion_id") or "")
        if ingestion_id:
            ingestion_ids_by_episode[episode_path].add(ingestion_id)

    return VideoInventory(
        title_to_uuid=title_to_uuid,
        camera_uuids_by_episode=dict(camera_uuids_by_episode),
        ingestion_ids_by_episode=dict(ingestion_ids_by_episode),
        duplicate_slots=duplicate_slots,
        unclassified_video_count=unclassified,
    )


def inventory_group_folder(folder: Any) -> GroupInventory:
    from encord.orm.storage import StorageItemType

    by_ingestion_episode: dict[tuple[str, str], str] = {}
    counts_by_ingestion_episode: Counter[tuple[str, str]] = Counter()
    camera_maps_by_group_uuid: dict[str, dict[str, str]] = {}
    validation_failures_by_group_uuid: dict[str, list[str]] = {}
    episode_counts: Counter[str] = Counter()
    for item in folder.list_items(
        page_size=1000,
        item_types=[StorageItemType.GROUP],
    ):
        metadata = item_metadata(item)
        episode_path = canonical_episode_path(metadata.get("episode_path"))
        if episode_path is None:
            continue
        episode_counts[episode_path] += 1
        ingestion_id = str(metadata.get("r2_expansion_ingestion_id") or "")
        if ingestion_id:
            group_uuid = str(item.uuid)
            key = (ingestion_id, episode_path)
            by_ingestion_episode.setdefault(
                key,
                group_uuid,
            )
            counts_by_ingestion_episode[key] += 1
            camera_map = metadata.get("camera_uuid_map")
            if isinstance(camera_map, dict):
                camera_maps_by_group_uuid[group_uuid] = {
                    str(camera): str(value) for camera, value in camera_map.items()
                }
            reasons = []
            if metadata.get("json_uuids") != []:
                reasons.append("json_uuids is not empty")
            if not isinstance(camera_map, dict) or set(camera_map) != set(CAMERA_ORDER):
                reasons.append("camera_uuid_map is incomplete")
            video_uuids = metadata.get("video_uuids")
            if not isinstance(video_uuids, list) or len(video_uuids) != len(
                CAMERA_ORDER
            ):
                reasons.append("video_uuids does not contain three videos")
            if reasons:
                validation_failures_by_group_uuid[group_uuid] = reasons
    return GroupInventory(
        group_uuids_by_ingestion_episode=by_ingestion_episode,
        group_counts_by_ingestion_episode=dict(counts_by_ingestion_episode),
        camera_maps_by_group_uuid=camera_maps_by_group_uuid,
        validation_failures_by_group_uuid=validation_failures_by_group_uuid,
        episode_counts=dict(episode_counts),
    )


def synthetic_video_metadata(
    *,
    episode_path: str,
    camera: str,
    r2_bucket: str,
    r2_key: str,
    ingestion_id: str,
) -> dict[str, Any]:
    parts = PurePosixPath(episode_path.strip("/")).parts
    episode_id, episode_index, task_name = episode_fields(episode_path)
    title = (
        episode_path
        + "videos/chunk-000/"
        + f"observation.images.{camera}/{episode_id}.mp4"
    )
    return {
        "Tag": "A",
        "Data Type": "video",
        "Extension": ".mp4",
        "source_family": "trossen-data",
        "source_key": title,
        "source_uri": f"r2://{r2_bucket}/{r2_key}",
        "file_ext": ".mp4",
        "metadata_file_role": "none",
        "camera_name": camera,
        "sensor_key": f"observation.images.{camera}",
        "video_key": f"observation.images.{camera}",
        "episode_path": episode_path,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "task_name": task_name,
        "environment": parts[3],
        "collection_operator": parts[4],
        "collection_datetime": parts[5],
        "collection_fps": 30,
        "fps": 30.0,
        "codebase_version": "v2.1",
        "robot_type": "trossen_ai_mobile",
        "trossen_subversion": "v1.0",
        "action_dim": 16,
        "state_dim": 16,
        "has_info_json": True,
        "has_tasks_jsonl": True,
        "has_episodes_jsonl": True,
        "has_episodes_stats_jsonl": True,
        "has_parquet": False,
        "recovered_from_mcap": True,
        "recovered_from_r2_uri": f"r2://{r2_bucket}/{r2_key}",
        "r2_expansion_ingestion_id": ingestion_id,
    }


def recovery_episode_from_r2(
    *,
    bucket: str,
    prefix: str,
    key: str,
    size: int,
    ingestion_id: str,
) -> RecoveryEpisode | None:
    normalized_prefix = prefix.strip("/") + "/"
    if not key.startswith(normalized_prefix) or not key.endswith(".mcap"):
        return None
    relative = key[len(normalized_prefix) : -len(".mcap")].strip("/")
    if not relative:
        return None
    episode_path = f"raw-feed/trossen-data/{relative}/"
    try:
        episode_id, episode_index, task_name = episode_fields(episode_path)
    except (TypeError, ValueError):
        return None
    videos = {}
    for camera in CAMERA_ORDER:
        title = (
            episode_path
            + "videos/chunk-000/"
            + f"observation.images.{camera}/{episode_id}.mp4"
        )
        videos[camera] = SourceItem(
            uuid="",
            title=title,
            client_metadata=synthetic_video_metadata(
                episode_path=episode_path,
                camera=camera,
                r2_bucket=bucket,
                r2_key=key,
                ingestion_id=ingestion_id,
            ),
            role=camera,
        )
    return RecoveryEpisode(
        project_hash="",
        dataset_hash="",
        data_hash=episode_path,
        data_title=episode_path,
        group_uuid="",
        group_name=episode_path.rstrip("/"),
        group_client_metadata={
            "probe": INGESTION_PROBE,
            "episode_path": episode_path,
            "source_r2_uri": f"r2://{bucket}/{key}",
            "r2_expansion_ingestion_id": ingestion_id,
        },
        episode_path=episode_path,
        episode_id=episode_id,
        episode_index=episode_index,
        task_name=task_name,
        r2_bucket=bucket,
        r2_key=key,
        r2_size=size,
        fps=30.0,
        videos=videos,
        metadata_items={},
    )


def selection_bucket(episode: RecoveryEpisode) -> tuple[str, str]:
    parts = PurePosixPath(episode.episode_path.strip("/")).parts
    return episode.task_name, parts[3]


def order_candidates(
    episodes: list[RecoveryEpisode],
    mode: SelectionMode,
) -> list[RecoveryEpisode]:
    ordered = sorted(episodes, key=lambda episode: episode.r2_key)
    if mode == SelectionMode.LEXICAL:
        return ordered
    buckets: dict[tuple[str, str], deque[RecoveryEpisode]] = defaultdict(deque)
    for episode in ordered:
        buckets[selection_bucket(episode)].append(episode)
    output: list[RecoveryEpisode] = []
    active_keys = deque(sorted(buckets))
    while active_keys:
        key = active_keys.popleft()
        output.append(buckets[key].popleft())
        if buckets[key]:
            active_keys.append(key)
    return output


def should_schedule(successful: int, active: int, target: int, depth: int) -> bool:
    return successful + active < target and active < depth


def fraction_float(value: Any) -> float | None:
    text = str(value or "")
    if not text or text == "0/0":
        return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def enrich_video_metadata(
    metadata: dict[str, Any],
    probe: dict[str, Any],
    frame_count: int | None,
    video_folder_hash: str,
) -> dict[str, Any]:
    output = dict(metadata)
    streams = probe.get("streams") or []
    stream = streams[0] if streams else {}
    fps = fraction_float(stream.get("avg_frame_rate")) or fraction_float(
        stream.get("r_frame_rate")
    )
    if fps:
        output["fps"] = fps
        output["collection_fps"] = round(fps, 6)
    output.update(
        {
            "video_codec": str(stream.get("codec_name") or ""),
            "video_width": int(stream.get("width") or 0),
            "video_height": int(stream.get("height") or 0),
            "video_has_audio": False,
            "source_folder_id": video_folder_hash,
        }
    )
    if frame_count is not None:
        output["video_frame_count"] = frame_count
    duration = stream.get("duration") or (probe.get("format") or {}).get("duration")
    if duration not in (None, ""):
        output["video_duration_seconds"] = float(duration)
    return output


def safe_unlink(path: Path, root: Path) -> None:
    resolved = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"Refusing to delete path outside {resolved_root}: {resolved}")
    resolved.unlink(missing_ok=True)


def safe_rmtree(path: Path, root: Path) -> None:
    resolved = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"Refusing to delete path outside {resolved_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def cleanup_artifacts(
    *,
    cache_path: str,
    episode_dir: str,
    cache_root: Path,
    output_root: Path,
) -> None:
    if episode_dir:
        safe_rmtree(Path(episode_dir), output_root)
    if cache_path:
        safe_unlink(Path(cache_path), cache_root)


def configure_tqdm_lock(lock: Any) -> None:
    if lock is not None:
        tqdm.set_lock(lock)


def run_thread_stage(
    *,
    input_queue: Any,
    output_queue: Any,
    terminal_queue: Any,
    max_workers: int,
    position: int,
    description: str,
    action: Callable[[Any], tuple[Any | None, TerminalEvent | None]],
) -> None:
    futures: dict[Any, Any] = {}
    input_closed = False
    counts: Counter[str] = Counter()
    with (
        tqdm(
            total=None,
            desc=description,
            position=position,
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format=(
                "{desc}: {n_fmt} episodes "
                "[{elapsed}, {rate_fmt}{postfix}]"
            ),
        ) as progress,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        while not input_closed or futures:
            while not input_closed and len(futures) < max_workers:
                try:
                    item = input_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                if item is None:
                    input_closed = True
                    break
                futures[executor.submit(action, item)] = item

            if not futures:
                continue
            done, _pending = wait(
                futures,
                timeout=0.2,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                item = futures.pop(future)
                try:
                    output, terminal = future.result()
                except Exception as exc:  # noqa: BLE001
                    episode = getattr(item, "episode", item)
                    terminal = TerminalEvent(
                        episode_path=str(getattr(episode, "episode_path", "[unknown]")),
                        status="failed",
                        stage=description.lower(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    output = None
                status = terminal.status if terminal else "succeeded"
                counts[status] += 1
                if terminal is not None:
                    terminal_queue.put(terminal)
                elif output is not None:
                    output_queue.put(output)
                progress.update(1)
                progress.set_postfix(dict(counts))
    output_queue.put(None)


def download_stage(
    input_queue: Any,
    output_queue: Any,
    terminal_queue: Any,
    progress_lock: Any,
    *,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    cache_root: str,
    workers: int,
    multipart_concurrency: int,
    multipart_threshold_mb: int,
    multipart_chunksize_mb: int,
    overwrite: bool,
) -> None:
    configure_tqdm_lock(progress_lock)
    client = r2_client(
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        max_pool_connections=max(32, workers * multipart_concurrency + 8),
    )
    transfer = transfer_config(
        multipart_concurrency=multipart_concurrency,
        multipart_threshold_mb=multipart_threshold_mb,
        multipart_chunksize_mb=multipart_chunksize_mb,
    )
    resolved_cache_root = Path(cache_root)

    def action(episode: RecoveryEpisode) -> tuple[Any | None, TerminalEvent | None]:
        result = cache_one(
            client,
            R2Object(episode.r2_bucket, episode.r2_key, episode.r2_size),
            resolved_cache_root,
            transfer,
            False,
            overwrite,
        )
        if result.action in {"failed", "size_conflict"}:
            return None, TerminalEvent(
                episode_path=episode.episode_path,
                status="failed",
                stage="download",
                error=result.error or result.action,
                r2_key=episode.r2_key,
            )
        return (
            DownloadedEpisode(
                episode=episode,
                cache_path=str(result.cache_path),
                download_action=result.action,
            ),
            None,
        )

    run_thread_stage(
        input_queue=input_queue,
        output_queue=output_queue,
        terminal_queue=terminal_queue,
        max_workers=workers,
        position=0,
        description="Downloading",
        action=action,
    )


def extract_stage(
    input_queue: Any,
    output_queue: Any,
    terminal_queue: Any,
    progress_lock: Any,
    *,
    output_root: str,
    cache_root: str,
    workers: int,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    overwrite: bool,
    cleanup_failed: bool,
) -> None:
    configure_tqdm_lock(progress_lock)
    resolved_output_root = Path(output_root)
    resolved_cache_root = Path(cache_root)

    def action(
        downloaded: DownloadedEpisode,
    ) -> tuple[Any | None, TerminalEvent | None]:
        result = extract_episode(
            ExtractionJob(
                episode=downloaded.episode,
                mcap_path=Path(downloaded.cache_path),
                output_root=resolved_output_root,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                overwrite=overwrite,
            )
        )
        if result.status == "failed":
            if cleanup_failed:
                cleanup_artifacts(
                    cache_path=downloaded.cache_path,
                    episode_dir=result.episode_dir,
                    cache_root=resolved_cache_root,
                    output_root=resolved_output_root,
                )
            return None, TerminalEvent(
                episode_path=downloaded.episode.episode_path,
                status="failed",
                stage="extract",
                error=result.error,
                r2_key=downloaded.episode.r2_key,
            )
        return (
            ExtractedEpisode(
                episode=downloaded.episode,
                cache_path=downloaded.cache_path,
                download_action=downloaded.download_action,
                episode_dir=result.episode_dir,
                details=result.details or {},
            ),
            None,
        )

    run_thread_stage(
        input_queue=input_queue,
        output_queue=output_queue,
        terminal_queue=terminal_queue,
        max_workers=workers,
        position=1,
        description="Extracting",
        action=action,
    )


def rollback_uploaded(folder: Any, uploaded_uuids: list[str]) -> str | None:
    if not uploaded_uuids:
        return None
    try:
        folder.delete_storage_items([UUID(value) for value in uploaded_uuids])
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def upload_extracted_episode(
    *,
    extracted: ExtractedEpisode,
    folder: Any,
    known_titles: dict[str, str],
    known_lock: Any,
    output_root: Path,
    cache_root: Path,
    video_folder_hash: str,
    cleanup_failed: bool,
) -> tuple[UploadedEpisode | None, TerminalEvent | None]:
    specs = {
        str(spec["role"]): spec
        for spec in output_specs(extracted.episode, output_root)
        if spec["data_type"] == "video"
    }
    uploaded_uuids: list[str] = []
    camera_uuid_map: dict[str, str] = {}
    try:
        for camera in CAMERA_ORDER:
            spec = specs[camera]
            title = str(spec["title"])
            with known_lock:
                existing_uuid = known_titles.get(title)
            if existing_uuid:
                camera_uuid_map[camera] = existing_uuid
                continue
            path = output_root / str(spec["relative_path"])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Extracted video is missing: {path}")
            probe = extracted.details.get("video_probes", {}).get(camera, {})
            frame_count = extracted.details.get("frame_counts", {}).get(camera)
            metadata = enrich_video_metadata(
                dict(spec.get("client_metadata") or {}),
                probe,
                int(frame_count) if frame_count is not None else None,
                video_folder_hash,
            )
            item_uuid = str(folder.upload_video(path, title, metadata))
            uploaded_uuids.append(item_uuid)
            camera_uuid_map[camera] = item_uuid
            with known_lock:
                known_titles[title] = item_uuid
        if set(camera_uuid_map) != set(CAMERA_ORDER):
            raise RuntimeError("Upload did not produce all three camera UUIDs.")
    except Exception as exc:  # noqa: BLE001
        rollback_error = rollback_uploaded(folder, uploaded_uuids)
        if cleanup_failed:
            cleanup_artifacts(
                cache_path=extracted.cache_path,
                episode_dir=extracted.episode_dir,
                cache_root=cache_root,
                output_root=output_root,
            )
        error = f"{type(exc).__name__}: {exc}"
        if rollback_error:
            error += f"; rollback failed: {rollback_error}"
        return None, TerminalEvent(
            episode_path=extracted.episode.episode_path,
            status="failed",
            stage="upload",
            error=error,
            r2_key=extracted.episode.r2_key,
            camera_uuid_map=camera_uuid_map,
        )
    return (
        UploadedEpisode(
            episode=extracted.episode,
            cache_path=extracted.cache_path,
            download_action=extracted.download_action,
            episode_dir=extracted.episode_dir,
            camera_uuid_map=camera_uuid_map,
            uploaded_uuids=uploaded_uuids,
        ),
        None,
    )


def upload_stage(
    input_queue: Any,
    output_queue: Any,
    terminal_queue: Any,
    progress_lock: Any,
    *,
    ssh_key_file: str,
    encord_domain: str,
    video_folder_hash: str,
    output_root: str,
    cache_root: str,
    existing_title_to_uuid: dict[str, str],
    workers: int,
    cleanup_failed: bool,
) -> None:
    configure_tqdm_lock(progress_lock)
    client = create_encord_client(Path(ssh_key_file), encord_domain)
    folder = client.get_storage_folder(video_folder_hash)
    known_titles = dict(existing_title_to_uuid)
    known_lock = Lock()
    resolved_output_root = Path(output_root)
    resolved_cache_root = Path(cache_root)

    def action(extracted: ExtractedEpisode) -> tuple[Any | None, TerminalEvent | None]:
        return upload_extracted_episode(
            extracted=extracted,
            folder=folder,
            known_titles=known_titles,
            known_lock=known_lock,
            output_root=resolved_output_root,
            cache_root=resolved_cache_root,
            video_folder_hash=video_folder_hash,
            cleanup_failed=cleanup_failed,
        )

    run_thread_stage(
        input_queue=input_queue,
        output_queue=output_queue,
        terminal_queue=terminal_queue,
        max_workers=workers,
        position=2,
        description="Uploading",
        action=action,
    )


def build_video_group(
    uploaded: UploadedEpisode,
    *,
    video_folder_hash: str,
) -> Any:
    from encord.orm.group_layout import DataUnitTile, LayoutGrid
    from encord.orm.storage import DataGroupCustom

    camera_map = uploaded.camera_uuid_map
    layout_contents = {
        f"camera_{camera}": UUID(camera_map[camera]) for camera in CAMERA_ORDER
    }
    right_side = LayoutGrid(
        direction="column",
        split_percentage=50,
        first=DataUnitTile(key="camera_cam_left_wrist"),
        second=DataUnitTile(key="camera_cam_right_wrist"),
    )
    metadata = dict(uploaded.episode.group_client_metadata)
    metadata.update(
        {
            "probe": INGESTION_PROBE,
            "episode_path": uploaded.episode.episode_path,
            "source_folder_id": video_folder_hash,
            "source_r2_uri": (
                f"r2://{uploaded.episode.r2_bucket}/{uploaded.episode.r2_key}"
            ),
            "video_uuids": [camera_map[camera] for camera in CAMERA_ORDER],
            "camera_uuid_map": {camera: camera_map[camera] for camera in CAMERA_ORDER},
            "json_uuids": [],
        }
    )
    return DataGroupCustom(
        name=uploaded.episode.episode_path.rstrip("/"),
        layout_contents=layout_contents,
        layout=LayoutGrid(
            direction="row",
            split_percentage=50,
            first=DataUnitTile(key="camera_cam_high"),
            second=right_side,
        ),
        client_metadata=metadata,
    )


def group_stage(
    input_queue: Any,
    terminal_queue: Any,
    progress_lock: Any,
    *,
    ssh_key_file: str,
    encord_domain: str,
    group_folder_hash: str,
    video_folder_hash: str,
    cache_root: str,
    output_root: str,
    existing_groups: dict[str, str],
    batch_size: int,
    episode_limit: int,
    initial_success_count: int,
    cleanup_after_group: bool,
    cleanup_failed: bool,
) -> None:
    configure_tqdm_lock(progress_lock)
    client = create_encord_client(Path(ssh_key_file), encord_domain)
    folder = client.get_storage_folder(group_folder_hash)
    known_groups = dict(existing_groups)
    resolved_cache_root = Path(cache_root)
    resolved_output_root = Path(output_root)
    pending: list[UploadedEpisode] = []
    input_closed = False
    success_count = initial_success_count

    def cleanup(uploaded: UploadedEpisode) -> None:
        cleanup_artifacts(
            cache_path=uploaded.cache_path,
            episode_dir=uploaded.episode_dir,
            cache_root=resolved_cache_root,
            output_root=resolved_output_root,
        )

    def emit_grouped(
        uploaded: UploadedEpisode,
        group_uuid: str,
        action: str,
    ) -> None:
        nonlocal success_count
        known_groups[uploaded.episode.episode_path] = group_uuid
        terminal_queue.put(
            TerminalEvent(
                episode_path=uploaded.episode.episode_path,
                status="grouped",
                stage="group",
                group_uuid=group_uuid,
                camera_uuid_map=uploaded.camera_uuid_map,
                r2_key=uploaded.episode.r2_key,
                details={"action": action},
            )
        )
        success_count += 1
        if cleanup_after_group:
            cleanup(uploaded)

    def flush(batch: list[UploadedEpisode]) -> None:
        to_create: list[UploadedEpisode] = []
        for uploaded in batch:
            existing_uuid = known_groups.get(uploaded.episode.episode_path)
            if existing_uuid:
                emit_grouped(uploaded, existing_uuid, "existing")
                continue
            to_create.append(uploaded)
        if not to_create:
            return

        last_error: Exception | None = None
        for attempt in range(1, 5):
            create_inputs = [
                build_video_group(
                    uploaded,
                    video_folder_hash=video_folder_hash,
                )
                for uploaded in to_create
            ]
            try:
                created = list(folder.create_data_groups(create_inputs))
                if len(created) != len(to_create):
                    raise RuntimeError(
                        f"Created {len(created)} groups from {len(to_create)} inputs."
                    )
                for uploaded, group_uuid in zip(to_create, created, strict=True):
                    emit_grouped(uploaded, str(group_uuid), "created")
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                live = inventory_group_folder(folder)
                remaining = []
                for uploaded in to_create:
                    ingestion_id = str(
                        uploaded.episode.group_client_metadata.get(
                            "r2_expansion_ingestion_id"
                        )
                        or ""
                    )
                    group_uuid = live.group_uuids_by_ingestion_episode.get(
                        (ingestion_id, uploaded.episode.episode_path)
                    )
                    if group_uuid:
                        emit_grouped(uploaded, group_uuid, "reconciled")
                    else:
                        remaining.append(uploaded)
                to_create = remaining
                if not to_create:
                    return
                if attempt < 4:
                    time.sleep(2**attempt)
        if last_error is not None:
            for uploaded in to_create:
                if cleanup_failed:
                    cleanup(uploaded)
                terminal_queue.put(
                    TerminalEvent(
                        episode_path=uploaded.episode.episode_path,
                        status="failed",
                        stage="group",
                        error=f"{type(last_error).__name__}: {last_error}",
                        camera_uuid_map=uploaded.camera_uuid_map,
                        r2_key=uploaded.episode.r2_key,
                    )
                )
            return

    with tqdm(
        total=episode_limit,
        initial=initial_success_count,
        desc="Grouping",
        unit="episode",
        position=3,
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress:
        while not input_closed:
            try:
                item = input_queue.get(timeout=0.5)
            except queue.Empty:
                item = ...
            if item is None:
                input_closed = True
            elif item is not ...:
                pending.append(item)
            if pending and (len(pending) >= batch_size or input_closed or item is ...):
                before = success_count
                flush(pending)
                pending = []
                progress.update(success_count - before)
                progress.set_postfix({"successful": success_count})


def load_state(
    path: Path,
    *,
    config: dict[str, Any],
    create: bool,
) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text())
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise typer.BadParameter(f"Unsupported expansion state schema: {path}")
        if state.get("kind") != STATE_KIND:
            raise typer.BadParameter(f"State belongs to another command: {path}")
        for key, expected in config.items():
            if state.get("config", {}).get(key) != expected:
                raise typer.BadParameter(
                    f"State configuration mismatch for {key}: "
                    f"{state.get('config', {}).get(key)!r} != {expected!r}. "
                    "Use a different --state-json for a new run."
                )
        return state
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "kind": STATE_KIND,
        "ingestion_id": str(uuid4()),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "planned",
        "config": config,
        "episodes": {},
    }
    if create:
        save_state(path, state)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json_atomic(path, state)


def grouped_state_entries(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        episode_path: value
        for episode_path, value in state.get("episodes", {}).items()
        if value.get("status") == "grouped"
    }


def terminal_state_entry(event: TerminalEvent) -> dict[str, Any]:
    return {
        "status": event.status,
        "stage": event.stage,
        "error": event.error,
        "group_uuid": event.group_uuid,
        "camera_uuid_map": event.camera_uuid_map,
        "r2_key": event.r2_key,
        "details": event.details,
        "updated_at": now_iso(),
    }


def candidate_size_summary(episodes: list[RecoveryEpisode]) -> dict[str, Any]:
    sizes = sorted(episode.r2_size for episode in episodes)
    if not sizes:
        return {"count": 0, "total_bytes": 0}

    def percentile(fraction: float) -> int:
        return sizes[min(math.floor((len(sizes) - 1) * fraction), len(sizes) - 1)]

    return {
        "count": len(sizes),
        "total_bytes": sum(sizes),
        "min_bytes": sizes[0],
        "p50_bytes": percentile(0.5),
        "p90_bytes": percentile(0.9),
        "max_bytes": sizes[-1],
    }


def build_r2_episodes(
    *,
    objects: dict[str, int],
    bucket: str,
    prefix: str,
    ingestion_id: str,
) -> tuple[list[RecoveryEpisode], int, int]:
    episodes_by_path: dict[str, RecoveryEpisode] = {}
    invalid = 0
    duplicates = 0
    for key, size in objects.items():
        episode = recovery_episode_from_r2(
            bucket=bucket,
            prefix=prefix,
            key=key,
            size=size,
            ingestion_id=ingestion_id,
        )
        if episode is None:
            invalid += 1
        else:
            if episode.episode_path in episodes_by_path:
                duplicates += 1
                continue
            episodes_by_path[episode.episode_path] = episode
    return list(episodes_by_path.values()), invalid, duplicates


def pending_group_jobs(
    *,
    episodes_by_path: dict[str, RecoveryEpisode],
    video_inventory: VideoInventory,
    group_inventory: GroupInventory,
    ingestion_id: str,
) -> list[UploadedEpisode]:
    pending = []
    for episode_path in sorted(video_inventory.complete_episodes):
        if ingestion_id not in video_inventory.ingestion_ids_by_episode.get(
            episode_path, set()
        ):
            continue
        if (ingestion_id, episode_path) in (
            group_inventory.group_uuids_by_ingestion_episode
        ):
            continue
        episode = episodes_by_path.get(episode_path)
        if episode is None:
            continue
        pending.append(
            UploadedEpisode(
                episode=episode,
                cache_path="",
                download_action="existing",
                episode_dir="",
                camera_uuid_map=video_inventory.camera_uuids_by_episode[episode_path],
                uploaded_uuids=[],
            )
        )
    return pending


def reconcile_state_with_live_items(
    *,
    state: dict[str, Any],
    video_inventory: VideoInventory,
    group_inventory: GroupInventory,
) -> bool:
    changed = False
    ingestion_id = str(state["ingestion_id"])
    episodes = state.setdefault("episodes", {})
    live_grouped_paths = {
        episode_path: group_uuid
        for (candidate_ingestion, episode_path), group_uuid in (
            group_inventory.group_uuids_by_ingestion_episode.items()
        )
        if candidate_ingestion == ingestion_id
        and episode_path in video_inventory.complete_episodes
    }
    for episode_path, group_uuid in live_grouped_paths.items():
        current = episodes.get(episode_path, {})
        if (
            current.get("status") != "grouped"
            or current.get("group_uuid") != group_uuid
        ):
            episodes[episode_path] = {
                "status": "grouped",
                "stage": "group",
                "error": None,
                "group_uuid": group_uuid,
                "camera_uuid_map": video_inventory.camera_uuids_by_episode[
                    episode_path
                ],
                "r2_key": current.get("r2_key"),
                "details": {"action": "adopted_live"},
                "updated_at": now_iso(),
            }
            changed = True
    for episode_path, entry in list(episodes.items()):
        if entry.get("status") != "grouped":
            continue
        if episode_path not in live_grouped_paths:
            episodes[episode_path] = {
                **entry,
                "status": "stale",
                "stage": "reconcile",
                "error": "Previously grouped episode is incomplete or missing live.",
                "updated_at": now_iso(),
            }
            changed = True
    return changed


def process_kwargs(
    target: Callable[..., None],
    kwargs: dict[str, Any],
    context: Any,
    name: str,
) -> Any:
    return context.Process(target=target, kwargs=kwargs, name=name)


def run_pipeline(
    *,
    candidates: list[RecoveryEpisode],
    pending_groups: list[UploadedEpisode],
    state: dict[str, Any],
    state_path: Path,
    report_path: Path,
    episode_limit: int,
    queue_depth: int,
    stage_processes: list[Any],
    download_queue: Any,
    group_queue: Any,
    terminal_queue: Any,
) -> tuple[int, bool]:
    grouped_entries = grouped_state_entries(state)
    successful = len(grouped_entries)
    active = 0
    candidate_cursor = 0
    pending_cursor = 0
    scheduled: set[str] = set(grouped_entries)
    failed_before = {
        episode_path
        for episode_path, value in state.get("episodes", {}).items()
        if value.get("status") == "failed"
    }

    for process in stage_processes:
        process.start()

    def schedule_one() -> bool:
        nonlocal active, candidate_cursor, pending_cursor
        while pending_cursor < len(pending_groups):
            uploaded = pending_groups[pending_cursor]
            pending_cursor += 1
            episode_path = uploaded.episode.episode_path
            if episode_path in scheduled:
                continue
            group_queue.put(uploaded)
            scheduled.add(episode_path)
            active += 1
            return True
        while candidate_cursor < len(candidates):
            episode = candidates[candidate_cursor]
            candidate_cursor += 1
            if (
                episode.episode_path in scheduled
                or episode.episode_path in failed_before
            ):
                continue
            download_queue.put(episode)
            scheduled.add(episode.episode_path)
            active += 1
            return True
        return False

    try:
        while should_schedule(successful, active, episode_limit, queue_depth):
            if not schedule_one():
                break

        while active:
            try:
                event: TerminalEvent = terminal_queue.get(timeout=5)
            except queue.Empty:
                failed_processes = [
                    process
                    for process in stage_processes
                    if process.exitcode not in (None, 0)
                ]
                if failed_processes:
                    details = ", ".join(
                        f"{process.name}={process.exitcode}"
                        for process in failed_processes
                    )
                    raise RuntimeError(f"Pipeline stage exited unexpectedly: {details}")
                continue

            active -= 1
            state["episodes"][event.episode_path] = terminal_state_entry(event)
            if event.status == "grouped":
                successful += 1
            save_state(state_path, state)

            while should_schedule(successful, active, episode_limit, queue_depth):
                if not schedule_one():
                    break

        exhausted = successful < episode_limit
        state["status"] = "incomplete" if exhausted else "complete"
        state["successful_group_count"] = successful
        state["completed_at"] = now_iso() if not exhausted else None
        save_state(state_path, state)
    except BaseException:
        state["status"] = "interrupted"
        save_state(state_path, state)
        for process in stage_processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        download_queue.put(None)
        for process in stage_processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()

    partial_report = {
        "kind": STATE_KIND,
        "generated_at": now_iso(),
        "status": state["status"],
        "successful_group_count": successful,
        "episode_limit": episode_limit,
        "candidate_cursor": candidate_cursor,
        "pending_group_cursor": pending_cursor,
        "state_json": str(state_path),
    }
    write_json_atomic(report_path, partial_report)
    return successful, not exhausted


def verify_completed_run(
    *,
    client: Any,
    video_folder_hash: str,
    group_folder_hash: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    video_inventory = inventory_video_folder(
        client.get_storage_folder(video_folder_hash)
    )
    group_inventory = inventory_group_folder(
        client.get_storage_folder(group_folder_hash)
    )
    ingestion_id = str(state["ingestion_id"])
    grouped_entries = grouped_state_entries(state)
    failures = []
    video_uuid_seen: set[str] = set()
    for episode_path, entry in grouped_entries.items():
        camera_map = video_inventory.camera_uuids_by_episode.get(episode_path, {})
        group_uuid = group_inventory.group_uuids_by_ingestion_episode.get(
            (ingestion_id, episode_path)
        )
        group_count = group_inventory.group_counts_by_ingestion_episode.get(
            (ingestion_id, episode_path),
            0,
        )
        reasons = []
        if set(camera_map) != set(CAMERA_ORDER):
            reasons.append("missing camera videos")
        if len(set(camera_map.values())) != len(CAMERA_ORDER):
            reasons.append("camera UUIDs are not distinct")
        if group_uuid is None:
            reasons.append("missing expansion group")
        elif group_count != 1:
            reasons.append(f"expected one expansion group, found {group_count}")
        if group_uuid in group_inventory.validation_failures_by_group_uuid:
            reasons.extend(
                group_inventory.validation_failures_by_group_uuid[group_uuid]
            )
        live_group_camera_map = group_inventory.camera_maps_by_group_uuid.get(
            group_uuid or "", {}
        )
        if group_uuid and live_group_camera_map != camera_map:
            reasons.append("group camera UUID map does not match live videos")
        for video_uuid in camera_map.values():
            if video_uuid in video_uuid_seen:
                reasons.append(f"video UUID reused across episodes: {video_uuid}")
            video_uuid_seen.add(video_uuid)
        if reasons:
            failures.append(
                {
                    "episode_path": episode_path,
                    "state_group_uuid": entry.get("group_uuid"),
                    "live_group_uuid": group_uuid,
                    "reasons": reasons,
                }
            )

    expected_count = int(state["config"]["episode_limit"])
    result = {
        "expected_episode_count": expected_count,
        "state_grouped_episode_count": len(grouped_entries),
        "verified_video_count": len(video_uuid_seen),
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    result["passed"] = (
        len(grouped_entries) == expected_count
        and len(video_uuid_seen) == expected_count * len(CAMERA_ORDER)
        and not failures
    )
    return result


def require_positive(name: str, value: int) -> None:
    if value < 1:
        raise typer.BadParameter(f"{name} must be at least 1.")


def main(
    video_folder_hash: Annotated[
        str,
        typer.Argument(help="Existing Encord folder that receives ungrouped videos."),
    ] = DEFAULT_VIDEO_FOLDER_HASH,
    data_group_folder_hash: Annotated[
        str,
        typer.Option(help="Existing Encord folder that receives video-only groups."),
    ] = DEFAULT_GROUP_FOLDER_HASH,
    ssh_key_file: Annotated[
        Path | None,
        typer.Option(
            "--ssh-key-file",
            "-k",
            envvar="ENCORD_SSH_KEY_FILE",
            help="Path to the Encord SSH private key.",
        ),
    ] = None,
    encord_domain: Annotated[
        str,
        typer.Option(help="Encord API domain."),
    ] = "https://api.encord.com",
    bucket: Annotated[
        str,
        typer.Option(envvar="R2_BUCKET", help="R2 bucket containing MCAPs."),
    ] = DEFAULT_BUCKET,
    r2_prefix: Annotated[
        str,
        typer.Option(help="R2 prefix containing canonical Trossen MCAPs."),
    ] = DEFAULT_R2_PREFIX,
    account_id: Annotated[
        str | None,
        typer.Option(envvar="CLOUDFLARE_ACCOUNT_ID", help="Cloudflare account ID."),
    ] = None,
    endpoint_url: Annotated[
        str | None,
        typer.Option(envvar="R2_ENDPOINT_URL", help="Full R2 S3 endpoint URL."),
    ] = None,
    access_key_id: Annotated[
        str | None,
        typer.Option(envvar="R2_ACCESS_KEY_ID", help="R2 access key ID."),
    ] = None,
    secret_access_key: Annotated[
        str | None,
        typer.Option(envvar="R2_SECRET_ACCESS_KEY", help="R2 secret access key."),
    ] = None,
    episode_limit: Annotated[
        int,
        typer.Option(help="New complete episodes/groups to add in this run."),
    ] = DEFAULT_EPISODE_LIMIT,
    selection: Annotated[
        SelectionMode,
        typer.Option(help="Candidate ordering strategy."),
    ] = SelectionMode.BALANCED,
    download_workers: Annotated[
        int,
        typer.Option(help="Concurrent R2 downloads inside the download stage."),
    ] = DEFAULT_DOWNLOAD_WORKERS,
    extract_workers: Annotated[
        int,
        typer.Option(help="Concurrent MCAP extractions inside the extraction stage."),
    ] = DEFAULT_EXTRACT_WORKERS,
    upload_workers: Annotated[
        int,
        typer.Option(help="Concurrent episode uploads inside the upload stage."),
    ] = DEFAULT_UPLOAD_WORKERS,
    group_batch_size: Annotated[
        int,
        typer.Option(help="Video-only data groups created per Encord API batch."),
    ] = DEFAULT_GROUP_BATCH_SIZE,
    queue_depth: Annotated[
        int,
        typer.Option(help="Maximum active episodes across the pipeline."),
    ] = DEFAULT_QUEUE_DEPTH,
    multipart_concurrency: Annotated[
        int,
        typer.Option(help="Multipart threads per active R2 download."),
    ] = DEFAULT_MULTIPART_CONCURRENCY,
    multipart_threshold_mb: Annotated[
        int,
        typer.Option(help="Multipart threshold in MiB."),
    ] = DEFAULT_MULTIPART_THRESHOLD_MB,
    multipart_chunksize_mb: Annotated[
        int,
        typer.Option(help="Multipart chunk size in MiB."),
    ] = DEFAULT_MULTIPART_CHUNKSIZE_MB,
    cache_root: Annotated[
        Path,
        typer.Option(help="Local R2 object cache root."),
    ] = R2_CACHE_ROOT,
    output_root: Annotated[
        Path | None,
        typer.Option(help="Extracted episode root."),
    ] = None,
    state_json: Annotated[
        Path,
        typer.Option(help="Atomic resume state for this expansion run."),
    ] = DEFAULT_STATE_JSON,
    report_json: Annotated[
        Path,
        typer.Option(help="Audit/final report JSON."),
    ] = DEFAULT_REPORT_JSON,
    ffmpeg_bin: Annotated[
        str,
        typer.Option(help="ffmpeg executable name or path."),
    ] = "ffmpeg",
    ffprobe_bin: Annotated[
        str,
        typer.Option(help="ffprobe executable name or path."),
    ] = "ffprobe",
    overwrite_downloads: Annotated[
        bool,
        typer.Option("--overwrite-downloads/--reuse-downloads"),
    ] = False,
    overwrite_extracted: Annotated[
        bool,
        typer.Option("--overwrite-extracted/--reuse-extracted"),
    ] = False,
    cleanup_after_group: Annotated[
        bool,
        typer.Option(
            "--cleanup-after-group/--keep-after-group",
            help="Delete local MCAP and extracted output after its group is confirmed.",
        ),
    ] = True,
    cleanup_failed: Annotated[
        bool,
        typer.Option(
            "--cleanup-failed/--keep-failed",
            help="Remove failed candidate artifacts after recording the error.",
        ),
    ] = True,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Run the four-stage mutation pipeline."),
    ] = False,
) -> None:
    for name, value in (
        ("--episode-limit", episode_limit),
        ("--download-workers", download_workers),
        ("--extract-workers", extract_workers),
        ("--upload-workers", upload_workers),
        ("--group-batch-size", group_batch_size),
        ("--queue-depth", queue_depth),
        ("--multipart-concurrency", multipart_concurrency),
        ("--multipart-threshold-mb", multipart_threshold_mb),
        ("--multipart-chunksize-mb", multipart_chunksize_mb),
    ):
        require_positive(name, value)
    if ssh_key_file is None:
        raise typer.BadParameter("Pass --ssh-key-file or set ENCORD_SSH_KEY_FILE.")
    if access_key_id is None:
        raise typer.BadParameter("Pass --access-key-id or set R2_ACCESS_KEY_ID.")
    if secret_access_key is None:
        raise typer.BadParameter(
            "Pass --secret-access-key or set R2_SECRET_ACCESS_KEY."
        )
    if shutil.which(ffmpeg_bin) is None:
        raise typer.BadParameter(f"ffmpeg executable was not found: {ffmpeg_bin}")
    if shutil.which(ffprobe_bin) is None:
        raise typer.BadParameter(f"ffprobe executable was not found: {ffprobe_bin}")

    resolved_state = state_json.expanduser().resolve()
    resolved_report = report_json.expanduser().resolve()
    resolved_cache = cache_root.expanduser().resolve()
    resolved_output = (
        output_root.expanduser().resolve()
        if output_root is not None
        else (RECOVERY_ROOT / bucket / f"folder-{video_folder_hash}").resolve()
    )
    config = {
        "video_folder_hash": video_folder_hash,
        "group_folder_hash": data_group_folder_hash,
        "bucket": bucket,
        "r2_prefix": r2_prefix,
        "episode_limit": episode_limit,
        "selection": selection.value,
    }
    state = load_state(resolved_state, config=config, create=apply)
    ingestion_id = str(state["ingestion_id"])

    typer.echo(f"Auditing Encord video folder {video_folder_hash}...")
    client = create_encord_client(ssh_key_file, encord_domain)
    video_folder = client.get_storage_folder(video_folder_hash)
    group_folder = client.get_storage_folder(data_group_folder_hash)
    video_inventory = inventory_video_folder(video_folder)
    group_inventory = inventory_group_folder(group_folder)
    if (
        reconcile_state_with_live_items(
            state=state,
            video_inventory=video_inventory,
            group_inventory=group_inventory,
        )
        and apply
    ):
        save_state(resolved_state, state)

    endpoint = r2_endpoint_url(account_id, endpoint_url)
    client_r2 = r2_client(
        endpoint_url=endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        max_pool_connections=max(32, download_workers * multipart_concurrency + 8),
    )
    typer.echo(f"Listing r2://{bucket}/{r2_prefix.strip('/')}/...")
    objects = list_r2_objects(client_r2, bucket, r2_prefix)
    r2_episodes, invalid_r2_objects, duplicate_r2_episodes = build_r2_episodes(
        objects=objects,
        bucket=bucket,
        prefix=r2_prefix,
        ingestion_id=ingestion_id,
    )
    episodes_by_path = {episode.episode_path: episode for episode in r2_episodes}
    failed_state_paths = {
        episode_path
        for episode_path, value in state.get("episodes", {}).items()
        if value.get("status") == "failed"
    }
    candidates = order_candidates(
        [
            episode
            for episode in r2_episodes
            if episode.episode_path not in video_inventory.complete_episodes
            and episode.episode_path not in failed_state_paths
        ],
        selection,
    )
    pending_groups = pending_group_jobs(
        episodes_by_path=episodes_by_path,
        video_inventory=video_inventory,
        group_inventory=group_inventory,
        ingestion_id=ingestion_id,
    )
    existing_success = len(grouped_state_entries(state))
    needed = max(episode_limit - existing_success, 0)
    projected = candidates[: max(needed - len(pending_groups), 0)]
    audit_report = {
        "kind": STATE_KIND,
        "generated_at": now_iso(),
        "apply": apply,
        "ingestion_id": ingestion_id,
        "config": config,
        "video_folder": {
            "video_count": len(video_inventory.title_to_uuid),
            "episode_count": len(video_inventory.camera_uuids_by_episode),
            "complete_episode_count": len(video_inventory.complete_episodes),
            "duplicate_slot_count": len(video_inventory.duplicate_slots),
            "unclassified_video_count": video_inventory.unclassified_video_count,
        },
        "group_folder": {
            "episode_count": len(group_inventory.episode_counts),
            "group_count": sum(group_inventory.episode_counts.values()),
        },
        "r2": {
            "object_count": len(objects),
            "valid_episode_count": len(r2_episodes),
            "invalid_object_count": invalid_r2_objects,
            "duplicate_episode_count": duplicate_r2_episodes,
            "new_candidate_count": len(candidates),
        },
        "resume": {
            "existing_success_count": existing_success,
            "pending_group_count": len(pending_groups),
            "remaining_success_count": needed,
        },
        "projected_downloads": candidate_size_summary(projected),
        "projected_task_counts": dict(
            Counter(episode.task_name for episode in projected)
        ),
        "projected_environment_counts": dict(
            Counter(selection_bucket(episode)[1] for episode in projected)
        ),
        "candidate_samples": [
            {
                "episode_path": episode.episode_path,
                "r2_key": episode.r2_key,
                "size": episode.r2_size,
            }
            for episode in projected[:100]
        ],
        "sufficient_candidates": (
            existing_success + len(pending_groups) + len(candidates) >= episode_limit
        ),
    }
    write_json_atomic(resolved_report, audit_report)
    typer.echo(
        f"Existing complete episodes: {len(video_inventory.complete_episodes):,}"
    )
    typer.echo(
        f"New R2 candidates: {len(candidates):,}; "
        f"pending groups from a prior partial run: {len(pending_groups):,}"
    )
    typer.echo(
        f"Target for this resume state: {episode_limit:,}; "
        f"already grouped: {existing_success:,}"
    )
    typer.echo(f"Audit report: {resolved_report}")
    if not audit_report["sufficient_candidates"]:
        raise typer.BadParameter(
            f"Only {existing_success + len(pending_groups) + len(candidates):,} "
            f"recoverable episodes are available for a target of {episode_limit:,}."
        )
    if not apply:
        typer.echo("Audit complete; no Encord or cache mutations were made.")
        return
    if state.get("status") == "complete" and existing_success >= episode_limit:
        verification = verify_completed_run(
            client=client,
            video_folder_hash=video_folder_hash,
            group_folder_hash=data_group_folder_hash,
            state=state,
        )
        write_json_atomic(
            resolved_report,
            {
                **audit_report,
                "apply": True,
                "status": (
                    "complete" if verification["passed"] else "verification_failed"
                ),
                "verification": verification,
            },
        )
        if not verification["passed"]:
            typer.echo(
                "Completed state failed live verification; no new items were scheduled.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo("This expansion state is already complete and verified.")
        return

    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    state["status"] = "running"
    save_state(resolved_state, state)

    context = multiprocessing.get_context("spawn")
    progress_lock = context.RLock()
    download_queue = context.Queue(maxsize=queue_depth)
    extract_queue = context.Queue(maxsize=queue_depth)
    upload_queue = context.Queue(maxsize=queue_depth)
    group_queue = context.Queue(maxsize=queue_depth)
    terminal_queue = context.Queue()

    existing_groups_for_ingestion = {
        episode_path: group_uuid
        for (candidate_ingestion, episode_path), group_uuid in (
            group_inventory.group_uuids_by_ingestion_episode.items()
        )
        if candidate_ingestion == ingestion_id
    }
    processes = [
        process_kwargs(
            download_stage,
            {
                "input_queue": download_queue,
                "output_queue": extract_queue,
                "terminal_queue": terminal_queue,
                "progress_lock": progress_lock,
                "endpoint_url": endpoint,
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
                "cache_root": str(resolved_cache),
                "workers": download_workers,
                "multipart_concurrency": multipart_concurrency,
                "multipart_threshold_mb": multipart_threshold_mb,
                "multipart_chunksize_mb": multipart_chunksize_mb,
                "overwrite": overwrite_downloads,
            },
            context,
            "r2-download-stage",
        ),
        process_kwargs(
            extract_stage,
            {
                "input_queue": extract_queue,
                "output_queue": upload_queue,
                "terminal_queue": terminal_queue,
                "progress_lock": progress_lock,
                "output_root": str(resolved_output),
                "cache_root": str(resolved_cache),
                "workers": extract_workers,
                "ffmpeg_bin": shutil.which(ffmpeg_bin) or ffmpeg_bin,
                "ffprobe_bin": shutil.which(ffprobe_bin) or ffprobe_bin,
                "overwrite": overwrite_extracted,
                "cleanup_failed": cleanup_failed,
            },
            context,
            "mcap-extract-stage",
        ),
        process_kwargs(
            upload_stage,
            {
                "input_queue": upload_queue,
                "output_queue": group_queue,
                "terminal_queue": terminal_queue,
                "progress_lock": progress_lock,
                "ssh_key_file": str(ssh_key_file.expanduser().resolve()),
                "encord_domain": encord_domain,
                "video_folder_hash": video_folder_hash,
                "output_root": str(resolved_output),
                "cache_root": str(resolved_cache),
                "existing_title_to_uuid": video_inventory.title_to_uuid,
                "workers": upload_workers,
                "cleanup_failed": cleanup_failed,
            },
            context,
            "encord-upload-stage",
        ),
        process_kwargs(
            group_stage,
            {
                "input_queue": group_queue,
                "terminal_queue": terminal_queue,
                "progress_lock": progress_lock,
                "ssh_key_file": str(ssh_key_file.expanduser().resolve()),
                "encord_domain": encord_domain,
                "group_folder_hash": data_group_folder_hash,
                "video_folder_hash": video_folder_hash,
                "cache_root": str(resolved_cache),
                "output_root": str(resolved_output),
                "existing_groups": existing_groups_for_ingestion,
                "batch_size": group_batch_size,
                "episode_limit": episode_limit,
                "initial_success_count": existing_success,
                "cleanup_after_group": cleanup_after_group,
                "cleanup_failed": cleanup_failed,
            },
            context,
            "encord-group-stage",
        ),
    ]
    successful, complete = run_pipeline(
        candidates=candidates,
        pending_groups=pending_groups,
        state=state,
        state_path=resolved_state,
        report_path=resolved_report,
        episode_limit=episode_limit,
        queue_depth=queue_depth,
        stage_processes=processes,
        download_queue=download_queue,
        group_queue=group_queue,
        terminal_queue=terminal_queue,
    )
    if not complete:
        typer.echo(
            f"Pipeline exhausted candidates at {successful:,}/{episode_limit:,} groups.",
            err=True,
        )
        raise typer.Exit(code=1)

    verification = verify_completed_run(
        client=create_encord_client(ssh_key_file, encord_domain),
        video_folder_hash=video_folder_hash,
        group_folder_hash=data_group_folder_hash,
        state=state,
    )
    final_report = {
        **audit_report,
        "generated_at": now_iso(),
        "apply": True,
        "status": "complete" if verification["passed"] else "verification_failed",
        "state_summary": {
            "successful_group_count": successful,
            "failed_candidate_count": sum(
                value.get("status") == "failed"
                for value in state.get("episodes", {}).values()
            ),
            "state_json": str(resolved_state),
        },
        "verification": verification,
    }
    write_json_atomic(resolved_report, final_report)
    typer.echo(
        f"Completed {successful:,} new episodes / "
        f"{successful * len(CAMERA_ORDER):,} videos."
    )
    typer.echo(f"Final report: {resolved_report}")
    if not verification["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
