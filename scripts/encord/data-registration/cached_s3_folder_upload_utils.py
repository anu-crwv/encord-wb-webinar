# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "pydicom",
#     "tqdm",
#     "typer",
# ]
# ///
"""Helpers for uploading local S3-cache files into an Encord storage folder."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any
from uuid import UUID

import typer
from encord import EncordUserClient
from encord.exceptions import AuthenticationError
from encord.orm.storage import (
    DataGroupGrid,
    DataUploadImageGroupFromItems,
    DataUploadItems,
)
from encord.storage import StorageFolder
from encord.storage import StorageItemType as DataType
from tqdm import tqdm

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_S3_CACHE_ROOT = REPO_ROOT / "exports" / "encord-dataset-export" / "_cache" / "s3"
FILE_MAP_FILENAME = "file_map.json"
PROJECT_RECOVERY_MANIFEST_FILENAME = "project_recovery_manifest.json"
MAX_WORKERS_DEFAULT = 10
SYSTEM_FILENAMES = {
    ".DS_Store",
    FILE_MAP_FILENAME,
    PROJECT_RECOVERY_MANIFEST_FILENAME,
}
EPISODE_NAME_RE = re.compile(r"^episode_(\d+)(?:_[0-9A-Za-z][0-9A-Za-z._-]*)?$")
REQUIRED_EPISODE_METADATA_FILES = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)

SUPPORTED_FORMATS: dict[DataType, set[str]] = {
    DataType.AUDIO: {".aac", ".eac3", ".flac", ".m4a", ".mp3", ".mpeg", ".wav", ".x-wav"},
    DataType.DICOM_FILE: {".dcm"},
    DataType.IMAGE: {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"},
    DataType.PLAIN_TEXT: {".csv", ".html", ".json", ".jsonl", ".md", ".txt", ".xml", ".yaml", ".yml"},
    DataType.PDF: {".pdf"},
    DataType.NIFTI: {".gz", ".nii", ".nii.gz"},
    DataType.VIDEO: {".3g2", ".3gp", ".avi", ".mkv", ".mj2", ".mov", ".mp4", ".webm"},
}
VIDEO_EXTENSIONS = SUPPORTED_FORMATS[DataType.VIDEO]
SKIP_EXTENSIONS = {".parquet"}
EXTENSION_TO_DATA_TYPE = {
    ext: data_type for data_type, extensions in SUPPORTED_FORMATS.items() for ext in extensions
}
ANNOTATION_NAME_HINTS = {
    "annotations",
    "annotation",
    "labels",
    "label",
    "instances",
    "coco",
    "yolo",
}


class EncordDomain(StrEnum):
    PROD = "prod"
    DEV = "dev"
    STAGING = "staging"
    PROD_US = "prod_us"

    def get_url(self) -> str:
        urls = {
            self.PROD: "https://api.encord.com",
            self.DEV: "https://dev.api.encord.com",
            self.STAGING: "https://staging.api.encord.com",
            self.PROD_US: "https://api.us.encord.com",
        }
        return urls[self]


class TitleMode(StrEnum):
    AUTO = "auto"
    SOURCE_KEY = "source-key"
    RELATIVE_PATH = "relative-path"
    FILENAME = "filename"


@dataclass(frozen=True)
class CachedSource:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass
class EpisodeContext:
    episode_path: str
    has_info_json: bool = False
    has_tasks_jsonl: bool = False
    has_episodes_jsonl: bool = False
    has_episodes_stats_jsonl: bool = False
    has_parquet: bool = False
    info: dict[str, Any] | None = None
    files: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeCompletenessSelection:
    roots: list[Path] | None
    available_count: int | None
    complete_count: int | None
    incomplete: list[dict[str, Any]]
    skipped_incomplete: list[dict[str, Any]]


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    data_type: DataType
    extension: str
    cached_source: CachedSource | None


@dataclass(frozen=True)
class FileInfo:
    title: str
    path: Path
    data_type: DataType
    client_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class UploadOutcome:
    status: str
    title: str
    path: str
    data_type: str
    storage_item_uuid: str | None = None
    error: str | None = None


def get_encord_client(ssh_key_file: Path, domain: EncordDomain | str) -> EncordUserClient:
    try:
        domain_url = domain.get_url() if isinstance(domain, EncordDomain) else str(domain).rstrip("/")
        if not domain_url.startswith(("http://", "https://")):
            raise typer.BadParameter(f"Custom domain must be a full URL, got: {domain_url}")
        return EncordUserClient.create_with_ssh_private_key(
            ssh_private_key_path=ssh_key_file.expanduser(),
            domain=domain_url,
        )
    except AuthenticationError:
        raise typer.BadParameter(f"Could not authenticate against Encord domain {domain!r}.") from None


def resolve_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise typer.BadParameter(f"{label} is not a directory: {resolved}")
    return resolved


def resolve_parent(path: Path) -> Path:
    return path.expanduser().resolve().parent


def normal_extension(path: Path) -> str:
    name = path.name.lower()
    for suffix in (".nii.gz", ".x-wav"):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def visible_file(path: Path) -> bool:
    return path.name not in SYSTEM_FILENAMES and not path.name.startswith(".")


def get_all_files_recursive(
    directory: Path,
    exclude_filenames: list[str] | None = None,
    roots: list[Path] | None = None,
) -> list[Path]:
    invalid_filenames = set(SYSTEM_FILENAMES)
    if exclude_filenames:
        invalid_filenames.update(exclude_filenames)
    search_roots = roots or [directory]
    return sorted(
        file
        for root in search_roots
        for file in root.rglob("*")
        if file.is_file() and file.name not in invalid_filenames and visible_file(file)
    )


def episode_dirs(data_dir: Path) -> list[Path]:
    found = sorted(path for path in data_dir.rglob("*") if path.is_dir() and EPISODE_NAME_RE.fullmatch(path.name))
    if EPISODE_NAME_RE.fullmatch(data_dir.name):
        found = [data_dir] + [path for path in found if path != data_dir]
    return found


def episode_incomplete_reasons(episode_dir: Path, required_video_count: int) -> dict[str, Any] | None:
    missing_metadata = [
        metadata_file for metadata_file in REQUIRED_EPISODE_METADATA_FILES if not (episode_dir / metadata_file).is_file()
    ]
    video_files = [
        path
        for path in episode_dir.rglob("*")
        if path.is_file() and visible_file(path) and normal_extension(path) in VIDEO_EXTENSIONS
    ]
    missing_video_count = max(required_video_count - len(video_files), 0)
    if not missing_metadata and missing_video_count == 0:
        return None
    return {
        "episode_dir": str(episode_dir),
        "missing_metadata": missing_metadata,
        "video_count": len(video_files),
        "required_video_count": required_video_count,
        "missing_video_count": missing_video_count,
    }


def select_complete_episode_roots(
    data_dir: Path,
    require_complete_episodes: bool,
    required_video_count: int,
) -> EpisodeCompletenessSelection:
    all_episode_dirs = episode_dirs(data_dir)
    if not all_episode_dirs:
        return EpisodeCompletenessSelection(
            roots=None,
            available_count=None,
            complete_count=None,
            incomplete=[],
            skipped_incomplete=[],
        )

    complete_dirs: list[Path] = []
    incomplete: list[dict[str, Any]] = []
    for episode_dir in all_episode_dirs:
        reason = episode_incomplete_reasons(episode_dir, required_video_count)
        if reason is None:
            complete_dirs.append(episode_dir)
        else:
            incomplete.append(reason)

    if require_complete_episodes:
        if not complete_dirs:
            raise typer.BadParameter(f"No complete episode_* directories found under {data_dir}.")
        roots: list[Path] | None = complete_dirs
        skipped_incomplete = incomplete
    else:
        roots = None
        skipped_incomplete = []

    return EpisodeCompletenessSelection(
        roots=roots,
        available_count=len(all_episode_dirs),
        complete_count=len(complete_dirs),
        incomplete=incomplete,
        skipped_incomplete=skipped_incomplete,
    )


def path_is_under_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            pass
    return False


def get_data_type(file: Path) -> DataType | None:
    extension = normal_extension(file)
    if extension in SKIP_EXTENSIONS:
        return None
    data_type = EXTENSION_TO_DATA_TYPE.get(extension)
    if data_type is None:
        logger.warning("Skipping %s - unsupported file type %r", file, extension or "[no extension]")
    return data_type


def cached_source_for_path(path: Path, cache_root: Path) -> CachedSource | None:
    resolved_root = cache_root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return None
    bucket = parts[0]
    key = "/".join(parts[1:])
    return CachedSource(bucket=bucket, key=key)


def parse_datetime_token(token: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    match = re.search(r"(20\d{12})", token)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def source_family_for_key(key: str) -> str | None:
    parts = PurePosixPath(key).parts
    if "raw-feed" in parts:
        idx = parts.index("raw-feed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[0] if parts else None


def parse_path_metadata(key: str) -> dict[str, Any]:
    parts = PurePosixPath(key).parts
    out: dict[str, Any] = {}

    family = source_family_for_key(key)
    if family:
        out["source_family"] = family

    if "raw-feed" in parts:
        idx = parts.index("raw-feed")
        family = parts[idx + 1] if idx + 1 < len(parts) else None
        if idx + 6 < len(parts) and family in {"trossen-data", "trossen-data-stationary"}:
            out["source_family"] = family
            out["task_name"] = parts[idx + 2]
            out["environment"] = parts[idx + 3]
            dt = parse_datetime_token(parts[idx + 5])
            if dt:
                out["collection_datetime"] = dt
        elif family == "egocentric" and idx + 3 < len(parts) and parts[idx + 2] == "Meta-POC":
            out["source_family"] = family
            out["environment"] = parts[idx + 2]
            out["task_name"] = parts[idx + 3]

    for i, part in enumerate(parts):
        episode_match = EPISODE_NAME_RE.fullmatch(part)
        if episode_match:
            out["episode_id"] = part
            out["episode_index"] = int(episode_match.group(1))
            out["episode_path"] = "/".join(parts[: i + 1]) + "/"
            break

    for part in parts:
        if part.startswith("observation.images."):
            out["sensor_key"] = part
            out["camera_name"] = part.removeprefix("observation.images.")
            break

    if "collection_datetime" not in out:
        for part in reversed(parts):
            dt = parse_datetime_token(part)
            if dt:
                out["collection_datetime"] = dt
                break

    return out


def episode_path_for_key(key: str) -> str:
    path_meta = parse_path_metadata(key)
    if path_meta.get("episode_path"):
        return str(path_meta["episode_path"])
    parent = str(PurePosixPath(key).parent)
    return parent + ("/" if parent and parent != "." else "")


def metadata_file_role(key: str) -> str:
    name = PurePosixPath(key).name.lower()
    if name == "info.json":
        return "info"
    if name == "tasks.jsonl":
        return "tasks"
    if name == "episodes.jsonl":
        return "episodes"
    if name == "episodes_stats.jsonl":
        return "episodes_stats"
    if name == "dataset_metadata.json":
        return "dataset_metadata"
    if name in {"metadata.json", "metadata.yaml", "metadata.yml"}:
        return "metadata"
    return "none"


def read_json_file(file: Path) -> Any:
    try:
        return json.loads(file.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON file %s", file)
        return None
    except OSError as exc:
        logger.warning("Could not read JSON file %s: %s", file, exc)
        return None


def build_episode_contexts(discovered: list[DiscoveredFile], cache_root: Path) -> dict[str, EpisodeContext]:
    return build_episode_contexts_from_paths([item.path for item in discovered], cache_root)


def build_episode_contexts_from_paths(files: list[Path], cache_root: Path) -> dict[str, EpisodeContext]:
    contexts: dict[str, EpisodeContext] = {}
    source_by_path: dict[Path, CachedSource] = {}
    for file in files:
        source = cached_source_for_path(file, cache_root)
        if source is None:
            continue
        source_by_path[file] = source
        key = source.key
        episode_path = episode_path_for_key(key)
        ctx = contexts.setdefault(episode_path, EpisodeContext(episode_path=episode_path))
        ctx.files.append(file)
        role = metadata_file_role(key)
        ctx.has_info_json = ctx.has_info_json or role == "info"
        ctx.has_tasks_jsonl = ctx.has_tasks_jsonl or role == "tasks"
        ctx.has_episodes_jsonl = ctx.has_episodes_jsonl or role == "episodes"
        ctx.has_episodes_stats_jsonl = ctx.has_episodes_stats_jsonl or role == "episodes_stats"
        ctx.has_parquet = ctx.has_parquet or normal_extension(file) == ".parquet"

    for ctx in contexts.values():
        info_file = next(
            (
                path
                for path in ctx.files
                if path.name == "info.json" or metadata_file_role(source_by_path[path].key) == "info"
            ),
            None,
        )
        if info_file is not None:
            info = read_json_file(info_file)
            if isinstance(info, dict):
                ctx.info = info
    return contexts


def add_if_present(metadata: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    metadata[key] = value


def cached_source_metadata(source: CachedSource, file: Path, ctx: EpisodeContext | None) -> dict[str, Any]:
    path_meta = parse_path_metadata(source.key)
    role = metadata_file_role(source.key)
    metadata: dict[str, Any] = {
        "source_key": source.key,
        "source_uri": source.uri,
        "file_ext": normal_extension(file),
        "metadata_file_role": role,
    }

    if ctx is not None:
        metadata.update(
            {
                "has_info_json": ctx.has_info_json,
                "has_tasks_jsonl": ctx.has_tasks_jsonl,
                "has_episodes_jsonl": ctx.has_episodes_jsonl,
                "has_episodes_stats_jsonl": ctx.has_episodes_stats_jsonl,
                "has_parquet": ctx.has_parquet,
            }
        )

    for field_name in (
        "source_family",
        "task_name",
        "environment",
        "collection_datetime",
        "episode_id",
        "episode_index",
        "episode_path",
        "camera_name",
        "sensor_key",
    ):
        add_if_present(metadata, field_name, path_meta.get(field_name))

    info = ctx.info if ctx and isinstance(ctx.info, dict) else {}
    add_if_present(metadata, "robot_type", info.get("robot_type"))
    add_if_present(metadata, "codebase_version", info.get("codebase_version"))
    add_if_present(metadata, "trossen_subversion", info.get("trossen_subversion"))
    add_if_present(metadata, "collection_fps", info.get("fps"))
    return metadata


def legacy_client_metadata(file: Path, data_type: DataType) -> dict[str, Any]:
    return {
        "Tag": "A",
        "Data Type": data_type.value,
        "Extension": file.suffix,
    }


def get_optional_client_metadata(
    file: Path,
    data_type: DataType,
    include: bool,
    include_legacy_fields: bool,
    contexts: dict[str, EpisodeContext],
    cache_root: Path,
) -> dict[str, Any] | None:
    if not include:
        return None

    metadata: dict[str, Any] = {}
    source = cached_source_for_path(file, cache_root)
    if source is not None:
        ctx = contexts.get(episode_path_for_key(source.key))
        metadata.update(cached_source_metadata(source, file, ctx))
    if include_legacy_fields:
        metadata.update(legacy_client_metadata(file, data_type))

    return metadata or None


def discover_files(
    data_dir: Path,
    cache_root: Path,
    roots: list[Path] | None = None,
) -> tuple[list[DiscoveredFile], Counter[str]]:
    discovered: list[DiscoveredFile] = []
    skipped: Counter[str] = Counter()
    for file in get_all_files_recursive(data_dir, roots=roots):
        extension = normal_extension(file)
        if extension in SKIP_EXTENSIONS:
            skipped[extension] += 1
            continue
        data_type = get_data_type(file)
        if data_type is None:
            skipped[extension or "[no extension]"] += 1
            continue
        discovered.append(
            DiscoveredFile(
                path=file,
                data_type=data_type,
                extension=extension,
                cached_source=cached_source_for_path(file, cache_root),
            )
        )
    return discovered, skipped


def title_for_file(
    file: Path,
    data_dir: Path,
    cache_root: Path,
    mode: TitleMode,
    mapped_title: str | None = None,
) -> str:
    if mapped_title:
        return mapped_title
    source = cached_source_for_path(file, cache_root)
    if mode in {TitleMode.AUTO, TitleMode.SOURCE_KEY} and source is not None:
        return source.key
    if mode == TitleMode.FILENAME:
        return file.name
    try:
        return file.relative_to(data_dir).as_posix()
    except ValueError:
        return file.name


def get_file_map(data_dir: Path) -> dict[str, Any] | None:
    file_map_file = data_dir / FILE_MAP_FILENAME
    if not file_map_file.is_file():
        return None
    file_map = read_json_file(file_map_file)
    if not isinstance(file_map, dict):
        raise typer.BadParameter(f"{FILE_MAP_FILENAME} must contain a JSON object.")
    return file_map


def recovery_file_entry(
    file_map: dict[str, Any] | None,
    data_dir: Path,
    file: Path,
) -> dict[str, Any] | None:
    if not file_map:
        return None
    files = file_map.get("files")
    if not isinstance(files, dict):
        return None
    try:
        relative_path = file.relative_to(data_dir).as_posix()
    except ValueError:
        return None
    entry = files.get(relative_path)
    return entry if isinstance(entry, dict) else None


def merged_file_metadata(
    inferred: dict[str, Any] | None,
    mapped: Any,
) -> dict[str, Any] | None:
    metadata = dict(inferred or {})
    if isinstance(mapped, dict):
        metadata.update(mapped)
    return metadata or None


def file_infos_from_extension_analysis(
    data_dir: Path,
    include_client_metadata: bool,
    include_legacy_client_metadata: bool,
    cache_root: Path,
    title_mode: TitleMode,
    episode_roots: list[Path] | None = None,
) -> tuple[list[FileInfo], Counter[str]]:
    discovered, skipped = discover_files(data_dir, cache_root, roots=episode_roots)
    all_files = get_all_files_recursive(data_dir, roots=episode_roots)
    contexts = build_episode_contexts_from_paths(all_files, cache_root)
    file_map = get_file_map(data_dir)
    file_infos: list[FileInfo] = []

    for item in discovered:
        mapped = recovery_file_entry(file_map, data_dir, item.path)
        inferred_metadata = get_optional_client_metadata(
            item.path,
            item.data_type,
            include=include_client_metadata,
            include_legacy_fields=include_legacy_client_metadata,
            contexts=contexts,
            cache_root=cache_root,
        )
        mapped_metadata = mapped.get("client_metadata") if mapped and include_client_metadata else None
        file_infos.append(
            FileInfo(
                title=title_for_file(
                    item.path,
                    data_dir,
                    cache_root,
                    title_mode,
                    mapped_title=str(mapped.get("title")) if mapped and mapped.get("title") else None,
                ),
                path=item.path,
                data_type=item.data_type,
                client_metadata=merged_file_metadata(inferred_metadata, mapped_metadata),
            )
        )
    return file_infos, skipped


def file_infos_from_folder_structure(
    data_dir: Path,
    include_client_metadata: bool,
    include_legacy_client_metadata: bool,
    cache_root: Path,
    title_mode: TitleMode,
    episode_roots: list[Path] | None = None,
) -> tuple[list[FileInfo], Counter[str]]:
    file_map = get_file_map(data_dir)
    if file_map is None:
        raise typer.BadParameter(f"Missing {FILE_MAP_FILENAME} for --folder-structure mode.")

    raw_discovered: list[DiscoveredFile] = []
    skipped: Counter[str] = Counter()
    for data_type in DataType:
        target_dir = data_dir / data_type.value
        if not target_dir.is_dir():
            continue
        for file in get_all_files_recursive(target_dir, exclude_filenames=[FILE_MAP_FILENAME]):
            if episode_roots is not None and not path_is_under_any(file, episode_roots):
                continue
            extension = normal_extension(file)
            if extension in SKIP_EXTENSIONS:
                skipped[extension] += 1
                continue
            raw_discovered.append(
                DiscoveredFile(
                    path=file,
                    data_type=data_type,
                    extension=extension,
                    cached_source=cached_source_for_path(file, cache_root),
                )
            )

    contexts = build_episode_contexts(raw_discovered, cache_root)
    file_infos: list[FileInfo] = []
    for item in raw_discovered:
        mapped_title = file_map.get(item.path.name)
        file_infos.append(
            FileInfo(
                title=title_for_file(item.path, data_dir, cache_root, title_mode, mapped_title=mapped_title),
                path=item.path,
                data_type=item.data_type,
                client_metadata=get_optional_client_metadata(
                    item.path,
                    item.data_type,
                    include=include_client_metadata,
                    include_legacy_fields=include_legacy_client_metadata,
                    contexts=contexts,
                    cache_root=cache_root,
                ),
            )
        )
    return file_infos, skipped


def fix_dicom_file(file_path: Path) -> None:
    import pydicom
    from pydicom.uid import generate_uid

    try:
        dicom_data = pydicom.dcmread(file_path)
    except Exception:
        dicom_data = pydicom.dcmread(file_path, force=True)
        with file_path.open("wb") as f:
            f.write(b"\x00" * 128 + b"DICM")
            dicom_data.save_as(f, write_like_original=True)
        dicom_data = pydicom.dcmread(file_path)

    changed = False
    for field_name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "PatientID"):
        if not dicom_data.get(field_name):
            setattr(dicom_data, field_name, generate_uid())
            changed = True
    if not dicom_data.get("Columns"):
        dicom_data.Columns = 512
        changed = True
    if not dicom_data.get("Rows"):
        dicom_data.Rows = 512
        changed = True
    if changed:
        dicom_data.save_as(file_path)


def upload_task(storage_folder: StorageFolder, file_info: FileInfo) -> UploadOutcome:
    try:
        uuid = None
        if file_info.data_type == DataType.IMAGE:
            uuid = storage_folder.upload_image(file_info.path, file_info.title, file_info.client_metadata)
        elif file_info.data_type in {DataType.VIDEO, DataType.IMAGE_SEQUENCE}:
            uuid = storage_folder.upload_video(file_info.path, file_info.title, file_info.client_metadata)
        elif file_info.data_type == DataType.AUDIO:
            uuid = storage_folder.upload_audio(file_info.path, file_info.title, file_info.client_metadata)
        elif file_info.data_type == DataType.PLAIN_TEXT:
            uuid = storage_folder.upload_text(file_info.path, file_info.title, file_info.client_metadata)
        elif file_info.data_type == DataType.PDF:
            uuid = storage_folder.upload_pdf(file_info.path, file_info.title, file_info.client_metadata)
        elif file_info.data_type == DataType.NIFTI:
            uuid = storage_folder.upload_nifti(file_info.path, file_info.title, file_info.client_metadata)
        elif file_info.data_type == DataType.DICOM_FILE:
            uuid = storage_folder.create_dicom_series([file_info.path], file_info.title, file_info.client_metadata)
        else:
            return UploadOutcome(
                status="unsupported",
                title=file_info.title,
                path=str(file_info.path),
                data_type=file_info.data_type.value,
                error=f"Unsupported data type: {file_info.data_type}",
            )
        return UploadOutcome(
            status="uploaded",
            title=file_info.title,
            path=str(file_info.path),
            data_type=file_info.data_type.value,
            storage_item_uuid=str(uuid) if uuid is not None else None,
        )
    except Exception as exc:
        return UploadOutcome(
            status="failed",
            title=file_info.title,
            path=str(file_info.path),
            data_type=file_info.data_type.value,
            error=f"{type(exc).__name__}: {exc}",
        )


def storage_item_type_value(item: Any) -> str:
    item_type = getattr(item, "type", None)
    return str(getattr(item_type, "value", item_type) or "")


def existing_file_title_mapping(items: list[Any]) -> dict[str, str]:
    group_types = {DataType.GROUP.value, DataType.IMAGE_GROUP.value}
    return {
        str(item.name): str(item.uuid)
        for item in items
        if storage_item_type_value(item) not in group_types
    }


def already_uploaded_titles(storage_folder: StorageFolder) -> set[str]:
    group_types = {DataType.GROUP.value, DataType.IMAGE_GROUP.value}
    return {
        str(item.name)
        for item in storage_folder.list_items(page_size=1000)
        if storage_item_type_value(item) not in group_types
    }


def existing_group_titles(items: list[Any], group_type: DataType) -> set[str]:
    return {
        str(item.name)
        for item in items
        if storage_item_type_value(item) == group_type.value
    }


def upload_files(
    storage_folder: StorageFolder,
    file_infos: list[FileInfo],
    max_workers: int,
    fix_dicom: bool,
) -> tuple[list[UploadOutcome], dict[str, str]]:
    title_to_uuid_mapping: dict[str, str] = {}
    mapping_lock = Lock()
    outcomes: list[UploadOutcome] = []

    def run_one(file_info: FileInfo) -> UploadOutcome:
        if fix_dicom and file_info.data_type == DataType.DICOM_FILE:
            fix_dicom_file(file_info.path)
        outcome = upload_task(storage_folder, file_info)
        if outcome.storage_item_uuid is not None:
            with mapping_lock:
                title_to_uuid_mapping[file_info.title] = outcome.storage_item_uuid
        return outcome

    with tqdm(total=len(file_infos), desc="Uploading files", dynamic_ncols=True) as progress:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_tasks = {executor.submit(run_one, file_info): file_info for file_info in file_infos}
            for future in as_completed(future_tasks):
                outcome = future.result()
                outcomes.append(outcome)
                if outcome.error:
                    logger.error("%s: %s", outcome.title, outcome.error)
                progress.update(1)
    return outcomes, title_to_uuid_mapping


def create_image_groups_from_file_map(
    file_map: dict[str, Any],
    storage_folder: StorageFolder,
    title_to_uuid_mapping: dict[str, str],
    existing_group_names: set[str] | None = None,
) -> list[str]:
    image_groups_from_file = file_map.get("image_groups")
    if not isinstance(image_groups_from_file, dict):
        return []

    created: list[str] = []
    image_groups_to_upload = []
    for group_name, content in image_groups_from_file.items():
        if existing_group_names and group_name in existing_group_names:
            logger.info("Skipping existing image group %s", group_name)
            continue
        items = content.get("items", []) if isinstance(content, dict) else []
        missing = [item for item in items if item not in title_to_uuid_mapping]
        if missing:
            logger.warning("Skipping image group %s; missing item titles: %s", group_name, missing)
            continue
        image_groups_to_upload.append(
            DataUploadImageGroupFromItems(
                image_items=[title_to_uuid_mapping[item] for item in items],
                title=group_name,
                create_video=False,
            )
        )
        created.append(group_name)

    if image_groups_to_upload:
        storage_folder.add_private_data_to_folder_start(
            integration_id=None,
            private_files=DataUploadItems(image_groups_from_items=image_groups_to_upload),
            ignore_errors=True,
        )
    return created


def create_trossen_recovery_group(
    *,
    group_name: str,
    content: dict[str, Any],
    storage_folder: StorageFolder,
    title_to_uuid_mapping: dict[str, str],
) -> None:
    from encord.orm.group_layout import DataUnitCarouselTile, DataUnitTile, LayoutGrid
    from encord.orm.storage import DataGroupCustom

    roles = content.get("roles")
    videos = roles.get("videos") if isinstance(roles, dict) else None
    metadata = roles.get("metadata") if isinstance(roles, dict) else None
    if not isinstance(videos, dict) or not isinstance(metadata, dict):
        raise TypeError(f"Recovery data group {group_name!r} is missing roles.videos/metadata.")

    required_cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    missing_roles = [camera for camera in required_cameras if camera not in videos]
    if missing_roles:
        raise ValueError(
            f"Recovery data group {group_name!r} is missing video roles: {missing_roles}"
        )

    layout_contents: dict[str, UUID] = {
        "camera_cam_high": UUID(title_to_uuid_mapping[videos["cam_high"]]),
        "camera_cam_left_wrist": UUID(title_to_uuid_mapping[videos["cam_left_wrist"]]),
        "camera_cam_right_wrist": UUID(title_to_uuid_mapping[videos["cam_right_wrist"]]),
    }
    metadata_keys = []
    for role in ("info", "tasks", "episodes", "episodes_stats"):
        title = metadata.get(role)
        if not title:
            continue
        key = f"metadata_{role}"
        layout_contents[key] = UUID(title_to_uuid_mapping[title])
        metadata_keys.append(key)

    wrist_grid = LayoutGrid(
        direction="row",
        split_percentage=50,
        first=DataUnitTile(key="camera_cam_left_wrist"),
        second=DataUnitTile(key="camera_cam_right_wrist"),
    )
    right_side = LayoutGrid(
        direction="column",
        split_percentage=50,
        first=wrist_grid,
        second=DataUnitCarouselTile(
            keys=metadata_keys,
            carousel_position="bottom",
            carousel_size=10,
        ),
    )

    client_metadata = dict(content.get("client_metadata") or {})
    original_source_folder_id = client_metadata.get("source_folder_id")
    if original_source_folder_id:
        client_metadata["recovery_original_source_folder_id"] = original_source_folder_id
    client_metadata["source_folder_id"] = str(storage_folder.uuid)
    client_metadata["video_uuids"] = [
        str(layout_contents[f"camera_{camera}"]) for camera in required_cameras
    ]
    client_metadata["json_uuids"] = [str(layout_contents[key]) for key in metadata_keys]

    storage_folder.create_data_group(
        DataGroupCustom(
            name=group_name,
            layout_contents=layout_contents,
            layout=LayoutGrid(
                direction="row",
                split_percentage=50,
                first=DataUnitTile(key="camera_cam_high"),
                second=right_side,
            ),
            client_metadata=client_metadata,
        )
    )


def create_data_groups_from_file_map(
    file_map: dict[str, Any],
    storage_folder: StorageFolder,
    title_to_uuid_mapping: dict[str, str],
    existing_group_names: set[str] | None = None,
) -> list[str]:
    data_groups = file_map.get("data_groups")
    if not isinstance(data_groups, dict):
        return []

    created: list[str] = []
    for group_name, content in data_groups.items():
        if existing_group_names and group_name in existing_group_names:
            logger.info("Skipping existing data group %s", group_name)
            continue
        items = content.get("items", []) if isinstance(content, dict) else []
        missing = [item for item in items if item not in title_to_uuid_mapping]
        if missing:
            logger.warning("Skipping data group %s; missing item titles: %s", group_name, missing)
            continue
        if content.get("layout") == "trossen-three-camera-metadata":
            create_trossen_recovery_group(
                group_name=group_name,
                content=content,
                storage_folder=storage_folder,
                title_to_uuid_mapping=title_to_uuid_mapping,
            )
        else:
            storage_folder.create_data_group(
                DataGroupGrid(
                    name=group_name,
                    layout_contents=[title_to_uuid_mapping[item] for item in items],
                )
            )
        created.append(group_name)
    return created


def looks_like_annotation_file(file_info: FileInfo) -> bool:
    path = PurePosixPath(file_info.title.lower())
    name = path.name
    if any(hint in name for hint in ANNOTATION_NAME_HINTS):
        return True
    return any(part in ANNOTATION_NAME_HINTS for part in path.parts)


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
    if hasattr(value, "__dict__"):
        return {key: json_safe(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def outcome_summary(outcomes: list[UploadOutcome]) -> dict[str, int]:
    return dict(Counter(outcome.status for outcome in outcomes))


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(report), indent=2))
