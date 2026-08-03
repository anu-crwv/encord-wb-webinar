# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord",
#     "pydicom",
#     "pyyaml",
#     "tqdm",
#     "typer",
#     "wandb>=0.18.0",
# ]
# ///
"""Upload videos from a W&B dataset artifact into an existing Encord storage folder."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path, PurePosixPath
import re
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

from encord.storage import StorageItemType as DataType
import typer
import yaml
from tqdm import tqdm

from cached_s3_folder_upload_utils import (
    MAX_WORKERS_DEFAULT,
    SCRIPT_DIR,
    EncordDomain,
    FileInfo,
    TitleMode,
    already_uploaded_titles,
    get_encord_client,
    outcome_summary,
    upload_files,
    write_report,
)


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WANDB_CONFIG = SCRIPT_DIR.parent / "wandb_config.yaml"
IMPORT_ROOT = REPO_ROOT / "exports" / "encord-wandb-import"
DEFAULT_DOWNLOAD_WORKERS = 8
VIDEO_EXTENSIONS = {".3g2", ".3gp", ".avi", ".mkv", ".mj2", ".mov", ".mp4", ".webm"}
META_ENTRY_PATHS = (
    "dataset/meta/source_dataset_items.json",
    "dataset/meta/source_dataset_manifest.json",
    "dataset/meta/info.json",
    "dataset/meta/episodes.jsonl",
)
SOURCE_ITEM_METADATA_FIELDS = (
    "episode_index",
    "data_hash",
    "data_group_uuid",
    "video_storage_item_uuid",
    "camera_name",
    "video_key",
    "artifact_path",
    "source_uri",
    "fps",
)
EPISODE_FILE_RE = re.compile(r"episode_(\d+)")


@dataclass(frozen=True)
class ArtifactVideo:
    artifact_path: str
    local_path: Path | None
    ref_target: str | None


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"{label} does not exist: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"{label} must contain a YAML object.")
    return loaded


def required(config: dict[str, Any], key: str, label: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise typer.BadParameter(f"{label} is missing required key: {key}")
    return value


def resolve_artifact_ref(wandb_config: dict[str, Any], artifact_ref: str | None) -> str:
    ref = artifact_ref or f"{required(wandb_config, 'source_artifact_name', 'W&B config')}:latest"
    if "/" in ref.split(":", 1)[0]:
        return ref
    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    return f"{entity}/{project}/{ref}"


def safe_cache_name(artifact_ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", artifact_ref).strip("_") or "wandb-artifact"


def artifact_attr(artifact: Any, name: str) -> Any:
    value = getattr(artifact, name, None)
    return value() if callable(value) else value


def artifact_entries(artifact: Any) -> dict[str, Any]:
    manifest = getattr(artifact, "manifest", None)
    entries = getattr(manifest, "entries", None)
    if not isinstance(entries, dict):
        raise typer.BadParameter("Could not read W&B artifact manifest entries.")
    return entries


def video_entry_names(entries: dict[str, Any]) -> list[str]:
    return sorted(name for name in entries if PurePosixPath(name).suffix.lower() in VIDEO_EXTENSIONS)


def find_metadata_entry(entries: dict[str, Any], target: str) -> str | None:
    if target in entries:
        return target
    target_path = PurePosixPath(target)
    matches = [
        name
        for name in entries
        if PurePosixPath(name).parts[-len(target_path.parts) :] == target_path.parts
    ]
    return sorted(matches)[0] if matches else None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Could not parse JSON metadata file {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Could not parse JSONL metadata file {path}:{line_number}: {exc}") from exc
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def download_entry(entry: Any, cache_dir: Path) -> Path:
    return Path(entry.download(root=str(cache_dir), skip_cache=True))


def load_artifact_metadata(
    artifact: Any,
    entries: dict[str, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_items": [],
        "source_items_by_path": {},
        "manifest": {},
        "info": {},
        "episodes": [],
    }
    for target in META_ENTRY_PATHS:
        name = find_metadata_entry(entries, target)
        if name is None:
            continue
        local_path = download_entry(artifact.get_entry(name), cache_dir)
        if target.endswith(".jsonl"):
            metadata["episodes"] = read_jsonl(local_path)
            continue
        loaded = read_json(local_path)
        if target.endswith("source_dataset_items.json") and isinstance(loaded, list):
            metadata["source_items"] = loaded
            metadata["source_items_by_path"] = {
                str(item.get("artifact_path")): item
                for item in loaded
                if isinstance(item, dict) and item.get("artifact_path")
            }
        elif target.endswith("source_dataset_manifest.json") and isinstance(loaded, dict):
            metadata["manifest"] = loaded
        elif target.endswith("info.json") and isinstance(loaded, dict):
            metadata["info"] = loaded
    return metadata


def entry_ref_target(entry: Any) -> str | None:
    try:
        value = entry.ref_target()
    except Exception:
        return None
    return str(value) if value else None


def source_key_from_uri(uri: str) -> str | None:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return unquote(parsed.path.lstrip("/")) or None
    if parsed.scheme in {"http", "https"}:
        return unquote(parsed.path.lstrip("/")) or None
    return None


def add_if_scalar(metadata: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    if isinstance(value, str | int | float | bool):
        metadata.setdefault(key, value)


def path_metadata(artifact_path: str) -> dict[str, Any]:
    path = PurePosixPath(artifact_path)
    out: dict[str, Any] = {
        "source_wandb_entry_path": artifact_path,
        "file_ext": path.suffix.lower(),
    }
    match = EPISODE_FILE_RE.search(path.stem)
    if match:
        out["episode_index"] = int(match.group(1))
        out["episode_id"] = f"episode_{int(match.group(1)):06d}"
    parts = path.parts
    if "videos" in parts:
        video_index = parts.index("videos")
        if video_index + 2 < len(parts):
            out["video_key"] = parts[video_index + 2]
            out["sensor_key"] = parts[video_index + 2]
            out["camera_name"] = parts[video_index + 2].removeprefix("observation.images.")
    return out


def build_client_metadata(
    video: ArtifactVideo,
    artifact_ref: str,
    artifact: Any,
    artifact_metadata: dict[str, Any],
    include_legacy_client_metadata: bool,
) -> dict[str, Any]:
    source_item = artifact_metadata["source_items_by_path"].get(video.artifact_path)
    metadata: dict[str, Any] = {}
    if isinstance(source_item, dict):
        source_client_metadata = source_item.get("client_metadata")
        if isinstance(source_client_metadata, dict):
            for key, value in source_client_metadata.items():
                add_if_scalar(metadata, str(key), value)
        for key in SOURCE_ITEM_METADATA_FIELDS:
            add_if_scalar(metadata, key, source_item.get(key))

    for key, value in path_metadata(video.artifact_path).items():
        add_if_scalar(metadata, key, value)

    manifest = artifact_metadata.get("manifest") or {}
    add_if_scalar(metadata, "source_wandb_artifact", artifact_ref)
    add_if_scalar(metadata, "source_wandb_artifact_name", artifact_attr(artifact, "name"))
    add_if_scalar(metadata, "source_wandb_artifact_version", artifact_attr(artifact, "version"))
    add_if_scalar(metadata, "source_wandb_artifact_digest", artifact_attr(artifact, "digest"))
    add_if_scalar(metadata, "source_encord_dataset_hash", manifest.get("encord_source_dataset_hash"))
    add_if_scalar(metadata, "source_encord_dataset_title", manifest.get("encord_dataset_title"))
    if video.ref_target:
        add_if_scalar(metadata, "source_wandb_ref_target", video.ref_target)
        if "source_uri" not in metadata:
            add_if_scalar(metadata, "source_uri", video.ref_target)
        source_key = source_key_from_uri(video.ref_target)
        if source_key:
            add_if_scalar(metadata, "source_key", source_key)
    elif "source_uri" in metadata and "source_key" not in metadata:
        source_key = source_key_from_uri(str(metadata["source_uri"]))
        if source_key:
            add_if_scalar(metadata, "source_key", source_key)

    if include_legacy_client_metadata:
        metadata.setdefault("Tag", "A")
        metadata.setdefault("Data Type", DataType.VIDEO.value)
        metadata.setdefault("Extension", PurePosixPath(video.artifact_path).suffix)
    return metadata


def title_for_video(video: ArtifactVideo, metadata: dict[str, Any], title_mode: TitleMode) -> str:
    if title_mode in {TitleMode.AUTO, TitleMode.SOURCE_KEY}:
        source_key = metadata.get("source_key")
        if source_key:
            return str(source_key)
        source_uri = metadata.get("source_uri")
        if isinstance(source_uri, str):
            key = source_key_from_uri(source_uri)
            if key:
                return key
        if title_mode == TitleMode.SOURCE_KEY:
            return video.artifact_path
    if title_mode == TitleMode.FILENAME:
        return PurePosixPath(video.artifact_path).name
    return video.artifact_path


def metadata_for_title(
    video: ArtifactVideo,
    artifact_ref: str,
    artifact: Any,
    artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    return build_client_metadata(
        video=video,
        artifact_ref=artifact_ref,
        artifact=artifact,
        artifact_metadata=artifact_metadata,
        include_legacy_client_metadata=False,
    )


def placeholder_file_info(
    video: ArtifactVideo,
    artifact_ref: str,
    artifact: Any,
    artifact_metadata: dict[str, Any],
    title_mode: TitleMode,
) -> FileInfo:
    metadata = metadata_for_title(video, artifact_ref, artifact, artifact_metadata)
    return FileInfo(
        title=title_for_video(video, metadata, title_mode),
        path=Path(video.artifact_path),
        data_type=DataType.VIDEO,
        client_metadata=None,
    )


def select_video_names(video_names: list[str], max_videos: int | None) -> list[str]:
    if max_videos is None:
        return video_names
    return video_names[:max_videos]


def download_videos(
    artifact: Any,
    videos: list[ArtifactVideo],
    cache_dir: Path,
    max_download_workers: int,
) -> list[ArtifactVideo]:
    if not videos:
        return []

    downloaded: list[ArtifactVideo | None] = [None] * len(videos)

    def run_one(index: int, video: ArtifactVideo) -> tuple[int, ArtifactVideo]:
        entry = artifact.get_entry(video.artifact_path)
        return index, ArtifactVideo(
            artifact_path=video.artifact_path,
            local_path=download_entry(entry, cache_dir),
            ref_target=video.ref_target,
        )

    with tqdm(total=len(videos), desc="Downloading W&B videos", unit="file", dynamic_ncols=True) as progress:
        with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
            futures = {
                executor.submit(run_one, index, video): index
                for index, video in enumerate(videos)
            }
            for future in as_completed(futures):
                index, video = future.result()
                downloaded[index] = video
                progress.update(1)

    return [video for video in downloaded if video is not None]


def dry_run_videos(artifact: Any, selected_names: list[str]) -> list[ArtifactVideo]:
    return [
        ArtifactVideo(
            artifact_path=name,
            local_path=None,
            ref_target=entry_ref_target(artifact.get_entry(name)),
        )
        for name in selected_names
    ]


def build_file_infos(
    videos: list[ArtifactVideo],
    artifact_ref: str,
    artifact: Any,
    artifact_metadata: dict[str, Any],
    include_client_metadata: bool,
    include_legacy_client_metadata: bool,
    title_mode: TitleMode,
) -> list[FileInfo]:
    file_infos: list[FileInfo] = []
    for video in videos:
        if video.local_path is None:
            continue
        title_metadata = metadata_for_title(video, artifact_ref, artifact, artifact_metadata)
        metadata = None
        if include_client_metadata:
            metadata = build_client_metadata(
                video=video,
                artifact_ref=artifact_ref,
                artifact=artifact,
                artifact_metadata=artifact_metadata,
                include_legacy_client_metadata=include_legacy_client_metadata,
            )
        file_infos.append(
            FileInfo(
                title=title_for_video(video, title_metadata, title_mode),
                path=video.local_path,
                data_type=DataType.VIDEO,
                client_metadata=metadata,
            )
        )
    return file_infos


def sample_video_report(
    videos: list[ArtifactVideo],
    artifact_ref: str,
    artifact: Any,
    artifact_metadata: dict[str, Any],
    include_legacy_client_metadata: bool,
    title_mode: TitleMode,
) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for video in videos[:20]:
        metadata = build_client_metadata(
            video=video,
            artifact_ref=artifact_ref,
            artifact=artifact,
            artifact_metadata=artifact_metadata,
            include_legacy_client_metadata=include_legacy_client_metadata,
        )
        sample.append(
            {
                "artifact_path": video.artifact_path,
                "title": title_for_video(video, metadata, title_mode),
                "ref_target": video.ref_target,
                "client_metadata": metadata,
            }
        )
    return sample


def build_report(
    *,
    dry_run: bool,
    folder_hash: str,
    requested_artifact_ref: str,
    artifact: Any,
    cache_dir: Path,
    all_video_count: int,
    selected_videos: list[ArtifactVideo],
    downloaded_video_count: int,
    artifact_metadata: dict[str, Any],
    include_legacy_client_metadata: bool,
    title_mode: TitleMode,
    skipped_existing: list[FileInfo] | None = None,
    outcomes: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "folder_hash": folder_hash,
        "requested_artifact_ref": requested_artifact_ref,
        "artifact_name": artifact_attr(artifact, "name"),
        "artifact_version": artifact_attr(artifact, "version"),
        "artifact_digest": artifact_attr(artifact, "digest"),
        "artifact_url": artifact_attr(artifact, "url"),
        "cache_dir": str(cache_dir),
        "video_count_in_artifact": all_video_count,
        "selected_video_count": len(selected_videos),
        "downloaded_video_count": downloaded_video_count,
        "source_metadata_item_count": len(artifact_metadata.get("source_items") or []),
        "skipped_existing_count": len(skipped_existing or []),
        "skipped_existing_titles": [item.title for item in (skipped_existing or [])[:500]],
        "summary": outcome_summary(outcomes or []),
        "sample_videos": sample_video_report(
            selected_videos,
            requested_artifact_ref,
            artifact,
            artifact_metadata,
            include_legacy_client_metadata,
            title_mode,
        ),
        "outcomes": outcomes or [],
    }


def default_report_json(dry_run: bool) -> Path:
    suffix = "dry_run_report" if dry_run else "upload_report"
    return SCRIPT_DIR / f"encord_wandb_artifact_{suffix}.json"


def main(
    folder_hash: Annotated[
        str,
        typer.Argument(help="Existing Encord storage folder UUID/hash to upload videos into."),
    ],
    artifact_ref: Annotated[
        str | None,
        typer.Option(
            "--artifact-ref",
            help="W&B dataset artifact ref. Defaults to <source_artifact_name>:latest from wandb_config.yaml.",
        ),
    ] = None,
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
    cache_dir: Annotated[
        Path | None,
        typer.Option(help="Local cache directory for downloaded W&B artifact entries."),
    ] = None,
    ssh_key_file: Annotated[
        Path | None,
        typer.Option(
            "--ssh-key-file",
            "-k",
            envvar="ENCORD_SSH_KEY_FILE",
            help="Path to the Encord SSH private key. Can also be supplied by ENCORD_SSH_KEY_FILE.",
        ),
    ] = None,
    domain: Annotated[
        EncordDomain,
        typer.Option("--domain", help="Encord environment."),
    ] = EncordDomain.PROD,
    title_mode: Annotated[
        TitleMode,
        typer.Option(
            "--title-mode",
            help="How uploaded item titles are chosen. auto prefers original source_key metadata.",
        ),
    ] = TitleMode.AUTO,
    max_videos: Annotated[
        int | None,
        typer.Option(help="Optional max videos to import, useful for smoke tests."),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Maximum concurrent Encord upload threads."),
    ] = MAX_WORKERS_DEFAULT,
    max_download_workers: Annotated[
        int,
        typer.Option("--max-download-workers", help="Maximum concurrent W&B artifact video downloads."),
    ] = DEFAULT_DOWNLOAD_WORKERS,
    include_client_metadata: Annotated[
        bool,
        typer.Option(
            "--include-client-metadata/--no-client-metadata",
            help="Attach recovered client metadata to uploaded videos.",
        ),
    ] = True,
    include_legacy_client_metadata: Annotated[
        bool,
        typer.Option(
            "--include-legacy-client-metadata/--no-legacy-client-metadata",
            help="Also include the original uploader's Tag, Data Type, and Extension metadata fields.",
        ),
    ] = True,
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing/--upload-existing", help="Skip artifact videos whose title already exists."),
    ] = True,
    report_json: Annotated[
        Path | None,
        typer.Option("--report-json", help="Path for upload or dry-run report JSON."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Inspect the W&B artifact and write a report without downloading videos/uploading."),
    ] = False,
) -> None:
    if not folder_hash:
        raise typer.BadParameter("Pass the target Encord storage folder hash.")
    if max_videos is not None and max_videos < 1:
        raise typer.BadParameter("--max-videos must be at least 1.")
    if max_workers < 1:
        raise typer.BadParameter("--max-workers must be at least 1.")
    if max_download_workers < 1:
        raise typer.BadParameter("--max-download-workers must be at least 1.")

    import wandb

    wandb_settings = load_yaml(wandb_config.expanduser().resolve(), "W&B config")
    resolved_artifact_ref = resolve_artifact_ref(wandb_settings, artifact_ref)
    cache_dir = (cache_dir or (IMPORT_ROOT / safe_cache_name(resolved_artifact_ref))).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_report = report_json or default_report_json(dry_run)
    output_report = output_report.expanduser().resolve()

    typer.echo(f"W&B artifact: {resolved_artifact_ref}")
    typer.echo(f"Cache dir: {cache_dir}")
    artifact = wandb.Api().artifact(resolved_artifact_ref, type="dataset")
    entries = artifact_entries(artifact)
    all_video_names = video_entry_names(entries)
    selected_names = select_video_names(all_video_names, max_videos)
    if not selected_names:
        raise typer.BadParameter(f"No video entries found in W&B artifact {resolved_artifact_ref}.")

    artifact_metadata = load_artifact_metadata(artifact, entries, cache_dir)
    selected_videos = dry_run_videos(artifact, selected_names)
    typer.echo(f"Video entries selected: {len(selected_videos):,}/{len(all_video_names):,}")
    typer.echo(f"Source metadata records found: {len(artifact_metadata.get('source_items') or []):,}")

    if dry_run:
        report = build_report(
            dry_run=True,
            folder_hash=folder_hash,
            requested_artifact_ref=resolved_artifact_ref,
            artifact=artifact,
            cache_dir=cache_dir,
            all_video_count=len(all_video_names),
            selected_videos=selected_videos,
            downloaded_video_count=0,
            artifact_metadata=artifact_metadata,
            include_legacy_client_metadata=include_legacy_client_metadata,
            title_mode=title_mode,
        )
        write_report(output_report, report)
        typer.echo(f"Dry-run report JSON: {output_report}")
        return

    if ssh_key_file is None:
        raise typer.BadParameter("Pass --ssh-key-file or set ENCORD_SSH_KEY_FILE.")

    client = get_encord_client(ssh_key_file=ssh_key_file, domain=domain)
    storage_folder = client.get_storage_folder(folder_hash)
    if skip_existing:
        existing_titles = already_uploaded_titles(storage_folder)
        skipped_existing = []
        videos_to_download = []
        for video in selected_videos:
            planned = placeholder_file_info(video, resolved_artifact_ref, artifact, artifact_metadata, title_mode)
            if planned.title in existing_titles:
                skipped_existing.append(planned)
            else:
                videos_to_download.append(video)
    else:
        skipped_existing = []
        videos_to_download = selected_videos

    typer.echo(f"Already in folder: {len(skipped_existing):,}")
    typer.echo(f"Downloading videos: {len(videos_to_download):,} with {max_download_workers:,} worker(s)")
    downloaded_videos = download_videos(
        artifact=artifact,
        videos=videos_to_download,
        cache_dir=cache_dir,
        max_download_workers=max_download_workers,
    )

    file_infos = build_file_infos(
        videos=downloaded_videos,
        artifact_ref=resolved_artifact_ref,
        artifact=artifact,
        artifact_metadata=artifact_metadata,
        include_client_metadata=include_client_metadata,
        include_legacy_client_metadata=include_legacy_client_metadata,
        title_mode=title_mode,
    )

    to_upload = file_infos
    typer.echo(f"Uploading ungrouped videos: {len(to_upload):,} with {max_workers:,} worker(s)")
    outcomes, _title_to_uuid_mapping = upload_files(
        storage_folder=storage_folder,
        file_infos=to_upload,
        max_workers=max_workers,
        fix_dicom=False,
    )
    report = build_report(
        dry_run=False,
        folder_hash=folder_hash,
        requested_artifact_ref=resolved_artifact_ref,
        artifact=artifact,
        cache_dir=cache_dir,
        all_video_count=len(all_video_names),
        selected_videos=selected_videos,
        downloaded_video_count=len(downloaded_videos),
        artifact_metadata=artifact_metadata,
        include_legacy_client_metadata=include_legacy_client_metadata,
        title_mode=title_mode,
        skipped_existing=skipped_existing,
        outcomes=outcomes,
    )
    write_report(output_report, report)
    summary = outcome_summary(outcomes)
    typer.echo(f"Upload summary: {summary}")
    typer.echo(f"Report JSON: {output_report}")
    if summary.get("failed", 0):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
