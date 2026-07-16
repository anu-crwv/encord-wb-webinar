# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "google-cloud-storage",
#     "tqdm",
#     "typer",
# ]
# ///
"""Re-encode Encord source S3 videos to normalized MP4s in GCS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import typer
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MAIN_WORKTREE_EXPORTS_ROOT = Path("/Users/encordsf/Desktop/encord-wb-webinar/exports")
DEFAULT_DATASET_EXPORT_ROOT = MAIN_WORKTREE_EXPORTS_ROOT / "encord-dataset-export"
DEFAULT_SHARED_S3_CACHE_ROOT = DEFAULT_DATASET_EXPORT_ROOT / "_cache" / "s3"
DEFAULT_OUTPUT_ROOT = MAIN_WORKTREE_EXPORTS_ROOT / "encord-video-reencode"
DEFAULT_GCP_PROJECT = "encord-operations"
DEFAULT_GCS_BUCKET = "encord-wandb-webinar"
DEFAULT_GCS_PREFIX = "re-encoded-robotics-data"
DEFAULT_SOURCE_BUCKET = "ego-data-collection-encord"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".3gp", ".3g2", ".mj2", ".avi"}


@dataclass(frozen=True)
class VideoSource:
    source_uri: str
    title: str
    storage_item_uuid: str
    parent_kind: str
    parent_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DiscoveryResult:
    kind: str
    title: str
    hash_value: str
    videos: list[VideoSource]
    skipped: list[dict[str, Any]]


@dataclass(frozen=True)
class CachedSource:
    path: Path
    cache_kind: str
    downloaded: bool


def timestamped_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def create_client() -> Any:
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def item_metadata(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "client_metadata", None) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def item_uuid(item: Any) -> str:
    return str(getattr(item, "uuid", "") or getattr(item, "uid", "") or "")


def item_title(item: Any) -> str:
    return str(getattr(item, "name", None) or getattr(item, "title", None) or item_uuid(item))


def item_type_name(item: Any) -> str:
    item_type = getattr(item, "item_type", "")
    return str(getattr(item_type, "name", item_type)).upper()


def is_video_item(item: Any) -> bool:
    return item_type_name(item) == "VIDEO"


def normalize_source_key(value: Any) -> str:
    path = str(value or "")
    if path.startswith("s3://") or path.startswith("http://") or path.startswith("https://"):
        return unquote(urlparse(path).path.lstrip("/"))
    return path.lstrip("/")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        return bucket, unquote(parsed.path.lstrip("/"))
    raise ValueError(f"Unsupported S3 URI format: {uri}")


def s3_identity(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"{bucket}/{key}"


def source_uri_from_metadata(metadata: dict[str, Any], default_source_bucket: str) -> str | None:
    for key in ("source_uri", "s3_uri", "source_s3_uri", "objectUrl", "object_url"):
        value = metadata.get(key)
        if value:
            return str(value)
    source_key = metadata.get("source_key")
    if source_key:
        return f"s3://{default_source_bucket}/{normalize_source_key(source_key)}"
    return None


def source_uri_for_item(item: Any, default_source_bucket: str) -> str | None:
    metadata = item_metadata(item)
    uri = source_uri_from_metadata(metadata, default_source_bucket)
    if uri:
        return uri
    title = item_title(item)
    return title if title.startswith("s3://") else None


def has_video_extension(uri: str) -> bool:
    try:
        _, key = parse_s3_uri(uri)
    except ValueError:
        return False
    return PurePosixPath(key).suffix.lower() in VIDEO_EXTENSIONS


def group_children(item: Any, client: Any) -> list[Any]:
    children_by_uuid: dict[str, Any] = {}
    try:
        for child in item.get_child_items():
            children_by_uuid[item_uuid(child)] = child
    except Exception:
        pass

    try:
        summary = item.get_summary()
    except Exception:
        return list(children_by_uuid.values())

    data_group = getattr(summary, "data_group", None)
    layout_contents = getattr(data_group, "layout_contents", None)
    if layout_contents:
        child_uuids = [
            child.uuid
            for child in layout_contents.values()
            if str(child.uuid) not in children_by_uuid
        ]
        if child_uuids:
            for child in client.get_storage_items(child_uuids):
                children_by_uuid[item_uuid(child)] = child
    return list(children_by_uuid.values())


def video_sources_from_storage_item(
    *,
    item: Any,
    client: Any,
    parent_kind: str,
    parent_id: str,
    default_source_bucket: str,
) -> tuple[list[VideoSource], list[dict[str, Any]]]:
    candidates = [item] if is_video_item(item) else [
        child for child in group_children(item, client) if is_video_item(child)
    ]
    videos: list[VideoSource] = []
    skipped: list[dict[str, Any]] = []
    for candidate in candidates:
        uri = source_uri_for_item(candidate, default_source_bucket)
        if not uri:
            skipped.append({
                "reason": "missing_source_uri",
                "storage_item_uuid": item_uuid(candidate),
                "title": item_title(candidate),
            })
            continue
        if not has_video_extension(uri):
            skipped.append({
                "reason": "unsupported_or_missing_video_extension",
                "storage_item_uuid": item_uuid(candidate),
                "title": item_title(candidate),
                "source_uri": uri,
            })
            continue
        videos.append(VideoSource(
            source_uri=uri,
            title=item_title(candidate),
            storage_item_uuid=item_uuid(candidate),
            parent_kind=parent_kind,
            parent_id=parent_id,
            metadata=item_metadata(candidate),
        ))
    return videos, skipped


def discover_dataset_videos(client: Any, dataset_hash: str, default_source_bucket: str) -> DiscoveryResult:
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    backing_ids = [row.backing_item_uuid for row in data_rows if getattr(row, "backing_item_uuid", None)]
    storage_items = {item_uuid(item): item for item in client.get_storage_items(backing_ids)} if backing_ids else {}

    videos: list[VideoSource] = []
    skipped: list[dict[str, Any]] = []
    for row in data_rows:
        backing_id = str(getattr(row, "backing_item_uuid", "") or "")
        item = storage_items.get(backing_id)
        if item is None:
            skipped.append({
                "reason": "missing_storage_item",
                "data_hash": str(row.uid),
                "data_title": str(row.title),
                "backing_item_uuid": backing_id,
            })
            continue
        found, item_skipped = video_sources_from_storage_item(
            item=item,
            client=client,
            parent_kind="dataset",
            parent_id=str(row.uid),
            default_source_bucket=default_source_bucket,
        )
        videos.extend(found)
        skipped.extend({"data_hash": str(row.uid), "data_title": str(row.title), **entry} for entry in item_skipped)

    return DiscoveryResult(
        kind="dataset",
        title=str(getattr(dataset, "title", dataset_hash)),
        hash_value=dataset_hash,
        videos=videos,
        skipped=skipped,
    )


def discover_folder_videos(client: Any, folder_hash: str, default_source_bucket: str) -> DiscoveryResult:
    folder = client.get_storage_folder(folder_hash)
    videos: list[VideoSource] = []
    skipped: list[dict[str, Any]] = []
    for item in folder.list_items():
        found, item_skipped = video_sources_from_storage_item(
            item=item,
            client=client,
            parent_kind="folder",
            parent_id=folder_hash,
            default_source_bucket=default_source_bucket,
        )
        videos.extend(found)
        skipped.extend(item_skipped)

    return DiscoveryResult(
        kind="folder",
        title=str(getattr(folder, "name", folder_hash)),
        hash_value=folder_hash,
        videos=videos,
        skipped=skipped,
    )


def discover_videos(client: Any, folder_or_dataset_hash: str, default_source_bucket: str) -> DiscoveryResult:
    attempts: list[str] = []
    dataset_result: DiscoveryResult | None = None
    folder_result: DiscoveryResult | None = None

    try:
        dataset_result = discover_dataset_videos(client, folder_or_dataset_hash, default_source_bucket)
    except Exception as exc:
        attempts.append(f"dataset: {exc}")

    try:
        folder_result = discover_folder_videos(client, folder_or_dataset_hash, default_source_bucket)
    except Exception as exc:
        attempts.append(f"folder: {exc}")

    if dataset_result and dataset_result.videos:
        if folder_result and folder_result.videos:
            typer.echo("Hash resolved as both dataset and folder; using dataset result.")
        return dataset_result
    if folder_result and folder_result.videos:
        return folder_result
    if dataset_result:
        return dataset_result
    if folder_result:
        return folder_result
    raise typer.BadParameter(
        "Could not resolve hash as an Encord dataset or storage folder. "
        + " | ".join(attempts)
    )


def dedupe_and_limit(videos: list[VideoSource], limit: int | None) -> tuple[list[VideoSource], list[dict[str, Any]]]:
    selected: list[VideoSource] = []
    skipped: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for video in videos:
        identity = s3_identity(video.source_uri)
        if identity in seen_sources:
            skipped.append({
                "reason": "duplicate_source_uri",
                "source_uri": video.source_uri,
                "storage_item_uuid": video.storage_item_uuid,
            })
            continue
        seen_sources.add(identity)
        selected.append(video)
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped


def s3_cache_path(cache_root: Path, bucket: str, key: str) -> Path:
    key_parts = key.split("/")
    if not bucket or not key or any(part == ".." for part in key_parts):
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    cache_parts = [part for part in key_parts if part not in {"", "."}]
    if not cache_parts:
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    return cache_root / bucket / Path(*cache_parts)


def s3_client(unsigned: bool) -> Any:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if unsigned:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def download_s3_to_shared_cache(client_s3: Any, uri: str, cache_root: Path) -> CachedSource:
    bucket, key = parse_s3_uri(uri)
    cache_path = s3_cache_path(cache_root, bucket, key)
    if cache_path.exists():
        return CachedSource(cache_path, "shared-s3-cache", False)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    try:
        client_s3.download_file(bucket, key, str(tmp_path))
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return CachedSource(cache_path, "downloaded-to-shared-s3-cache", True)


def manifest_source_uri_candidates(item: dict[str, Any], default_source_bucket: str) -> list[str]:
    candidates: list[str] = []
    for key in ("source_uri", "source_s3_uri", "s3_uri", "objectUrl", "object_url"):
        value = item.get(key)
        if value:
            candidates.append(str(value))

    metadata = item.get("client_metadata") or {}
    if isinstance(metadata, dict):
        uri = source_uri_from_metadata(metadata, default_source_bucket)
        if uri:
            candidates.append(uri)

    return list(dict.fromkeys(candidates))


class LegacyExportVideoIndex:
    def __init__(self, export_root: Path, default_source_bucket: str):
        self.export_root = export_root
        self.default_source_bucket = default_source_bucket
        self._by_identity: dict[str, Path] | None = None

    def lookup(self, uri: str) -> Path | None:
        if self._by_identity is None:
            self._by_identity = self._build()
        return self._by_identity.get(s3_identity(uri))

    def _build(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not self.export_root.exists():
            return index

        run_dirs = [
            path for path in self.export_root.iterdir()
            if path.is_dir() and path.name != "_cache"
        ]
        for run_dir in sorted(run_dirs, reverse=True):
            items_path = run_dir / "dataset" / "meta" / "source_dataset_items.json"
            if not items_path.exists():
                continue
            try:
                items = json.loads(items_path.read_text())
            except Exception as exc:
                typer.echo(f"Warning: could not read legacy export manifest {items_path}: {exc}", err=True)
                continue
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                artifact_path = item.get("artifact_path")
                if not artifact_path:
                    continue
                local_path = run_dir / str(artifact_path)
                if not local_path.exists():
                    continue
                for candidate in manifest_source_uri_candidates(item, self.default_source_bucket):
                    try:
                        index.setdefault(s3_identity(candidate), local_path)
                    except ValueError:
                        continue
        return index


def resolve_cached_source(
    *,
    client_s3: Any,
    source_uri: str,
    shared_cache_root: Path,
    legacy_index: LegacyExportVideoIndex,
) -> CachedSource:
    bucket, key = parse_s3_uri(source_uri)
    shared_path = s3_cache_path(shared_cache_root, bucket, key)
    if shared_path.exists():
        return CachedSource(shared_path, "shared-s3-cache", False)

    legacy_path = legacy_index.lookup(source_uri)
    if legacy_path is not None:
        return CachedSource(legacy_path, "legacy-export-cache", False)

    return download_s3_to_shared_cache(client_s3, source_uri, shared_cache_root)


def ffmpeg_encoders(ffmpeg_bin: str) -> set[str]:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        line.split()[-1]
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith(" ")
    } | {
        token
        for line in result.stdout.splitlines()
        for token in line.split()
        if token.startswith("h264_") or token == "libx264"
    }


def choose_encoder(ffmpeg_bin: str, requested: str, allow_cpu_fallback: bool) -> str:
    if requested != "auto":
        return requested

    encoders = ffmpeg_encoders(ffmpeg_bin)
    for encoder in ("h264_videotoolbox", "h264_nvenc", "h264_amf"):
        if encoder in encoders:
            return encoder
    if allow_cpu_fallback and "libx264" in encoders:
        return "libx264"
    raise typer.BadParameter(
        "No supported GPU H.264 ffmpeg encoder found. "
        "Install ffmpeg with h264_videotoolbox, h264_nvenc, or h264_amf, "
        "or pass --allow-cpu-fallback."
    )


def encoder_args(encoder: str, video_bitrate: str) -> list[str]:
    if encoder == "h264_videotoolbox":
        return ["-c:v", encoder, "-b:v", video_bitrate, "-pix_fmt", "yuv420p"]
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p4", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_amf":
        return ["-c:v", encoder, "-quality", "quality", "-rc", "cqp", "-qp_i", "23", "-qp_p", "23", "-pix_fmt", "yuv420p"]
    if encoder == "libx264":
        return ["-c:v", encoder, "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]
    return ["-c:v", encoder, "-b:v", video_bitrate, "-pix_fmt", "yuv420p"]


def reencode_video(
    *,
    ffmpeg_bin: str,
    source_path: Path,
    output_path: Path,
    encoder: str,
    video_bitrate: str,
    hwaccel: bool,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp.mp4")
    command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y"]
    if hwaccel:
        command.extend(["-hwaccel", "auto"])
    command.extend(["-i", str(source_path), "-map", "0:v:0", "-an"])
    command.extend(encoder_args(encoder, video_bitrate))
    command.extend(["-movflags", "+faststart", "-f", "mp4", str(tmp_path)])
    try:
        subprocess.run(command, check=True)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def gcs_blob_name(prefix: str, source_uri: str) -> str:
    bucket, key = parse_s3_uri(source_uri)
    key_path = PurePosixPath(key).with_suffix(".mp4")
    clean_prefix = prefix.strip("/")
    path = PurePosixPath(clean_prefix) / bucket / key_path if clean_prefix else PurePosixPath(bucket) / key_path
    if any(part == ".." for part in path.parts):
        raise ValueError(f"Unsafe GCS blob path for {source_uri}")
    return path.as_posix()


def local_output_path(output_dir: Path, blob_name: str) -> Path:
    path = PurePosixPath(blob_name)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"Unsafe local output path for GCS blob: {blob_name}")
    return output_dir / "encoded" / Path(*path.parts)


def gcs_bucket(project: str, bucket_name: str) -> Any:
    from google.cloud import storage

    return storage.Client(project=project).bucket(bucket_name)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))


def process_videos(
    *,
    videos: list[VideoSource],
    output_dir: Path,
    shared_cache_root: Path,
    legacy_export_root: Path,
    default_source_bucket: str,
    unsigned_s3: bool,
    gcp_project: str,
    gcs_bucket_name: str,
    gcs_prefix: str,
    ffmpeg_bin: str,
    encoder: str,
    video_bitrate: str,
    allow_cpu_fallback: bool,
    hwaccel: bool,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    client_s3 = None if dry_run else s3_client(unsigned_s3)
    legacy_index = None if dry_run else LegacyExportVideoIndex(legacy_export_root, default_source_bucket)
    selected_encoder = "dry-run" if dry_run else choose_encoder(
        ffmpeg_bin,
        encoder,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    bucket = None if dry_run else gcs_bucket(gcp_project, gcs_bucket_name)

    rows: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    with tqdm(videos, desc="Re-encoding videos", unit="video", dynamic_ncols=True, mininterval=5.0) as progress:
        for video in progress:
            row = {
                "source_uri": video.source_uri,
                "title": video.title,
                "storage_item_uuid": video.storage_item_uuid,
                "parent_kind": video.parent_kind,
                "parent_id": video.parent_id,
            }
            try:
                blob_name = gcs_blob_name(gcs_prefix, video.source_uri)
                row["gcs_uri"] = f"gs://{gcs_bucket_name}/{blob_name}"
                row["encoder"] = selected_encoder

                if blob_name in seen_targets:
                    rows.append({**row, "status": "skipped", "reason": "duplicate_target_blob"})
                    continue
                seen_targets.add(blob_name)

                blob = None if bucket is None else bucket.blob(blob_name)
                if blob is not None and blob.exists() and not overwrite:
                    rows.append({**row, "status": "skipped", "reason": "target_exists"})
                    continue

                if dry_run:
                    rows.append({**row, "status": "dry_run"})
                    continue

                cached = resolve_cached_source(
                    client_s3=client_s3,
                    source_uri=video.source_uri,
                    shared_cache_root=shared_cache_root,
                    legacy_index=legacy_index,
                )
                encoded_path = local_output_path(output_dir, blob_name)
                reencode_video(
                    ffmpeg_bin=ffmpeg_bin,
                    source_path=cached.path,
                    output_path=encoded_path,
                    encoder=selected_encoder,
                    video_bitrate=video_bitrate,
                    hwaccel=hwaccel,
                    overwrite=overwrite,
                )
                blob.upload_from_filename(str(encoded_path), content_type="video/mp4")
                rows.append({
                    **row,
                    "status": "uploaded",
                    "cache_kind": cached.cache_kind,
                    "downloaded": cached.downloaded,
                    "local_source_path": str(cached.path),
                    "local_encoded_path": str(encoded_path),
                })
            except Exception as exc:
                rows.append({**row, "status": "failed", "error": str(exc)})
    return rows


def summarize(
    rows: list[dict[str, Any]],
    discovery: DiscoveryResult,
    output_dir: Path,
    selected_video_count: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return {
        "input_kind": discovery.kind,
        "input_hash": discovery.hash_value,
        "input_title": discovery.title,
        "discovered_videos": len(discovery.videos),
        "selected_videos": selected_video_count,
        "manifest_rows": len(rows),
        "skipped_during_discovery": len(discovery.skipped),
        "status_counts": counts,
        "output_dir": str(output_dir),
    }


def main(
    folder_or_dataset_hash: Annotated[str, typer.Argument(help="Encord storage folder hash or dataset hash.")],
    limit: Annotated[int | None, typer.Option(help="Maximum number of unique source videos to process.")] = None,
    gcp_project: Annotated[str, typer.Option(help="GCP project for the output bucket.")] = DEFAULT_GCP_PROJECT,
    gcs_bucket_name: Annotated[str, typer.Option("--gcs-bucket", help="GCS bucket for re-encoded videos.")] = DEFAULT_GCS_BUCKET,
    gcs_prefix: Annotated[str, typer.Option(help="GCS prefix for re-encoded videos.")] = DEFAULT_GCS_PREFIX,
    shared_s3_cache_root: Annotated[
        Path,
        typer.Option(help="Shared S3 cache root mirroring s3://bucket/key."),
    ] = DEFAULT_SHARED_S3_CACHE_ROOT,
    legacy_export_root: Annotated[
        Path,
        typer.Option(help="Dataset export root to search for old per-run cached videos."),
    ] = DEFAULT_DATASET_EXPORT_ROOT,
    output_root: Annotated[Path, typer.Option(help="Local run output root for manifests and encoded files.")] = DEFAULT_OUTPUT_ROOT,
    default_source_bucket: Annotated[
        str,
        typer.Option(help="Bucket to use when Encord metadata only has source_key."),
    ] = DEFAULT_SOURCE_BUCKET,
    ffmpeg_bin: Annotated[str, typer.Option(help="ffmpeg binary path/name.")] = "ffmpeg",
    encoder: Annotated[
        str,
        typer.Option(help="H.264 encoder to use, or auto for h264_videotoolbox/h264_nvenc/h264_amf."),
    ] = "auto",
    video_bitrate: Annotated[str, typer.Option(help="Bitrate for bitrate-based GPU encoders.")] = "5M",
    allow_cpu_fallback: Annotated[
        bool,
        typer.Option(help="Allow libx264 if no supported GPU H.264 encoder is available."),
    ] = False,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing local outputs and GCS blobs.")] = False,
    unsigned_s3: Annotated[bool, typer.Option(help="Use unsigned S3 requests for public source buckets.")] = False,
    hwaccel: Annotated[bool, typer.Option(help="Pass -hwaccel auto to ffmpeg.")] = True,
    dry_run: Annotated[bool, typer.Option(help="Resolve and dedupe videos without downloading, encoding, or uploading.")] = False,
) -> None:
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be positive when provided.")

    output_dir = timestamped_output_dir(output_root)
    typer.echo(f"Writing run manifest to {output_dir}")

    client = create_client()
    discovery = discover_videos(client, folder_or_dataset_hash, default_source_bucket)
    typer.echo(
        f"Resolved {folder_or_dataset_hash} as {discovery.kind} "
        f"{discovery.title!r}; found {len(discovery.videos)} video source(s)."
    )
    if discovery.skipped:
        typer.echo(f"Skipped {len(discovery.skipped)} item(s) during discovery.")

    selected, duplicate_skips = dedupe_and_limit(discovery.videos, limit)
    typer.echo(f"Selected {len(selected)} unique source video(s).")

    rows = [
        {
            "status": "skipped",
            **entry,
        }
        for entry in discovery.skipped + duplicate_skips
    ]
    if selected:
        rows.extend(process_videos(
            videos=selected,
            output_dir=output_dir,
            shared_cache_root=shared_s3_cache_root,
            legacy_export_root=legacy_export_root,
            default_source_bucket=default_source_bucket,
            unsigned_s3=unsigned_s3,
            gcp_project=gcp_project,
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix=gcs_prefix,
            ffmpeg_bin=ffmpeg_bin,
            encoder=encoder,
            video_bitrate=video_bitrate,
            allow_cpu_fallback=allow_cpu_fallback,
            hwaccel=hwaccel,
            overwrite=overwrite,
            dry_run=dry_run,
        ))
    else:
        typer.echo("No unique source videos selected; skipping encode/upload.")

    write_jsonl(output_dir / "manifest.jsonl", rows)
    summary = summarize(rows, discovery, output_dir, len(selected))
    write_json(output_dir / "summary.json", summary)
    typer.echo(f"summary: {summary['status_counts']}")
    typer.echo(f"manifest: {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    typer.run(main)
