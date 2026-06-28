# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Backfill the shared S3 video cache from a partial dataset export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import UUID

import typer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
EXPORT_ROOT = REPO_ROOT / "exports/encord-dataset-export"
S3_CACHE_ROOT = EXPORT_ROOT / "_cache" / "s3"
CHUNK_SIZE = 1000
CAMERA_ORDER = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CAMERA_TO_DROID_KEY = {
    "cam_high": "exterior_image_1_left",
    "cam_left_wrist": "wrist_image_left",
    "cam_right_wrist": "wrist_image_right",
}


def client_from_env():
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def item_metadata(item: Any) -> dict[str, Any]:
    return getattr(item, "client_metadata", None) or {}


def source_uri(item: Any) -> str:
    metadata = item_metadata(item)
    uri = metadata.get("source_uri") or metadata.get("s3_uri") or metadata.get("source_s3_uri")
    if uri:
        return str(uri)
    source_key = metadata.get("source_key")
    if source_key:
        return f"s3://ego-data-collection-encord/{source_key}"
    raise ValueError(f"No S3 source URI found for item {item.uuid} ({item.name})")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        return bucket, unquote(parsed.path.lstrip("/"))
    raise ValueError(f"Unsupported S3 URI format: {uri}")


def s3_cache_path(bucket: str, key: str) -> Path:
    key_parts = key.split("/")
    if not bucket or not key or any(part == ".." for part in key_parts):
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    cache_parts = [part for part in key_parts if part not in {"", "."}]
    if not cache_parts:
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    return S3_CACHE_ROOT / bucket / Path(*cache_parts)


def group_children(item: Any, client: Any) -> list[Any]:
    children_by_uuid = {str(child.uuid): child for child in item.get_child_items()}

    try:
        summary = item.get_summary()
    except Exception:
        return list(children_by_uuid.values())

    if summary.data_group is not None:
        child_uuids = [
            child.uuid
            for child in summary.data_group.layout_contents.values()
            if str(child.uuid) not in children_by_uuid
        ]
        if child_uuids:
            for child in client.get_storage_items(child_uuids):
                children_by_uuid[str(child.uuid)] = child

    return list(children_by_uuid.values())


def video_children_by_camera(group_item: Any, client: Any) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    videos = {}
    for child in group_children(group_item, client):
        if child.item_type != StorageItemType.VIDEO:
            continue
        camera_name = item_metadata(child).get("camera_name")
        if camera_name:
            videos[str(camera_name)] = child
    return videos


def lerobot_video_path(episode_index: int, camera_name: str) -> Path:
    chunk = episode_index // CHUNK_SIZE
    video_key = f"observation.images.{CAMERA_TO_DROID_KEY[camera_name]}"
    return Path("dataset") / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def cache_local_video(local_path: Path, cache_path: Path, dry_run: bool) -> str:
    if cache_path.exists():
        if cache_path.stat().st_size == local_path.stat().st_size:
            return "already_cached"
        return "size_conflict"
    if dry_run:
        return "would_cache"
    return link_or_copy(local_path, cache_path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main(
    dataset_hash: Annotated[UUID, typer.Option(help="Encord dataset hash used by the failed export.")],
    export_dir: Annotated[Path, typer.Option(help="Partial export directory containing dataset/videos.")],
    first_episode_index: Annotated[
        int,
        typer.Option(help="First local episode index in the partial export."),
    ] = 0,
    dry_run: Annotated[bool, typer.Option(help="Preview cache actions without writing files.")] = True,
    progress_every: Annotated[int, typer.Option(help="Print progress every N data groups.")] = 100,
) -> None:
    export_dir = export_dir.expanduser().resolve()
    dataset_dir = export_dir / "dataset"
    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        raise typer.BadParameter(f"Export videos directory does not exist: {videos_dir}")

    client = client_from_env()
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    typer.echo(f"Found {len(data_rows)} Encord data groups in dataset {dataset_hash}.")
    typer.echo(f"Backfilling from {export_dir}")
    typer.echo(f"Shared S3 video cache: {S3_CACHE_ROOT}")

    backing_ids = [row.backing_item_uuid for row in data_rows if getattr(row, "backing_item_uuid", None)]
    typer.echo(f"Resolving {len(backing_ids)} backing storage items...")
    group_items_by_uuid = {
        str(item.uuid): item for item in client.get_storage_items(backing_ids)
    } if backing_ids else {}

    actions: dict[str, int] = {
        "would_cache": 0,
        "linked": 0,
        "copied": 0,
        "already_cached": 0,
        "size_conflict": 0,
        "missing_local": 0,
        "missing_group": 0,
        "missing_camera": 0,
        "missing_source_uri": 0,
        "empty_local": 0,
    }
    cached_items = []
    skipped_items = []

    for offset, row in enumerate(data_rows):
        if offset and offset % progress_every == 0:
            typer.echo(f"Checked {offset}/{len(data_rows)} data groups...")

        episode_index = first_episode_index + offset
        group_item = group_items_by_uuid.get(str(row.backing_item_uuid))
        if group_item is None:
            actions["missing_group"] += len(CAMERA_ORDER)
            skipped_items.append({
                "episode_index": episode_index,
                "data_hash": str(row.uid),
                "reason": "missing_group",
            })
            continue

        videos = video_children_by_camera(group_item, client)
        for camera_name in CAMERA_ORDER:
            relative_path = lerobot_video_path(episode_index, camera_name)
            local_path = export_dir / relative_path
            if not local_path.exists():
                actions["missing_local"] += 1
                continue
            if local_path.stat().st_size == 0:
                actions["empty_local"] += 1
                skipped_items.append({
                    "episode_index": episode_index,
                    "camera_name": camera_name,
                    "artifact_path": relative_path.as_posix(),
                    "reason": "empty_local",
                })
                continue

            item = videos.get(camera_name)
            if item is None:
                actions["missing_camera"] += 1
                skipped_items.append({
                    "episode_index": episode_index,
                    "camera_name": camera_name,
                    "artifact_path": relative_path.as_posix(),
                    "data_hash": str(row.uid),
                    "data_group_uuid": str(group_item.uuid),
                    "reason": "missing_camera",
                })
                continue

            try:
                uri = source_uri(item)
                bucket, key = parse_s3_uri(uri)
            except ValueError as exc:
                actions["missing_source_uri"] += 1
                skipped_items.append({
                    "episode_index": episode_index,
                    "camera_name": camera_name,
                    "artifact_path": relative_path.as_posix(),
                    "data_hash": str(row.uid),
                    "data_group_uuid": str(group_item.uuid),
                    "video_storage_item_uuid": str(item.uuid),
                    "reason": "missing_source_uri",
                    "error": str(exc),
                })
                continue

            cache_path = s3_cache_path(bucket, key)
            action = cache_local_video(local_path, cache_path, dry_run=dry_run)
            actions[action] += 1
            record = {
                "episode_index": episode_index,
                "data_hash": str(row.uid),
                "data_group_uuid": str(group_item.uuid),
                "video_storage_item_uuid": str(item.uuid),
                "camera_name": camera_name,
                "artifact_path": relative_path.as_posix(),
                "source_uri": uri,
                "cache_path": str(cache_path),
                "local_path": str(local_path),
                "local_size": local_path.stat().st_size,
                "action": action,
            }
            if action == "size_conflict":
                record["cache_size"] = cache_path.stat().st_size
                skipped_items.append({**record, "reason": action})
            else:
                cached_items.append(record)

    summary = {
        "backfilled_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "dataset_hash": str(dataset_hash),
        "dataset_title": dataset.title,
        "export_dir": str(export_dir),
        "s3_cache_root": str(S3_CACHE_ROOT),
        "first_episode_index": first_episode_index,
        "action_counts": actions,
        "cached_item_count": len(cached_items),
        "skipped_item_count": len(skipped_items),
        "cached_items": cached_items,
        "skipped_items": skipped_items,
    }

    typer.echo(json.dumps({
        "dry_run": dry_run,
        "action_counts": actions,
        "cached_item_count": len(cached_items),
        "skipped_item_count": len(skipped_items),
    }, indent=2))

    if not dry_run:
        manifest_path = export_dir / "cache_backfill_manifest.json"
        write_json(manifest_path, summary)
        typer.echo(f"Wrote {manifest_path}")


if __name__ == "__main__":
    typer.run(main)
