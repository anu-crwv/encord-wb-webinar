# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "encord==0.1.199",
#     "typer",
# ]
# ///
"""Resolve the webinar's Encord data groups into stable S3 references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import typer

CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_TO_FEATURE = {
    "cam_high": "exterior_image_1_left",
    "cam_left_wrist": "wrist_image_left",
    "cam_right_wrist": "wrist_image_right",
}
EPISODE_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")
SOURCE_URI_KEYS = ("source_uri", "s3_uri", "source_s3_uri")
PARQUET_URI_KEYS = ("source_parquet_uri", "parquet_uri")


@dataclass(frozen=True)
class VideoReference:
    camera_name: str
    source_uri: str
    artifact_path: str
    storage_item_uuid: str


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    data_hash: str
    data_title: str
    group_uuid: str
    label_hash: str | None
    caption: str
    episode_path: str
    parquet_uri: str
    info_uri: str
    videos: tuple[VideoReference, ...]


def create_encord_client(ssh_key_file: str, domain: str | None = None) -> Any:
    """Create the public Encord SDK client from a private-key file path."""
    from encord import EncordUserClient

    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def item_metadata(item: Any) -> dict[str, Any]:
    value = getattr(item, "client_metadata", None) or {}
    return dict(value) if isinstance(value, dict) else {}


def metadata_uri(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return bucket/key for canonical or virtual-hosted S3 object URLs."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3" and parsed.netloc and parsed.path.lstrip("/"):
        return parsed.netloc, unquote(parsed.path.lstrip("/"))
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        key = unquote(parsed.path.lstrip("/"))
        if bucket and key:
            return bucket, key
    raise typer.BadParameter(f"Expected an S3 object URI, got: {uri}")


def canonical_s3_uri(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"s3://{bucket}/{key}"


def episode_path_from_value(value: Any) -> str | None:
    text = str(value or "")
    if "://" in text:
        text = unquote(urlparse(text).path.lstrip("/"))
    parts = [part for part in text.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_RE.fullmatch(part):
            return "/".join(parts[: index + 1]) + "/"
    return None


def episode_path_for_item(item: Any) -> str | None:
    metadata = item_metadata(item)
    if metadata.get("episode_path"):
        candidate = str(metadata["episode_path"]).strip("/") + "/"
        if episode_path_from_value(candidate):
            return candidate
    for key in (*SOURCE_URI_KEYS, "source_key"):
        path = episode_path_from_value(metadata.get(key))
        if path:
            return path
    return episode_path_from_value(getattr(item, "name", None))


def source_uri_for_item(item: Any) -> str:
    uri = metadata_uri(item_metadata(item), SOURCE_URI_KEYS)
    if not uri:
        raise typer.BadParameter(
            f"Storage item {getattr(item, 'uuid', 'unknown')} lacks source_uri metadata"
        )
    return canonical_s3_uri(uri)


def source_parquet_uri(item: Any, video_uri: str, episode_path: str) -> str:
    direct = metadata_uri(item_metadata(item), PARQUET_URI_KEYS)
    if direct:
        return canonical_s3_uri(direct)
    bucket, _ = parse_s3_uri(video_uri)
    episode_id = PurePosixPath(episode_path.rstrip("/")).name
    return f"s3://{bucket}/{episode_path.rstrip('/')}/data/chunk-000/{episode_id}.parquet"


def source_info_uri(video_uri: str, episode_path: str) -> str:
    bucket, _ = parse_s3_uri(video_uri)
    return f"s3://{bucket}/{episode_path.rstrip('/')}/meta/info.json"


def group_children(item: Any, client: Any) -> list[Any]:
    """Load data-group children, including UUID-only layout entries."""
    children = list(item.get_child_items())
    by_uuid = {str(child.uuid): child for child in children}
    try:
        summary = item.get_summary()
    except (AttributeError, TypeError):
        return list(by_uuid.values())
    layout = getattr(getattr(summary, "data_group", None), "layout_contents", {}) or {}
    missing = [child.uuid for child in layout.values() if str(child.uuid) not in by_uuid]
    if missing:
        for child in client.get_storage_items(missing):
            by_uuid[str(child.uuid)] = child
    return list(by_uuid.values())


def camera_name(item: Any) -> str | None:
    value = item_metadata(item).get("camera_name")
    return str(value) if value else None


def video_children_by_camera(group_item: Any, client: Any) -> dict[str, Any]:
    videos: dict[str, Any] = {}
    for child in group_children(group_item, client):
        name = camera_name(child)
        if name in CAMERA_ORDER:
            if name in videos:
                raise typer.BadParameter(f"Data group {group_item.uuid} has duplicate camera {name}")
            videos[name] = child
    missing = [name for name in CAMERA_ORDER if name not in videos]
    if missing:
        raise typer.BadParameter(f"Data group {group_item.uuid} is missing cameras: {missing}")
    return videos


def video_artifact_path(episode_index: int, camera: str) -> str:
    feature = CAMERA_TO_FEATURE[camera]
    chunk = episode_index // 1000
    return (
        f"dataset/videos/chunk-{chunk:03d}/observation.images.{feature}/"
        f"episode_{episode_index:06d}.mp4"
    )
