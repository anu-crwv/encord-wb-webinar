# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create an Encord data-group dataset from a single-video Encord dataset."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import os
import re
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import UUID

import typer


DEFAULT_GROUPS_FOLDER = UUID("fae47d1a-0c23-4332-9ab1-9e37e8e44b06")
EPISODE_DIR_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")
EPISODE_BASE_RE = re.compile(r"^(episode_\d+)(?:_[A-Za-z0-9]+)?$")


def create_client():
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def item_metadata(item: Any) -> dict[str, Any]:
    return getattr(item, "client_metadata", None) or {}


def metadata_hint(metadata: dict[str, Any]) -> str:
    interesting = {
        key: metadata.get(key)
        for key in ["episode_path", "episode_id", "source_key", "source_uri", "camera_name", "sensor_key", "file_ext"]
        if key in metadata
    }
    return f"keys={sorted(metadata.keys())[:20]} interesting={interesting}"


def normalize_source_path(value: Any) -> str:
    path = str(value or "")
    if path.startswith("s3://"):
        return urlparse(path).path.lstrip("/")
    if path.startswith("http://") or path.startswith("https://"):
        return urlparse(path).path.lstrip("/")
    return path.lstrip("/")


def episode_path_from_source(value: Any) -> str | None:
    parts = [part for part in normalize_source_path(value).split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_DIR_RE.match(part):
            return "/".join(parts[: index + 1]) + "/"
    return None


def normalized_episode_path(episode_path: str) -> str:
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    if not parts:
        return episode_path
    match = EPISODE_BASE_RE.match(parts[-1])
    if match:
        parts[-1] = match.group(1)
    return "/".join(parts) + "/"


def episode_id_from_path(episode_path: str) -> str | None:
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    if not parts:
        return None
    match = EPISODE_BASE_RE.match(parts[-1])
    return match.group(1) if match else None


def episode_path_from_metadata(metadata: dict[str, Any], fallback_name: Any = None) -> str | None:
    episode_path = metadata.get("episode_path")
    if episode_path:
        return str(episode_path)

    for key in ["source_key", "source_uri", "s3_uri", "source_s3_uri", "objectUrl", "object_url"]:
        derived = episode_path_from_source(metadata.get(key))
        if derived:
            return derived

    return episode_path_from_source(fallback_name)


def episode_keys_from_path(episode_path: str) -> list[str]:
    keys = [episode_path, normalized_episode_path(episode_path)]
    episode_id = episode_id_from_path(episode_path)
    if episode_id:
        keys.append(episode_id)
    return list(dict.fromkeys(keys))


def episode_keys_from_metadata(metadata: dict[str, Any], fallback_name: Any = None) -> list[str]:
    keys = []
    episode_path = episode_path_from_metadata(metadata, fallback_name)
    if episode_path:
        keys.extend(episode_keys_from_path(episode_path))
    if metadata.get("episode_id"):
        keys.append(str(metadata["episode_id"]))
    return list(dict.fromkeys(keys))


def episode_path_from_item(item: Any) -> str | None:
    return episode_path_from_metadata(item_metadata(item), getattr(item, "name", None))


def episode_keys_from_item(item: Any, client: Any | None = None, debug: bool = False) -> list[str]:
    metadata = item_metadata(item)
    keys = episode_keys_from_metadata(metadata, getattr(item, "name", None))
    if keys:
        if debug:
            typer.echo(f"    keys from item metadata/source path: {keys}")
        return keys

    child_items = list(item.get_child_items())
    if debug:
        typer.echo(f"    get_child_items returned {len(child_items)} children.")
    for child in child_items:
        child_keys = episode_keys_from_metadata(item_metadata(child), getattr(child, "name", None))
        if debug:
            typer.echo(f"    child via get_child_items: {child.uuid} | {child.item_type} | {child.name}")
            typer.echo(f"      {metadata_hint(item_metadata(child))}")
            typer.echo(f"      keys={child_keys}")
        if child_keys:
            return child_keys

    if client is not None:
        layout_children = group_layout_children(item, client, debug=debug)
        if debug:
            typer.echo(f"    group layout resolved {len(layout_children)} children.")
        for child in layout_children:
            child_keys = episode_keys_from_metadata(item_metadata(child), getattr(child, "name", None))
            if debug:
                typer.echo(f"    child via layout: {child.uuid} | {child.item_type} | {child.name}")
                typer.echo(f"      {metadata_hint(item_metadata(child))}")
                typer.echo(f"      keys={child_keys}")
            if child_keys:
                return child_keys

    return []


def group_layout_children(item: Any, client: Any, debug: bool = False) -> list[Any]:
    try:
        data_group = item.get_summary().data_group
    except Exception as exc:
        if debug:
            typer.echo(f"    get_summary failed: {type(exc).__name__}: {exc}")
        return []
    if data_group is None:
        if debug:
            typer.echo("    summary has no data_group layout.")
        return []

    child_uuids = [child.uuid for child in data_group.layout_contents.values()]
    if debug:
        typer.echo(f"    layout child UUIDs: {[str(child_uuid) for child_uuid in child_uuids]}")
    if not child_uuids:
        return []
    try:
        return client.get_storage_items(child_uuids)
    except Exception as exc:
        if debug:
            typer.echo(f"    get_storage_items failed: {type(exc).__name__}: {exc}")
        return []


def load_group_by_episode(folder: Any, client: Any, debug: bool = False, debug_limit: int = 5) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    groups_by_episode = {}
    collisions: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    typer.echo(f"Scanning data-group folder {folder.uuid}...")
    for group_item in folder.list_items(page_size=1000, item_types=[StorageItemType.GROUP]):
        scanned += 1
        item_debug = debug and scanned <= debug_limit
        if item_debug:
            typer.echo("")
            typer.echo(f"  Debug group {scanned}: {group_item.uuid} | {group_item.item_type} | {group_item.name}")
            typer.echo(f"    group metadata: {metadata_hint(item_metadata(group_item))}")
        keys = episode_keys_from_item(group_item, client, debug=item_debug)
        if item_debug:
            typer.echo(f"    resolved keys: {keys}")
        for key in keys:
            existing_group = groups_by_episode.get(key)
            if existing_group is not None and existing_group.uuid != group_item.uuid:
                collisions[key].append(str(group_item.uuid))
                groups_by_episode.pop(key, None)
                continue
            groups_by_episode[key] = group_item
    typer.echo(f"Scanned {scanned} group items.")
    typer.echo(f"Found {len(groups_by_episode)} data-group match keys.")
    if collisions:
        typer.echo(f"Skipped {sum(len(value) for value in collisions.values())} duplicate ambiguous group keys.")
    return groups_by_episode


def default_dataset_title(source_title: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{source_title} - grouped videos - {timestamp}"


def main(
    source_dataset_hash: Annotated[UUID, typer.Option(help="Encord dataset hash containing single-video rows.")],
    group_folder_id: Annotated[
        UUID,
        typer.Option(help="Storage folder containing existing 3-video data groups."),
    ] = DEFAULT_GROUPS_FOLDER,
    output_dataset_title: Annotated[str | None, typer.Option(help="Title for the created Encord dataset.")] = None,
    limit: Annotated[int | None, typer.Option(help="Optional max number of matched data groups to link.")] = None,
    debug: Annotated[bool, typer.Option(help="Print match keys for the first few source and group items.")] = False,
    debug_limit: Annotated[int, typer.Option(help="Number of source/group items to print when --debug is set.")] = 5,
) -> None:
    from encord.orm.dataset import StorageLocation

    client = create_client()
    source_dataset = client.get_dataset(source_dataset_hash)
    group_folder = client.get_storage_folder(group_folder_id)

    groups_by_episode = load_group_by_episode(group_folder, client, debug=debug, debug_limit=debug_limit)
    data_rows = list(source_dataset.data_rows)
    source_items = client.get_storage_items([row.backing_item_uuid for row in data_rows])

    matched_group_uuids = []
    seen_source_keys = set()
    matched_group_uuid_set = set()
    missing_keys: list[list[str]] = []
    missing = 0
    for index, item in enumerate(source_items, start=1):
        item_debug = debug and index <= debug_limit
        if item_debug:
            typer.echo("")
            typer.echo(f"  Debug source {index}: {item.uuid} | {item.item_type} | {item.name}")
            typer.echo(f"    metadata: {metadata_hint(item_metadata(item))}")
        keys = episode_keys_from_item(item, debug=item_debug)
        if item_debug:
            typer.echo(f"    resolved keys: {keys}")
        source_key = keys[0] if keys else str(item.uuid)
        if not keys or source_key in seen_source_keys:
            continue
        seen_source_keys.add(source_key)
        group_item = next((groups_by_episode[key] for key in keys if key in groups_by_episode), None)
        if group_item is None:
            missing += 1
            if len(missing_keys) < debug_limit:
                missing_keys.append(keys)
            continue
        if group_item.uuid in matched_group_uuid_set:
            continue
        matched_group_uuid_set.add(group_item.uuid)
        matched_group_uuids.append(group_item.uuid)
        if limit is not None and len(matched_group_uuids) >= limit:
            break

    typer.echo(f"Matched {len(matched_group_uuids)} data groups from {len(seen_source_keys)} source episodes.")
    if missing:
        typer.echo(f"Missing group matches for {missing} source episodes.")
        if missing_keys:
            typer.echo(f"First missing keys: {missing_keys}")
    if not matched_group_uuids:
        typer.echo("No matching data groups found. No dataset created.", err=True)
        raise typer.Exit(1)

    title = output_dataset_title or default_dataset_title(source_dataset.title)
    typer.echo(f"Creating dataset: {title}")
    response = client.create_dataset(
        dataset_title=title,
        dataset_type=StorageLocation.CORD_STORAGE,
        create_backing_folder=False,
    )
    output_dataset = client.get_dataset(str(response.dataset_hash))
    output_dataset.link_items(matched_group_uuids)

    typer.echo(f"Created dataset: {response.dataset_hash}")
    typer.echo(f"Linked data groups: {len(matched_group_uuids)}")


if __name__ == "__main__":
    typer.run(main)
