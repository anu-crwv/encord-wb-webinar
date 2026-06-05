# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Try nested carousel data groups by matching Encord client metadata."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import UUID

import typer


DEFAULT_ALL_DATA_FOLDER = UUID("cdb6587a-d00b-4446-a3a9-16d2b8babbda")
DEFAULT_GROUPS_FOLDER = UUID("fae47d1a-0c23-4332-9ab1-9e37e8e44b06")
EPISODE_DIR_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")


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
    keys = sorted(metadata.keys())
    interesting = {
        key: metadata.get(key)
        for key in ["episode_path", "episode_id", "source_key", "source_uri", "camera_name", "sensor_key", "file_ext"]
        if key in metadata
    }
    return f"keys={keys[:20]} interesting={interesting}"


def normalize_source_path(value: Any) -> str:
    path = str(value or "")
    if path.startswith("s3://"):
        parsed = urlparse(path)
        return parsed.path.lstrip("/")
    if path.startswith("http://") or path.startswith("https://"):
        parsed = urlparse(path)
        return parsed.path.lstrip("/")
    return path.lstrip("/")


def derive_episode_path_from_source(value: Any) -> str | None:
    source_path = normalize_source_path(value)
    parts = [part for part in source_path.split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_DIR_RE.match(part):
            return "/".join(parts[: index + 1]) + "/"
    return None


def episode_path_from_metadata(metadata: dict[str, Any], fallback_name: Any = None) -> str | None:
    episode_path = metadata.get("episode_path")
    if episode_path:
        return str(episode_path)

    for key in ["source_key", "source_uri", "objectUrl", "object_url"]:
        derived = derive_episode_path_from_source(metadata.get(key))
        if derived:
            return derived

    return derive_episode_path_from_source(fallback_name)


def episode_path_from_item(item: Any, client: Any | None = None, debug: bool = False) -> str | None:
    metadata = item_metadata(item)
    episode_path = episode_path_from_metadata(metadata, getattr(item, "name", None))
    if episode_path:
        if debug:
            typer.echo(f"    episode_path from item metadata/source path: {episode_path}")
        return str(episode_path)

    child_items = list(item.get_child_items())
    if debug:
        typer.echo(f"    get_child_items returned {len(child_items)} children.")
    for child in child_items:
        child_episode_path = episode_path_from_metadata(item_metadata(child), getattr(child, "name", None))
        if debug:
            typer.echo(f"    child via get_child_items: {child.uuid} | {child.item_type} | {child.name}")
            typer.echo(f"      {metadata_hint(item_metadata(child))}")
            typer.echo(f"      derived_episode_path={child_episode_path}")
        if child_episode_path:
            return str(child_episode_path)

    if client is not None:
        layout_children = group_layout_children(item, client, debug=debug)
        if debug:
            typer.echo(f"    group layout resolved {len(layout_children)} children.")
        for child in layout_children:
            child_episode_path = episode_path_from_metadata(item_metadata(child), getattr(child, "name", None))
            if debug:
                typer.echo(f"    child via layout: {child.uuid} | {child.item_type} | {child.name}")
                typer.echo(f"      {metadata_hint(item_metadata(child))}")
                typer.echo(f"      derived_episode_path={child_episode_path}")
            if child_episode_path:
                return str(child_episode_path)

    return None


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


def is_json_metadata_item(item: Any) -> bool:
    metadata = item_metadata(item)
    title = str(getattr(item, "name", "") or "")
    source_key = str(metadata.get("source_key") or title)
    file_ext = str(metadata.get("file_ext") or Path(source_key).suffix).lower()
    role = str(metadata.get("metadata_file_role") or "none")
    return file_ext in {".json", ".jsonl"} or role != "none"


def load_json_items_by_episode(folder: Any) -> dict[str, list[Any]]:
    from encord.orm.storage import StorageItemType

    by_episode: dict[str, list[Any]] = defaultdict(list)
    typer.echo(f"Scanning all-data folder {folder.uuid} for JSON metadata items...")
    for item in folder.list_items(page_size=1000, item_types=[StorageItemType.PLAIN_TEXT]):
        episode_path = episode_path_from_item(item)
        if episode_path and is_json_metadata_item(item):
            by_episode[episode_path].append(item)
    typer.echo(f"Found JSON metadata for {len(by_episode)} episodes.")
    return by_episode


def load_group_items(folder: Any, client: Any, debug: bool = False, debug_limit: int = 5) -> list[tuple[str, Any]]:
    from encord.orm.storage import StorageItemType

    groups = []
    scanned = 0
    typer.echo(f"Scanning groups folder {folder.uuid} for existing video groups...")
    for item in folder.list_items(page_size=1000, item_types=[StorageItemType.GROUP]):
        scanned += 1
        item_debug = debug and scanned <= debug_limit
        if item_debug:
            typer.echo("")
            typer.echo(f"  Debug group {scanned}: {item.uuid} | {item.item_type} | {item.name}")
            typer.echo(f"    group metadata: {metadata_hint(item_metadata(item))}")
        episode_path = episode_path_from_item(item, client, debug=item_debug)
        if item_debug:
            typer.echo(f"    resolved episode_path: {episode_path}")
        if episode_path:
            groups.append((episode_path, item))
    typer.echo(f"Scanned {scanned} group items.")
    typer.echo(f"Found {len(groups)} groups with episode_path from group or child file metadata.")
    return groups


def group_name(episode_path: str) -> str:
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    return f"nested-carousel-{parts[-1] if parts else episode_path}"


def default_output_folder_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"nested-carousel-output-{timestamp}"


def main(
    all_data_folder_id: Annotated[
        UUID,
        typer.Option(help="Folder containing the ungrouped videos and JSON/text metadata items."),
    ] = DEFAULT_ALL_DATA_FOLDER,
    groups_folder_id: Annotated[
        UUID,
        typer.Option(help="Folder containing the existing grouped 3-camera video data groups."),
    ] = DEFAULT_GROUPS_FOLDER,
    limit: Annotated[int, typer.Option(help="Max matched groups to create unless --no-limit is passed.")] = 5,
    no_limit: Annotated[
        bool,
        typer.Option("--no-limit", help="Create nested carousel groups for every match."),
    ] = False,
    output_folder_name: Annotated[
        str | None,
        typer.Option(help="Name for the new output folder. Defaults to nested-carousel-output-<timestamp>."),
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Print capped diagnostics while matching group metadata.")] = False,
    debug_limit: Annotated[int, typer.Option(help="Number of group items to inspect in --debug output.")] = 5,
    dataset_hash: Annotated[UUID | None, typer.Option(help="Optional dataset to link created groups into.")] = None,
) -> None:
    from encord.orm.storage import DataGroupCarousel

    client = create_client()
    all_data_folder = client.get_storage_folder(all_data_folder_id)
    groups_folder = client.get_storage_folder(groups_folder_id)

    json_by_episode = load_json_items_by_episode(all_data_folder)
    group_items = load_group_items(groups_folder, client, debug=debug, debug_limit=debug_limit)

    matches = []
    for episode_path, group_item in group_items:
        json_items = json_by_episode.get(episode_path, [])
        if json_items:
            matches.append((episode_path, group_item, json_items))

    typer.echo(f"Matched {len(matches)} existing groups to JSON metadata items.")
    selected = matches if no_limit else matches[:limit]
    typer.echo(f"Creating {len(selected)} nested carousel groups.")
    if not selected:
        typer.echo("No matching groups to create. No output folder created.")
        return

    output_name = output_folder_name or default_output_folder_name()
    output_folder = client.create_storage_folder(
        name=output_name,
        description="Nested carousel data group probe output.",
        client_metadata={
            "probe": "nested-carousel-data-group-output",
            "all_data_folder_id": str(all_data_folder_id),
            "groups_folder_id": str(groups_folder_id),
            "matched_group_count": len(matches),
            "created_group_limit": None if no_limit else limit,
        },
    )
    typer.echo(f"Output folder: {output_folder.uuid} | {output_folder.name}")

    dataset = client.get_dataset(dataset_hash) if dataset_hash is not None else None

    for index, (episode_path, group_item, json_items) in enumerate(selected, start=1):
        layout_contents = [group_item.uuid, *[item.uuid for item in json_items]]
        typer.echo("")
        typer.echo(f"[{index}/{len(selected)}] {episode_path}")
        typer.echo(f"  tile 1 group: {group_item.uuid} | {group_item.name}")
        for tile_index, item in enumerate(json_items, start=2):
            typer.echo(f"  tile {tile_index} json: {item.uuid} | {item.name}")

        try:
            created_uuid = output_folder.create_data_group(
                DataGroupCarousel(
                    name=group_name(episode_path),
                    layout_contents=layout_contents,
                    client_metadata={
                        "probe": "nested-carousel-data-group",
                        "episode_path": episode_path,
                        "inner_group_uuid": str(group_item.uuid),
                        "json_uuids": [str(item.uuid) for item in json_items],
                    },
                )
            )
        except Exception as exc:
            typer.echo("  Encord rejected this nested carousel group.")
            typer.echo(f"  {type(exc).__name__}: {exc}")
            continue

        typer.echo(f"  created: {created_uuid}")
        if dataset is not None:
            dataset.link_items([created_uuid])
            typer.echo(f"  linked to dataset: {dataset_hash}")


if __name__ == "__main__":
    typer.run(main)
