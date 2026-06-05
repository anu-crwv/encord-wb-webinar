# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create custom data groups from raw Encord video and metadata items."""

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


def load_raw_items_by_episode(folder: Any, debug: bool = False, debug_limit: int = 5) -> dict[str, dict[str, list[Any]]]:
    from encord.orm.storage import StorageItemType

    by_episode: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"videos": [], "jsons": []})
    scanned = 0
    skipped = 0
    typer.echo(f"Scanning raw source folder {folder.uuid} for videos and JSON metadata...")
    for item in folder.list_items(page_size=1000, item_types=[StorageItemType.VIDEO, StorageItemType.PLAIN_TEXT]):
        scanned += 1
        item_debug = debug and scanned <= debug_limit
        if item_debug:
            typer.echo("")
            typer.echo(f"  Debug raw item {scanned}: {item.uuid} | {item.item_type} | {item.name}")
            typer.echo(f"    metadata: {metadata_hint(item_metadata(item))}")
        episode_path = episode_path_from_item(item)
        if item_debug:
            typer.echo(f"    resolved episode_path: {episode_path}")
        if not episode_path:
            skipped += 1
            continue
        if item.item_type == StorageItemType.VIDEO:
            by_episode[episode_path]["videos"].append(item)
        elif is_json_metadata_item(item):
            by_episode[episode_path]["jsons"].append(item)
    typer.echo(f"Scanned {scanned} raw items; skipped {skipped} with no episode path.")
    typer.echo(f"Found {len(by_episode)} episodes with raw videos and/or JSON metadata.")
    return by_episode


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CHUNK_RE = re.compile(r"^chunk-\d+$")


def source_path(item: Any) -> str:
    metadata = item_metadata(item)
    return normalize_source_path(metadata.get("source_key") or metadata.get("source_uri") or getattr(item, "name", ""))


def path_parts(value: str) -> list[str]:
    return [part for part in value.strip("/").split("/") if part]


def first_metadata_value(items: list[Any], key: str) -> str | None:
    for item in items:
        value = item_metadata(item).get(key)
        if value not in (None, ""):
            return str(value)
    return None


def task_name_from_path(episode_path: str) -> str | None:
    parts = path_parts(episode_path)
    date_index = next((index for index, part in enumerate(parts) if DATE_RE.match(part)), None)
    if date_index is not None and date_index >= 3:
        return parts[date_index - 3]
    return None


def date_from_path(episode_path: str) -> str | None:
    return next((part for part in path_parts(episode_path) if DATE_RE.match(part)), None)


def chunk_from_items(items: list[Any]) -> str | None:
    for item in items:
        chunk = next((part for part in path_parts(source_path(item)) if CHUNK_RE.match(part)), None)
        if chunk:
            return chunk
    return None


def clean_name_part(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("/", "-")).strip()


def group_name(episode_path: str, video_items: list[Any]) -> str:
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    episode = parts[-1] if parts else episode_path
    date = (first_metadata_value(video_items, "collection_datetime") or date_from_path(episode_path) or "")[:10]
    name_parts = [
        first_metadata_value(video_items, "task_name") or task_name_from_path(episode_path),
        date or None,
        episode,
        chunk_from_items(video_items),
    ]
    name = " | ".join(clean_name_part(part) for part in name_parts if part)
    return name[:120] if name else clean_name_part(episode_path)[:120]


def default_output_folder_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"data-groups-output-{timestamp}"


def video_sort_key(item: Any) -> tuple[int, str]:
    camera_name = str(item_metadata(item).get("camera_name") or "")
    order = {
        "cam_high": 0,
        "cam_left_wrist": 1,
        "cam_right_wrist": 2,
    }
    return order.get(camera_name, 99), str(getattr(item, "name", ""))


def video_items_by_camera(video_items: list[Any]) -> dict[str, Any]:
    return {str(item_metadata(item).get("camera_name") or ""): item for item in video_items}


def sanitize_layout_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_")
    return key or "item"


def unique_key(base: str, existing: set[str]) -> str:
    key = base
    index = 2
    while key in existing:
        key = f"{base}_{index}"
        index += 1
    existing.add(key)
    return key


def metadata_role(item: Any) -> str:
    metadata = item_metadata(item)
    role = str(metadata.get("metadata_file_role") or "")
    if role and role != "none":
        return role

    source_key = str(metadata.get("source_key") or getattr(item, "name", ""))
    name = Path(source_key).name
    if name == "info.json":
        return "info_json"
    if name == "tasks.jsonl":
        return "tasks_jsonl"
    if name == "episodes.jsonl":
        return "episodes_jsonl"
    if name == "episodes_stats.jsonl":
        return "episodes_stats_jsonl"
    return sanitize_layout_key(name)


def metadata_sort_key(item: Any) -> tuple[int, str]:
    order = {
        "info_json": 0,
        "info": 0,
        "tasks_jsonl": 1,
        "tasks": 1,
        "episodes_jsonl": 2,
        "episodes": 2,
        "episodes_stats_jsonl": 3,
        "episodes_stats": 3,
    }
    role = metadata_role(item)
    return order.get(role, 99), role


def build_custom_group(episode_path: str, source_folder_id: UUID, video_items: list[Any], json_items: list[Any]) -> Any:
    from encord.orm.group_layout import DataUnitCarouselTile, DataUnitTile, LayoutGrid
    from encord.orm.storage import DataGroupCustom

    layout_contents: dict[str, UUID] = {}
    used_keys: set[str] = set()

    by_camera = video_items_by_camera(video_items)
    high_key = unique_key("camera_cam_high", used_keys)
    left_key = unique_key("camera_cam_left_wrist", used_keys)
    right_key = unique_key("camera_cam_right_wrist", used_keys)
    layout_contents[high_key] = by_camera["cam_high"].uuid
    layout_contents[left_key] = by_camera["cam_left_wrist"].uuid
    layout_contents[right_key] = by_camera["cam_right_wrist"].uuid

    json_keys = []
    for item in sorted(json_items, key=metadata_sort_key):
        key = unique_key(f"metadata_{sanitize_layout_key(metadata_role(item))}", used_keys)
        layout_contents[key] = item.uuid
        json_keys.append(key)

    json_carousel = DataUnitCarouselTile(keys=json_keys, carousel_position="bottom", carousel_size=10)
    wrist_grid = LayoutGrid(
        direction="row",
        split_percentage=50,
        first=DataUnitTile(key=left_key),
        second=DataUnitTile(key=right_key),
    )
    right_side = LayoutGrid(direction="column", split_percentage=50, first=wrist_grid, second=json_carousel)

    return DataGroupCustom(
        name=group_name(episode_path, video_items),
        layout_contents=layout_contents,
        layout=LayoutGrid(
            direction="row",
            split_percentage=50,
            first=DataUnitTile(key=high_key),
            second=right_side,
        ),
        client_metadata={
            "probe": "custom-carousel-data-group",
            "episode_path": episode_path,
            "source_folder_id": str(source_folder_id),
            "video_uuids": [str(item.uuid) for item in video_items],
            "json_uuids": [str(item.uuid) for item in json_items],
        },
    )


def main(
    all_data_folder_id: Annotated[
        UUID,
        typer.Option(help="Folder containing the ungrouped videos and JSON/text metadata items."),
    ] = DEFAULT_ALL_DATA_FOLDER,
    limit: Annotated[int, typer.Option(help="Max matched groups to create unless --no-limit is passed.")] = 5,
    no_limit: Annotated[
        bool,
        typer.Option("--no-limit", help="Create custom carousel groups for every match."),
    ] = False,
    output_folder_name: Annotated[str | None, typer.Option(help="Name for the new output folder.")] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Print capped diagnostics while matching raw items.")] = False,
    debug_limit: Annotated[int, typer.Option(help="Number of raw items to inspect in --debug output.")] = 5,
    dataset_hash: Annotated[UUID | None, typer.Option(help="Optional dataset to link created groups into.")] = None,
) -> None:

    client = create_client()
    all_data_folder = client.get_storage_folder(all_data_folder_id)

    matches = []
    raw_by_episode = load_raw_items_by_episode(all_data_folder, debug=debug, debug_limit=debug_limit)
    for episode_path, items in sorted(raw_by_episode.items()):
        if items["videos"] and items["jsons"]:
            matches.append((episode_path, sorted(items["videos"], key=video_sort_key), items["jsons"]))

    typer.echo(f"Matched {len(matches)} episodes with videos and JSON metadata.")
    selected = matches if no_limit else matches[:limit]
    typer.echo(f"Creating {len(selected)} custom carousel groups.")
    if not selected:
        typer.echo("No matching groups to create. No output folder created.")
        return

    output_name = output_folder_name or default_output_folder_name()
    output_folder = client.create_storage_folder(
        name=output_name,
        description="Custom carousel data group output.",
        client_metadata={
            "probe": "custom-carousel-data-group-output",
            "all_data_folder_id": str(all_data_folder_id),
            "matched_episode_count": len(matches),
            "created_group_limit": None if no_limit else limit,
        },
    )
    typer.echo(f"Output folder: {output_folder.uuid} | {output_folder.name}")

    dataset = client.get_dataset(dataset_hash) if dataset_hash is not None else None

    for index, (episode_path, video_items, json_items) in enumerate(selected, start=1):
        typer.echo("")
        typer.echo(f"[{index}/{len(selected)}] {episode_path}")
        for tile_index, item in enumerate(video_items, start=1):
            camera_name = item_metadata(item).get("camera_name")
            typer.echo(f"  video {tile_index}: {item.uuid} | {camera_name} | {item.name}")
        for tile_index, item in enumerate(json_items, start=1):
            typer.echo(f"  json {tile_index}: {item.uuid} | {item.name}")

        cameras = set(video_items_by_camera(video_items))
        expected_cameras = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        if cameras != expected_cameras:
            typer.echo(f"  skipping: expected cameras {sorted(expected_cameras)}, found {sorted(cameras)}")
            continue

        try:
            created_uuid = output_folder.create_data_group(
                build_custom_group(episode_path, all_data_folder_id, video_items, json_items)
            )
        except Exception as exc:
            typer.echo("  Encord rejected this custom carousel group.")
            typer.echo(f"  {type(exc).__name__}: {exc}")
            continue

        typer.echo(f"  created: {created_uuid}")
        if dataset is not None:
            dataset.link_items([created_uuid])
            typer.echo(f"  linked to dataset: {dataset_hash}")


if __name__ == "__main__":
    typer.run(main)
