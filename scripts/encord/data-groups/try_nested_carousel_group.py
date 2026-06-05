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
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer


DEFAULT_ALL_DATA_FOLDER = UUID("cdb6587a-d00b-4446-a3a9-16d2b8babbda")
DEFAULT_GROUPS_FOLDER = UUID("fae47d1a-0c23-4332-9ab1-9e37e8e44b06")


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


def episode_path_from_item(item: Any) -> str | None:
    metadata = item_metadata(item)
    episode_path = metadata.get("episode_path")
    if episode_path:
        return str(episode_path)

    for child in item.get_child_items():
        child_episode_path = item_metadata(child).get("episode_path")
        if child_episode_path:
            return str(child_episode_path)

    return None


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


def load_group_items(folder: Any) -> list[tuple[str, Any]]:
    from encord.orm.storage import StorageItemType

    groups = []
    typer.echo(f"Scanning groups folder {folder.uuid} for existing video groups...")
    for item in folder.list_items(page_size=1000, item_types=[StorageItemType.GROUP]):
        episode_path = episode_path_from_item(item)
        if episode_path:
            groups.append((episode_path, item))
    typer.echo(f"Found {len(groups)} groups with episode_path metadata.")
    return groups


def group_name(episode_path: str) -> str:
    parts = [part for part in episode_path.rstrip("/").split("/") if part]
    return f"nested-carousel-{parts[-1] if parts else episode_path}"


def main(
    all_data_folder_id: Annotated[
        UUID,
        typer.Option(help="Folder containing the ungrouped videos and JSON/text metadata items."),
    ] = DEFAULT_ALL_DATA_FOLDER,
    groups_folder_id: Annotated[
        UUID,
        typer.Option(help="Folder containing the existing grouped 3-camera video data groups."),
    ] = DEFAULT_GROUPS_FOLDER,
    limit: Annotated[int, typer.Option(help="Max matched groups to try. Use 0 for all.")] = 5,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Preview matches only unless --no-dry-run is passed."),
    ] = True,
    dataset_hash: Annotated[UUID | None, typer.Option(help="Optional dataset to link created groups into.")] = None,
) -> None:
    from encord.orm.storage import DataGroupCarousel

    client = create_client()
    all_data_folder = client.get_storage_folder(all_data_folder_id)
    groups_folder = client.get_storage_folder(groups_folder_id)

    json_by_episode = load_json_items_by_episode(all_data_folder)
    group_items = load_group_items(groups_folder)

    matches = []
    for episode_path, group_item in group_items:
        json_items = json_by_episode.get(episode_path, [])
        if json_items:
            matches.append((episode_path, group_item, json_items))

    typer.echo(f"Matched {len(matches)} existing groups to JSON metadata items.")
    selected = matches if limit == 0 else matches[:limit]
    typer.echo(f"{'Dry-running' if dry_run else 'Creating'} {len(selected)} nested carousel groups.")

    dataset = client.get_dataset(dataset_hash) if dataset_hash is not None and not dry_run else None

    for index, (episode_path, group_item, json_items) in enumerate(selected, start=1):
        layout_contents = [group_item.uuid, *[item.uuid for item in json_items]]
        typer.echo("")
        typer.echo(f"[{index}/{len(selected)}] {episode_path}")
        typer.echo(f"  tile 1 group: {group_item.uuid} | {group_item.name}")
        for tile_index, item in enumerate(json_items, start=2):
            typer.echo(f"  tile {tile_index} json: {item.uuid} | {item.name}")

        if dry_run:
            continue

        try:
            created_uuid = groups_folder.create_data_group(
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
