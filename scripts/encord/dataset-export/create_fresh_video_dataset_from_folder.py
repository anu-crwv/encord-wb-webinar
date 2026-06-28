# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create an Encord dataset from recursive folder videos missing from another dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer


ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"
PAGE_SIZE = 1000


def normalized_uuid(value: Any) -> str:
    return str(UUID(str(value)))


def create_client() -> Any:
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get(ENCORD_SSH_KEY_ENV)
    if not ssh_key_file:
        raise typer.BadParameter(f"Set {ENCORD_SSH_KEY_ENV} to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def load_source_dataset_item_uuids(client: Any, source_dataset_hash: UUID) -> set[str]:
    dataset = client.get_dataset(source_dataset_hash)
    data_rows = list(dataset.data_rows)
    item_uuids = {
        normalized_uuid(row.backing_item_uuid)
        for row in data_rows
        if getattr(row, "backing_item_uuid", None) is not None
    }

    typer.echo(f"Source dataset rows: {len(data_rows):,}")
    typer.echo(f"Source dataset backing items: {len(item_uuids):,}")
    return item_uuids


def load_recursive_folder_videos(client: Any, folder_hash: UUID) -> list[Any]:
    from encord.orm.storage import StorageItemType

    folder = client.get_storage_folder(folder_hash)
    videos = list(folder.find_items(item_types=[StorageItemType.VIDEO], page_size=PAGE_SIZE))
    typer.echo(f"Recursive folder videos: {len(videos):,}")
    return videos


def resolve_collection_by_name(client: Any, collection_name: str) -> Any:
    matches = [collection for collection in client.list_collections(page_size=PAGE_SIZE) if collection.name == collection_name]

    if not matches:
        raise typer.BadParameter(f"No Encord Index collection found with name: {collection_name!r}")
    if len(matches) > 1:
        details = ", ".join(
            f"{collection.name!r} ({collection.uuid}, top_level_folder={collection.top_level_folder_uuid})"
            for collection in matches
        )
        raise typer.BadParameter(f"Multiple Encord Index collections found with name {collection_name!r}: {details}")

    collection = matches[0]
    typer.echo(f"Resolved collection {collection.name!r}: {collection.uuid}")
    return collection


def load_collection_video_uuids(client: Any, collection_name: str) -> set[str]:
    from encord.orm.storage import StorageItemType

    collection = resolve_collection_by_name(client, collection_name)
    video_uuids = set()
    skipped_non_videos = 0

    for item in collection.list_items(page_size=PAGE_SIZE):
        if getattr(item, "item_type", None) == StorageItemType.VIDEO:
            video_uuids.add(normalized_uuid(item.uuid))
        else:
            skipped_non_videos += 1

    typer.echo(f"Collection videos: {len(video_uuids):,}")
    if skipped_non_videos:
        typer.echo(f"Collection non-videos skipped: {skipped_non_videos:,}")
    return video_uuids


def apply_collection_filter(folder_videos: list[Any], collection_uuids: set[str] | None) -> list[Any]:
    if collection_uuids is None:
        return folder_videos

    filtered = [item for item in folder_videos if normalized_uuid(item.uuid) in collection_uuids]
    typer.echo(f"Folder videos also in collection: {len(filtered):,}")
    return filtered


def fresh_video_uuids(candidate_videos: list[Any], source_item_uuids: set[str]) -> list[UUID]:
    fresh = []
    seen = set()

    for item in candidate_videos:
        item_uuid = normalized_uuid(item.uuid)
        if item_uuid in source_item_uuids or item_uuid in seen:
            continue
        fresh.append(UUID(item_uuid))
        seen.add(item_uuid)

    typer.echo(f"Fresh videos: {len(fresh):,}")
    return fresh


def create_and_link_dataset(client: Any, output_dataset_title: str, item_uuids: list[UUID]) -> str:
    from encord.orm.dataset import StorageLocation

    response = client.create_dataset(
        dataset_title=output_dataset_title,
        dataset_type=StorageLocation.CORD_STORAGE,
        create_backing_folder=False,
    )
    dataset_hash = str(response.dataset_hash)
    output_dataset = client.get_dataset(dataset_hash)
    output_dataset.link_items(item_uuids)
    return dataset_hash


def main(
    folder_hash: Annotated[UUID, typer.Option(help="Encord storage folder UUID to scan recursively.")],
    source_dataset_hash: Annotated[UUID, typer.Option(help="Existing Encord dataset hash to compare against.")],
    output_dataset_title: Annotated[str, typer.Option(help="Title for the new Encord dataset.")],
    collection_name: Annotated[
        str | None,
        typer.Option(help="Optional Encord Index collection name used to filter candidate videos."),
    ] = None,
) -> None:
    client = create_client()

    source_item_uuids = load_source_dataset_item_uuids(client, source_dataset_hash)
    folder_videos = load_recursive_folder_videos(client, folder_hash)

    collection_uuids = load_collection_video_uuids(client, collection_name) if collection_name is not None else None
    candidate_videos = apply_collection_filter(folder_videos, collection_uuids)
    fresh_uuids = fresh_video_uuids(candidate_videos, source_item_uuids)

    if not fresh_uuids:
        typer.echo("No fresh videos found. No dataset created.")
        return

    typer.echo(f"Creating dataset: {output_dataset_title}")
    dataset_hash = create_and_link_dataset(client, output_dataset_title, fresh_uuids)
    typer.echo(f"Created dataset: {dataset_hash}")
    typer.echo(f"Linked videos: {len(fresh_uuids):,}")


if __name__ == "__main__":
    typer.run(main)
