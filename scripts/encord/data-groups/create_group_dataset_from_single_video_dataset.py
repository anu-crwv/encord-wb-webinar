# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create an Encord data-group dataset from a single-video Encord dataset."""

from __future__ import annotations

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


def episode_path_from_item(item: Any) -> str | None:
    metadata = item_metadata(item)
    if metadata.get("episode_path"):
        return str(metadata["episode_path"])
    for key in ["source_key", "source_uri", "s3_uri", "source_s3_uri", "object_url"]:
        episode_path = episode_path_from_source(metadata.get(key))
        if episode_path:
            return episode_path
    return episode_path_from_source(getattr(item, "name", None))


def group_episode_path(group_item: Any) -> str | None:
    if episode_path := episode_path_from_item(group_item):
        return episode_path
    for child in group_item.get_child_items():
        if episode_path := episode_path_from_item(child):
            return episode_path
    return None


def load_group_by_episode(folder: Any) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    groups_by_episode = {}
    typer.echo(f"Scanning data-group folder {folder.uuid}...")
    for group_item in folder.list_items(page_size=1000, item_types=[StorageItemType.GROUP]):
        episode_path = group_episode_path(group_item)
        if episode_path:
            groups_by_episode[episode_path] = group_item
    typer.echo(f"Found {len(groups_by_episode)} data groups with episode keys.")
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
) -> None:
    from encord.orm.dataset import StorageLocation

    client = create_client()
    source_dataset = client.get_dataset(source_dataset_hash)
    group_folder = client.get_storage_folder(group_folder_id)

    groups_by_episode = load_group_by_episode(group_folder)
    data_rows = list(source_dataset.data_rows)
    source_items = client.get_storage_items([row.backing_item_uuid for row in data_rows])

    matched_group_uuids = []
    seen_episodes = set()
    missing = 0
    for item in source_items:
        episode_path = episode_path_from_item(item)
        if not episode_path or episode_path in seen_episodes:
            continue
        seen_episodes.add(episode_path)
        group_item = groups_by_episode.get(episode_path)
        if group_item is None:
            missing += 1
            continue
        matched_group_uuids.append(group_item.uuid)
        if limit is not None and len(matched_group_uuids) >= limit:
            break

    typer.echo(f"Matched {len(matched_group_uuids)} data groups from {len(seen_episodes)} source episodes.")
    if missing:
        typer.echo(f"Missing group matches for {missing} source episodes.")
    if not matched_group_uuids:
        raise typer.BadParameter("No matching data groups found. No dataset created.")

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
