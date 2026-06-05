# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Try creating a carousel data group whose first tile is another data group."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID

import typer


def create_client():
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")

    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")

    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def main(
    folder_id: Annotated[UUID, typer.Option(help="Storage folder where the new data group should be created.")],
    inner_group_uuid: Annotated[UUID, typer.Option(help="Existing 3-camera data group UUID to use as carousel tile 1.")],
    json_uuid: Annotated[
        Optional[list[UUID]], typer.Option("--json-uuid", help="JSON/text item UUID to add as another carousel tile.")
    ] = None,
    group_name: Annotated[str, typer.Option(help="Name for the probe data group.")] = "nested-carousel-probe",
    dataset_hash: Annotated[UUID | None, typer.Option(help="Optional dataset to link the created group into.")] = None,
) -> None:
    from encord.orm.storage import DataGroupCarousel

    client = create_client()
    folder = client.get_storage_folder(folder_id)

    json_uuids = json_uuid or []
    layout_contents = [inner_group_uuid, *json_uuids]
    typer.echo(f"Creating carousel group {group_name!r} with {len(layout_contents)} tiles...")
    typer.echo(f"  tile 1: existing data group {inner_group_uuid}")
    for index, item_uuid in enumerate(json_uuids, start=2):
        typer.echo(f"  tile {index}: json/text item {item_uuid}")

    try:
        created_uuid = folder.create_data_group(
            DataGroupCarousel(
                name=group_name,
                layout_contents=layout_contents,
                client_metadata={
                    "probe": "nested-carousel-data-group",
                    "inner_group_uuid": str(inner_group_uuid),
                    "json_uuids": [str(item_uuid) for item_uuid in json_uuids],
                },
            )
        )
    except Exception as exc:
        typer.echo("Encord rejected the nested carousel group.")
        typer.echo(f"{type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc

    typer.echo(f"Created data group: {created_uuid}")

    if dataset_hash is not None:
        typer.echo(f"Linking created group into dataset {dataset_hash}...")
        dataset = client.get_dataset(dataset_hash)
        dataset.link_items([created_uuid])
        typer.echo("Linked created group to dataset.")

    created_item = client.get_storage_item(created_uuid)
    typer.echo("Created item children:")
    for child in created_item.get_child_items():
        typer.echo(f"- {child.uuid} | {child.item_type} | {child.name}")


if __name__ == "__main__":
    typer.run(main)
