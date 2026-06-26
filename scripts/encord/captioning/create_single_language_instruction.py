# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create one Language Instruction caption for every video row from an Encord folder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer

from encord import EncordUserClient
from encord.constants.enums import DataType
from encord.objects import Classification
from encord.objects.frames import Range


CLASSIFICATION_TITLE = "Language Instruction"
BUNDLE_SIZE = 100
PAGE_SIZE = 1000


def client_from_env() -> EncordUserClient:
    key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def is_video(row: Any) -> bool:
    data_type = getattr(row, "data_type", None)
    return data_type == DataType.VIDEO or str(data_type).lower() in {"video", "data_type.video", "video/mp4"}


def full_range(row: Any) -> Range:
    frames = int(getattr(row, "number_of_frames", 0) or 0)
    return Range(start=0, end=max(frames - 1, 0))


def chunks(rows: list[Any]) -> list[list[Any]]:
    return [rows[index : index + BUNDLE_SIZE] for index in range(0, len(rows), BUNDLE_SIZE)]


def folder_video_uuids(client: EncordUserClient, folder_hash: UUID) -> set[str]:
    from encord.orm.storage import StorageItemType

    folder = client.get_storage_folder(folder_hash)
    return {
        str(item.uuid)
        for item in folder.find_items(item_types=[StorageItemType.VIDEO], page_size=PAGE_SIZE)
    }


def data_hashes_for_folder(project: Any, client: EncordUserClient, folder_hash: UUID) -> set[str]:
    folder_items = folder_video_uuids(client, folder_hash)
    datasets = list(project.list_datasets())
    data_hashes = set()

    for dataset_ref in datasets:
        dataset = client.get_dataset(str(dataset_ref.dataset_hash))
        for row in dataset.data_rows:
            if str(getattr(row, "backing_item_uuid", "")) in folder_items:
                data_hashes.add(str(row.uid))

    return data_hashes


def has_language_instruction(row: Any, classification_title: str) -> bool:
    for instance in row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        if getattr(ontology_item, "title", None) == classification_title:
            return True
    return False


def add_instruction(row: Any, instruction: str, classification_title: str, overwrite: bool) -> None:
    classification = row.ontology_structure.get_child_by_title(
        title=classification_title,
        type_=Classification,
    )
    if classification is None:
        raise RuntimeError(f"Classification not found in ontology: {classification_title}")

    instance = classification.create_instance()
    instance.set_answer(answer=instruction)
    instance.set_for_frames(frames=full_range(row), overwrite=overwrite)
    row.add_classification_instance(instance, force=overwrite)


def main(
    project_hash: Annotated[str, typer.Argument(help="Encord project hash.")],
    folder_hash: Annotated[UUID, typer.Argument(help="Encord storage folder UUID.")],
    instruction: Annotated[str, typer.Option(help="Language Instruction text to apply to every matching video.")],
    classification_title: Annotated[str, typer.Option(help="Text classification title in the ontology.")] = CLASSIFICATION_TITLE,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing Language Instruction captions.")] = False,
) -> None:
    typer.echo("Connecting to Encord...")
    client = client_from_env()
    project = client.get_project(project_hash)

    data_hashes = data_hashes_for_folder(project, client, folder_hash)
    rows = [row for row in project.list_label_rows_v2() if is_video(row) and str(row.data_hash) in data_hashes]
    if not rows:
        typer.echo("No matching video label rows found.")
        return

    typer.echo(f"Found {len(rows)} matching video label rows.")
    for chunk in chunks(rows):
        with project.create_bundle(bundle_size=len(chunk)) as bundle:
            for row in chunk:
                row.initialise_labels(bundle=bundle)

    touched = []
    skipped = 0
    for row in rows:
        if has_language_instruction(row, classification_title) and not overwrite:
            skipped += 1
            continue
        add_instruction(row, instruction, classification_title, overwrite)
        touched.append(row)

    for chunk in chunks(touched):
        with project.create_bundle(bundle_size=len(chunk)) as bundle:
            for row in chunk:
                row.save(bundle=bundle)

    typer.echo(f"Updated {len(touched)} label rows; skipped {skipped}.")


if __name__ == "__main__":
    typer.run(main)
