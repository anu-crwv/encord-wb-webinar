# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create Language Instruction captions from Encord task metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import typer

from encord import EncordUserClient
from encord.constants.enums import DataType
from encord.objects import Classification
from encord.objects.frames import Range


CLASSIFICATION_TITLE = "Language Instruction"
BUNDLE_SIZE = 100
TASK_TO_CAPTION = {
    "Pour nuts & bolts": "pour the nuts and bolts into the tray",
    "Batteries": "place the batteries into the tray",
    "Pour Coffee 2": "pour the coffee into the cup",
    "Sort glue by type": "sort the glue bottles by type",
    "Sort tape & safety glasses (2)": "sort the tape and safety glasses",
    "Microfiber towels": "fold the microfiber towels",
    "Coil wire": "coil the wire",
    "Plug ethernet cable into network device": "plug the ethernet cable into the network switch",
    "Plug ethernet cable into network device 2": "plug the ethernet cable into the network switch",
    "Plug ethernet cable into network switch 3": "plug the ethernet cable into the network switch",
}


def log_progress(action: str, index: int, total: int) -> None:
    if index == total or index % 50 == 0:
        typer.echo(f"{action} {index}/{total}...")


def row_chunks(rows: list[Any]) -> list[list[Any]]:
    return [rows[index : index + BUNDLE_SIZE] for index in range(0, len(rows), BUNDLE_SIZE)]


def client_from_env() -> EncordUserClient:
    key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def is_video(label_row: Any) -> bool:
    data_type = getattr(label_row, "data_type", None)
    return data_type == DataType.VIDEO or str(data_type).lower() in {"video", "data_type.video", "video/mp4"}


def full_range(label_row: Any) -> Range:
    frames = int(getattr(label_row, "number_of_frames", 0) or 0)
    return Range(start=0, end=max(frames - 1, 0))


def language_instruction_instances(label_row: Any) -> list[Any]:
    instances = []
    for instance in label_row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        if getattr(ontology_item, "title", None) == CLASSIFICATION_TITLE:
            instances.append(instance)
    return instances


def has_language_instruction(label_row: Any) -> bool:
    return bool(language_instruction_instances(label_row))


def task_by_data_hash(project: Any, client: EncordUserClient) -> dict[str, str]:
    typer.echo("Loading attached dataset metadata...")
    datasets = list(project.list_datasets())
    if len(datasets) != 1:
        raise typer.BadParameter(f"Expected exactly one attached dataset, found {len(datasets)}.")

    dataset = client.get_dataset(str(datasets[0].dataset_hash))
    data_rows = list(dataset.data_rows)
    typer.echo(f"Found {len(data_rows)} dataset rows.")
    storage_items = {
        str(item.uuid): item
        for item in client.get_storage_items([row.backing_item_uuid for row in data_rows])
    }
    tasks = {}
    for row in data_rows:
        item = storage_items.get(str(row.backing_item_uuid))
        metadata = getattr(item, "client_metadata", None) or {}
        if metadata.get("task_name"):
            tasks[str(row.uid)] = str(metadata["task_name"])
    typer.echo(f"Found task metadata for {len(tasks)} rows.")
    return tasks


def main(
    project_hash: Annotated[str, typer.Argument(help="Encord project hash.")],
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing Language Instruction captions.")] = False,
) -> None:
    typer.echo("Connecting to Encord...")
    client = client_from_env()
    project = client.get_project(project_hash)
    tasks = task_by_data_hash(project, client)
    typer.echo("Listing video label rows...")
    rows = [row for row in project.list_label_rows_v2() if is_video(row)]
    total_video_rows = len(rows)
    typer.echo(f"Found {total_video_rows} video label rows.")

    captions_by_hash = {
        str(row.data_hash): TASK_TO_CAPTION[tasks[str(row.data_hash)]]
        for row in rows
        if str(row.data_hash) in tasks and tasks[str(row.data_hash)] in TASK_TO_CAPTION
    }
    rows = [row for row in rows if str(row.data_hash) in captions_by_hash]
    skipped = total_video_rows - len(rows)
    if not rows:
        typer.echo("No rows matched the task-to-caption mapping.")
        return

    typer.echo(f"Preparing captions for {len(rows)} mapped rows.")
    if overwrite:
        typer.echo("Overwrite enabled: clearing existing labels with empty include hash sets...")
        cleared = 0
        for chunk in row_chunks(rows):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.initialise_labels(
                        include_object_feature_hashes=set(),
                        include_classification_feature_hashes=set(),
                        overwrite=True,
                        bundle=bundle,
                    )
            cleared += len(chunk)
            log_progress("Cleared", cleared, len(rows))
    else:
        typer.echo("Initializing existing labels...")
        initialized = 0
        for chunk in row_chunks(rows):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.initialise_labels(bundle=bundle)
            initialized += len(chunk)
            log_progress("Initialized", initialized, len(rows))

    typer.echo("Preparing caption updates...")
    touched = []
    for index, row in enumerate(rows, start=1):
        caption = captions_by_hash[str(row.data_hash)]
        if has_language_instruction(row) and not overwrite:
            skipped += 1
            continue

        classification = row.ontology_structure.get_child_by_title(
            title=CLASSIFICATION_TITLE,
            type_=Classification,
        )
        if classification is None:
            raise RuntimeError(f"Classification not found in ontology: {CLASSIFICATION_TITLE}")

        instance = classification.create_instance()
        instance.set_answer(answer=caption)
        instance.set_for_frames(frames=full_range(row), overwrite=overwrite)
        row.add_classification_instance(instance, force=overwrite)
        touched.append(row)
        log_progress("Prepared", index, len(rows))

    if touched:
        typer.echo(f"Saving {len(touched)} updated rows...")
        saved = 0
        for chunk in row_chunks(touched):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.save(bundle=bundle)
            saved += len(chunk)
            log_progress("Saved", saved, len(touched))

    typer.echo(f"Updated {len(touched)} label rows; skipped {skipped}.")


if __name__ == "__main__":
    typer.run(main)
