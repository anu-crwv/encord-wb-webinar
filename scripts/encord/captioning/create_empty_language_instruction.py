# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Add an empty full-video Language Instruction text classification to every video in a project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from encord import EncordUserClient
from encord.constants.enums import DataType
from encord.objects import Classification
from encord.objects.frames import Range


def is_video_label_row(label_row) -> bool:
    data_type = getattr(label_row, "data_type", None)
    if data_type == DataType.VIDEO:
        return True
    return str(data_type).lower() in {"video", "data_type.video", "video/mp4"}


def full_video_range(label_row) -> Range:
    number_of_frames = int(getattr(label_row, "number_of_frames", 0) or 0)
    return Range(start=0, end=max(number_of_frames - 1, 0))


def has_classification(label_row, classification_title: str) -> bool:
    for instance in label_row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        if getattr(ontology_item, "title", None) == classification_title:
            return True
    return False


def main(
    project_hash: Annotated[str, typer.Argument(help="Encord project hash.")],
    classification_title: Annotated[
        str,
        typer.Option(help="Text classification title in the project ontology."),
    ] = "Language Instruction",
    bundle_size: Annotated[int, typer.Option(help="SDK bundle size for label init/save.")] = 100,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite overlapping existing Language Instruction classifications."),
    ] = False,
    dry_run: Annotated[bool, typer.Option(help="Print intended changes without saving.")] = False,
) -> None:
    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")

    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")

    client = EncordUserClient.create_with_ssh_private_key(key_path.read_text())
    project = client.get_project(project_hash)
    label_rows = [row for row in project.list_label_rows_v2() if is_video_label_row(row)]

    if not label_rows:
        typer.echo("No video label rows found.")
        return

    touched = []
    skipped = []

    with project.create_bundle(bundle_size=min(bundle_size, len(label_rows))) as init_bundle:
        for label_row in label_rows:
            label_row.initialise_labels(bundle=init_bundle)

    for label_row in label_rows:
        if has_classification(label_row, classification_title) and not overwrite:
            skipped.append(label_row.data_title)
            continue

        ontology_structure = label_row.ontology_structure
        if ontology_structure is None:
            raise RuntimeError(f"Ontology structure was not initialized for {label_row.data_title}")

        classification = ontology_structure.get_child_by_title(
            title=classification_title,
            type_=Classification,
        )
        if classification is None:
            raise RuntimeError(f"Classification not found in ontology: {classification_title}")

        instance = classification.create_instance()
        instance.set_answer(answer="")
        instance.set_for_frames(frames=full_video_range(label_row), overwrite=overwrite)

        if not dry_run:
            label_row.add_classification_instance(instance, force=overwrite)

        touched.append(label_row)

    if not dry_run and touched:
        with project.create_bundle(bundle_size=min(bundle_size, len(touched))) as save_bundle:
            for label_row in touched:
                label_row.save(bundle=save_bundle)

    typer.echo(
        f"{'Would update' if dry_run else 'Updated'} {len(touched)} video label rows; "
        f"skipped {len(skipped)} existing rows."
    )
    if skipped:
        typer.echo("Skipped existing rows:")
        for title in skipped:
            typer.echo(f"  {title}")


if __name__ == "__main__":
    typer.run(main)
