# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Apply one full-video Language Instruction text classification to video label rows."""

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


BUNDLE_SIZE = 100
DEFAULT_CLASSIFICATION_TITLE = "Language Instruction"


def client_from_env() -> EncordUserClient:
    key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def is_video_label_row(label_row: Any) -> bool:
    data_type = getattr(label_row, "data_type", None)
    return data_type == DataType.VIDEO or str(data_type).lower() in {"video", "data_type.video", "video/mp4"}


def full_video_range(label_row: Any) -> Range:
    frames = int(getattr(label_row, "number_of_frames", 0) or 0)
    return Range(start=0, end=max(frames - 1, 0))


def row_chunks(rows: list[Any], bundle_size: int) -> list[list[Any]]:
    return [rows[index : index + bundle_size] for index in range(0, len(rows), bundle_size)]


def log_progress(action: str, index: int, total: int, progress_every: int) -> None:
    if progress_every > 0 and (index == total or index % progress_every == 0):
        typer.echo(f"{action} {index}/{total}...")


def has_classification(label_row: Any, classification_title: str) -> bool:
    for instance in label_row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        if getattr(ontology_item, "title", None) == classification_title:
            return True
    return False


def workflow_stage_data_hashes(project: Any, workflow_stage_id: str) -> set[str]:
    try:
        stage_uuid = UUID(workflow_stage_id)
    except ValueError as exc:
        raise typer.BadParameter(f"--workflow-stage-id must be a UUID, got: {workflow_stage_id}") from exc

    try:
        stage = project.workflow.get_stage(uuid=stage_uuid)
    except Exception as exc:
        raise typer.BadParameter(f"Could not load workflow stage {workflow_stage_id}: {exc}") from exc

    task_count = 0
    data_hashes = set()
    for task in stage.get_tasks():
        task_count += 1
        data_hash = getattr(task, "data_hash", None)
        if data_hash is not None:
            data_hashes.add(str(data_hash))

    typer.echo(
        f"Filtering to workflow stage '{stage.title}' ({stage.uuid}); "
        f"found {task_count} tasks and {len(data_hashes)} unique data hashes."
    )
    return data_hashes


def resolve_project_hash(project_hash: str | None, project_hash_option: str | None) -> str:
    if project_hash and project_hash_option and project_hash != project_hash_option:
        raise typer.BadParameter("Pass the project hash either positionally or via --project-hash, not both.")
    resolved = project_hash or project_hash_option
    if not resolved:
        raise typer.BadParameter("Missing project hash. Pass PROJECT_HASH or --project-hash.")
    return resolved


def main(
    project_hash: Annotated[str | None, typer.Argument(help="Encord project hash.")] = None,
    project_hash_option: Annotated[
        str | None,
        typer.Option("--project-hash", help="Encord project hash. Alternative to the positional PROJECT_HASH."),
    ] = None,
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", "-i", help="Language instruction text to apply to matching video rows."),
    ] = None,
    workflow_stage_id: Annotated[
        str | None,
        typer.Option(
            "--workflow-stage-id",
            "--workflow-stage-uuid",
            help="Only apply the instruction to label rows whose workflow task is currently in this stage.",
        ),
    ] = None,
    classification_title: Annotated[
        str,
        typer.Option(help="Text classification title in the project ontology."),
    ] = DEFAULT_CLASSIFICATION_TITLE,
    bundle_size: Annotated[int, typer.Option(help="SDK bundle size for label init/save.")] = BUNDLE_SIZE,
    progress_every: Annotated[int, typer.Option(help="Print progress every N label rows.")] = 500,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite overlapping existing Language Instruction classifications."),
    ] = False,
    dry_run: Annotated[bool, typer.Option(help="Print intended changes without saving.")] = False,
) -> None:
    if bundle_size < 1:
        raise typer.BadParameter("--bundle-size must be at least 1.")
    if not instruction:
        raise typer.BadParameter("Missing --instruction.")

    resolved_project_hash = resolve_project_hash(project_hash, project_hash_option)

    typer.echo("Connecting to Encord...")
    client = client_from_env()
    project = client.get_project(resolved_project_hash)

    stage_data_hashes = workflow_stage_data_hashes(project, workflow_stage_id) if workflow_stage_id else None
    if stage_data_hashes is not None and not stage_data_hashes:
        typer.echo("No workflow tasks with data hashes found in that stage.")
        return

    typer.echo("Listing video label rows...")
    all_rows = (
        project.list_label_rows_v2(data_hashes=sorted(stage_data_hashes))
        if stage_data_hashes is not None
        else project.list_label_rows_v2()
    )
    label_rows = [row for row in all_rows if is_video_label_row(row)]
    if not label_rows:
        typer.echo("No matching video label rows found.")
        return

    typer.echo(f"Preparing to apply instruction to {len(label_rows)} video label rows.")
    if dry_run:
        typer.echo("Dry run: no labels will be saved.")

    if overwrite:
        typer.echo("Overwrite enabled: clearing existing labels with empty include hash sets...")
        cleared = 0
        for chunk in row_chunks(label_rows, bundle_size):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.initialise_labels(
                        include_object_feature_hashes=set(),
                        include_classification_feature_hashes=set(),
                        overwrite=True,
                        bundle=bundle,
                    )
            cleared += len(chunk)
            log_progress("Cleared", cleared, len(label_rows), progress_every)
    else:
        typer.echo("Initializing existing labels...")
        initialized = 0
        for chunk in row_chunks(label_rows, bundle_size):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.initialise_labels(bundle=bundle)
            initialized += len(chunk)
            log_progress("Initialized", initialized, len(label_rows), progress_every)

    touched = []
    skipped_existing = []
    typer.echo("Preparing instruction updates...")
    for index, label_row in enumerate(label_rows, start=1):
        if has_classification(label_row, classification_title) and not overwrite:
            skipped_existing.append(label_row.data_title)
            log_progress("Prepared", index, len(label_rows), progress_every)
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
        instance.set_answer(answer=instruction)
        instance.set_for_frames(frames=full_video_range(label_row), overwrite=overwrite)

        if not dry_run:
            label_row.add_classification_instance(instance, force=overwrite)

        touched.append(label_row)
        log_progress("Prepared", index, len(label_rows), progress_every)

    if not dry_run and touched:
        typer.echo(f"Saving {len(touched)} updated rows...")
        saved = 0
        for chunk in row_chunks(touched, bundle_size):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for label_row in chunk:
                    label_row.save(bundle=bundle)
            saved += len(chunk)
            log_progress("Saved", saved, len(touched), progress_every)

    verb = "Would update" if dry_run else "Updated"
    typer.echo(f"{verb} {len(touched)} label rows; skipped {len(skipped_existing)} existing rows.")


if __name__ == "__main__":
    typer.run(main)
