# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Move all workflow tasks from one Encord workflow node to another."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from encord import EncordUserClient


def flush_batch(project, batch: list, target_node_uuid: UUID, dry_run: bool) -> None:
    if dry_run or not batch:
        return
    with project.create_bundle(bundle_size=len(batch)) as bundle:
        for task in batch:
            task.move(destination_stage_uuid=target_node_uuid, bundle=bundle)


def main(
    project_hash: Annotated[str, typer.Argument(help="Encord project hash.")],
    source_node_hash: Annotated[str, typer.Argument(help="Source workflow node UUID/hash.")],
    target_node_hash: Annotated[str, typer.Argument(help="Target workflow node UUID/hash.")],
    batch_size: Annotated[
        int,
        typer.Option(help="Move tasks in batches. Must be <= 500 because Encord move bundles are capped at 500."),
    ] = 500,
    max_tasks: Annotated[int | None, typer.Option(help="Optional cap for test runs.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Count tasks without moving them.")] = False,
    progress_every: Annotated[int, typer.Option(help="Print progress every N tasks.")] = 500,
) -> None:
    if batch_size < 1 or batch_size > 500:
        raise typer.BadParameter("--batch-size must be between 1 and 500")

    source_uuid = UUID(source_node_hash)
    target_uuid = UUID(target_node_hash)
    if source_uuid == target_uuid:
        raise typer.BadParameter("Source and target workflow nodes are the same.")

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")

    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")

    client = EncordUserClient.create_with_ssh_private_key(key_path.read_text())
    project = client.get_project(project_hash)
    source_stage = project.workflow.get_stage(uuid=source_uuid)
    target_stage = project.workflow.get_stage(uuid=target_uuid)

    typer.echo(f"Source: {source_stage.title} ({source_stage.uuid})")
    typer.echo(f"Target: {target_stage.title} ({target_stage.uuid})")
    if dry_run:
        typer.echo("Dry run: no tasks will be moved.")

    moved = 0
    batch = []

    for task in source_stage.get_tasks():
        batch.append(task)
        moved += 1

        if len(batch) >= batch_size:
            flush_batch(project, batch, target_uuid, dry_run)
            batch = []

        if progress_every > 0 and moved % progress_every == 0:
            verb = "Would move" if dry_run else "Moved"
            typer.echo(f"{verb} {moved} tasks...")

        if max_tasks is not None and moved >= max_tasks:
            break

    flush_batch(project, batch, target_uuid, dry_run)

    verb = "Would move" if dry_run else "Moved"
    typer.echo(f"{verb} {moved} tasks from '{source_stage.title}' to '{target_stage.title}'.")


if __name__ == "__main__":
    typer.run(main)
