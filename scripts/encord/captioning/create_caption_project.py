# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "encord==0.1.199",
#     "typer",
# ]
# ///
"""Create a captioning project from an existing Encord dataset and ontology."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from encord.objects import Classification

CLASSIFICATION_TITLES = (
    "Language Instruction 1",
    "Language Instruction 2",
    "Language Instruction 3",
)


def create_client(ssh_key_file: Path, domain: str | None = None) -> Any:
    from encord import EncordUserClient

    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"SSH key file does not exist: {ssh_key_file}")
    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def validate_caption_ontology(ontology: Any) -> None:
    missing = []
    for title in CLASSIFICATION_TITLES:
        try:
            ontology.structure.get_child_by_title(
                title=title,
                type_=Classification,
            )
        except Exception:
            missing.append(title)
    if missing:
        raise typer.BadParameter(
            "Ontology is missing caption classifications: " + ", ".join(missing)
        )


def create_caption_project(
    client: Any,
    *,
    dataset_hash: str,
    ontology_hash: str,
    project_title: str,
    project_description: str,
    workflow_template_hash: str | None,
) -> str:
    dataset = client.get_dataset(dataset_hash)
    ontology = client.get_ontology(ontology_hash)
    validate_caption_ontology(ontology)

    row_count = len(list(dataset.data_rows))
    if row_count == 0:
        raise typer.BadParameter(f"Dataset {dataset_hash} contains no data rows.")

    typer.echo(f"Project: {project_title}")
    typer.echo(f"Dataset: {dataset_hash} ({row_count} rows)")
    typer.echo(f"Ontology: {ontology_hash}")
    for title in CLASSIFICATION_TITLES:
        typer.echo(f"  {title}: present")
    if workflow_template_hash:
        typer.echo(f"Workflow template: {workflow_template_hash}")

    project_kwargs: dict[str, Any] = {
        "project_title": project_title,
        "dataset_hashes": [dataset_hash],
        "project_description": project_description,
        "ontology_hash": ontology_hash,
    }
    if workflow_template_hash:
        project_kwargs["workflow_template_hash"] = workflow_template_hash
    project_hash = str(client.create_project(**project_kwargs))

    project = client.get_project(project_hash)
    attached = [str(item.dataset_hash) for item in project.list_datasets()]
    if attached != [dataset_hash]:
        raise RuntimeError(
            f"Created project {project_hash} has unexpected datasets: {attached}"
        )
    typer.echo(f"Created caption project: {project_hash}")
    return project_hash


def main(
    ssh_key_file: Annotated[
        Path,
        typer.Option(help="Path to the Encord SSH private-key file."),
    ],
    dataset_hash: Annotated[
        str,
        typer.Option(help="Dataset to attach to the caption project."),
    ],
    ontology_hash: Annotated[
        str,
        typer.Option(help="Existing ontology with Language Instruction 1/2/3."),
    ],
    project_title: Annotated[
        str,
        typer.Option(help="Title for the new caption project."),
    ],
    project_description: Annotated[
        str,
        typer.Option(help="Description for the new caption project."),
    ] = "Robot demonstration captioning project.",
    workflow_template_hash: Annotated[
        str | None,
        typer.Option(help="Optional Encord workflow template for caption review."),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(help="Optional Encord API domain, for example https://api.us.encord.com."),
    ] = None,
) -> None:
    client = create_client(ssh_key_file.expanduser(), domain)
    create_caption_project(
        client,
        dataset_hash=dataset_hash,
        ontology_hash=ontology_hash,
        project_title=project_title,
        project_description=project_description,
        workflow_template_hash=workflow_template_hash,
    )


if __name__ == "__main__":
    typer.run(main)
