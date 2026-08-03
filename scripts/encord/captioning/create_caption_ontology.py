# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "encord==0.1.199",
#     "typer",
# ]
# ///
"""Create the caption ontology and optionally a dataset-attached Encord project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from encord.objects.attributes import TextAttribute
from encord.objects.ontology_structure import OntologyStructure

DEFAULT_ONTOLOGY_JSON = Path(__file__).with_name("caption_ontology.json")
CLASSIFICATION_TITLES = (
    "Language Instruction 1",
    "Language Instruction 2",
    "Language Instruction 3",
)


@dataclass(frozen=True)
class CaptionClassification:
    name: str
    required: bool


@dataclass(frozen=True)
class CaptionOntologySpec:
    title: str
    description: str
    classifications: tuple[CaptionClassification, ...]


def required_string(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise typer.BadParameter(f"Ontology JSON field {field!r} must be a non-empty string.")
    return text


def load_ontology_spec(path: Path) -> CaptionOntologySpec:
    if not path.is_file():
        raise typer.BadParameter(f"Ontology JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid ontology JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Ontology JSON must contain one top-level object.")

    unexpected = set(payload) - {"title", "description", "classifications"}
    if unexpected:
        raise typer.BadParameter(
            f"Ontology JSON has unsupported fields: {sorted(unexpected)}"
        )
    raw_classifications = payload.get("classifications")
    if not isinstance(raw_classifications, list):
        raise typer.BadParameter("Ontology JSON 'classifications' must be a list.")

    classifications = []
    for index, raw in enumerate(raw_classifications):
        if not isinstance(raw, dict):
            raise typer.BadParameter(f"Classification {index} must be an object.")
        extra = set(raw) - {"name", "type", "required"}
        if extra:
            raise typer.BadParameter(
                f"Classification {index} has unsupported fields: {sorted(extra)}"
            )
        if raw.get("type") != "text":
            raise typer.BadParameter(
                f"Classification {index} must use type 'text'."
            )
        required = raw.get("required", False)
        if not isinstance(required, bool):
            raise typer.BadParameter(
                f"Classification {index} 'required' must be true or false."
            )
        classifications.append(
            CaptionClassification(
                name=required_string(raw.get("name"), f"classifications[{index}].name"),
                required=required,
            )
        )

    titles = tuple(classification.name for classification in classifications)
    if titles != CLASSIFICATION_TITLES:
        raise typer.BadParameter(
            "Caption ontology classifications must be exactly, in order: "
            + ", ".join(CLASSIFICATION_TITLES)
        )
    return CaptionOntologySpec(
        title=required_string(payload.get("title"), "title"),
        description=str(payload.get("description") or "").strip(),
        classifications=tuple(classifications),
    )


def build_ontology_structure(spec: CaptionOntologySpec) -> OntologyStructure:
    structure = OntologyStructure()
    for classification in spec.classifications:
        node = structure.add_classification()
        node.add_attribute(
            TextAttribute,
            classification.name,
            required=classification.required,
        )
    structure.to_dict()
    return structure


def create_client(ssh_key_file: Path, domain: str | None = None) -> Any:
    from encord import EncordUserClient

    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"SSH key file does not exist: {ssh_key_file}")
    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def validate_project_options(
    dataset_hash: str | None,
    project_title: str | None,
    workflow_template_hash: str | None,
) -> None:
    if bool(dataset_hash) != bool(project_title):
        raise typer.BadParameter(
            "Pass --dataset-hash and --project-title together to create a project."
        )
    if workflow_template_hash and not project_title:
        raise typer.BadParameter(
            "--workflow-template-hash requires --dataset-hash and --project-title."
        )


def create_caption_ontology(
    *,
    ontology_json: Path,
    ssh_key_file: Path,
    dataset_hash: str | None,
    project_title: str | None,
    project_description: str,
    workflow_template_hash: str | None,
    domain: str | None,
) -> dict[str, str | None]:
    validate_project_options(dataset_hash, project_title, workflow_template_hash)
    spec = load_ontology_spec(ontology_json)
    structure = build_ontology_structure(spec)

    typer.echo(f"Ontology: {spec.title}")
    for classification in spec.classifications:
        required = "required" if classification.required else "optional"
        typer.echo(f"  {classification.name}: text ({required})")
    if project_title:
        typer.echo(f"Project: {project_title}")
        typer.echo(f"Dataset: {dataset_hash}")
        if workflow_template_hash:
            typer.echo(f"Workflow template: {workflow_template_hash}")

    client = create_client(ssh_key_file.expanduser(), domain)
    if dataset_hash:
        client.get_dataset(dataset_hash)

    ontology = client.create_ontology(
        title=spec.title,
        description=spec.description,
        structure=structure,
    )
    ontology_hash = str(ontology.ontology_hash)
    typer.echo(f"Created ontology: {ontology_hash}")

    project_hash = None
    if project_title and dataset_hash:
        project_kwargs: dict[str, Any] = {
            "project_title": project_title,
            "dataset_hashes": [dataset_hash],
            "project_description": project_description,
            "ontology_hash": ontology_hash,
        }
        if workflow_template_hash:
            project_kwargs["workflow_template_hash"] = workflow_template_hash
        project_hash = str(client.create_project(**project_kwargs))
        typer.echo(f"Created project: {project_hash}")

    return {"ontology_hash": ontology_hash, "project_hash": project_hash}


def main(
    ssh_key_file: Annotated[
        Path,
        typer.Option(help="Path to the Encord SSH private-key file."),
    ],
    ontology_json: Annotated[
        Path | None,
        typer.Option(help="Optional caption ontology JSON specification."),
    ] = None,
    dataset_hash: Annotated[
        str | None,
        typer.Option(help="Dataset for the optional new caption project."),
    ] = None,
    project_title: Annotated[
        str | None,
        typer.Option(help="Create a project with this title and the new ontology."),
    ] = None,
    project_description: Annotated[
        str,
        typer.Option(help="Description for the optional new project."),
    ] = "Robot demonstration captioning project.",
    workflow_template_hash: Annotated[
        str | None,
        typer.Option(help="Optional Encord workflow template for the new project."),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(help="Optional Encord API domain, for example https://api.us.encord.com."),
    ] = None,
) -> None:
    create_caption_ontology(
        ontology_json=(ontology_json or DEFAULT_ONTOLOGY_JSON).expanduser(),
        ssh_key_file=ssh_key_file,
        dataset_hash=dataset_hash,
        project_title=project_title,
        project_description=project_description,
        workflow_template_hash=workflow_template_hash,
        domain=domain,
    )


if __name__ == "__main__":
    typer.run(main)
