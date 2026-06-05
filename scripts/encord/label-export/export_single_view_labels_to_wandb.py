# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "pyyaml",
#     "typer",
#     "wandb>=0.18.0",
# ]
# ///
"""Version one Encord project's source dataset and labels in W&B."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WANDB_CONFIG = SCRIPT_DIR.parent / "wandb_config.yaml"
EXPORT_ROOT = REPO_ROOT / "exports/encord-label-export"


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"{label} does not exist: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"{label} must contain a YAML object")
    return loaded


def required(config: dict[str, Any], key: str, label: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise typer.BadParameter(f"{label} is missing required key: {key}")
    return value


def create_client():
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def get_single_project_dataset(client: Any, project_hash: str):
    typer.echo(f"Loading Encord project {project_hash}...")
    project = client.get_project(project_hash)
    typer.echo("Finding attached dataset...")
    datasets = list(project.list_datasets())
    if not datasets:
        raise typer.BadParameter(f"Project {project_hash} has no attached datasets.")
    if len(datasets) > 1:
        details = ", ".join(f"{item.title} ({item.dataset_hash})" for item in datasets)
        raise typer.BadParameter(
            f"Project {project_hash} has multiple attached datasets: {details}. "
            "This exporter supports exactly one dataset for now."
        )
    typer.echo(f"Using dataset {datasets[0].dataset_hash} ({datasets[0].title}).")
    return project, datasets[0]


def export_labels(project: Any) -> list[dict[str, Any]]:
    typer.echo("Listing label rows...")
    label_rows = list(project.list_label_rows_v2())
    typer.echo(f"Found {len(label_rows)} label rows.")
    if label_rows:
        typer.echo("Initializing labels...")
        progress_interval = max(1, min(100, len(label_rows) // 10 or 1))
        with project.create_bundle(bundle_size=min(100, len(label_rows))) as bundle:
            for index, label_row in enumerate(label_rows, start=1):
                label_row.initialise_labels(bundle=bundle)
                if index % progress_interval == 0:
                    typer.echo(f"Initialized {index}/{len(label_rows)} label rows.")
        if len(label_rows) % progress_interval:
            typer.echo(f"Initialized {len(label_rows)} label rows.")

    typer.echo("Serializing labels...")
    labels = []
    for label_row in label_rows:
        row = label_row.to_encord_dict()
        if isinstance(row, dict):
            row.setdefault("data_hash", getattr(label_row, "data_hash", None))
            row.setdefault("label_hash", getattr(label_row, "label_hash", None))
            row.setdefault("data_title", getattr(label_row, "data_title", None))
        labels.append(row)
    return labels


def read_dataset_metadata(client: Any, dataset_hash: str) -> dict[str, dict[str, Any]]:
    typer.echo("Loading dataset metadata...")
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    typer.echo(f"Found {len(data_rows)} data rows.")
    backing_ids = [row.backing_item_uuid for row in data_rows if getattr(row, "backing_item_uuid", None)]
    typer.echo(f"Fetching client metadata for {len(backing_ids)} storage items...")
    storage_items = {str(item.uuid): item for item in client.get_storage_items(backing_ids)} if backing_ids else {}

    metadata_by_hash: dict[str, dict[str, Any]] = {}
    for row in data_rows:
        item = storage_items.get(str(getattr(row, "backing_item_uuid", "")))
        metadata_by_hash[str(row.uid)] = {
            "data_hash": row.uid,
            "data_title": row.title,
            "data_type": str(getattr(row, "data_type", "")),
            "encord_storage_item_uuid": str(getattr(row, "backing_item_uuid", "")),
            "client_metadata": getattr(item, "client_metadata", None) or {},
        }
    typer.echo(f"Collected metadata for {len(metadata_by_hash)} data rows.")
    return metadata_by_hash


def source_s3_uri(client_meta: dict[str, Any]) -> Any:
    return client_meta.get("s3_uri") or client_meta.get("source_s3_uri") or client_meta.get("object_url")


def source_dataset_items(metadata_by_hash: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for data_hash, row in sorted(metadata_by_hash.items()):
        client_meta = row.get("client_metadata") or {}
        items.append({
            "data_hash": data_hash,
            "data_title": row.get("data_title"),
            "data_type": row.get("data_type"),
            "encord_storage_item_uuid": row.get("encord_storage_item_uuid"),
            "source_s3_uri": source_s3_uri(client_meta),
            "client_metadata": client_meta,
        })
    return items


def preview_rows(labels: list[dict[str, Any]], metadata_by_hash: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        data_hash = str(label.get("data_hash") or "")
        data_meta = metadata_by_hash.get(data_hash, {})
        client_meta = data_meta.get("client_metadata") or {}
        rows.append({
            "data_hash": data_hash,
            "data_title": label.get("data_title") or data_meta.get("data_title"),
            "label_hash": label.get("label_hash"),
            "episode_id": client_meta.get("episode_id"),
            "episode_path": client_meta.get("episode_path"),
            "camera_name": client_meta.get("camera_name"),
            "source_s3_uri": source_s3_uri(client_meta),
        })
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def make_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = EXPORT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def log_to_wandb(
    *,
    wandb_config: dict[str, Any],
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    import wandb

    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    source_name = required(wandb_config, "source_artifact_name", "W&B config")
    label_name = required(wandb_config, "label_artifact_name", "W&B config")
    table_name = required(wandb_config, "table_name", "W&B config")

    source_manifest = output_dir / "source_dataset_manifest.json"
    source_manifest_data = json.loads(source_manifest.read_text())
    source_items_path = output_dir / "source_dataset_items.json"
    labels_path = output_dir / "encord_labels.json"
    preview_path = output_dir / "label_preview_rows.json"

    with wandb.init(entity=entity, project=project, job_type="encord-label-export") as run:
        typer.echo(f"Logging to W&B run {run.url}...")
        source_ref = metadata.get("source_artifact_ref")
        if source_ref:
            typer.echo(f"Using existing source artifact {source_ref}.")
            run.use_artifact(source_ref)
        else:
            typer.echo(f"Logging source dataset artifact {source_name}...")
            source_artifact = wandb.Artifact(
                source_name,
                type="dataset",
                metadata={
                    "encord_project_hash": source_manifest_data.get("encord_project_hash"),
                    "encord_dataset_hash": source_manifest_data.get("encord_dataset_hash"),
                    "source_s3_prefix": metadata.get("source_s3_prefix"),
                    "source_dataset_note": metadata.get("source_dataset_note"),
                    "curation_status": metadata.get("curation_status"),
                },
                description=str(metadata.get("source_dataset_note", "")),
            )
            source_artifact.add_file(str(source_manifest), name="source_dataset_manifest.json")
            source_artifact.add_file(str(source_items_path), name="source_dataset_items.json")
            logged_source = run.log_artifact(source_artifact, aliases=["latest"])
            logged_source.wait()
            source_ref = f"{source_name}:{logged_source.version}"
            typer.echo(f"Logged source dataset artifact {source_ref}.")

        typer.echo(f"Logging labels artifact {label_name}...")
        label_artifact = wandb.Artifact(
            label_name,
            type="labels",
            metadata={
                "encord_project_hash": source_manifest_data.get("encord_project_hash"),
                "encord_dataset_hash": source_manifest_data.get("encord_dataset_hash"),
                "source_dataset_artifact": source_ref,
                "label_version_note": metadata.get("label_version_note"),
                "captioning_method": metadata.get("captioning_method"),
                "qc_status": metadata.get("qc_status"),
            },
            description=str(metadata.get("label_version_note", "")),
        )
        label_artifact.add_file(str(labels_path), name="encord_labels.json")
        label_artifact.add_file(str(preview_path), name="label_preview_rows.json")
        logged_labels = run.log_artifact(label_artifact, aliases=["latest", "single-view"])
        logged_labels.wait()
        labels_ref = f"{label_name}:{logged_labels.version}"
        typer.echo(f"Logged labels artifact {labels_ref}.")

        typer.echo("Logging preview table...")
        table = wandb.Table(columns=["data_hash", "data_title", "label_hash", "episode_id", "episode_path", "camera_name", "source_s3_uri"])
        for row in json.loads(preview_path.read_text()):
            table.add_data(row.get("data_hash"), row.get("data_title"), row.get("label_hash"), row.get("episode_id"), row.get("episode_path"), row.get("camera_name"), row.get("source_s3_uri"))
        run.log({table_name: table})
        typer.echo("Logged preview table.")

        return {"source_dataset_artifact": source_ref, "labels_artifact": labels_ref, "run_url": run.url}


def main(
    metadata_yaml: Annotated[Path, typer.Option(help="Required YAML notes for this dataset/label version.")],
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
) -> None:
    typer.echo("Loading config...")
    metadata = load_yaml(metadata_yaml, "metadata YAML")
    wandb_settings = load_yaml(wandb_config, "W&B config")
    project_hash = str(required(metadata, "encord_project_hash", "metadata YAML"))

    client = create_client()
    project, project_dataset = get_single_project_dataset(client, project_hash)
    dataset_hash = str(project_dataset.dataset_hash)

    output_dir = make_output_dir()
    typer.echo("Exporting labels and source metadata...")
    labels = export_labels(project)
    data_metadata = read_dataset_metadata(client, dataset_hash)
    dataset_items = source_dataset_items(data_metadata)
    rows = preview_rows(labels, data_metadata)

    source_manifest = {
        "encord_project_hash": project_hash,
        "encord_project_title": project.title,
        "encord_dataset_hash": dataset_hash,
        "encord_dataset_title": project_dataset.title,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_item_count": len(dataset_items),
        "label_row_count": len(labels),
        **metadata,
    }
    typer.echo(f"Writing local export files to {output_dir}...")
    write_json(output_dir / "source_dataset_manifest.json", source_manifest)
    write_json(output_dir / "source_dataset_items.json", dataset_items)
    write_json(output_dir / "encord_labels.json", {"export_info": source_manifest, "label_rows": labels})
    write_json(output_dir / "encord_data_metadata.json", data_metadata)
    write_json(output_dir / "label_preview_rows.json", rows)
    typer.echo("Wrote local export files.")

    lineage = log_to_wandb(wandb_config=wandb_settings, metadata=metadata, output_dir=output_dir)
    write_json(output_dir / "wandb_lineage.json", lineage)

    typer.echo(f"exported {len(labels)} label rows")
    typer.echo(f"dataset: {dataset_hash} ({project_dataset.title})")
    typer.echo(f"source artifact: {lineage['source_dataset_artifact']}")
    typer.echo(f"labels artifact: {lineage['labels_artifact']}")
    typer.echo(f"local files: {output_dir}")
    typer.echo(f"run: {lineage['run_url']}")


if __name__ == "__main__":
    typer.run(main)
