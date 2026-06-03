# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "typer",
# ]
# ///
"""Create the Encord client metadata schema from a registration JSON.

Run:
    uv run --script scripts/encord/data-registration/create_metadata_schema.py \
      registration.json --dry-run
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from encord.metadata_schema import MetadataSchemaError
from encord.user_client import EncordUserClient

MAX_ENUM_VALUES = 255
DEFAULT_REGISTRATION_JSON = "registration.json"
ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"

UPLOAD_KEYS = ["images", "videos", "audio", "text", "pdfs", "image_groups", "scenes", "data_groups"]

ENUM_FIELDS = {
    "source_family",
    "task_name",
    "environment",
    "file_ext",
    "metadata_file_role",
    "camera_name",
    "sensor_key",
    "robot_type",
    "codebase_version",
    "trossen_subversion",
    "video_codec",
}

SCALAR_FIELDS: dict[str, str] = {
    "collection_datetime": "datetime",
    "has_info_json": "boolean",
    "has_tasks_jsonl": "boolean",
    "has_episodes_jsonl": "boolean",
    "has_episodes_stats_jsonl": "boolean",
    "has_parquet": "boolean",
    "video_has_audio": "boolean",
    "video_width": "number",
    "video_height": "number",
    "collection_fps": "number",
    "state_dim": "number",
    "action_dim": "number",
    "episode_index": "number",
    "source_key": "text",
    "source_uri": "text",
    "episode_path": "text",
    "episode_id": "varchar",
}


def load_registration_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"Registration JSON not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise typer.BadParameter("Registration JSON must be an object with upload category lists.")
    return data


def iter_client_metadata(registration: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    item_count = 0
    metadata_items: list[dict[str, Any]] = []
    for key in UPLOAD_KEYS:
        items = registration.get(key, [])
        if not isinstance(items, list):
            raise typer.BadParameter(f"Registration JSON field {key!r} must be a list.")
        for item in items:
            item_count += 1
            if not isinstance(item, dict):
                continue
            metadata = item.get("clientMetadata")
            if isinstance(metadata, dict):
                metadata_items.append(metadata)
    return item_count, metadata_items


def collect_enum_values(registration: dict[str, Any]) -> tuple[dict[str, set[str]], int, int]:
    values: dict[str, set[str]] = defaultdict(set)
    item_count, metadata_items = iter_client_metadata(registration)

    for metadata in metadata_items:
        for field in ENUM_FIELDS:
            value = metadata.get(field)
            if value is None or value == "":
                continue
            values[field].add(str(value))

    return values, item_count, len(metadata_items)


def connect_client() -> EncordUserClient:
    ssh_key_file = os.environ.get(ENCORD_SSH_KEY_ENV)
    if not ssh_key_file:
        raise typer.BadParameter(f"Set {ENCORD_SSH_KEY_ENV} to the path of your Encord SSH private key.")
    return EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_key_file)


def apply_schema(client: EncordUserClient, enum_values: dict[str, set[str]], dry_run: bool) -> None:
    schema = client.metadata_schema()
    changes: list[str] = []

    for field in sorted(ENUM_FIELDS):
        values = sorted(v for v in enum_values.get(field, set()) if v)
        if not values:
            changes.append(f"SKIP enum {field}: no values in registration JSON")
            continue
        if len(values) > MAX_ENUM_VALUES:
            raise typer.BadParameter(
                f"Enum field {field} has {len(values)} values, exceeding Encord limit {MAX_ENUM_VALUES}"
            )

        existing_type = schema.get_key_type(field)
        if existing_type is None:
            changes.append(f"ADD enum {field}: {len(values)} values")
            if not dry_run:
                schema.add_enum(field, values=values)
        elif existing_type != "enum":
            raise MetadataSchemaError(f"{field} exists as {existing_type}, expected enum")
        else:
            existing_values = set(schema.get_enum_options(field))
            missing = sorted(set(values) - existing_values)
            if missing:
                changes.append(f"ADD enum values {field}: {missing}")
                if not dry_run:
                    schema.add_enum_options(field, values=missing)

    for field, data_type in sorted(SCALAR_FIELDS.items()):
        existing_type = schema.get_key_type(field)
        if existing_type is None:
            changes.append(f"ADD scalar {field}: {data_type}")
            if not dry_run:
                schema.add_scalar(field, data_type=data_type)
        elif existing_type != data_type:
            raise MetadataSchemaError(f"{field} exists as {existing_type}, expected {data_type}")

    for change in changes:
        typer.echo(change)
    if not changes:
        typer.echo("Schema already up to date.")
    if not dry_run:
        schema.save()
        typer.echo("Saved metadata schema.")


def preview_schema(enum_values: dict[str, set[str]], item_count: int, metadata_count: int) -> None:
    typer.echo(f"Registration items: {item_count}")
    typer.echo(f"Items with clientMetadata: {metadata_count}")
    typer.echo("Enum fields:")
    for field in sorted(ENUM_FIELDS):
        values = sorted(v for v in enum_values.get(field, set()) if v)
        if values:
            typer.echo(f"  {field}: {len(values)} values")
        else:
            typer.echo(f"  {field}: no values in registration JSON")

    typer.echo("Scalar fields:")
    for field, data_type in sorted(SCALAR_FIELDS.items()):
        typer.echo(f"  {field}: {data_type}")


def main(
    registration_json: Annotated[
        Path,
        typer.Argument(help="Registration JSON to inspect for clientMetadata enum values."),
    ] = Path(DEFAULT_REGISTRATION_JSON),
    dry_run: Annotated[bool, typer.Option("--dry-run/--apply", help="Print schema values without saving.")] = False,
) -> None:
    typer.echo(f"Inspecting {registration_json} for clientMetadata enum values...")
    registration = load_registration_json(registration_json)
    enum_values, item_count, metadata_count = collect_enum_values(registration)

    if dry_run:
        preview_schema(enum_values, item_count, metadata_count)
        return

    client = connect_client()
    apply_schema(client, enum_values, dry_run=dry_run)


if __name__ == "__main__":
    typer.run(main)
