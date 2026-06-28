# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "typer",
# ]
# ///
"""Update one Encord client metadata field across a dataset.

Set your Encord key once:
    export ENCORD_SSH_KEY_FILE=/path/to/encord_key

Run:
    uv run --script scripts/encord/data-registration/update_dataset_metadata_field.py \
      <metadata_field> <new_value> <dataset_hash> --limit 3
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Any
from uuid import UUID

import typer
from encord.user_client import EncordUserClient

ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"
PROGRESS_INTERVAL = 500
MAX_UPDATE_ATTEMPTS = 5
RETRY_BASE_SECONDS = 2
ERROR_PREVIEW_LIMIT = 5


def get_client() -> EncordUserClient:
    ssh_key_file = os.environ.get(ENCORD_SSH_KEY_ENV)
    if not ssh_key_file:
        raise typer.BadParameter(f"Set {ENCORD_SSH_KEY_ENV} to the path of your Encord SSH private key.")
    return EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_key_file)


def group_children(item: Any, client: EncordUserClient) -> list[Any]:
    children_by_uuid: dict[str, Any] = {}

    get_child_items = getattr(item, "get_child_items", None)
    if callable(get_child_items):
        try:
            for child in get_child_items():
                children_by_uuid[str(child.uuid)] = child
        except Exception:
            pass

    try:
        data_group = item.get_summary().data_group
    except Exception:
        return list(children_by_uuid.values())

    if data_group is None:
        return list(children_by_uuid.values())

    child_uuids = [
        child.uuid
        for child in data_group.layout_contents.values()
        if str(child.uuid) not in children_by_uuid
    ]
    if child_uuids:
        for child in client.get_storage_items(child_uuids):
            children_by_uuid[str(child.uuid)] = child

    return list(children_by_uuid.values())


def update_item(item: Any, metadata_field: str, new_value: str) -> tuple[bool, str | None]:
    metadata = dict(item.client_metadata or {})
    if metadata.get(metadata_field) == new_value:
        return False, None

    metadata[metadata_field] = new_value
    for attempt in range(1, MAX_UPDATE_ATTEMPTS + 1):
        try:
            item.update(client_metadata=metadata)
            return True, None
        except Exception as exc:
            if attempt == MAX_UPDATE_ATTEMPTS:
                return False, str(exc)
            retry_after = getattr(exc, "retry_after", None)
            if not isinstance(retry_after, int | float) or retry_after <= 0:
                retry_after = RETRY_BASE_SECONDS * attempt
            time.sleep(retry_after)

    return False, "update failed"


def is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def resolve_args(first: str, second: str, third: str) -> tuple[str, str, str]:
    if is_uuid(third):
        return third, first, second
    if is_uuid(first):
        return first, second, third
    raise typer.BadParameter(
        "Could not identify the dataset hash. Pass arguments as: <metadata_field> <new_value> <dataset_hash>."
    )


def main(
    metadata_field: Annotated[str, typer.Argument(help="Metadata field to overwrite.")],
    new_value: Annotated[str, typer.Argument(help="New string value for the metadata field.")],
    dataset_hash: Annotated[str, typer.Argument(help="Encord dataset hash.")],
    limit: Annotated[int | None, typer.Option(help="Optional number of dataset rows to update.")] = None,
) -> None:
    dataset_hash, metadata_field, new_value = resolve_args(metadata_field, new_value, dataset_hash)
    typer.echo("Connecting to Encord...")
    client = get_client()
    typer.echo(f"Loading dataset {dataset_hash}...")
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    selected_rows = data_rows[:limit] if limit is not None else data_rows
    typer.echo(f"Rows selected: {len(selected_rows)} of {len(data_rows)}")

    backing_item_uuids = [row.backing_item_uuid for row in selected_rows]
    typer.echo("Resolving backing items...")
    storage_items = client.get_storage_items(backing_item_uuids) if backing_item_uuids else []
    typer.echo(f"Backing items: {len(storage_items)}")

    updated = 0
    skipped_current = 0
    failed = 0
    processed = 0
    error_examples: list[str] = []
    data_groups_expanded = 0
    seen_target_uuids: set[str] = set()
    typer.echo(f"Updating {metadata_field!r}...")
    for item in storage_items:
        children = group_children(item, client)
        if children:
            data_groups_expanded += 1
        targets = children or [item]
        for target in targets:
            target_uuid = str(target.uuid)
            if target_uuid in seen_target_uuids:
                continue
            seen_target_uuids.add(target_uuid)
            did_update, error = update_item(target, metadata_field, new_value)
            processed += 1
            if error is not None:
                failed += 1
                if len(error_examples) < ERROR_PREVIEW_LIMIT:
                    error_examples.append(f"{target_uuid}: {error}")
            elif did_update:
                updated += 1
            else:
                skipped_current += 1

            if processed % PROGRESS_INTERVAL == 0:
                typer.echo(
                    f"Processed {processed} items; updated={updated}, skipped={skipped_current}, failed={failed}"
                )

    typer.echo(
        f"Done. Items processed: {processed}. Updated: {updated}. Skipped: {skipped_current}. "
        f"Failed: {failed}. Data groups expanded: {data_groups_expanded}."
    )
    for error in error_examples:
        typer.echo(f"Failed item: {error}", err=True)
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
