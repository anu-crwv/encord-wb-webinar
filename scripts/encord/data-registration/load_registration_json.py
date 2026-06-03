# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "typer",
# ]
# ///
"""Load a registration JSON into an Encord storage folder.

Set your Encord key once:
    export ENCORD_SSH_KEY_FILE=/path/to/encord_key

Run:
    uv run --script scripts/encord/data-registration/load_registration_json.py
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from encord.orm.dataset import LongPollingStatus
from encord.user_client import EncordUserClient

DEFAULT_REGISTRATION_JSON = "registration.json"
DEFAULT_FOLDER_HASH = ""
DEFAULT_INTEGRATION_HASH = "1a2117d0-7ce1-46b7-a426-48231247585c"
ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"


def get_client() -> EncordUserClient:
    ssh_key_file = os.environ.get(ENCORD_SSH_KEY_ENV)
    if not ssh_key_file:
        raise typer.BadParameter(f"Set {ENCORD_SSH_KEY_ENV} to the path of your Encord SSH private key.")
    return EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_key_file)


def get_or_create_folder(client: EncordUserClient, folder_hash: str) -> object:
    if folder_hash:
        return client.get_storage_folder(folder_hash)
    folder_name = date.today().isoformat()
    typer.echo(f"No folder hash provided; creating storage folder {folder_name!r}.")
    return client.create_storage_folder(name=folder_name)


def upload_registration_json(storage_folder, integration_hash: str, registration_json: Path) -> None:
    upload_job_id = storage_folder.add_private_data_to_folder_start(
        integration_id=integration_hash,
        private_files=str(registration_json),
        ignore_errors=True,
    )
    result = storage_folder.add_private_data_to_folder_get_result(upload_job_id)

    if result.status == LongPollingStatus.DONE:
        typer.echo("Upload finished.")
        if result.unit_errors:
            typer.echo("Some items failed:")
            for err in result.unit_errors:
                typer.echo(f"- {err.error}: {len(err.object_urls)} object URLs")
        return

    if result.status == LongPollingStatus.PENDING:
        raise RuntimeError(f"Upload timed out. Pending units: {result.units_pending_count}")

    raise RuntimeError(f"Upload failed: {result.errors}")


def main(
    registration_json: Annotated[
        Path,
        typer.Argument(help="Path to the registration JSON."),
    ] = Path(DEFAULT_REGISTRATION_JSON),
    folder_hash: Annotated[
        str,
        typer.Option("--folder-hash", help="Existing Encord storage folder hash. Creates today's folder if empty."),
    ] = DEFAULT_FOLDER_HASH,
    integration_hash: Annotated[
        str,
        typer.Option("--integration-hash", help="Encord cloud integration hash."),
    ] = DEFAULT_INTEGRATION_HASH,
) -> None:
    if not registration_json.exists():
        raise typer.BadParameter(f"Registration JSON not found: {registration_json}")
    if not integration_hash:
        raise typer.BadParameter("Set DEFAULT_INTEGRATION_HASH in this script or pass --integration-hash.")

    client = get_client()
    storage_folder = get_or_create_folder(client, folder_hash)
    upload_registration_json(storage_folder, integration_hash, registration_json)
    typer.echo(f"Storage folder hash: {storage_folder.uuid}")


if __name__ == "__main__":
    typer.run(main)
