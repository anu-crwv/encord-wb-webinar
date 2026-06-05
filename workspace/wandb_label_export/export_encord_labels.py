# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from encord.user_client import EncordUserClient


def _read_ssh_key(ssh_key_file: str) -> str:
    if not ssh_key_file:
        raise typer.BadParameter("Pass --ssh-key-file /path/to/your/encord_ssh_private_key")

    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")

    return key_path.read_text()


def main(
    project_hash: Annotated[str, typer.Option(help="Encord project hash to export labels from.")],
    ssh_key_file: Annotated[
        str,
        typer.Option(help="Path to the Encord SDK SSH private key file."),
    ] = "",
    output_json: Annotated[
        Path,
        typer.Option(help="Where to write the exported Encord label JSON."),
    ] = Path("encord_labels.json"),
    bundle_size: Annotated[
        int,
        typer.Option(help="Number of label rows to initialise per Encord bundle."),
    ] = 100,
) -> None:
    """Export all label rows from an Encord project into a local JSON file."""

    if bundle_size < 1:
        raise typer.BadParameter("--bundle-size must be at least 1")

    client = EncordUserClient.create_with_ssh_private_key(_read_ssh_key(ssh_key_file))
    project = client.get_project(project_hash)

    label_rows = list(project.list_label_rows_v2())
    if not label_rows:
        typer.echo("No label rows found in this project.")
        raise typer.Exit(1)

    effective_bundle_size = min(bundle_size, len(label_rows))
    typer.echo(f"Initialising {len(label_rows)} label rows with bundle size {effective_bundle_size}...")

    with project.create_bundle(bundle_size=effective_bundle_size) as bundle:
        for label_row in label_rows:
            label_row.initialise_labels(bundle=bundle)

    payload = {
        "export_info": {
            "source": "encord",
            "project_hash": project_hash,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "label_row_count": len(label_rows),
        },
        "label_rows": [label_row.to_encord_dict() for label_row in label_rows],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    typer.echo(f"Wrote {len(label_rows)} label rows to {output_json}")


if __name__ == "__main__":
    typer.run(main)
