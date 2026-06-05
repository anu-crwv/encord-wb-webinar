# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "typer",
# ]
# ///
"""Update Cloud Synced Folder item metadata from a registration JSON.

Set your Encord key once:
    export ENCORD_SSH_KEY_FILE=/path/to/encord_key

Run:
    uv run --script update_cloud_synced_folder_metadata.py --folder-hash <folder_uuid>
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

import typer
from encord.http.bundle import Bundle
from encord.storage import StorageItemType
from encord.user_client import EncordUserClient

DEFAULT_REGISTRATION_JSON = "registration.json"
ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"
UPLOAD_KEYS = ["images", "videos", "audio", "text", "pdfs", "image_groups", "scenes", "data_groups"]
ITEM_TYPES = [
    StorageItemType.IMAGE,
    StorageItemType.VIDEO,
    StorageItemType.AUDIO,
    StorageItemType.PLAIN_TEXT,
    StorageItemType.PDF,
    StorageItemType.GROUP,
    StorageItemType.IMAGE_GROUP,
    StorageItemType.IMAGE_SEQUENCE,
    StorageItemType.SCENE,
]


@dataclass(frozen=True)
class RegistrationMetadata:
    title: str
    source_key: str
    object_url: str
    client_metadata: dict[str, Any]


def get_client() -> EncordUserClient:
    ssh_key_file = os.environ.get(ENCORD_SSH_KEY_ENV)
    if not ssh_key_file:
        raise typer.BadParameter(f"Set {ENCORD_SSH_KEY_ENV} to the path of your Encord SSH private key.")
    return EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_key_file)


def report_path_for(registration_json: Path) -> Path:
    return registration_json.with_name(f"{registration_json.stem}_metadata_update_report.json")


def normalize_key(value: str | None) -> str:
    if not value:
        return ""
    value = unquote(value).strip()
    if value.startswith("s3://"):
        parsed = urlparse(value)
        value = parsed.path.lstrip("/")
    elif value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        value = parsed.path.lstrip("/")
    return value.strip("/")


def object_url_to_key(object_url: str | None) -> str:
    return normalize_key(object_url)


def load_registration_metadata(path: Path) -> list[RegistrationMetadata]:
    if not path.exists():
        raise typer.BadParameter(f"Registration JSON not found: {path}")
    with path.open() as f:
        registration = json.load(f)
    if not isinstance(registration, dict):
        raise typer.BadParameter("Registration JSON must be an object with upload category lists.")

    records: list[RegistrationMetadata] = []
    for category in UPLOAD_KEYS:
        items = registration.get(category, [])
        if not isinstance(items, list):
            raise typer.BadParameter(f"Registration JSON field {category!r} must be a list.")
        for item in items:
            if not isinstance(item, dict):
                continue
            client_metadata = item.get("clientMetadata")
            if not isinstance(client_metadata, dict) or not client_metadata:
                continue
            title = str(item.get("title") or "")
            source_key = str(client_metadata.get("source_key") or title)
            object_url = str(item.get("objectUrl") or client_metadata.get("source_uri") or "")
            records.append(
                RegistrationMetadata(
                    title=title,
                    source_key=source_key,
                    object_url=object_url,
                    client_metadata=client_metadata,
                )
            )
    return records


def registration_lookup(records: list[RegistrationMetadata]) -> tuple[dict[str, RegistrationMetadata], dict[str, list[RegistrationMetadata]]]:
    exact: dict[str, RegistrationMetadata] = {}
    basename: dict[str, list[RegistrationMetadata]] = {}

    for record in records:
        for key in candidate_keys(record.title, record.source_key, record.object_url):
            exact.setdefault(key, record)
        name = PurePosixPath(record.source_key or record.title).name
        if name:
            basename.setdefault(name, []).append(record)

    return exact, basename


def candidate_keys(*values: str | None) -> set[str]:
    keys: set[str] = set()
    for value in values:
        normalized = normalize_key(value)
        if not normalized:
            continue
        keys.add(normalized)
        parts = PurePosixPath(normalized).parts
        if "raw-feed" in parts:
            keys.add("/".join(parts[parts.index("raw-feed") :]))
    return keys


def item_candidate_keys(item) -> set[str]:
    keys = candidate_keys(getattr(item, "name", None), getattr(item, "url", None))
    for key in list(keys):
        parts = PurePosixPath(key).parts
        if "raw-feed" in parts:
            keys.add("/".join(parts[parts.index("raw-feed") :]))
    return keys


def metadata_is_current(existing: dict[str, Any], target: dict[str, Any]) -> bool:
    return all(existing.get(key) == value for key, value in target.items())


def merged_metadata(existing: dict[str, Any] | None, target: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(target)
    return merged


def match_registration(item, exact: dict[str, RegistrationMetadata], basename: dict[str, list[RegistrationMetadata]]) -> tuple[RegistrationMetadata | None, str]:
    for key in item_candidate_keys(item):
        record = exact.get(key)
        if record:
            return record, "exact"

    name = getattr(item, "name", "")
    matches = basename.get(name, [])
    if len(matches) == 1:
        return matches[0], "basename"
    if len(matches) > 1:
        return None, "ambiguous_basename"
    return None, "missing"


def update_items(
    folder,
    records: list[RegistrationMetadata],
    dry_run: bool,
    page_size: int,
    bundle_size: int,
    progress_interval: int,
) -> dict[str, Any]:
    exact, basename = registration_lookup(records)
    report: dict[str, Any] = {
        "registration_records": len(records),
        "listed_items": 0,
        "matched_exact": 0,
        "matched_basename": 0,
        "updated": 0,
        "would_update": 0,
        "skipped_current": 0,
        "missing_registration": [],
        "ambiguous": [],
        "errors": [],
        "dry_run": dry_run,
    }

    bundle = Bundle(bundle_size=bundle_size)
    pending = 0
    pending_items: list[dict[str, str]] = []

    def echo_progress(force: bool = False) -> None:
        listed = report["listed_items"]
        if not listed:
            return
        if not force and (progress_interval <= 0 or listed % progress_interval != 0):
            return
        matched = report["matched_exact"] + report["matched_basename"]
        changed = report["would_update"] if dry_run else report["updated"] + pending
        action = "would_update" if dry_run else "updated_or_queued"
        typer.echo(
            f"Progress: listed={listed:,} matched={matched:,} {action}={changed:,} "
            f"skipped={report['skipped_current']:,} missing={len(report['missing_registration']):,} "
            f"ambiguous={len(report['ambiguous']):,} errors={len(report['errors']):,}",
            err=True,
        )

    def flush() -> None:
        nonlocal bundle, pending, pending_items
        if pending and not dry_run:
            try:
                bundle.execute()
                report["updated"] += pending
            except Exception as exc:
                report["errors"].append(
                    {
                        "error": str(exc),
                        "item_count": pending,
                        "items": pending_items,
                    }
                )
        bundle = Bundle(bundle_size=bundle_size)
        pending = 0
        pending_items = []

    for item in folder.find_items(item_types=ITEM_TYPES, page_size=page_size):
        report["listed_items"] += 1
        record, match_type = match_registration(item, exact, basename)
        if record is None:
            detail = {"item_uuid": str(item.uuid), "name": item.name, "url": item.url, "reason": match_type}
            if match_type == "ambiguous_basename":
                report["ambiguous"].append(detail)
            else:
                report["missing_registration"].append(detail)
            echo_progress()
            continue

        if match_type == "exact":
            report["matched_exact"] += 1
        else:
            report["matched_basename"] += 1

        existing = item.client_metadata or {}
        if metadata_is_current(existing, record.client_metadata):
            report["skipped_current"] += 1
            echo_progress()
            continue

        if dry_run:
            report["would_update"] += 1
            echo_progress()
            continue

        try:
            item.update(client_metadata=merged_metadata(existing, record.client_metadata), bundle=bundle)
            pending += 1
            pending_items.append({"item_uuid": str(item.uuid), "name": item.name})
            if pending >= bundle_size:
                flush()
        except Exception as exc:
            report["errors"].append({"item_uuid": str(item.uuid), "name": item.name, "error": str(exc)})

        echo_progress()

    flush()
    echo_progress(force=True)
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main(
    registration_json: Annotated[
        Path,
        typer.Argument(help="Registration JSON with desired clientMetadata."),
    ] = Path(DEFAULT_REGISTRATION_JSON),
    folder_hash: Annotated[
        str,
        typer.Option("--folder-hash", help="Cloud Synced Folder hash to update."),
    ] = "cdb6587a-d00b-4446-a3a9-16d2b8babbda",
    report_json: Annotated[
        Path | None,
        typer.Option("--report-json", help="Path for update report JSON."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Plan updates without writing metadata."),
    ] = False,
    page_size: Annotated[
        int,
        typer.Option("--page-size", help="Encord item listing page size."),
    ] = 1000,
    bundle_size: Annotated[
        int,
        typer.Option("--bundle-size", help="Number of item metadata patches per bundled API call."),
    ] = 1000,
    progress_interval: Annotated[
        int,
        typer.Option("--progress-interval", help="Print progress every N listed items. Use 0 to disable."),
    ] = 2000,
) -> None:
    if not folder_hash:
        raise typer.BadParameter("Pass --folder-hash for the Cloud Synced Folder to update.")

    records = load_registration_metadata(registration_json)
    if not records:
        raise typer.BadParameter(f"No clientMetadata records found in {registration_json}.")

    client = get_client()
    folder = client.get_storage_folder(folder_hash)
    report = update_items(
        folder,
        records,
        dry_run=dry_run,
        page_size=page_size,
        bundle_size=bundle_size,
        progress_interval=progress_interval,
    )
    report["folder_hash"] = folder_hash
    report["registration_json"] = str(registration_json)

    output = report_json or report_path_for(registration_json)
    write_report(output, report)

    typer.echo(f"Listed items: {report['listed_items']}")
    if dry_run:
        typer.echo(f"Would update: {report['would_update']}")
    else:
        typer.echo(f"Updated: {report['updated']}")
    typer.echo(f"Skipped current: {report['skipped_current']}")
    typer.echo(f"Missing registration: {len(report['missing_registration'])}")
    typer.echo(f"Ambiguous matches: {len(report['ambiguous'])}")
    typer.echo(f"Errors: {len(report['errors'])}")
    typer.echo(f"Report JSON: {output}")


if __name__ == "__main__":
    typer.run(main)
