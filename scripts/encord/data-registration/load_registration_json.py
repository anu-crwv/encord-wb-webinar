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

import json
import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

import typer
from encord.orm.dataset import LongPollingStatus
from encord.user_client import EncordUserClient

DEFAULT_REGISTRATION_JSON = "registration.json"
DEFAULT_FOLDER_HASH = ""
DEFAULT_INTEGRATION_HASH = "1a2117d0-7ce1-46b7-a426-48231247585c"
ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"
DRY_RUN_ITEMS_PER_CATEGORY = 5
UPLOAD_KEYS = ["images", "videos", "audio", "text", "pdfs", "image_groups", "scenes", "data_groups"]
STALE_CLIENT_METADATA_KEYS = {"video_width", "video_height", "video_codec", "video_has_audio"}


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


def default_report_path(registration_json: Path, dry_run: bool) -> Path:
    suffix = "report_dry_run" if dry_run else "upload_report"
    return registration_json.with_name(f"{registration_json.stem}_{suffix}.json")


def dry_run_json_path(registration_json: Path) -> Path:
    return registration_json.with_name(f"{registration_json.stem}_dry_run.json")


def looks_like_dry_run_json(registration_json: Path) -> bool:
    return registration_json.stem.endswith("_dry_run")


def evenly_sample(items: list[Any], max_items: int) -> list[Any]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return items[:max_items]
    last = len(items) - 1
    indexes = [round(i * last / (max_items - 1)) for i in range(max_items)]
    return [items[index] for index in indexes]


def build_dry_run_registration_json(registration_json: Path, output_json: Path) -> Path:
    source = json.loads(registration_json.read_text())
    subset: dict[str, Any] = {}
    counts: dict[str, int] = {}

    for key, value in source.items():
        if isinstance(value, list):
            sampled = evenly_sample(value, DRY_RUN_ITEMS_PER_CATEGORY)
            subset[key] = sampled
            if sampled:
                counts[key] = len(sampled)
        else:
            subset[key] = value

    output_json.write_text(json.dumps(subset, indent=2))
    typer.echo(f"Dry run JSON: {output_json}")
    typer.echo(f"Dry run item counts: {counts}")
    return output_json


def load_registration_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"Registration JSON not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Registration JSON must be an object: {path}")
    return data


def iter_upload_items(registration: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for category in UPLOAD_KEYS:
        items = registration.get(category, [])
        if not isinstance(items, list):
            raise typer.BadParameter(f"Registration JSON field {category!r} must be a list.")
        for item in items:
            if isinstance(item, dict):
                yield category, item


def validate_registration_json(registration_json: Path, allow_invalid: bool) -> None:
    registration = load_registration_json(registration_json)
    stale_client_metadata = 0
    jsonl_without_text_metadata = 0
    videos_without_video_metadata = 0

    for category, item in iter_upload_items(registration):
        title = str(item.get("title") or item.get("objectUrl") or "")
        client_metadata = item.get("clientMetadata") or {}
        if isinstance(client_metadata, dict) and STALE_CLIENT_METADATA_KEYS & set(client_metadata):
            stale_client_metadata += 1
        if category == "text" and title.lower().endswith(".jsonl") and not item.get("textMetadata"):
            jsonl_without_text_metadata += 1
        if category == "videos" and not item.get("videoMetadata"):
            videos_without_video_metadata += 1

    blocking_issues = []
    if stale_client_metadata:
        blocking_issues.append(f"{stale_client_metadata} item(s) contain stale video fields in clientMetadata")
    if jsonl_without_text_metadata:
        blocking_issues.append(f"{jsonl_without_text_metadata} .jsonl item(s) are missing textMetadata")
    if videos_without_video_metadata:
        blocking_issues.append(f"{videos_without_video_metadata} video item(s) are missing videoMetadata")

    if blocking_issues and not allow_invalid:
        message = "; ".join(blocking_issues)
        raise typer.BadParameter(
            f"Registration JSON looks stale: {message}. Rebuild it with build_registration_json.py, "
            "or pass --allow-invalid-registration-json to upload it anyway."
        )

    if blocking_issues:
        typer.echo(f"Registration validation warning: {'; '.join(blocking_issues)}.", err=True)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {k: json_safe(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def unit_error_to_dict(error: Any) -> dict[str, Any]:
    object_urls = list(getattr(error, "object_urls", []) or [])
    return {
        "error": str(getattr(error, "error", error)),
        "object_url_count": len(object_urls),
        "object_urls": object_urls,
        "failed_objects": [parse_object_url(url) for url in object_urls],
    }


def parse_object_url(object_url: str) -> dict[str, str]:
    parsed = urlparse(object_url)
    host = parsed.netloc
    bucket = host.split(".s3.", 1)[0] if ".s3." in host else ""
    key = unquote(parsed.path.lstrip("/"))
    return {
        "object_url": object_url,
        "bucket": bucket,
        "key": key,
        "extension": Path(key).suffix.lower(),
    }


def build_report(result: Any, registration_json: Path, storage_folder: Any) -> dict[str, Any]:
    unit_errors = [unit_error_to_dict(error) for error in getattr(result, "unit_errors", []) or []]
    return {
        "registration_json": str(registration_json),
        "storage_folder_hash": str(getattr(storage_folder, "uuid", "")),
        "status": json_safe(getattr(result, "status", None)),
        "errors": json_safe(getattr(result, "errors", [])),
        "units_pending_count": json_safe(getattr(result, "units_pending_count", 0)),
        "unit_error_count": len(unit_errors),
        "failed_object_url_count": sum(error["object_url_count"] for error in unit_errors),
        "unit_errors": unit_errors,
    }


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))


def upload_registration_json(
    storage_folder,
    integration_hash: str,
    registration_json: Path,
    report_json: Path,
) -> None:
    upload_job_id = storage_folder.add_private_data_to_folder_start(
        integration_id=integration_hash,
        private_files=str(registration_json),
        ignore_errors=True,
    )
    result = storage_folder.add_private_data_to_folder_get_result(upload_job_id)
    report = build_report(result, registration_json, storage_folder)
    write_report(report_json, report)

    if result.status == LongPollingStatus.DONE:
        typer.echo("Upload finished.")
        if report["unit_error_count"]:
            typer.echo(
                f"{report['failed_object_url_count']} object URLs failed across "
                f"{report['unit_error_count']} error group(s)."
            )
        typer.echo(f"Report JSON: {report_json}")
        return

    if result.status == LongPollingStatus.PENDING:
        raise RuntimeError(f"Upload timed out. Pending units: {result.units_pending_count}. Report JSON: {report_json}")

    raise RuntimeError(f"Upload failed. Report JSON: {report_json}")


def main(
    registration_json_arg: Annotated[
        Path | None,
        typer.Argument(help="Path to the registration JSON."),
    ] = None,
    registration_json_option: Annotated[
        Path | None,
        typer.Option("--registration-json", help="Path to the registration JSON."),
    ] = None,
    folder_hash: Annotated[
        str,
        typer.Option("--folder-hash", help="Existing Encord storage folder hash. Creates today's folder if empty."),
    ] = DEFAULT_FOLDER_HASH,
    integration_hash: Annotated[
        str,
        typer.Option("--integration-hash", help="Encord cloud integration hash."),
    ] = DEFAULT_INTEGRATION_HASH,
    report_json: Annotated[
        Path | None,
        typer.Option("--report-json", help="Path for the upload result and failure report JSON."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Upload a small subset of the registration JSON."),
    ] = False,
    allow_invalid_registration_json: Annotated[
        bool,
        typer.Option("--allow-invalid-registration-json", help="Upload even if validation detects stale JSON."),
    ] = False,
) -> None:
    if registration_json_arg and registration_json_option and registration_json_arg != registration_json_option:
        raise typer.BadParameter("Pass the registration JSON either positionally or via --registration-json, not both.")
    registration_json = registration_json_option or registration_json_arg or Path(DEFAULT_REGISTRATION_JSON)

    if not registration_json.exists():
        raise typer.BadParameter(f"Registration JSON not found: {registration_json}")
    if not integration_hash:
        raise typer.BadParameter("Set DEFAULT_INTEGRATION_HASH in this script or pass --integration-hash.")

    if dry_run and looks_like_dry_run_json(registration_json):
        raise typer.BadParameter(
            f"{registration_json} already looks like a dry-run JSON. Upload it without --dry-run, "
            "or pass the full registration JSON with --dry-run."
        )

    upload_json = (
        build_dry_run_registration_json(registration_json, dry_run_json_path(registration_json))
        if dry_run
        else registration_json
    )
    validate_registration_json(upload_json, allow_invalid=allow_invalid_registration_json)

    client = get_client()
    storage_folder = get_or_create_folder(client, folder_hash)
    report_json = report_json or default_report_path(registration_json, dry_run)
    upload_registration_json(storage_folder, integration_hash, upload_json, report_json)
    typer.echo(f"Storage folder hash: {storage_folder.uuid}")


if __name__ == "__main__":
    typer.run(main)
