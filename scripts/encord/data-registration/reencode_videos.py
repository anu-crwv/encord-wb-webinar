# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "requests",
#     "tqdm",
#     "typer",
# ]
# ///
"""Download Encord videos, re-encode them to MP4, and upload them to a regular folder.

Set your Encord key once:
    export ENCORD_SSH_KEY_FILE=/path/to/encord_key

Run from this directory:
    uv run --script reencode_videos.py <dataset_or_folder_hash>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Semaphore
from typing import Annotated, Any, Iterable, Sequence
from uuid import UUID

import requests
import typer
from encord.constants.enums import DataType
from encord.http.utils import CloudUploadSettings
from encord.orm.dataset import DataLinkDuplicatesBehavior
from encord.storage import StorageFolder, StorageItem, StorageItemType
from encord.user_client import EncordUserClient
from tqdm import tqdm

ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"
WORKERS_ENV = "ENCORD_REENCODE_WORKERS"
UPLOAD_WORKERS_ENV = "ENCORD_REENCODE_UPLOAD_WORKERS"
TMPDIR_ENV = "ENCORD_REENCODE_TMPDIR"
REPORT_NAME = "reencode_videos_report.json"
DOWNLOAD_CHUNK_SIZE = 16 * 1024 * 1024
BULK_GET_CHUNK_SIZE = 500
LINK_CHUNK_SIZE = 1_000
REQUEST_TIMEOUT = (30, 300)
UPLOAD_SETTINGS = CloudUploadSettings(max_retries=5, backoff_factor=1.0, allow_failures=False)


@dataclass(frozen=True)
class Source:
    kind: str
    dataset: Any | None = None
    folder: StorageFolder | None = None


@dataclass(frozen=True)
class VideoJob:
    item: StorageItem
    source_title: str
    dataset_data_hash: str | None = None
    target_folder: StorageFolder | None = None


@dataclass(frozen=True)
class ProcessedVideo:
    source_item_uuid: str
    source_title: str
    source_data_hash: str | None
    new_item_uuid: str
    new_title: str
    target_folder_uuid: str
    audio_mode: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_client() -> EncordUserClient:
    ssh_key_file = os.environ.get(ENCORD_SSH_KEY_ENV)
    if not ssh_key_file:
        raise typer.BadParameter(f"Set {ENCORD_SSH_KEY_ENV} to the path of your Encord SSH private key.")
    return EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_key_file)


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise typer.BadParameter("ffmpeg is required on PATH.")


def worker_count() -> int:
    raw = os.environ.get(WORKERS_ENV, "4")
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise typer.BadParameter(f"{WORKERS_ENV} must be an integer.") from exc


def upload_worker_count(workers: int) -> int:
    raw = os.environ.get(UPLOAD_WORKERS_ENV, "2")
    try:
        return min(workers, max(1, int(raw)))
    except ValueError as exc:
        raise typer.BadParameter(f"{UPLOAD_WORKERS_ENV} must be an integer.") from exc


def temp_root() -> Path | None:
    raw = os.environ.get(TMPDIR_ENV)
    if not raw:
        return None
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunked(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def resolve_source(client: EncordUserClient, source_hash: str) -> Source:
    dataset_error = ""
    try:
        return Source(kind="dataset", dataset=client.get_dataset(source_hash))
    except Exception as exc:  # noqa: BLE001 - we want to fall through to folder lookup.
        dataset_error = str(exc)

    folder_error = ""
    try:
        return Source(kind="folder", folder=client.get_storage_folder(source_hash))
    except Exception as exc:  # noqa: BLE001 - report both lookup failures below.
        folder_error = str(exc)

    raise typer.BadParameter(
        "Hash was not found as a dataset or storage folder. "
        f"Dataset lookup error: {dataset_error}; folder lookup error: {folder_error}"
    )


def already_reencoded(title: str) -> bool:
    return PurePosixPath(title).name.lower().endswith(" (re-encoded).mp4")


def reencoded_title(title: str) -> str:
    path = PurePosixPath(title)
    name = path.name or "video"
    suffix = PurePosixPath(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    new_name = f"{stem} (re-encoded).mp4"
    if str(path.parent) == ".":
        return new_name
    return str(path.with_name(new_name))


def item_title(item: StorageItem, fallback: str | None = None) -> str:
    return str(getattr(item, "name", None) or fallback or getattr(item, "uuid", "video"))


def source_title(source: Source, source_hash: str) -> str:
    if source.kind == "dataset":
        return str(getattr(source.dataset, "title", None) or source_hash)
    return str(getattr(source.folder, "name", None) or source_hash)


def create_output_folder(client: EncordUserClient, source: Source, source_hash: str) -> StorageFolder:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{source_title(source, source_hash)} re-encoded {timestamp}"
    return client.create_storage_folder(name=name)


def get_storage_items_lenient(
    client: EncordUserClient,
    item_ids: Sequence[str],
    skipped: list[dict[str, str]],
) -> list[StorageItem]:
    try:
        return client.get_storage_items(item_ids, sign_url=False)
    except Exception:  # noqa: BLE001 - fall back to per-item lookup below.
        items: list[StorageItem] = []
        for item_id in item_ids:
            try:
                items.append(client.get_storage_item(item_id, sign_url=False))
            except Exception as exc:  # noqa: BLE001 - keep going when one item is inaccessible.
                skipped.append(
                    {"source_item_uuid": item_id, "reason": "storage_item_lookup_failed", "error": str(exc)}
                )
        return items


def discover_dataset_jobs(
    client: EncordUserClient,
    dataset: Any,
) -> tuple[list[VideoJob], list[dict[str, str]]]:
    rows = dataset.list_data_rows(data_types=[DataType.VIDEO])
    skipped: list[dict[str, str]] = []
    row_by_item_uuid: dict[str, Any] = {}

    for row in rows:
        if not row.backing_item_uuid:
            skipped.append({"data_hash": row.uid, "title": row.title, "reason": "missing_backing_item_uuid"})
            continue
        row_by_item_uuid.setdefault(str(row.backing_item_uuid), row)

    jobs: list[VideoJob] = []
    item_ids = list(row_by_item_uuid)
    for ids in chunked(item_ids, BULK_GET_CHUNK_SIZE):
        for item in get_storage_items_lenient(client, ids, skipped):
            row = row_by_item_uuid.get(str(item.uuid))
            title = item_title(item, getattr(row, "title", None))
            if already_reencoded(title):
                skipped.append({"source_item_uuid": str(item.uuid), "title": title, "reason": "already_reencoded"})
                continue
            jobs.append(VideoJob(item=item, source_title=title, dataset_data_hash=getattr(row, "uid", None)))

    return jobs, skipped


def discover_folder_jobs(folder: StorageFolder) -> tuple[list[VideoJob], list[dict[str, str]]]:
    jobs: list[VideoJob] = []
    skipped: list[dict[str, str]] = []

    for item in folder.list_items(item_types=[StorageItemType.VIDEO], page_size=1_000):
        title = item_title(item)
        if already_reencoded(title):
            skipped.append({"source_item_uuid": str(item.uuid), "title": title, "reason": "already_reencoded"})
            continue
        jobs.append(VideoJob(item=item, source_title=title, target_folder=folder))

    return jobs, skipped


def download_item(item: StorageItem, output_path: Path) -> None:
    signed_url = item.get_signed_url(refetch=True)
    if not signed_url:
        raise RuntimeError("No signed URL available for source item.")

    with requests.get(signed_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with output_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    output.write(chunk)


def ffmpeg_command(input_path: Path, output_path: Path, copy_audio: bool) -> list[str]:
    audio_args = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "128k"]
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        *audio_args,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def run_ffmpeg(input_path: Path, output_path: Path) -> str:
    result = subprocess.run(
        ffmpeg_command(input_path, output_path, copy_audio=True),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return "copy"

    if output_path.exists():
        output_path.unlink()

    fallback = subprocess.run(
        ffmpeg_command(input_path, output_path, copy_audio=False),
        capture_output=True,
        text=True,
    )
    if fallback.returncode == 0:
        return "aac"

    stderr = (fallback.stderr or result.stderr or "").strip()
    raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")


def with_target_folder(jobs: list[VideoJob], target_folder: StorageFolder) -> list[VideoJob]:
    return [replace(job, target_folder=target_folder) for job in jobs]


def process_video(job: VideoJob, root: Path | None, upload_gate: Semaphore) -> ProcessedVideo:
    title = job.source_title
    new_title = reencoded_title(title)
    source_suffix = PurePosixPath(title).suffix or ".video"

    with tempfile.TemporaryDirectory(prefix="encord-reencode-", dir=root) as item_tmp:
        workdir = Path(item_tmp)
        input_path = workdir / f"source{source_suffix}"
        output_path = workdir / "output.mp4"

        download_item(job.item, input_path)
        audio_mode = run_ffmpeg(input_path, output_path)

        if job.target_folder is None:
            raise RuntimeError("Missing output storage folder.")

        with upload_gate:
            new_item_uuid = job.target_folder.upload_video(
                output_path,
                title=new_title,
                cloud_upload_settings=UPLOAD_SETTINGS,
            )

    return ProcessedVideo(
        source_item_uuid=str(job.item.uuid),
        source_title=title,
        source_data_hash=job.dataset_data_hash,
        new_item_uuid=str(new_item_uuid),
        new_title=new_title,
        target_folder_uuid=str(job.target_folder.uuid),
        audio_mode=audio_mode,
    )


def process_jobs(
    jobs: list[VideoJob],
    workers: int,
    upload_workers: int,
    root: Path | None,
) -> tuple[list[ProcessedVideo], list[dict[str, str]]]:
    processed: list[ProcessedVideo] = []
    failed: list[dict[str, str]] = []
    upload_gate = Semaphore(upload_workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video, job, root, upload_gate): job for job in jobs}
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Re-encoding videos",
            unit="video",
            dynamic_ncols=True,
        )
        for future in progress:
            job = futures[future]
            try:
                processed.append(future.result())
            except Exception as exc:  # noqa: BLE001 - one bad video should not stop the batch.
                error = str(exc)
                failed.append(
                    {
                        "source_item_uuid": str(job.item.uuid),
                        "title": job.source_title,
                        "error": error,
                    }
                )
            progress.set_postfix(uploaded=len(processed), failed=len(failed))

    return processed, failed


def link_dataset_items(dataset: Any, processed: list[ProcessedVideo]) -> tuple[int, list[dict[str, str]]]:
    linked = 0
    errors: list[dict[str, str]] = []
    item_uuids = [UUID(item.new_item_uuid) for item in processed]
    chunks = list(chunked(item_uuids, LINK_CHUNK_SIZE))

    for ids in tqdm(chunks, desc="Linking dataset", unit="chunk", dynamic_ncols=True):
        try:
            rows = dataset.link_items(list(ids), duplicates_behavior=DataLinkDuplicatesBehavior.SKIP)
            linked += len(rows)
        except Exception as exc:  # noqa: BLE001 - keep the upload report even if linking fails.
            errors.append({"item_uuids": [str(item_id) for item_id in ids], "error": str(exc)})

    return linked, errors


def report_path() -> Path:
    return Path(__file__).with_name(REPORT_NAME)


def write_report(report: dict[str, Any]) -> Path:
    path = report_path()
    path.write_text(json.dumps(report, indent=2))
    return path


def main(
    source_hash: Annotated[str, typer.Argument(help="Encord dataset hash or storage folder UUID.")],
) -> None:
    require_ffmpeg()
    workers = worker_count()
    upload_workers = upload_worker_count(workers)
    root = temp_root()
    client = get_client()
    source = resolve_source(client, source_hash)

    if source.kind == "dataset":
        jobs, skipped = discover_dataset_jobs(client, source.dataset)
    else:
        jobs, skipped = discover_folder_jobs(source.folder)

    typer.echo(f"Source: {source.kind} {source_hash}")
    typer.echo(
        f"Videos to process: {len(jobs):,}; skipped: {len(skipped):,}; "
        f"workers: {workers}; upload_workers: {upload_workers}"
    )

    if not jobs:
        report = {
            "source_hash": source_hash,
            "source_kind": source.kind,
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "workers": workers,
            "upload_workers": upload_workers,
            "found_videos": 0,
            "skipped": skipped,
            "processed": [],
            "failed": [],
            "linked_to_dataset": 0,
            "link_errors": [],
        }
        path = write_report(report)
        typer.echo(f"No videos to process. Report JSON: {path}")
        return

    output_folder = create_output_folder(client, source, source_hash)
    jobs = with_target_folder(jobs, output_folder)
    typer.echo(f"Output folder: {output_folder.name} ({output_folder.uuid})")

    report: dict[str, Any] = {
        "source_hash": source_hash,
        "source_kind": source.kind,
        "output_folder_hash": str(output_folder.uuid),
        "output_folder_name": output_folder.name,
        "started_at": utc_now(),
        "workers": workers,
        "upload_workers": upload_workers,
        "found_videos": len(jobs),
        "skipped": skipped,
        "processed": [],
        "failed": [],
        "linked_to_dataset": 0,
        "link_errors": [],
    }

    processed, failed = process_jobs(jobs, workers, upload_workers, root)
    report["processed"] = [video.__dict__ for video in processed]
    report["failed"] = failed

    if source.kind == "dataset" and processed:
        linked, link_errors = link_dataset_items(source.dataset, processed)
        report["linked_to_dataset"] = linked
        report["link_errors"] = link_errors

    report["finished_at"] = utc_now()
    path = write_report(report)

    typer.echo(
        f"Done. uploaded={len(processed):,} failed={len(failed):,} "
        f"linked_to_dataset={report['linked_to_dataset']:,}"
    )
    typer.echo(f"Report JSON: {path}")

    if failed or report["link_errors"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
