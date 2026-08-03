# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "tqdm",
#     "typer",
# ]
# ///
"""Download an S3 prefix into the shared dataset-export S3 cache."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Annotated, Any, Callable
from urllib.parse import unquote, urlparse
from uuid import uuid4

import typer
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
EXPORT_ROOT = REPO_ROOT / "exports/encord-dataset-export"
S3_CACHE_ROOT = EXPORT_ROOT / "_cache" / "s3"

S3_DOWNLOAD_ATTEMPTS = 4
S3_RETRY_BASE_SECONDS = 3.0
DEFAULT_WORKERS = 32
DEFAULT_MULTIPART_CONCURRENCY = 4
DEFAULT_MULTIPART_THRESHOLD_MB = 64
DEFAULT_MULTIPART_CHUNKSIZE_MB = 32


@dataclass(frozen=True)
class S3Object:
    bucket: str
    key: str
    size: int


@dataclass(frozen=True)
class DownloadResult:
    key: str
    cache_path: Path
    size: int
    action: str
    error: str | None = None
    retries: int = 0


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise typer.BadParameter("S3 URI must include a bucket name.")
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        if not bucket:
            raise typer.BadParameter("S3 URL must include a bucket name.")
        return bucket, unquote(parsed.path.lstrip("/"))
    raise typer.BadParameter(f"Unsupported S3 URI format: {uri}")


def s3_client(
    profile: str | None,
    unsigned: bool,
    max_pool_connections: int,
    region_name: str | None = None,
):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    config: dict[str, Any] = {
        "max_pool_connections": max_pool_connections,
        "retries": {"max_attempts": 10, "mode": "standard"},
    }
    if unsigned:
        config["signature_version"] = UNSIGNED
    return session.client("s3", region_name=region_name, config=Config(**config))


def transfer_config(
    multipart_concurrency: int,
    multipart_threshold_mb: int,
    multipart_chunksize_mb: int,
):
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=multipart_threshold_mb * 1024 * 1024,
        multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
        max_concurrency=multipart_concurrency,
        use_threads=multipart_concurrency > 1,
    )


def bucket_region(client_s3: Any, bucket: str) -> str:
    response = client_s3.head_bucket(Bucket=bucket)
    headers = response["ResponseMetadata"]["HTTPHeaders"]
    return headers.get("x-amz-bucket-region", "us-east-1")


def list_top_level(client_s3: Any, bucket: str, prefix: str) -> tuple[int, int, int]:
    paginator = client_s3.get_paginator("list_objects_v2")
    files = 0
    folders = 0
    total_bytes = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        page_files = [obj for obj in page.get("Contents", []) if not obj["Key"].endswith("/")]
        files += len(page_files)
        folders += len(page.get("CommonPrefixes", []))
        total_bytes += sum(int(obj["Size"]) for obj in page_files)
    return files, folders, total_bytes


def list_recursive(
    client_s3: Any,
    bucket: str,
    prefix: str,
    max_objects: int | None,
) -> list[S3Object]:
    paginator = client_s3.get_paginator("list_objects_v2")
    objects: list[S3Object] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            objects.append(S3Object(bucket, key, int(obj["Size"])))
            if max_objects is not None and len(objects) >= max_objects:
                return objects
    return objects


def s3_cache_path(cache_root: Path, bucket: str, key: str) -> Path:
    parts = [part for part in key.split("/") if part not in {"", "."}]
    if not bucket or not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    return cache_root / bucket / Path(*parts)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def retry_reason(exc: BaseException) -> str:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return str(code) if code else type(exc).__name__


def retry_call(label: str, call: Callable[[], Any]) -> tuple[Any, int]:
    retries = 0
    for attempt in range(1, S3_DOWNLOAD_ATTEMPTS + 1):
        try:
            return call(), retries
        except Exception as exc:
            if attempt == S3_DOWNLOAD_ATTEMPTS:
                raise
            retries += 1
            sleep_seconds = S3_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            typer.echo(
                f"Warning: {label} failed with {retry_reason(exc)}: {exc}; "
                f"retrying in {sleep_seconds:.0f}s ({attempt}/{S3_DOWNLOAD_ATTEMPTS})",
                err=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Unreachable retry state for {label}")


def download_to_cache(
    client_s3: Any,
    obj: S3Object,
    cache_root: Path,
    boto_transfer_config: Any,
    dry_run: bool,
    overwrite: bool,
) -> DownloadResult:
    cache_path = s3_cache_path(cache_root, obj.bucket, obj.key)
    exists = cache_path.exists()

    if exists and not overwrite:
        cached_size = cache_path.stat().st_size
        if cached_size == obj.size:
            return DownloadResult(obj.key, cache_path, obj.size, "cached")
        return DownloadResult(
            obj.key,
            cache_path,
            obj.size,
            "size_conflict",
            f"existing file is {cached_size} bytes, S3 object is {obj.size} bytes",
        )

    if dry_run:
        action = "would_overwrite" if exists else "would_download"
        return DownloadResult(obj.key, cache_path, obj.size, action)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    try:
        client_s3.download_file(
            obj.bucket,
            obj.key,
            str(tmp_path),
            Config=boto_transfer_config,
        )
        downloaded_size = tmp_path.stat().st_size
        if downloaded_size != obj.size:
            raise RuntimeError(f"downloaded {downloaded_size} bytes, expected {obj.size} bytes")
        os.replace(tmp_path, cache_path)
        return DownloadResult(obj.key, cache_path, obj.size, "overwritten" if exists else "downloaded")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def cache_one(
    client_s3: Any,
    obj: S3Object,
    cache_root: Path,
    boto_transfer_config: Any,
    dry_run: bool,
    overwrite: bool,
) -> DownloadResult:
    try:
        result, retries = retry_call(
            f"download s3://{obj.bucket}/{obj.key}",
            lambda: download_to_cache(
                client_s3,
                obj,
                cache_root,
                boto_transfer_config,
                dry_run,
                overwrite,
            ),
        )
        return DownloadResult(
            result.key,
            result.cache_path,
            result.size,
            result.action,
            result.error,
            retries,
        )
    except Exception as exc:
        return DownloadResult(
            obj.key,
            s3_cache_path(cache_root, obj.bucket, obj.key),
            obj.size,
            "failed",
            f"{type(exc).__name__}: {exc}",
            S3_DOWNLOAD_ATTEMPTS - 1,
        )


def progress_postfix(
    counts: Counter[str],
    retries: int,
    completed_files: int,
    total_files: int,
) -> dict[str, Any]:
    postfix: dict[str, Any] = {"files": f"{completed_files}/{total_files}", "retries": retries}
    postfix.update(counts)
    return postfix


def download_objects(
    client_s3: Any,
    objects: list[S3Object],
    cache_root: Path,
    boto_transfer_config: Any,
    workers: int,
    dry_run: bool,
    overwrite: bool,
) -> list[DownloadResult]:
    total_bytes = sum(obj.size for obj in objects)
    results: list[DownloadResult] = []
    counts: Counter[str] = Counter()
    retry_count = 0
    completed_files = 0

    with tqdm(
        total=total_bytes,
        desc="Caching S3 bytes",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=5.0,
    ) as progress:
        if workers == 1:
            for obj in objects:
                result = cache_one(
                    client_s3,
                    obj,
                    cache_root,
                    boto_transfer_config,
                    dry_run,
                    overwrite,
                )
                results.append(result)
                completed_files += 1
                counts[result.action] += 1
                retry_count += result.retries
                if result.action != "failed":
                    progress.update(result.size)
                progress.set_postfix(
                    progress_postfix(counts, retry_count, completed_files, len(objects)),
                    refresh=False,
                )
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            object_iter = iter(objects)
            pending: set[Future[DownloadResult]] = set()
            max_pending = workers * 4

            def submit_next() -> bool:
                try:
                    obj = next(object_iter)
                except StopIteration:
                    return False
                pending.add(
                    pool.submit(
                        cache_one,
                        client_s3,
                        obj,
                        cache_root,
                        boto_transfer_config,
                        dry_run,
                        overwrite,
                    )
                )
                return True

            for _ in range(max_pending):
                if not submit_next():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    completed_files += 1
                    counts[result.action] += 1
                    retry_count += result.retries
                    if result.action != "failed":
                        progress.update(result.size)
                    progress.set_postfix(
                        progress_postfix(counts, retry_count, completed_files, len(objects)),
                        refresh=False,
                    )

                while len(pending) < max_pending and submit_next():
                    pass

    return results


def cache_prefix_path(cache_root: Path, bucket: str, prefix: str) -> Path:
    clean_prefix = prefix.strip("/")
    if not clean_prefix:
        return cache_root / bucket
    return cache_root / bucket / Path(*clean_prefix.split("/"))


def print_failures(results: list[DownloadResult], max_examples: int = 20) -> None:
    failures = [result for result in results if result.action in {"failed", "size_conflict"}]
    if not failures:
        return
    typer.echo("\nFailures/conflicts:", err=True)
    for result in failures[:max_examples]:
        typer.echo(f"  {result.action}: {result.key} -> {result.error}", err=True)
    if len(failures) > max_examples:
        typer.echo(f"  ... {len(failures) - max_examples} more", err=True)


def auto_max_pool_connections(workers: int, multipart_concurrency: int) -> int:
    return max(64, workers * multipart_concurrency + workers + 16)


def main(
    s3_uri: Annotated[
        str,
        typer.Argument(help="S3 folder/prefix to cache, e.g. s3://bucket/path/to/folder/"),
    ],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="AWS profile name. Omit to use default AWS resolution."),
    ] = None,
    cache_root: Annotated[
        Path,
        typer.Option(help="Shared S3 cache root mirroring s3://bucket/key."),
    ] = S3_CACHE_ROOT,
    workers: Annotated[int, typer.Option(help="Parallel file download workers.")] = DEFAULT_WORKERS,
    multipart_concurrency: Annotated[
        int,
        typer.Option(help="Per-large-object multipart download threads."),
    ] = DEFAULT_MULTIPART_CONCURRENCY,
    multipart_threshold_mb: Annotated[
        int,
        typer.Option(help="Object size threshold for multipart downloads, in MiB."),
    ] = DEFAULT_MULTIPART_THRESHOLD_MB,
    multipart_chunksize_mb: Annotated[
        int,
        typer.Option(help="Multipart download chunk size, in MiB."),
    ] = DEFAULT_MULTIPART_CHUNKSIZE_MB,
    max_pool_connections: Annotated[
        int | None,
        typer.Option(help="S3 HTTP connection pool size. Defaults from workers and multipart settings."),
    ] = None,
    max_objects: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of objects to list/download."),
    ] = None,
    skip_top_level_summary: Annotated[
        bool,
        typer.Option(help="Skip the extra Delimiter='/' listing pass once you trust the prefix."),
    ] = False,
    unsigned_s3: Annotated[
        bool,
        typer.Option(help="Use unsigned S3 requests for public buckets."),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Redownload and replace cache files that already exist."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="List planned cache writes without downloading."),
    ] = False,
) -> None:
    if workers < 1:
        raise typer.BadParameter("workers must be at least 1")
    if multipart_concurrency < 1:
        raise typer.BadParameter("multipart concurrency must be at least 1")
    if multipart_threshold_mb < 1:
        raise typer.BadParameter("multipart threshold must be at least 1 MiB")
    if multipart_chunksize_mb < 1:
        raise typer.BadParameter("multipart chunk size must be at least 1 MiB")
    if max_pool_connections is None:
        max_pool_connections = auto_max_pool_connections(workers, multipart_concurrency)
    if max_pool_connections < workers:
        raise typer.BadParameter("max pool connections must be at least workers")

    bucket, prefix = parse_s3_uri(s3_uri)
    if prefix and not prefix.endswith("/"):
        typer.echo(
            "Warning: prefix does not end with '/'; S3 will match any key beginning with "
            f"{prefix!r}.",
            err=True,
        )

    cache_root = cache_root.expanduser().resolve()
    bootstrap_s3 = s3_client(profile, unsigned_s3, max_pool_connections)
    region = bucket_region(bootstrap_s3, bucket)
    client_s3 = s3_client(profile, unsigned_s3, max_pool_connections, region_name=region)
    boto_transfer_config = transfer_config(
        multipart_concurrency,
        multipart_threshold_mb,
        multipart_chunksize_mb,
    )

    typer.echo(f"Bucket: {bucket}")
    typer.echo(f"Prefix: {prefix or '[bucket root]'}")
    typer.echo(f"Region: {region}")
    typer.echo(f"Cache root: {cache_root}")
    typer.echo(f"Prefix cache path: {cache_prefix_path(cache_root, bucket, prefix)}")
    typer.echo(
        "Download settings: "
        f"workers={workers}, multipart_concurrency={multipart_concurrency}, "
        f"multipart_threshold={multipart_threshold_mb} MiB, "
        f"multipart_chunksize={multipart_chunksize_mb} MiB, "
        f"max_pool_connections={max_pool_connections}"
    )

    if skip_top_level_summary:
        typer.echo("Skipping top-level summary listing.")
    else:
        files, folders, direct_bytes = list_top_level(client_s3, bucket, prefix)
        typer.echo(
            f"Top-level under prefix: {files:,} files, {folders:,} folders, "
            f"{format_bytes(direct_bytes)} directly in this folder."
        )

    objects = list_recursive(client_s3, bucket, prefix, max_objects)
    total_bytes = sum(obj.size for obj in objects)
    typer.echo(f"Recursive objects selected: {len(objects):,} ({format_bytes(total_bytes)})")
    if max_objects is not None and len(objects) == max_objects:
        typer.echo(f"Stopped at --max-objects={max_objects:,}; prefix may contain more.", err=True)
    if not objects:
        typer.echo("No objects found.")
        return
    if dry_run:
        typer.echo("Dry run enabled; no files will be downloaded.")

    started_at = time.monotonic()
    results = download_objects(
        client_s3,
        objects,
        cache_root,
        boto_transfer_config,
        workers,
        dry_run,
        overwrite,
    )
    elapsed = max(time.monotonic() - started_at, 0.001)

    counts = Counter(result.action for result in results)
    retries = sum(result.retries for result in results)
    downloaded_bytes = sum(
        result.size for result in results if result.action in {"downloaded", "overwritten"}
    )

    typer.echo("\nSummary:")
    for action in sorted(counts):
        typer.echo(f"  {action}: {counts[action]:,}")
    typer.echo(f"  retries: {retries:,}")
    typer.echo(f"  downloaded: {format_bytes(downloaded_bytes)}")
    typer.echo(f"  elapsed: {elapsed:.1f}s")
    typer.echo(f"  average download rate: {downloaded_bytes / 1024 / 1024 / elapsed:.1f} MiB/s")

    print_failures(results)
    if counts.get("failed", 0) or counts.get("size_conflict", 0):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
