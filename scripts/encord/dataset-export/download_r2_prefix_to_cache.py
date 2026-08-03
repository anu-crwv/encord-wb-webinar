# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "tqdm",
#     "typer",
# ]
# ///
"""Download a Cloudflare R2 prefix into the shared dataset-export R2 cache."""

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
R2_CACHE_ROOT = EXPORT_ROOT / "_cache" / "r2"

R2_DOWNLOAD_ATTEMPTS = 4
R2_RETRY_BASE_SECONDS = 3.0
DEFAULT_WORKERS = 32
DEFAULT_MULTIPART_CONCURRENCY = 4
DEFAULT_MULTIPART_THRESHOLD_MB = 64
DEFAULT_MULTIPART_CHUNKSIZE_MB = 32


@dataclass(frozen=True)
class R2Object:
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


def parse_r2_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "r2":
        if not parsed.netloc:
            raise typer.BadParameter("R2 URI must include a bucket name.")
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise typer.BadParameter("S3-style R2 URI must include a bucket name.")
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"}:
        parts = parsed.path.lstrip("/").split("/", 1)
        if len(parts) != 2 or not parts[0]:
            raise typer.BadParameter("R2 HTTPS URL must look like https://.../<bucket>/<prefix>")
        return parts[0], unquote(parts[1])
    raise typer.BadParameter(f"Unsupported R2 URI format: {uri}")


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def r2_endpoint_url(account_id: str | None, endpoint_url: str | None) -> str:
    if endpoint_url:
        return endpoint_url.rstrip("/")
    if not account_id:
        raise typer.BadParameter(
            "Set CLOUDFLARE_ACCOUNT_ID/R2_ACCOUNT_ID or pass --account-id/--endpoint-url."
        )
    return f"https://{account_id}.r2.cloudflarestorage.com"


def r2_client(
    *,
    endpoint_url: str,
    access_key_id: str | None,
    secret_access_key: str | None,
    max_pool_connections: int,
):
    import boto3
    from botocore.config import Config

    if not access_key_id:
        raise typer.BadParameter(
            "Set R2_ACCESS_KEY_ID/CLOUDFLARE_R2_ACCESS_KEY_ID/AWS_ACCESS_KEY_ID "
            "or pass --access-key-id."
        )
    if not secret_access_key:
        raise typer.BadParameter(
            "Set R2_SECRET_ACCESS_KEY/CLOUDFLARE_R2_SECRET_ACCESS_KEY/AWS_SECRET_ACCESS_KEY "
            "or pass --secret-access-key."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name="auto",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(
            max_pool_connections=max_pool_connections,
            retries={"max_attempts": 10, "mode": "standard"},
            signature_version="s3v4",
        ),
    )


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


def list_top_level(client_r2: Any, bucket: str, prefix: str) -> tuple[int, int, int]:
    paginator = client_r2.get_paginator("list_objects_v2")
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
    client_r2: Any,
    bucket: str,
    prefix: str,
    max_objects: int | None,
) -> list[R2Object]:
    paginator = client_r2.get_paginator("list_objects_v2")
    objects: list[R2Object] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            objects.append(R2Object(bucket, key, int(obj["Size"])))
            if max_objects is not None and len(objects) >= max_objects:
                return objects
    return objects


def r2_cache_path(cache_root: Path, bucket: str, key: str) -> Path:
    parts = [part for part in key.split("/") if part not in {"", "."}]
    if not bucket or not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe R2 cache path for r2://{bucket}/{key}")
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
    for attempt in range(1, R2_DOWNLOAD_ATTEMPTS + 1):
        try:
            return call(), retries
        except Exception as exc:
            if attempt == R2_DOWNLOAD_ATTEMPTS:
                raise
            retries += 1
            sleep_seconds = R2_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            typer.echo(
                f"Warning: {label} failed with {retry_reason(exc)}: {exc}; "
                f"retrying in {sleep_seconds:.0f}s ({attempt}/{R2_DOWNLOAD_ATTEMPTS})",
                err=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Unreachable retry state for {label}")


def download_to_cache(
    client_r2: Any,
    obj: R2Object,
    cache_root: Path,
    boto_transfer_config: Any,
    dry_run: bool,
    overwrite: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> DownloadResult:
    cache_path = r2_cache_path(cache_root, obj.bucket, obj.key)
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
            f"existing file is {cached_size} bytes, R2 object is {obj.size} bytes",
        )

    if dry_run:
        action = "would_overwrite" if exists else "would_download"
        return DownloadResult(obj.key, cache_path, obj.size, action)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    transferred_bytes = 0

    def on_transfer(bytes_amount: int) -> None:
        nonlocal transferred_bytes
        transferred_bytes += bytes_amount
        if progress_callback is not None:
            progress_callback(bytes_amount)

    try:
        client_r2.download_file(
            obj.bucket,
            obj.key,
            str(tmp_path),
            Config=boto_transfer_config,
            Callback=on_transfer if progress_callback is not None else None,
        )
        downloaded_size = tmp_path.stat().st_size
        if downloaded_size != obj.size:
            raise RuntimeError(f"downloaded {downloaded_size} bytes, expected {obj.size} bytes")
        os.replace(tmp_path, cache_path)
        return DownloadResult(obj.key, cache_path, obj.size, "overwritten" if exists else "downloaded")
    except Exception:
        if transferred_bytes and progress_callback is not None:
            progress_callback(-transferred_bytes)
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def cache_one(
    client_r2: Any,
    obj: R2Object,
    cache_root: Path,
    boto_transfer_config: Any,
    dry_run: bool,
    overwrite: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> DownloadResult:
    try:
        result, retries = retry_call(
            f"download r2://{obj.bucket}/{obj.key}",
            lambda: download_to_cache(
                client_r2,
                obj,
                cache_root,
                boto_transfer_config,
                dry_run,
                overwrite,
                progress_callback,
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
            r2_cache_path(cache_root, obj.bucket, obj.key),
            obj.size,
            "failed",
            f"{type(exc).__name__}: {exc}",
            R2_DOWNLOAD_ATTEMPTS - 1,
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
    client_r2: Any,
    objects: list[R2Object],
    cache_root: Path,
    boto_transfer_config: Any,
    workers: int,
    dry_run: bool,
    overwrite: bool,
    byte_progress: bool,
) -> list[DownloadResult]:
    total_bytes = sum(obj.size for obj in objects)
    results: list[DownloadResult] = []
    counts: Counter[str] = Counter()
    retry_count = 0
    completed_files = 0

    with tqdm(
        total=total_bytes,
        desc="Caching R2 bytes",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=5.0,
    ) as progress:
        progress_callback = progress.update if byte_progress else None
        if workers == 1:
            for obj in objects:
                result = cache_one(
                    client_r2,
                    obj,
                    cache_root,
                    boto_transfer_config,
                    dry_run,
                    overwrite,
                    progress_callback,
                )
                results.append(result)
                completed_files += 1
                counts[result.action] += 1
                retry_count += result.retries
                if result.action != "failed" and not (
                    byte_progress and result.action in {"downloaded", "overwritten"}
                ):
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
                        client_r2,
                        obj,
                        cache_root,
                        boto_transfer_config,
                        dry_run,
                        overwrite,
                        progress_callback,
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
                    if result.action != "failed" and not (
                        byte_progress and result.action in {"downloaded", "overwritten"}
                    ):
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
    r2_uri: Annotated[
        str,
        typer.Argument(help="R2 folder/prefix to cache, e.g. r2://bucket/path/to/folder/"),
    ],
    account_id: Annotated[
        str | None,
        typer.Option(help="Cloudflare account ID. Defaults to CLOUDFLARE_ACCOUNT_ID or R2_ACCOUNT_ID."),
    ] = None,
    endpoint_url: Annotated[
        str | None,
        typer.Option(help="Full R2 S3 endpoint URL. Overrides --account-id."),
    ] = None,
    access_key_id: Annotated[
        str | None,
        typer.Option(help="R2 access key ID. Defaults to R2/CLOUDFLARE/AWS env vars."),
    ] = None,
    secret_access_key: Annotated[
        str | None,
        typer.Option(help="R2 secret access key. Defaults to R2/CLOUDFLARE/AWS env vars."),
    ] = None,
    cache_root: Annotated[
        Path,
        typer.Option(help="Shared R2 cache root mirroring r2://bucket/key."),
    ] = R2_CACHE_ROOT,
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
        typer.Option(help="R2 HTTP connection pool size. Defaults from workers and multipart settings."),
    ] = None,
    max_objects: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of objects to list/download."),
    ] = None,
    skip_top_level_summary: Annotated[
        bool,
        typer.Option(help="Skip the extra Delimiter='/' listing pass once you trust the prefix."),
    ] = False,
    byte_progress: Annotated[
        bool,
        typer.Option(
            help="Update tqdm during each R2 transfer; disable for huge many-file runs if it slows things down."
        ),
    ] = True,
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

    bucket, prefix = parse_r2_uri(r2_uri)
    if prefix and not prefix.endswith("/"):
        typer.echo(
            "Warning: prefix does not end with '/'; R2 will match any key beginning with "
            f"{prefix!r}.",
            err=True,
        )

    account_id = account_id or env_first("CLOUDFLARE_ACCOUNT_ID", "R2_ACCOUNT_ID")
    endpoint = r2_endpoint_url(account_id, endpoint_url)
    access_key_id = access_key_id or env_first(
        "R2_ACCESS_KEY_ID",
        "CLOUDFLARE_R2_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
    )
    secret_access_key = secret_access_key or env_first(
        "R2_SECRET_ACCESS_KEY",
        "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
    )

    cache_root = cache_root.expanduser().resolve()
    client_r2 = r2_client(
        endpoint_url=endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        max_pool_connections=max_pool_connections,
    )
    boto_transfer_config = transfer_config(
        multipart_concurrency,
        multipart_threshold_mb,
        multipart_chunksize_mb,
    )

    typer.echo(f"Bucket: {bucket}")
    typer.echo(f"Prefix: {prefix or '[bucket root]'}")
    typer.echo(f"Endpoint: {endpoint}")
    typer.echo(f"Cache root: {cache_root}")
    typer.echo(f"Prefix cache path: {cache_prefix_path(cache_root, bucket, prefix)}")
    typer.echo(
        "Download settings: "
        f"workers={workers}, multipart_concurrency={multipart_concurrency}, "
        f"multipart_threshold={multipart_threshold_mb} MiB, "
        f"multipart_chunksize={multipart_chunksize_mb} MiB, "
        f"max_pool_connections={max_pool_connections}, "
        f"byte_progress={byte_progress}"
    )

    if skip_top_level_summary:
        typer.echo("Skipping top-level summary listing.")
    else:
        files, folders, direct_bytes = list_top_level(client_r2, bucket, prefix)
        typer.echo(
            f"Top-level under prefix: {files:,} files, {folders:,} folders, "
            f"{format_bytes(direct_bytes)} directly in this folder."
        )

    objects = list_recursive(client_r2, bucket, prefix, max_objects)
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
        client_r2,
        objects,
        cache_root,
        boto_transfer_config,
        workers,
        dry_run,
        overwrite,
        byte_progress,
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
