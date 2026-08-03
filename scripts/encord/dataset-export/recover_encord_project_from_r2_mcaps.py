# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "mcap-protobuf-support",
#     "tqdm",
#     "typer",
# ]
# ///
"""Download an Encord project's R2 MCAPs while extracting upload-ready episodes."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from download_r2_prefix_to_cache import (
    DEFAULT_MULTIPART_CHUNKSIZE_MB,
    DEFAULT_MULTIPART_CONCURRENCY,
    DEFAULT_MULTIPART_THRESHOLD_MB,
    R2_CACHE_ROOT,
    auto_max_pool_connections,
    r2_client,
    r2_endpoint_url,
)
from r2_mcap_recovery_utils import (
    FILE_MAP_FILENAME,
    PROJECT_MANIFEST_FILENAME,
    bind_r2_sizes,
    build_file_map,
    build_project_manifest,
    create_encord_client,
    discover_project_episodes,
    list_r2_objects,
    run_download_extract_pipeline,
    write_json_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RECOVERY_ROOT = REPO_ROOT / "exports" / "encord-dataset-export" / "recovered" / "r2"
DEFAULT_PROJECT_HASH = "08411c84-7e66-4ad9-a63d-d948f9e821a1"
DEFAULT_BUCKET = "trossen-robotics-data"
DEFAULT_R2_PREFIX = "trossen-data-mobile"
DEFAULT_DOWNLOAD_WORKERS = 16
DEFAULT_EXTRACT_WORKERS = 4


def require_positive(name: str, value: int) -> None:
    if value < 1:
        raise typer.BadParameter(f"{name} must be at least 1.")


def resolve_program(value: str, label: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise typer.BadParameter(f"{label} executable was not found: {value}")
    return resolved


def main(
    project_hash: Annotated[
        str,
        typer.Argument(
            help="Encord project hash whose data groups identify the MCAP episodes.",
            envvar="ENCORD_PROJECT_HASH",
        ),
    ] = DEFAULT_PROJECT_HASH,
    ssh_key_file: Annotated[
        Path | None,
        typer.Option(
            "--ssh-key-file",
            "-k",
            envvar="ENCORD_SSH_KEY_FILE",
            help="Path to the Encord SSH private key.",
        ),
    ] = None,
    dataset_hash: Annotated[
        str | None,
        typer.Option(
            help="Attached dataset hash. Required only when the project has multiple datasets."
        ),
    ] = None,
    encord_domain: Annotated[
        str,
        typer.Option(help="Encord API domain."),
    ] = "https://api.encord.com",
    bucket: Annotated[
        str,
        typer.Option(
            envvar="R2_BUCKET", help="R2 bucket containing the Trossen MCAP files."
        ),
    ] = DEFAULT_BUCKET,
    r2_prefix: Annotated[
        str,
        typer.Option(help="R2 prefix corresponding to raw-feed/trossen-data."),
    ] = DEFAULT_R2_PREFIX,
    account_id: Annotated[
        str | None,
        typer.Option(envvar="CLOUDFLARE_ACCOUNT_ID", help="Cloudflare account ID."),
    ] = None,
    endpoint_url: Annotated[
        str | None,
        typer.Option(envvar="R2_ENDPOINT_URL", help="Full R2 S3 endpoint URL."),
    ] = None,
    access_key_id: Annotated[
        str | None,
        typer.Option(envvar="R2_ACCESS_KEY_ID", help="R2 access key ID."),
    ] = None,
    secret_access_key: Annotated[
        str | None,
        typer.Option(envvar="R2_SECRET_ACCESS_KEY", help="R2 secret access key."),
    ] = None,
    cache_root: Annotated[
        Path,
        typer.Option(help="R2 cache root, which mirrors bucket and object key paths."),
    ] = R2_CACHE_ROOT,
    output_root: Annotated[
        Path | None,
        typer.Option(
            help="Upload-ready output root. Defaults under exports/encord-dataset-export/recovered."
        ),
    ] = None,
    episode_path_contains: Annotated[
        str | None,
        typer.Option(
            help="Only recover episodes whose source episode path contains this text."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of complete project episodes to recover."),
    ] = None,
    download_workers: Annotated[
        int,
        typer.Option(help="Concurrent R2 object download workers."),
    ] = DEFAULT_DOWNLOAD_WORKERS,
    extract_workers: Annotated[
        int,
        typer.Option(help="Spawned MCAP extraction worker processes."),
    ] = DEFAULT_EXTRACT_WORKERS,
    multipart_concurrency: Annotated[
        int,
        typer.Option(help="Multipart download threads per large MCAP."),
    ] = DEFAULT_MULTIPART_CONCURRENCY,
    multipart_threshold_mb: Annotated[
        int,
        typer.Option(help="Multipart threshold in MiB."),
    ] = DEFAULT_MULTIPART_THRESHOLD_MB,
    multipart_chunksize_mb: Annotated[
        int,
        typer.Option(help="Multipart chunk size in MiB."),
    ] = DEFAULT_MULTIPART_CHUNKSIZE_MB,
    ffmpeg_bin: Annotated[
        str,
        typer.Option(help="ffmpeg executable name or path."),
    ] = "ffmpeg",
    ffprobe_bin: Annotated[
        str,
        typer.Option(help="ffprobe executable name or path."),
    ] = "ffprobe",
    allow_missing: Annotated[
        bool,
        typer.Option(
            "--allow-missing/--require-all",
            help="Continue with matched MCAPs when any project episodes are absent from R2.",
        ),
    ] = False,
    overwrite_downloads: Annotated[
        bool,
        typer.Option(
            "--overwrite-downloads/--reuse-downloads",
            help="Replace same-key MCAP cache files instead of reusing size-matched files.",
        ),
    ] = False,
    overwrite_extracted: Annotated[
        bool,
        typer.Option(
            "--overwrite-extracted/--reuse-extracted",
            help="Re-extract episodes with valid completion markers.",
        ),
    ] = False,
    byte_progress: Annotated[
        bool,
        typer.Option(
            "--byte-progress/--no-byte-progress",
            help="Update the download bar for each transferred byte.",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Resolve the project-to-R2 manifest without downloading or writing outputs."
        ),
    ] = False,
) -> None:
    require_positive("--download-workers", download_workers)
    require_positive("--extract-workers", extract_workers)
    require_positive("--multipart-concurrency", multipart_concurrency)
    require_positive("--multipart-threshold-mb", multipart_threshold_mb)
    require_positive("--multipart-chunksize-mb", multipart_chunksize_mb)
    if limit is not None:
        require_positive("--limit", limit)
    if ssh_key_file is None:
        raise typer.BadParameter("Pass --ssh-key-file or set ENCORD_SSH_KEY_FILE.")
    if access_key_id is None:
        raise typer.BadParameter("Pass --access-key-id or set R2_ACCESS_KEY_ID.")
    if secret_access_key is None:
        raise typer.BadParameter(
            "Pass --secret-access-key or set R2_SECRET_ACCESS_KEY."
        )

    resolved_cache_root = cache_root.expanduser().resolve()
    resolved_output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else (RECOVERY_ROOT / bucket / project_hash).resolve()
    )
    endpoint = r2_endpoint_url(account_id, endpoint_url)
    max_pool_connections = auto_max_pool_connections(
        download_workers,
        multipart_concurrency,
    )

    typer.echo(f"Loading Encord project {project_hash}...")
    encord_client = create_encord_client(ssh_key_file, encord_domain)
    try:
        project_info, project_episodes = discover_project_episodes(
            client=encord_client,
            project_hash=project_hash,
            dataset_hash=dataset_hash,
            r2_bucket=bucket,
            r2_prefix=r2_prefix,
            episode_path_contains=episode_path_contains,
            limit=limit,
        )
    except ValueError as exc:
        typer.echo(f"Could not resolve project episodes: {exc}", err=True)
        raise typer.Exit(code=1) from None

    source_warning_summary = project_info.get("source_warning_summary") or {}
    source_warnings = project_info.get("source_warnings") or []
    if source_warning_summary:
        warning_preview_limit = 10
        typer.echo(
            f"Source group metadata repairs/warnings: {source_warning_summary}",
            err=True,
        )
        for warning in source_warnings[:warning_preview_limit]:
            typer.echo(
                f"  [{warning['kind']}] {warning.get('episode_path') or warning.get('data_hash')}: "
                f"{warning['message']}",
                err=True,
            )
        if len(source_warnings) > warning_preview_limit:
            remaining = len(source_warnings) - warning_preview_limit
            suffix = (
                "not shown during this dry run"
                if dry_run
                else "recorded in the recovery manifest"
            )
            typer.echo(
                f"  ... {remaining} more {suffix}.",
                err=True,
            )
    if not project_episodes:
        raise typer.BadParameter("No project episodes matched the selected filters.")

    typer.echo(f"Listing r2://{bucket}/{r2_prefix.strip('/')}/...")
    client_r2 = r2_client(
        endpoint_url=endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        max_pool_connections=max_pool_connections,
    )
    r2_objects = list_r2_objects(client_r2, bucket, r2_prefix)
    episodes, missing = bind_r2_sizes(project_episodes, r2_objects)
    total_bytes = sum(episode.r2_size for episode in episodes)

    typer.echo(
        f"Matched MCAPs: {len(episodes):,}/{len(project_episodes):,} "
        f"({total_bytes / 1_000_000_000:.2f} GB)"
    )
    if missing:
        typer.echo(f"Missing MCAPs: {len(missing):,}", err=True)
        for episode in missing[:20]:
            typer.echo(f"  r2://{bucket}/{episode.r2_key}", err=True)
        if len(missing) > 20:
            typer.echo(f"  ... {len(missing) - 20} more", err=True)
        if not allow_missing:
            raise typer.BadParameter(
                "Some project episodes are missing from R2. Pass --allow-missing to recover only matches."
            )

    if dry_run:
        typer.echo(f"R2 cache root: {resolved_cache_root}")
        typer.echo(f"Upload-ready output root: {resolved_output_root}")
        typer.echo("Dry run complete; no files were written.")
        return

    ffmpeg_path = resolve_program(ffmpeg_bin, "ffmpeg")
    ffprobe_path = resolve_program(ffprobe_bin, "ffprobe")
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    initial_manifest = build_project_manifest(
        project_info=project_info,
        episodes=episodes,
        missing=missing,
        cache_root=resolved_cache_root,
        output_root=resolved_output_root,
    )
    manifest_path = resolved_output_root / PROJECT_MANIFEST_FILENAME
    write_json_atomic(manifest_path, initial_manifest)

    typer.echo(f"R2 cache root: {resolved_cache_root}")
    typer.echo(f"Upload-ready output root: {resolved_output_root}")
    typer.echo(
        f"Pipeline: {download_workers} download threads -> "
        f"{extract_workers} spawned extractor processes"
    )
    download_results, extraction_results = run_download_extract_pipeline(
        episodes=episodes,
        client_r2=client_r2,
        cache_root=resolved_cache_root,
        output_root=resolved_output_root,
        download_workers=download_workers,
        extract_workers=extract_workers,
        multipart_concurrency=multipart_concurrency,
        multipart_threshold_mb=multipart_threshold_mb,
        multipart_chunksize_mb=multipart_chunksize_mb,
        overwrite_downloads=overwrite_downloads,
        overwrite_extracted=overwrite_extracted,
        byte_progress=byte_progress,
        ffmpeg_bin=ffmpeg_path,
        ffprobe_bin=ffprobe_path,
    )

    file_map = build_file_map(
        episodes=episodes,
        extraction_results=extraction_results,
        output_root=resolved_output_root,
    )
    write_json_atomic(resolved_output_root / FILE_MAP_FILENAME, file_map)
    final_manifest = build_project_manifest(
        project_info=project_info,
        episodes=episodes,
        missing=missing,
        cache_root=resolved_cache_root,
        output_root=resolved_output_root,
        download_results=download_results,
        extraction_results=extraction_results,
    )
    write_json_atomic(manifest_path, final_manifest)

    download_counts = Counter(result["action"] for result in download_results.values())
    extraction_counts = Counter(result.status for result in extraction_results.values())
    typer.echo(f"Download summary: {dict(download_counts)}")
    typer.echo(f"Extraction summary: {dict(extraction_counts)}")
    typer.echo(f"Upload file map: {resolved_output_root / FILE_MAP_FILENAME}")
    typer.echo(f"Recovery manifest: {manifest_path}")

    failed_downloads = sum(
        count
        for action, count in download_counts.items()
        if action in {"failed", "size_conflict"}
    )
    failed_extractions = extraction_counts.get("failed", 0)
    if failed_downloads or failed_extractions:
        for result in extraction_results.values():
            if result.error:
                typer.echo(f"  {result.data_hash}: {result.error}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
