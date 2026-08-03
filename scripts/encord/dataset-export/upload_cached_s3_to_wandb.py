# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "pyyaml",
#     "tqdm",
#     "typer",
#     "wandb>=0.18.0",
# ]
# ///
"""Upload a local cached S3 folder to W&B while preserving its subfolder layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
from pathlib import Path
import re
from typing import Annotated, Any

import click
import typer
from tqdm import tqdm

from export_encord_dataset_to_wandb import (
    DEFAULT_EXPORT_CONFIG,
    DEFAULT_WANDB_CONFIG,
    EXPORT_ROOT,
    S3_CACHE_ROOT,
    WANDB_UPLOAD_HEARTBEAT_SECONDS,
    configured_tags,
    format_bytes,
    has_errno,
    load_yaml,
    make_output_dir,
    required,
    wait_for_wandb_artifact,
    wandb_data_dir_hint,
    write_json,
)


SYSTEM_FILENAMES = {".DS_Store"}
EPISODE_DIR_RE = re.compile(r"^episode_\d+(?:_[0-9A-Za-z][0-9A-Za-z._-]*)?$")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".3gp", ".3g2", ".mj2", ".avi"}
REQUIRED_METADATA_FILES = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)
IGNORED_EXPORT_CONFIG_KEYS = {
    "dataset_hash",
    "encord_dataset_hash",
    "folder_hash",
    "integration_hash",
    "limit",
    "unsigned_s3",
    "base_artifact_ref",
    "wandb_upload_heartbeat_seconds",
}


@dataclass(frozen=True)
class CacheFile:
    path: Path
    artifact_name: str
    source_uri: str
    size: int


@dataclass(frozen=True)
class EpisodeSelection:
    roots: list[Path]
    available_count: int | None
    complete_count: int | None
    incomplete: list[dict[str, Any]]
    skipped_incomplete: list[dict[str, Any]]


def default_description(export_config: dict[str, Any], summary: dict[str, Any]) -> str:
    return str(
        export_config.get("description")
        or f"Cached S3 upload from {summary['cache_dir']} ({summary['file_count']} files)"
    )


def resolve_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise typer.BadParameter(f"{label} is not a directory: {resolved}")
    return resolved


def artifact_name_for_path(path: Path, artifact_root: Path, artifact_prefix: str) -> str:
    try:
        relative = path.relative_to(artifact_root)
    except ValueError:
        raise typer.BadParameter(
            f"Cache file is not under artifact root: {path} is outside {artifact_root}"
        ) from None

    name = relative.as_posix()
    prefix = artifact_prefix.strip("/")
    return f"{prefix}/{name}" if prefix else name


def s3_uri_for_cache_path(path: Path, s3_cache_root: Path) -> str:
    try:
        relative = path.relative_to(s3_cache_root)
    except ValueError:
        raise typer.BadParameter(
            f"Cache file is not under S3 cache root: {path} is outside {s3_cache_root}. "
            "Pass --s3-cache-root if your local cache root is different."
        ) from None

    parts = relative.parts
    if len(parts) < 2:
        raise typer.BadParameter(
            f"S3 cache file must be under <s3-cache-root>/<bucket>/<key>: {path}"
        )
    bucket = parts[0]
    key = "/".join(parts[1:])
    return f"s3://{bucket}/{key}"


def discover_cache_files(
    cache_dir: Path,
    artifact_root: Path,
    artifact_prefix: str,
    s3_cache_root: Path,
    include_system_files: bool,
    max_files: int | None,
    max_episodes: int | None,
    require_complete_episodes: bool,
    required_video_count: int,
) -> tuple[list[CacheFile], dict[str, Any]]:
    skipped = {
        "hidden_files": 0,
        "system_files": 0,
        "non_files": 0,
    }
    files: list[CacheFile] = []
    selection = select_episode_roots(
        cache_dir=cache_dir,
        max_episodes=max_episodes,
        require_complete_episodes=require_complete_episodes,
        required_video_count=required_video_count,
    )
    if max_files is not None and require_complete_episodes and selection.available_count is not None:
        raise typer.BadParameter(
            "--max-files can cut off files inside a selected episode. Use --max-episodes "
            "for complete-episode smoke tests, or pass --allow-incomplete-episodes to opt out."
        )
    skipped["incomplete_episodes"] = len(selection.skipped_incomplete)
    for search_root in selection.roots:
        for path in sorted(search_root.rglob("*")):
            if not path.is_file():
                skipped["non_files"] += 1
                continue
            if not include_system_files and path.name in SYSTEM_FILENAMES:
                skipped["system_files"] += 1
                continue
            if not include_system_files and path.name.startswith("."):
                skipped["hidden_files"] += 1
                continue
            files.append(
                CacheFile(
                    path=path,
                    artifact_name=artifact_name_for_path(path, artifact_root, artifact_prefix),
                    source_uri=s3_uri_for_cache_path(path, s3_cache_root),
                    size=path.stat().st_size,
                )
            )
            if max_files is not None and len(files) >= max_files:
                break
        if max_files is not None and len(files) >= max_files:
            break
    skipped = {key: value for key, value in skipped.items() if value}
    skipped["_episode_selection"] = {
        "available_episode_count": selection.available_count,
        "complete_episode_count": selection.complete_count,
        "incomplete_episode_count": len(selection.incomplete),
        "incomplete_episodes": selection.incomplete[:200],
        "selected_episode_count": len(selection.roots) if selection.available_count is not None else None,
        "skipped_incomplete_episode_count": len(selection.skipped_incomplete),
        "skipped_incomplete_episodes": selection.skipped_incomplete[:200],
    }
    return files, skipped


def visible_file(path: Path) -> bool:
    return path.name not in SYSTEM_FILENAMES and not path.name.startswith(".")


def episode_dirs(cache_dir: Path) -> list[Path]:
    episode_dirs = sorted(path for path in cache_dir.rglob("*") if path.is_dir() and EPISODE_DIR_RE.match(path.name))
    if EPISODE_DIR_RE.match(cache_dir.name):
        episode_dirs = [cache_dir] + [path for path in episode_dirs if path != cache_dir]
    return episode_dirs


def episode_incomplete_reasons(episode_dir: Path, required_video_count: int) -> dict[str, Any] | None:
    missing_metadata = [name for name in REQUIRED_METADATA_FILES if not (episode_dir / name).is_file()]
    video_files = [
        path
        for path in episode_dir.rglob("*")
        if path.is_file() and visible_file(path) and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    missing_video_count = max(required_video_count - len(video_files), 0)
    if not missing_metadata and missing_video_count == 0:
        return None
    return {
        "episode_dir": str(episode_dir),
        "missing_metadata": missing_metadata,
        "video_count": len(video_files),
        "required_video_count": required_video_count,
        "missing_video_count": missing_video_count,
    }


def select_episode_roots(
    *,
    cache_dir: Path,
    max_episodes: int | None,
    require_complete_episodes: bool,
    required_video_count: int,
) -> EpisodeSelection:
    all_episode_dirs = episode_dirs(cache_dir)
    if not all_episode_dirs:
        if max_episodes is not None:
            raise typer.BadParameter(f"No episode_* directories found under {cache_dir}.")
        return EpisodeSelection(
            roots=[cache_dir],
            available_count=None,
            complete_count=None,
            incomplete=[],
            skipped_incomplete=[],
        )

    complete_dirs: list[Path] = []
    skipped_incomplete: list[dict[str, Any]] = []
    for episode_dir in all_episode_dirs:
        reason = episode_incomplete_reasons(episode_dir, required_video_count)
        if reason is None:
            complete_dirs.append(episode_dir)
        else:
            skipped_incomplete.append(reason)

    if require_complete_episodes:
        if not complete_dirs:
            raise typer.BadParameter(f"No complete episode_* directories found under {cache_dir}.")
        selected = complete_dirs[:max_episodes] if max_episodes is not None else complete_dirs
        skipped = skipped_incomplete
    else:
        selected = all_episode_dirs[:max_episodes] if max_episodes is not None else [cache_dir]
        skipped = []

    return EpisodeSelection(
        roots=selected,
        available_count=len(all_episode_dirs),
        complete_count=len(complete_dirs),
        incomplete=skipped_incomplete,
        skipped_incomplete=skipped,
    )


def cache_size(files: list[CacheFile]) -> tuple[int, int]:
    return len(files), sum(file.size for file in files)


def add_cache_files_to_artifact(artifact: Any, files: list[CacheFile]) -> None:
    for file in tqdm(files, desc="Registering cache file uploads", unit="file", dynamic_ncols=True):
        artifact.add_file(
            str(file.path),
            name=file.artifact_name,
            policy="immutable",
            skip_cache=True,
        )


def add_cache_references_to_artifact(
    artifact: Any,
    files: list[CacheFile],
    checksum: bool,
) -> None:
    for file in tqdm(files, desc="Registering S3 references", unit="file", dynamic_ncols=True):
        artifact.add_reference(
            file.source_uri,
            name=file.artifact_name,
            checksum=checksum,
        )


def ignored_export_config_keys(export_config: dict[str, Any]) -> list[str]:
    return sorted(IGNORED_EXPORT_CONFIG_KEYS & set(export_config))


def build_summary(
    *,
    cache_dir: Path,
    artifact_root: Path,
    artifact_prefix: str,
    s3_cache_root: Path,
    files: list[CacheFile],
    skipped: dict[str, Any],
    export_config: dict[str, Any],
    wandb_config: dict[str, Any],
    ignored_cli_args: list[str],
    upload_file_bytes: bool,
    reference_checksum: bool,
    dry_run: bool,
) -> dict[str, Any]:
    file_count, total_bytes = cache_size(files)
    sample_count = min(20, len(files))
    episode_selection = skipped.pop("_episode_selection", {})
    return {
        "source": "cached_s3",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "cache_dir": str(cache_dir),
        "artifact_root": str(artifact_root),
        "artifact_prefix": artifact_prefix.strip("/"),
        "s3_cache_root": str(s3_cache_root),
        "path_mode": "artifact-root-relative",
        "storage_mode": "wandb_file_upload" if upload_file_bytes else "s3_reference",
        "reference_checksum": reference_checksum if not upload_file_bytes else None,
        "file_count": file_count,
        "selected_episode_count": episode_selection.get("selected_episode_count"),
        "available_episode_count": episode_selection.get("available_episode_count"),
        "complete_episode_count": episode_selection.get("complete_episode_count"),
        "incomplete_episode_count": episode_selection.get("incomplete_episode_count", 0),
        "incomplete_episodes": episode_selection.get("incomplete_episodes", []),
        "skipped_incomplete_episode_count": episode_selection.get("skipped_incomplete_episode_count", 0),
        "skipped_incomplete_episodes": episode_selection.get("skipped_incomplete_episodes", []),
        "total_bytes": total_bytes,
        "total_size": format_bytes(total_bytes),
        "skipped": skipped,
        "ignored_export_config_keys": ignored_export_config_keys(export_config),
        "ignored_cli_args": ignored_cli_args,
        "wandb_entity": wandb_config.get("entity"),
        "wandb_project": wandb_config.get("project"),
        "artifact_name": wandb_config.get("source_artifact_name"),
        "sample_artifact_paths": [files[index].artifact_name for index in range(sample_count)],
        "sample_source_uris": [files[index].source_uri for index in range(sample_count)],
    }


def cache_upload_run_name(artifact_name: str, summary: dict[str, Any]) -> str:
    cache_leaf = Path(str(summary["cache_dir"])).name or "cache"
    return f"cached-s3-{artifact_name}-{cache_leaf}-{summary['file_count']}files"


def log_cache_to_wandb(
    *,
    wandb_config: dict[str, Any],
    files: list[CacheFile],
    summary: dict[str, Any],
    tags: list[str],
    description: str,
    upload_heartbeat_seconds: int,
    upload_file_bytes: bool,
    reference_checksum: bool,
) -> dict[str, str]:
    import wandb

    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    artifact_name = required(wandb_config, "source_artifact_name", "W&B config")
    file_count, total_bytes = cache_size(files)

    typer.echo(f"Preparing W&B artifact with {file_count:,} entries ({format_bytes(total_bytes)} in local cache).")
    if upload_file_bytes:
        typer.echo("Uploading file bytes to W&B with local artifact caching disabled.")
    else:
        typer.echo("Logging S3 reference entries only; W&B will not stage or upload cache file bytes.")
        if not reference_checksum:
            typer.echo("S3 reference checksum lookup disabled; W&B will not scan S3 object metadata.")
    if upload_heartbeat_seconds > 0:
        typer.echo(f"Will print W&B upload/finalization heartbeat every {upload_heartbeat_seconds}s.")

    try:
        with wandb.init(
            entity=entity,
            project=project,
            job_type="cached-s3-upload",
            name=cache_upload_run_name(str(artifact_name), summary),
        ) as run:
            artifact = wandb.Artifact(
                str(artifact_name),
                type="dataset",
                metadata=summary,
                description=description,
            )
            if upload_file_bytes:
                add_cache_files_to_artifact(artifact, files)
            else:
                add_cache_references_to_artifact(artifact, files, checksum=reference_checksum)
            logged = run.log_artifact(artifact, aliases=["latest"], tags=tags)
            wait_for_wandb_artifact(
                logged,
                file_count=file_count,
                total_bytes=total_bytes,
                run_url=run.url,
                heartbeat_seconds=upload_heartbeat_seconds,
            )
            return {"dataset_artifact": f"{artifact_name}:{logged.version}", "run_url": run.url}
    except OSError as exc:
        if has_errno(exc, errno.ENOSPC):
            raise click.ClickException(
                "W&B ran out of local disk space while preparing the artifact. "
                f"This upload contains {file_count:,} files totaling {format_bytes(total_bytes)}. "
                f"Set WANDB_DATA_DIR to a directory with enough free space and rerun; current target is "
                f"{wandb_data_dir_hint()}. Example: "
                "WANDB_DATA_DIR=/Volumes/big-disk/wandb-data uv run --script "
                "scripts/encord/dataset-export/upload_cached_s3_to_wandb.py"
            ) from exc
        raise


def main(
    cache_dir: Annotated[
        Path,
        typer.Argument(help="Local folder to upload recursively."),
    ],
    dataset_hash: Annotated[
        str | None,
        typer.Option(help="Ignored; accepted for command compatibility with the Encord dataset exporter."),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help="Root used for artifact-relative paths. Defaults to the selected upload folder.",
        ),
    ] = None,
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            help="Optional prefix to add inside the W&B artifact. Empty preserves selected-folder-relative paths.",
        ),
    ] = "",
    s3_cache_root: Annotated[
        Path,
        typer.Option(
            "--s3-cache-root",
            help="Local root that mirrors s3://<bucket>/<key> for W&B reference URIs.",
        ),
    ] = S3_CACHE_ROOT,
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
    export_config: Annotated[
        Path,
        typer.Option(help="Dataset export config YAML for tags and description."),
    ] = DEFAULT_EXPORT_CONFIG,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Local directory for upload summary files. Defaults to a timestamped export dir."),
    ] = None,
    max_files: Annotated[
        int | None,
        typer.Option(help="Optional max files to register, useful for smoke tests."),
    ] = None,
    max_episodes: Annotated[
        int | None,
        typer.Option(help="Optional max episode_* folders to include recursively."),
    ] = None,
    require_complete_episodes: Annotated[
        bool,
        typer.Option(
            "--require-complete-episodes/--allow-incomplete-episodes",
            help="Only upload episodes with required metadata and enough video files.",
        ),
    ] = True,
    required_video_count: Annotated[
        int,
        typer.Option(help="Minimum visible video files required for an episode to be complete."),
    ] = 3,
    limit: Annotated[
        int | None,
        typer.Option(help="Ignored; use --max-files for local cache smoke tests."),
    ] = None,
    unsigned_s3: Annotated[
        bool,
        typer.Option(help="Ignored; this command reads the local cache and does not call S3."),
    ] = False,
    base_artifact_ref: Annotated[
        str | None,
        typer.Option(help="Ignored; cached-tree uploads always create a fresh artifact version."),
    ] = None,
    folder_hash: Annotated[
        str | None,
        typer.Option(help="Ignored; this command does not call Encord."),
    ] = None,
    include_system_files: Annotated[
        bool,
        typer.Option(help="Include local system files such as .DS_Store."),
    ] = False,
    upload_file_bytes: Annotated[
        bool,
        typer.Option(
            "--upload-file-bytes/--s3-references",
            help="Upload local bytes to W&B storage instead of logging S3 references only.",
        ),
    ] = False,
    reference_checksum: Annotated[
        bool,
        typer.Option(
            "--reference-checksum/--no-reference-checksum",
            help="Ask W&B to read S3 object metadata for reference checksums.",
        ),
    ] = False,
    wandb_upload_heartbeat_seconds: Annotated[
        int,
        typer.Option(help="Seconds between W&B artifact upload/finalization heartbeat messages; set 0 to disable."),
    ] = WANDB_UPLOAD_HEARTBEAT_SECONDS,
    dry_run: Annotated[
        bool,
        typer.Option(help="Build summary and list files without logging to W&B."),
    ] = False,
) -> None:
    if max_files is not None and max_files < 1:
        raise typer.BadParameter("--max-files must be at least 1.")
    if max_episodes is not None and max_episodes < 1:
        raise typer.BadParameter("--max-episodes must be at least 1.")
    if required_video_count < 1:
        raise typer.BadParameter("--required-video-count must be at least 1.")
    if wandb_upload_heartbeat_seconds < 0:
        raise typer.BadParameter("W&B upload heartbeat seconds must be 0 or greater.")

    cache_dir = resolve_dir(cache_dir, "cache_dir")
    artifact_root = resolve_dir(artifact_root or cache_dir, "artifact_root")
    s3_cache_root = resolve_dir(s3_cache_root, "s3_cache_root")
    try:
        cache_dir.relative_to(artifact_root)
    except ValueError:
        raise typer.BadParameter(f"--cache-dir must be inside --artifact-root: {cache_dir} not under {artifact_root}") from None

    export_settings = load_yaml(export_config, "Dataset export config")
    ignored_keys = ignored_export_config_keys(export_settings)
    if ignored_keys:
        typer.echo(f"Ignoring Encord-only export config key(s): {', '.join(ignored_keys)}", err=True)
    tags = configured_tags(export_settings)
    wandb_settings = load_yaml(wandb_config, "W&B config")
    output_dir = output_dir.expanduser().resolve() if output_dir else make_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ignored_cli_args = [
        name
        for name, value in {
            "--dataset-hash": dataset_hash,
            "--limit": limit,
            "--unsigned-s3": unsigned_s3,
            "--base-artifact-ref": base_artifact_ref,
            "--folder-hash": folder_hash,
        }.items()
        if value not in (None, False)
    ]
    if ignored_cli_args:
        typer.echo(f"Ignoring Encord-only CLI option(s): {', '.join(ignored_cli_args)}", err=True)

    files, skipped = discover_cache_files(
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        artifact_prefix=artifact_prefix,
        s3_cache_root=s3_cache_root,
        include_system_files=include_system_files,
        max_files=max_files,
        max_episodes=max_episodes,
        require_complete_episodes=require_complete_episodes,
        required_video_count=required_video_count,
    )
    if not files:
        raise typer.BadParameter(f"No files found under {cache_dir}.")

    summary = build_summary(
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        artifact_prefix=artifact_prefix,
        s3_cache_root=s3_cache_root,
        files=files,
        skipped=skipped,
        export_config=export_settings,
        wandb_config=wandb_settings,
        ignored_cli_args=ignored_cli_args,
        upload_file_bytes=upload_file_bytes,
        reference_checksum=reference_checksum,
        dry_run=dry_run,
    )
    description = default_description(export_settings, summary)
    write_json(output_dir / "cached_s3_upload_summary.json", summary)

    typer.echo(f"Cache dir: {cache_dir}")
    typer.echo(f"Artifact root: {artifact_root}")
    typer.echo(f"S3 cache root: {s3_cache_root}")
    typer.echo(f"W&B storage mode: {summary['storage_mode']}")
    if summary["available_episode_count"] is not None:
        typer.echo(
            f"Episodes selected: {summary['selected_episode_count']:,}/"
            f"{summary['complete_episode_count']:,} complete "
            f"({summary['available_episode_count']:,} available)"
        )
        if summary["skipped_incomplete_episode_count"]:
            typer.echo(f"Skipped incomplete episodes: {summary['skipped_incomplete_episode_count']:,}")
        elif summary["incomplete_episode_count"]:
            typer.echo(f"Incomplete episodes allowed: {summary['incomplete_episode_count']:,}")
    typer.echo(f"Files selected: {summary['file_count']:,} ({summary['total_size']})")
    typer.echo(f"Local summary: {output_dir / 'cached_s3_upload_summary.json'}")

    if dry_run:
        typer.echo("Dry run enabled; not logging to W&B.")
        return

    lineage = log_cache_to_wandb(
        wandb_config=wandb_settings,
        files=files,
        summary=summary,
        tags=tags,
        description=description,
        upload_heartbeat_seconds=wandb_upload_heartbeat_seconds,
        upload_file_bytes=upload_file_bytes,
        reference_checksum=reference_checksum,
    )
    write_json(output_dir / "wandb_lineage.json", lineage)

    typer.echo(f"dataset artifact: {lineage['dataset_artifact']}")
    typer.echo(f"run: {lineage['run_url']}")
    typer.echo(f"local files: {output_dir}")


if __name__ == "__main__":
    typer.run(main)
