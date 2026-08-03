# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "pydicom",
#     "tqdm",
#     "typer",
# ]
# ///
"""Upload files from the local shared S3 cache into an existing Encord folder.

Run from the cache folder when you want the CLI to take only the Encord folder hash:

    uv run --script scripts/encord/data-registration/upload_cached_s3_folder_to_encord.py <folder_hash>
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from cached_s3_folder_upload_utils import (
    DEFAULT_S3_CACHE_ROOT,
    FILE_MAP_FILENAME,
    MAX_WORKERS_DEFAULT,
    SCRIPT_DIR,
    EncordDomain,
    EpisodeCompletenessSelection,
    FileInfo,
    TitleMode,
    create_data_groups_from_file_map,
    create_image_groups_from_file_map,
    existing_file_title_mapping,
    existing_group_titles,
    file_infos_from_extension_analysis,
    file_infos_from_folder_structure,
    get_encord_client,
    get_file_map,
    looks_like_annotation_file,
    outcome_summary,
    resolve_dir,
    resolve_parent,
    select_complete_episode_roots,
    upload_files,
    write_report,
)
from encord.storage import StorageItemType as DataType

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def default_report_json(data_dir: Path, dry_run: bool) -> Path:
    suffix = "dry_run_report" if dry_run else "upload_report"
    return SCRIPT_DIR / f"encord_s3_cache_{suffix}.json"


def load_file_infos(
    data_dir: Path,
    cache_root: Path,
    include_client_metadata: bool,
    include_legacy_client_metadata: bool,
    use_folder_structure: bool,
    title_mode: TitleMode,
    require_complete_episodes: bool,
    required_video_count: int,
) -> tuple[list[FileInfo], dict[str, int], EpisodeCompletenessSelection]:
    episode_selection = select_complete_episode_roots(
        data_dir=data_dir,
        require_complete_episodes=require_complete_episodes,
        required_video_count=required_video_count,
    )
    if use_folder_structure:
        file_infos, skipped = file_infos_from_folder_structure(
            data_dir=data_dir,
            include_client_metadata=include_client_metadata,
            include_legacy_client_metadata=include_legacy_client_metadata,
            cache_root=cache_root,
            title_mode=title_mode,
            episode_roots=episode_selection.roots,
        )
    else:
        file_infos, skipped = file_infos_from_extension_analysis(
            data_dir=data_dir,
            include_client_metadata=include_client_metadata,
            include_legacy_client_metadata=include_legacy_client_metadata,
            cache_root=cache_root,
            title_mode=title_mode,
            episode_roots=episode_selection.roots,
        )
    return file_infos, dict(skipped), episode_selection


def filter_already_uploaded(
    file_infos: list[FileInfo],
    existing_titles: set[str],
) -> tuple[list[FileInfo], list[FileInfo]]:
    to_upload = []
    skipped_existing = []
    for file_info in file_infos:
        if file_info.title in existing_titles:
            skipped_existing.append(file_info)
        else:
            to_upload.append(file_info)
    return to_upload, skipped_existing


def episode_completeness_report(selection: EpisodeCompletenessSelection) -> dict:
    return {
        "available_episode_count": selection.available_count,
        "complete_episode_count": selection.complete_count,
        "incomplete_episode_count": len(selection.incomplete),
        "incomplete_episodes": selection.incomplete[:500],
        "skipped_incomplete_episode_count": len(selection.skipped_incomplete),
        "skipped_incomplete_episodes": selection.skipped_incomplete[:500],
    }


def build_dry_run_report(
    folder_hash: str,
    data_dir: Path,
    cache_root: Path,
    file_infos: list[FileInfo],
    skipped_unsupported: dict[str, int],
    episode_selection: EpisodeCompletenessSelection,
    annotation_candidates: list[FileInfo],
    include_client_metadata: bool,
    include_legacy_client_metadata: bool,
) -> dict:
    counts_by_type: dict[str, int] = {}
    with_cache_metadata = 0
    for file_info in file_infos:
        counts_by_type[file_info.data_type.value] = counts_by_type.get(file_info.data_type.value, 0) + 1
        metadata = file_info.client_metadata or {}
        if metadata.get("source_uri"):
            with_cache_metadata += 1

    return {
        "dry_run": True,
        "folder_hash": folder_hash,
        "data_dir": str(data_dir),
        "s3_cache_root": str(cache_root),
        "discovered_count": len(file_infos),
        "counts_by_data_type": counts_by_type,
        "with_s3_source_metadata": with_cache_metadata,
        "episode_completeness": episode_completeness_report(episode_selection),
        "skipped_unsupported": skipped_unsupported,
        "include_client_metadata": include_client_metadata,
        "include_legacy_client_metadata": include_legacy_client_metadata,
        "potential_annotation_files": [item.title for item in annotation_candidates[:100]],
        "sample_items": [
            {
                "title": item.title,
                "path": str(item.path),
                "data_type": item.data_type.value,
                "client_metadata": item.client_metadata,
            }
            for item in file_infos[:20]
        ],
    }


def build_upload_report(
    folder_hash: str,
    data_dir: Path,
    cache_root: Path,
    discovered_count: int,
    skipped_unsupported: dict[str, int],
    episode_selection: EpisodeCompletenessSelection,
    skipped_existing: list[FileInfo],
    outcomes,
    created_data_groups: list[str],
    created_image_groups: list[str],
    annotation_candidates: list[FileInfo],
    include_client_metadata: bool,
    include_legacy_client_metadata: bool,
) -> dict:
    return {
        "dry_run": False,
        "folder_hash": folder_hash,
        "data_dir": str(data_dir),
        "s3_cache_root": str(cache_root),
        "discovered_count": discovered_count,
        "episode_completeness": episode_completeness_report(episode_selection),
        "skipped_existing_count": len(skipped_existing),
        "skipped_existing_titles": [item.title for item in skipped_existing[:500]],
        "skipped_unsupported": skipped_unsupported,
        "summary": outcome_summary(outcomes),
        "include_client_metadata": include_client_metadata,
        "include_legacy_client_metadata": include_legacy_client_metadata,
        "created_data_groups": created_data_groups,
        "created_image_groups": created_image_groups,
        "potential_annotation_files": [item.title for item in annotation_candidates[:100]],
        "outcomes": outcomes,
    }


def maybe_echo_annotation_candidates(annotation_candidates: list[FileInfo]) -> None:
    if not annotation_candidates:
        return
    typer.echo(
        f"Found {len(annotation_candidates)} file(s) that look like labels/annotations. "
        "They will be uploaded as data if their extension is supported; upload labels separately if needed.",
        err=True,
    )
    for item in annotation_candidates[:20]:
        typer.echo(f"  {item.title}", err=True)
    if len(annotation_candidates) > 20:
        typer.echo(f"  ... {len(annotation_candidates) - 20} more", err=True)


def main(
    folder_hash: Annotated[
        str,
        typer.Argument(help="Existing Encord storage folder UUID/hash to upload into."),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-d",
            help="Local directory to upload. Defaults to the directory where this command is run.",
        ),
    ] = Path("."),
    ssh_key_file: Annotated[
        Path | None,
        typer.Option(
            "--ssh-key-file",
            "-k",
            envvar="ENCORD_SSH_KEY_FILE",
            help="Path to the Encord SSH private key. Can also be supplied by ENCORD_SSH_KEY_FILE.",
        ),
    ] = None,
    domain: Annotated[
        EncordDomain,
        typer.Option("--domain", help="Encord environment."),
    ] = EncordDomain.PROD,
    s3_cache_root: Annotated[
        Path,
        typer.Option(
            "--s3-cache-root",
            help="Root that mirrors s3://bucket/key for source_uri/source_key metadata.",
        ),
    ] = DEFAULT_S3_CACHE_ROOT,
    title_mode: Annotated[
        TitleMode,
        typer.Option(
            "--title-mode",
            help="How uploaded item titles are chosen. auto uses S3 source keys when data is under the cache root.",
        ),
    ] = TitleMode.AUTO,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Maximum concurrent upload threads."),
    ] = MAX_WORKERS_DEFAULT,
    include_client_metadata: Annotated[
        bool,
        typer.Option(
            "--include-client-metadata/--no-client-metadata",
            help="Attach client metadata to uploaded storage items.",
        ),
    ] = True,
    include_legacy_client_metadata: Annotated[
        bool,
        typer.Option(
            "--include-legacy-client-metadata/--no-legacy-client-metadata",
            help="Also include the original uploader's Tag, Data Type, and Extension metadata fields.",
        ),
    ] = True,
    use_folder_structure: Annotated[
        bool,
        typer.Option(
            "--folder-structure/--detect-types",
            help=f"Use {FILE_MAP_FILENAME} and data-type subfolders instead of extension detection.",
        ),
    ] = False,
    require_complete_episodes: Annotated[
        bool,
        typer.Option(
            "--require-complete-episodes/--allow-incomplete-episodes",
            help="Only upload files from episodes with required metadata and enough video files.",
        ),
    ] = True,
    required_video_count: Annotated[
        int,
        typer.Option(help="Minimum visible video files required for an episode to be complete."),
    ] = 3,
    fix_dicom: Annotated[
        bool,
        typer.Option("--fix-dicom/--no-fix-dicom", help="Apply DICOM file corrections before DICOM upload."),
    ] = False,
    report_json: Annotated[
        Path | None,
        typer.Option("--report-json", help="Path for upload or dry-run report JSON."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Discover files and write a report without uploading."),
    ] = False,
) -> None:
    if not folder_hash:
        raise typer.BadParameter("Pass the target Encord storage folder hash.")
    if max_workers < 1:
        raise typer.BadParameter("--max-workers must be at least 1.")
    if required_video_count < 1:
        raise typer.BadParameter("--required-video-count must be at least 1.")

    data_dir = resolve_dir(data_dir, "data_dir")
    cache_root = s3_cache_root.expanduser().resolve()
    output_report = report_json or default_report_json(data_dir, dry_run)
    if not output_report.is_absolute():
        output_report = resolve_parent(output_report) / output_report.name

    file_infos, skipped_unsupported, episode_selection = load_file_infos(
        data_dir=data_dir,
        cache_root=cache_root,
        include_client_metadata=include_client_metadata,
        include_legacy_client_metadata=include_legacy_client_metadata,
        use_folder_structure=use_folder_structure,
        title_mode=title_mode,
        require_complete_episodes=require_complete_episodes,
        required_video_count=required_video_count,
    )
    annotation_candidates = [item for item in file_infos if looks_like_annotation_file(item)]

    typer.echo(f"Data dir: {data_dir}")
    typer.echo(f"S3 cache root: {cache_root}")
    if episode_selection.available_count is not None:
        selected_episode_count = (
            episode_selection.complete_count
            if require_complete_episodes
            else episode_selection.available_count
        )
        typer.echo(
            f"Episodes selected: {selected_episode_count:,}/"
            f"{episode_selection.complete_count:,} complete "
            f"({episode_selection.available_count:,} available)"
        )
        if episode_selection.skipped_incomplete:
            typer.echo(f"Skipped incomplete episodes: {len(episode_selection.skipped_incomplete):,}")
        elif episode_selection.incomplete:
            typer.echo(f"Incomplete episodes allowed: {len(episode_selection.incomplete):,}")
    typer.echo(f"Discovered uploadable files: {len(file_infos):,}")
    if skipped_unsupported:
        typer.echo(f"Skipped unsupported files by extension: {skipped_unsupported}")
    maybe_echo_annotation_candidates(annotation_candidates)

    if dry_run:
        report = build_dry_run_report(
            folder_hash=folder_hash,
            data_dir=data_dir,
            cache_root=cache_root,
            file_infos=file_infos,
            skipped_unsupported=skipped_unsupported,
            episode_selection=episode_selection,
            annotation_candidates=annotation_candidates,
            include_client_metadata=include_client_metadata,
            include_legacy_client_metadata=include_legacy_client_metadata,
        )
        write_report(output_report, report)
        typer.echo(f"Dry-run report JSON: {output_report}")
        return

    if ssh_key_file is None:
        raise typer.BadParameter("Pass --ssh-key-file or set ENCORD_SSH_KEY_FILE.")

    client = get_encord_client(ssh_key_file=ssh_key_file, domain=domain)
    storage_folder = client.get_storage_folder(folder_hash)
    existing_items = list(storage_folder.list_items())
    existing_title_to_uuid = existing_file_title_mapping(existing_items)
    existing_titles = set(existing_title_to_uuid)
    existing_data_groups = existing_group_titles(existing_items, DataType.GROUP)
    existing_image_groups = existing_group_titles(existing_items, DataType.IMAGE_GROUP)
    to_upload, skipped_existing = filter_already_uploaded(file_infos, existing_titles)

    typer.echo(f"Already in folder: {len(skipped_existing):,}")
    typer.echo(f"Uploading: {len(to_upload):,}")

    outcomes, uploaded_title_to_uuid = upload_files(
        storage_folder=storage_folder,
        file_infos=to_upload,
        max_workers=max_workers,
        fix_dicom=fix_dicom,
    )
    selected_titles = {file_info.title for file_info in file_infos}
    title_to_uuid_mapping = {
        title: uuid
        for title, uuid in existing_title_to_uuid.items()
        if title in selected_titles
    }
    title_to_uuid_mapping.update(uploaded_title_to_uuid)

    created_data_groups: list[str] = []
    created_image_groups: list[str] = []
    file_map = get_file_map(data_dir)
    if file_map:
        created_data_groups = create_data_groups_from_file_map(
            file_map,
            storage_folder,
            title_to_uuid_mapping,
            existing_group_names=existing_data_groups,
        )
        created_image_groups = create_image_groups_from_file_map(
            file_map,
            storage_folder,
            title_to_uuid_mapping,
            existing_group_names=existing_image_groups,
        )

    report = build_upload_report(
        folder_hash=folder_hash,
        data_dir=data_dir,
        cache_root=cache_root,
        discovered_count=len(file_infos),
        skipped_unsupported=skipped_unsupported,
        episode_selection=episode_selection,
        skipped_existing=skipped_existing,
        outcomes=outcomes,
        created_data_groups=created_data_groups,
        created_image_groups=created_image_groups,
        annotation_candidates=annotation_candidates,
        include_client_metadata=include_client_metadata,
        include_legacy_client_metadata=include_legacy_client_metadata,
    )
    write_report(output_report, report)

    summary = outcome_summary(outcomes)
    typer.echo(f"Upload summary: {summary}")
    typer.echo(f"Skipped existing: {len(skipped_existing):,}")
    typer.echo(f"Report JSON: {output_report}")
    if summary.get("failed", 0):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
