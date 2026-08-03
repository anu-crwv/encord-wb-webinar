# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "boto3",
#     "encord==0.1.199",
#     "numpy",
#     "pyarrow",
#     "typer",
#     "wandb==0.28.1",
# ]
# ///
"""Publish the webinar's prepared Encord data as one train-ready W&B artifact."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import lerobot
import typer
from encord_source import (
    CAMERA_ORDER,
    EpisodePlan,
    VideoReference,
    create_encord_client,
    episode_path_for_item,
    item_metadata,
    parse_s3_uri,
    source_info_uri,
    source_parquet_uri,
    source_uri_for_item,
    video_artifact_path,
    video_children_by_camera,
)

EXPORTER_SCHEMA_VERSION = 1
LANGUAGE_INSTRUCTION_PATTERN = re.compile(
    r"^language instruction(?:\s*([123]))?$", re.IGNORECASE
)
LANGUAGE_INSTRUCTION_VALUE_PATTERN = re.compile(
    r"^language_instruction(?:_([123]))?$", re.IGNORECASE
)


def language_instruction_index(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    for pattern in (LANGUAGE_INSTRUCTION_PATTERN, LANGUAGE_INSTRUCTION_VALUE_PATTERN):
        match = pattern.fullmatch(value.strip())
        if match:
            return int(match.group(1) or 1)
    return None


def strings_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings_from(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings_from(item)]
    return []


def language_instruction_candidates(value: Any) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    if isinstance(value, dict):
        index = language_instruction_index(value.get("name"))
        if index is None:
            index = language_instruction_index(value.get("value"))
        if index is not None and "answers" in value:
            candidates.extend((index, text) for text in strings_from(value["answers"]))
        for key, child in value.items():
            child_index = language_instruction_index(key)
            if child_index is not None:
                candidates.extend((child_index, text) for text in strings_from(child))
            candidates.extend(language_instruction_candidates(child))
    elif isinstance(value, list):
        for child in value:
            candidates.extend(language_instruction_candidates(child))
    return candidates


def caption_text(label: dict[str, Any]) -> str | None:
    candidates = language_instruction_candidates(label)
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def export_labels(project: Any) -> list[dict[str, Any]]:
    label_rows = list(project.list_label_rows_v2())
    if label_rows:
        with project.create_bundle(bundle_size=min(100, len(label_rows))) as bundle:
            for label_row in label_rows:
                label_row.initialise_labels(bundle=bundle)

    labels: list[dict[str, Any]] = []
    for label_row in label_rows:
        serialized = label_row.to_encord_dict()
        if not isinstance(serialized, dict):
            raise typer.BadParameter(
                f"Label row {getattr(label_row, 'data_hash', 'unknown')} did not serialize to a mapping"
            )
        row = dict(serialized)
        row.setdefault("data_hash", getattr(label_row, "data_hash", None))
        row.setdefault("label_hash", getattr(label_row, "label_hash", None))
        row.setdefault("data_title", getattr(label_row, "data_title", None))
        labels.append(row)
    return labels


def build_caption_map(labels: list[dict[str, Any]]) -> dict[str, tuple[str | None, str]]:
    result: dict[str, tuple[str | None, str]] = {}
    for label in labels:
        data_hash = str(label.get("data_hash") or "")
        caption = caption_text(label)
        if not data_hash or not caption:
            continue
        if data_hash in result:
            raise typer.BadParameter(f"Duplicate label rows for data hash {data_hash}")
        result[data_hash] = (
            str(label["label_hash"]) if label.get("label_hash") else None,
            caption,
        )
    return result


def validate_project_dataset(project: Any, dataset_hash: str) -> None:
    attached = list(project.list_datasets())
    if len(attached) != 1:
        hashes = sorted(str(item.dataset_hash) for item in attached)
        raise typer.BadParameter(
            f"The caption project must contain exactly one dataset; attached datasets: {hashes}"
        )
    attached_hash = str(attached[0].dataset_hash)
    if attached_hash != dataset_hash:
        raise typer.BadParameter(
            f"Project dataset {attached_hash} does not match --dataset-hash {dataset_hash}"
        )


def build_episode_plan(
    client: Any,
    dataset: Any,
    captions: dict[str, tuple[str | None, str]],
    limit: int | None,
) -> list[EpisodePlan]:
    rows = list(dataset.data_rows)
    if limit is not None:
        rows = rows[:limit]

    plans: list[EpisodePlan] = []
    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    for episode_index, row in enumerate(rows):
        data_hash = str(row.uid)
        if data_hash in seen_hashes:
            raise typer.BadParameter(f"Duplicate dataset data hash: {data_hash}")
        seen_hashes.add(data_hash)
        if data_hash not in captions:
            raise typer.BadParameter(f"Dataset row {data_hash} has no Language Instruction label")

        group = client.get_storage_item(row.backing_item_uuid)
        children = video_children_by_camera(group, client)
        first_video = children[CAMERA_ORDER[0]]
        episode_path = episode_path_for_item(group) or episode_path_for_item(first_video)
        if not episode_path:
            raise typer.BadParameter(f"Data group {group.uuid} has no canonical episode_path")
        if episode_path in seen_paths:
            raise typer.BadParameter(f"Duplicate episode_path: {episode_path}")
        seen_paths.add(episode_path)

        first_uri = source_uri_for_item(first_video)
        parquet_uri = source_parquet_uri(group, first_uri, episode_path)
        group_metadata = item_metadata(group)
        if not any(group_metadata.get(key) for key in ("source_parquet_uri", "parquet_uri")):
            parquet_uri = source_parquet_uri(first_video, first_uri, episode_path)
        label_hash, caption = captions[data_hash]
        videos = tuple(
            VideoReference(
                camera_name=camera,
                source_uri=source_uri_for_item(children[camera]),
                artifact_path=video_artifact_path(episode_index, camera),
                storage_item_uuid=str(children[camera].uuid),
            )
            for camera in CAMERA_ORDER
        )
        plans.append(
            EpisodePlan(
                episode_index=episode_index,
                data_hash=data_hash,
                data_title=str(row.title),
                group_uuid=str(group.uuid),
                label_hash=label_hash,
                caption=caption,
                episode_path=episode_path,
                parquet_uri=parquet_uri,
                info_uri=source_info_uri(first_uri, episode_path),
                videos=videos,
            )
        )
    if not plans:
        raise typer.BadParameter("No exportable Encord episodes were found")
    return plans


def create_s3_client(profile: str | None) -> Any:
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3")


def read_s3_bytes(client: Any, uri: str) -> bytes:
    bucket, key = parse_s3_uri(uri)
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        close = getattr(body, "close", None)
        if close:
            close()


def read_s3_json(client: Any, uri: str) -> dict[str, Any]:
    try:
        value = json.loads(read_s3_bytes(client, uri))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Source metadata is not valid JSON: {uri}") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"Source metadata is not a JSON object: {uri}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_train_ready_dataset(
    plans: list[EpisodePlan],
    output_root: Path,
    s3: Any,
) -> tuple[list[VideoReference], dict[str, Any]]:
    import pyarrow.parquet as pq

    dataset_dir = output_root / "dataset"
    tasks: list[dict[str, Any]] = []
    task_ids: dict[str, int] = {}
    episodes: list[dict[str, Any]] = []
    parquet_paths: list[Path] = []
    references: list[VideoReference] = []
    first_table: Any | None = None
    first_source_info: dict[str, Any] | None = None
    expected_fps: float | None = None
    total_frames = 0

    for plan in plans:
        source_info, fps = lerobot.validate_source_info(plan.info_uri, read_s3_json(s3, plan.info_uri))
        if expected_fps is None:
            expected_fps = fps
            first_source_info = source_info
        elif fps != expected_fps:
            raise typer.BadParameter(
                f"Source episodes have inconsistent FPS values: {expected_fps} and {fps}"
            )

        if plan.caption not in task_ids:
            task_ids[plan.caption] = len(task_ids)
            tasks.append({"task_index": task_ids[plan.caption], "task": plan.caption})
        task_id = task_ids[plan.caption]

        table = pq.read_table(BytesIO(read_s3_bytes(s3, plan.parquet_uri)))
        table = lerobot.rewrite_episode_table(
            table,
            episode_index=plan.episode_index,
            global_start=total_frames,
            task_id=task_id,
        )
        if first_table is None:
            first_table = table
        parquet_path = (
            dataset_dir
            / "data"
            / f"chunk-{plan.episode_index // 1000:03d}"
            / f"episode_{plan.episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        parquet_paths.append(parquet_path)

        total_frames += table.num_rows
        references.extend(plan.videos)
        episodes.append(
            {
                "episode_index": plan.episode_index,
                "tasks": [plan.caption],
                "length": table.num_rows,
                "encord_data_hash": plan.data_hash,
                "encord_data_group_uuid": plan.group_uuid,
                "encord_label_hash": plan.label_hash,
                "episode_path": plan.episode_path,
                "source_parquet_uri": plan.parquet_uri,
            }
        )

    assert first_table is not None and first_source_info is not None
    meta_dir = dataset_dir / "meta"
    write_jsonl(meta_dir / "tasks.jsonl", tasks)
    write_jsonl(meta_dir / "episodes.jsonl", episodes)
    info = lerobot.build_info(
        source_info=first_source_info,
        first_table=first_table,
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(tasks),
    )
    modality = lerobot.build_modality(info)
    stats_columns = lerobot.stats_columns(info, set(first_table.column_names))
    stats = lerobot.compute_stats(parquet_paths, stats_columns, info)
    relative_stats = lerobot.compute_relative_stats(parquet_paths, modality)
    write_json(meta_dir / "info.json", info)
    write_json(meta_dir / "modality.json", modality)
    write_json(meta_dir / "embodiment.json", lerobot.build_embodiment(info))
    write_json(meta_dir / "stats.json", stats)
    write_json(meta_dir / "relative_stats_dreamzero.json", relative_stats)
    return references, {
        "episode_count": len(episodes),
        "task_count": len(tasks),
        "frame_count": total_frames,
        "video_reference_count": len(references),
        "stats_columns": stats_columns,
    }


def build_export_manifest(
    plans: list[EpisodePlan],
    summary: dict[str, Any],
    dataset_hash: str,
    project_hash: str,
    wandb_entity: str,
    wandb_project: str,
    artifact_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXPORTER_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "encord_dataset_hash": dataset_hash,
        "encord_project_hash": project_hash,
        "wandb_entity": wandb_entity,
        "wandb_project": wandb_project,
        "artifact_name": artifact_name,
        "storage_mode": "s3_reference_media",
        **summary,
        "episodes": [
            {
                "episode_index": plan.episode_index,
                "data_hash": plan.data_hash,
                "group_uuid": plan.group_uuid,
                "label_hash": plan.label_hash,
                "episode_path": plan.episode_path,
                "source_parquet_uri": plan.parquet_uri,
                "source_info_uri": plan.info_uri,
                "videos": [
                    {
                        "camera_name": video.camera_name,
                        "storage_item_uuid": video.storage_item_uuid,
                        "source_uri": video.source_uri,
                        "artifact_path": video.artifact_path,
                    }
                    for video in plan.videos
                ],
            }
            for plan in plans
        ],
    }


def add_video_references(artifact: Any, references: list[VideoReference]) -> None:
    for reference in references:
        artifact.add_reference(reference.source_uri, name=reference.artifact_path)


def publish_artifact(
    output_root: Path,
    references: list[VideoReference],
    manifest: dict[str, Any],
    wandb_entity: str,
    wandb_project: str,
    artifact_name: str,
    aliases: list[str],
) -> tuple[str, str | None]:
    import wandb

    wandb_dir = output_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        job_type="encord-dataset-export",
        name=f"encord-export-{manifest['encord_dataset_hash'][:8]}",
        dir=str(wandb_dir),
    ) as run:
        artifact = wandb.Artifact(
            artifact_name,
            type="dataset",
            metadata={key: value for key, value in manifest.items() if key != "episodes"},
            description="Train-ready Encord dataset with externally referenced S3 video media.",
        )
        add_video_references(artifact, references)
        artifact.add_dir(str(output_root / "dataset" / "data"), name="dataset/data", skip_cache=True)
        artifact.add_dir(str(output_root / "dataset" / "meta"), name="dataset/meta", skip_cache=True)
        artifact.add_file(
            str(output_root / "encord_export_manifest.json"),
            name="encord_export_manifest.json",
            skip_cache=True,
        )
        logged = run.log_artifact(artifact, aliases=aliases)
        logged.wait()
        qualified_name = f"{wandb_entity}/{wandb_project}/{artifact_name}:{logged.version}"
        return qualified_name, run.url


def run_export(
    *,
    ssh_key_file: Path,
    dataset_hash: str,
    project_hash: str,
    wandb_entity: str,
    wandb_project: str,
    artifact_name: str,
    aliases: list[str],
    aws_profile: str | None,
    domain: str | None,
    limit: int | None,
    apply: bool,
    client_factory: Callable[[str, str | None], Any] = create_encord_client,
    s3_factory: Callable[[str | None], Any] = create_s3_client,
) -> dict[str, Any]:
    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"Encord SSH private key does not exist: {ssh_key_file}")
    if not aliases or any(not alias.strip() for alias in aliases):
        raise typer.BadParameter("At least one non-empty W&B artifact alias is required")

    client = client_factory(str(ssh_key_file), domain)
    dataset = client.get_dataset(dataset_hash)
    project = client.get_project(project_hash)
    validate_project_dataset(project, dataset_hash)
    captions = build_caption_map(export_labels(project))
    plans = build_episode_plan(client, dataset, captions, limit)

    with TemporaryDirectory(prefix="encord-wandb-") as temporary_directory:
        output_root = Path(temporary_directory)
        references, summary = build_train_ready_dataset(plans, output_root, s3_factory(aws_profile))
        manifest = build_export_manifest(
            plans=plans,
            summary=summary,
            dataset_hash=dataset_hash,
            project_hash=project_hash,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            artifact_name=artifact_name,
        )
        write_json(output_root / "encord_export_manifest.json", manifest)
        typer.echo(
            f"Validated {summary['episode_count']} episodes, {summary['frame_count']} frames, "
            f"and {summary['video_reference_count']} S3 video references."
        )
        if not apply:
            typer.echo("Validation complete. Pass --apply to publish the W&B artifact.")
            return manifest

        qualified_name, run_url = publish_artifact(
            output_root=output_root,
            references=references,
            manifest=manifest,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            artifact_name=artifact_name,
            aliases=aliases,
        )
        typer.echo(f"Artifact: {qualified_name}")
        if run_url:
            typer.echo(f"Run: {run_url}")
        return manifest


def main(
    ssh_key_file: Annotated[Path, typer.Option(help="Path to the Encord SSH private key file.")],
    dataset_hash: Annotated[str, typer.Option(help="Prepared Encord data-group dataset ID.")],
    project_hash: Annotated[str, typer.Option(help="Encord caption project ID.")],
    wandb_entity: Annotated[str, typer.Option(help="Destination W&B entity.")],
    wandb_project: Annotated[str, typer.Option(help="Destination W&B project.")],
    artifact_name: Annotated[
        str, typer.Option(help="W&B dataset artifact collection name.")
    ] = "encord-train-ready",
    alias: Annotated[
        list[str] | None, typer.Option("--alias", help="Artifact alias; repeat for multiple aliases.")
    ] = None,
    aws_profile: Annotated[
        str | None, typer.Option(help="Optional AWS profile for source metadata and parquet reads.")
    ] = None,
    domain: Annotated[
        str | None, typer.Option(help="Optional Encord API domain, for example the US deployment.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Maximum episodes for a validation or publish smoke run.")
    ] = None,
    apply: Annotated[
        bool, typer.Option("--apply", help="Publish after the complete dataset passes validation.")
    ] = False,
) -> None:
    run_export(
        ssh_key_file=ssh_key_file,
        dataset_hash=dataset_hash,
        project_hash=project_hash,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        artifact_name=artifact_name,
        aliases=alias or ["latest"],
        aws_profile=aws_profile,
        domain=domain,
        limit=limit,
        apply=apply,
    )


if __name__ == "__main__":
    typer.run(main)
