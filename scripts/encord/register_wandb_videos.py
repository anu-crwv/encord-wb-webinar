# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "encord==0.1.199",
#     "typer",
#     "wandb==0.28.1",
# ]
# ///
"""Register a train-ready W&B artifact's external S3 videos in Encord."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import quote, unquote, urlparse

import typer

VIDEO_EXTENSIONS = {".3g2", ".3gp", ".avi", ".mkv", ".mj2", ".mov", ".mp4", ".webm"}
CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FEATURE_TO_CAMERA = {
    "exterior_image_1_left": "cam_high",
    "wrist_image_left": "cam_left_wrist",
    "wrist_image_right": "cam_right_wrist",
}
CAMERA_TO_FEATURE = {camera: feature for feature, camera in FEATURE_TO_CAMERA.items()}
VIDEO_PATH_RE = re.compile(
    r"^dataset/videos/chunk-\d+/observation\.images\.([^/]+)/episode_(\d+)\.[A-Za-z0-9]+$"
)
EPISODE_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ArtifactVideo:
    object_url: str
    episode_index: int
    title: str
    client_metadata: dict[str, Any]


def create_encord_client(ssh_key_file: Path, domain: str | None = None) -> Any:
    from encord import EncordUserClient

    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"SSH key file does not exist: {ssh_key_file}")
    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def artifact_entries(artifact: Any) -> dict[str, Any]:
    entries = getattr(getattr(artifact, "manifest", None), "entries", None)
    if not isinstance(entries, dict):
        raise typer.BadParameter("Could not read the W&B artifact manifest.")
    return entries


def entry_reference(entry: Any, artifact_path: str) -> str:
    value = getattr(entry, "ref", None)
    if value:
        return str(value)
    try:
        value = entry.ref_target()
    except (AttributeError, ValueError) as exc:
        raise typer.BadParameter(
            f"W&B video entry is stored in the artifact instead of referenced: {artifact_path}"
        ) from exc
    if not value:
        raise typer.BadParameter(f"W&B video entry has no external reference: {artifact_path}")
    return str(value)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3" and parsed.netloc and parsed.path.lstrip("/"):
        return parsed.netloc, unquote(parsed.path.lstrip("/"))
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        key = unquote(parsed.path.lstrip("/"))
        if bucket and key:
            return bucket, key
    raise typer.BadParameter(f"Expected an external S3 reference, got: {uri}")


def canonical_s3_uri(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"s3://{bucket}/{key}"


def encord_object_url(uri: str, region: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"https://{bucket}.s3.{region}.amazonaws.com/{quote(key, safe='/-_.~')}"


def episode_path_from_uri(uri: str) -> str:
    _bucket, key = parse_s3_uri(uri)
    parts = [part for part in key.split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_RE.fullmatch(part):
            return "/".join(parts[: index + 1]) + "/"
    raise typer.BadParameter(f"S3 video reference contains no episode path: {uri}")


def task_name_from_episode_path(episode_path: str) -> str | None:
    parts = [part for part in episode_path.strip("/").split("/") if part]
    if "raw-feed" not in parts:
        return None
    index = parts.index("raw-feed")
    if index + 2 < len(parts) and parts[index + 1] in {
        "trossen-data",
        "trossen-data-stationary",
    }:
        return parts[index + 2]
    return None


def collection_date_from_episode_path(episode_path: str) -> str | None:
    return next(
        (part for part in episode_path.strip("/").split("/") if DATE_RE.fullmatch(part)),
        None,
    )


def path_identity(artifact_path: str) -> tuple[int, str]:
    match = VIDEO_PATH_RE.fullmatch(artifact_path)
    if match is None:
        raise typer.BadParameter(
            "Video entry does not match the train-ready artifact layout: "
            f"{artifact_path}"
        )
    feature, episode = match.groups()
    camera = FEATURE_TO_CAMERA.get(feature)
    if camera is None:
        raise typer.BadParameter(
            f"Video entry uses an unsupported camera feature {feature!r}: {artifact_path}"
        )
    return int(episode), camera


def artifact_value(artifact: Any, name: str) -> Any:
    value = getattr(artifact, name, None)
    return value() if callable(value) else value


def build_registration_plan(
    artifact: Any,
    artifact_ref: str,
    s3_region: str,
) -> list[ArtifactVideo]:
    if not re.fullmatch(r"[a-z0-9-]+", s3_region):
        raise typer.BadParameter(f"Invalid AWS region: {s3_region}")

    entries = artifact_entries(artifact)
    names = sorted(
        name
        for name in entries
        if name.startswith("dataset/videos/")
        and PurePosixPath(name).suffix.lower() in VIDEO_EXTENSIONS
    )
    if not names:
        raise typer.BadParameter(
            f"W&B artifact {artifact_ref} has no videos under dataset/videos/."
        )

    plans: list[ArtifactVideo] = []
    seen_uris: set[str] = set()
    episode_cameras: dict[int, dict[str, ArtifactVideo]] = defaultdict(dict)
    episode_paths: dict[int, set[str]] = defaultdict(set)
    for name in names:
        episode_index, camera = path_identity(name)
        source_uri = canonical_s3_uri(entry_reference(entries[name], name))
        if source_uri in seen_uris:
            raise typer.BadParameter(f"Duplicate W&B S3 video reference: {source_uri}")
        seen_uris.add(source_uri)
        if camera in episode_cameras[episode_index]:
            raise typer.BadParameter(
                f"Episode {episode_index} contains duplicate {camera} video entries."
            )

        episode_path = episode_path_from_uri(source_uri)
        episode_paths[episode_index].add(episode_path)
        _bucket, source_key = parse_s3_uri(source_uri)
        metadata: dict[str, Any] = {
            "camera_name": camera,
            "sensor_key": f"observation.images.{CAMERA_TO_FEATURE[camera]}",
            "episode_index": episode_index,
            "episode_path": episode_path,
            "source_uri": source_uri,
            "source_key": source_key,
            "source_wandb_artifact": artifact_ref,
            "source_wandb_entry_path": name,
        }
        task_name = task_name_from_episode_path(episode_path)
        collection_date = collection_date_from_episode_path(episode_path)
        if task_name:
            metadata["task_name"] = task_name
        if collection_date:
            metadata["collection_datetime"] = collection_date
        artifact_digest = artifact_value(artifact, "digest")
        if artifact_digest:
            metadata["source_wandb_artifact_digest"] = str(artifact_digest)

        plan = ArtifactVideo(
            object_url=encord_object_url(source_uri, s3_region),
            episode_index=episode_index,
            title=source_key,
            client_metadata=metadata,
        )
        episode_cameras[episode_index][camera] = plan
        plans.append(plan)

    errors = []
    expected = set(CAMERA_ORDER)
    for episode_index in sorted(episode_cameras):
        cameras = set(episode_cameras[episode_index])
        if cameras != expected:
            errors.append(
                f"episode {episode_index}: expected {list(CAMERA_ORDER)}, found {sorted(cameras)}"
            )
        if len(episode_paths[episode_index]) != 1:
            errors.append(
                f"episode {episode_index}: references resolve to multiple source episode paths"
            )
    if errors:
        raise typer.BadParameter("Invalid three-camera artifact:\n  - " + "\n  - ".join(errors[:20]))
    return plans


def select_episodes(plans: list[ArtifactVideo], limit: int | None) -> list[ArtifactVideo]:
    episode_indexes = sorted({plan.episode_index for plan in plans})
    selected = set(episode_indexes[:limit] if limit is not None else episode_indexes)
    return [plan for plan in plans if plan.episode_index in selected]


def clean_resource_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "-", value).strip(" .-")
    return (name or "W&B referenced videos")[:120]


def find_integration(client: Any, title: str) -> Any:
    matches = list(client.get_cloud_integrations(filter_integration_titles=[title]))
    if len(matches) != 1:
        raise typer.BadParameter(
            f"Expected exactly one Encord cloud integration named {title!r}, found {len(matches)}."
        )
    return matches[0]


def register_videos(
    *,
    artifact_ref: str,
    s3_region: str,
    ssh_key_file: Path,
    integration_name: str,
    storage_folder_name: str | None,
    dataset_title: str | None,
    domain: str | None,
    limit: int | None,
) -> None:
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be at least 1.")

    import wandb

    artifact = wandb.Api().artifact(artifact_ref, type="dataset")
    all_plans = build_registration_plan(artifact, artifact_ref, s3_region)
    plans = select_episodes(all_plans, limit)
    episode_count = len({plan.episode_index for plan in plans})
    typer.echo(
        f"Validated {len(all_plans)} external video references across "
        f"{len(all_plans) // len(CAMERA_ORDER)} episodes."
    )
    typer.echo(f"Selected {len(plans)} videos across {episode_count} episodes.")
    typer.echo("Media transfer: none; Encord will register the existing S3 object URLs.")

    from encord.orm.dataset import LongPollingStatus, StorageLocation
    from encord.orm.storage import DataUploadItems, DataUploadVideo

    client = create_encord_client(ssh_key_file.expanduser(), domain)
    integration = find_integration(client, integration_name)
    resource_base = clean_resource_name(artifact_ref)
    folder = client.create_storage_folder(
        name=storage_folder_name or f"{resource_base} references",
        description="External S3 videos selected from a W&B dataset artifact.",
        client_metadata={
            "source_wandb_artifact": artifact_ref,
            "storage_mode": "external_s3_references",
        },
    )
    upload = DataUploadItems(
        videos=[
            DataUploadVideo(
                object_url=plan.object_url,
                title=plan.title,
                client_metadata=plan.client_metadata,
            )
            for plan in plans
        ]
    )
    job_id = folder.add_private_data_to_folder_start(
        integration_id=str(integration.id),
        private_files=upload,
        ignore_errors=False,
    )
    upload_result = folder.add_private_data_to_folder_get_result(job_id)
    if upload_result.status != LongPollingStatus.DONE:
        raise RuntimeError(
            f"Encord registration ended with {upload_result.status}: {upload_result.errors}"
        )
    if upload_result.units_error_count or len(upload_result.items_with_names) != len(plans):
        raise RuntimeError(
            "Encord registration did not return one item for every W&B reference: "
            f"{len(upload_result.items_with_names)}/{len(plans)} created, "
            f"{upload_result.units_error_count} errors."
        )

    dataset_response = client.create_dataset(
        dataset_title=dataset_title or resource_base,
        dataset_type=StorageLocation.CORD_STORAGE,
        dataset_description=f"External video references from W&B artifact {artifact_ref}.",
        create_backing_folder=False,
    )
    dataset_hash = str(dataset_response.dataset_hash)
    dataset = client.get_dataset(dataset_hash)
    dataset.link_items([item.item_uuid for item in upload_result.items_with_names])
    typer.echo(f"Storage folder: {folder.uuid}")
    typer.echo(f"Dataset: {dataset_hash}")


def main(
    artifact_ref: Annotated[
        str,
        typer.Option(help="Fully qualified W&B dataset artifact, including alias or version."),
    ],
    s3_region: Annotated[
        str,
        typer.Option(help="AWS region used to form permanent Encord object URLs."),
    ],
    ssh_key_file: Annotated[
        Path,
        typer.Option(help="Path to the Encord SSH private-key file."),
    ],
    integration_name: Annotated[
        str,
        typer.Option(help="Encord S3 integration name."),
    ],
    storage_folder_name: Annotated[
        str | None,
        typer.Option(help="Optional title for the new Encord storage folder."),
    ] = None,
    dataset_title: Annotated[
        str | None,
        typer.Option(help="Optional title for the new Encord dataset."),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(help="Optional Encord API domain, for example the US deployment."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum complete episodes to register after full validation."),
    ] = None,
) -> None:
    register_videos(
        artifact_ref=artifact_ref,
        s3_region=s3_region,
        ssh_key_file=ssh_key_file,
        integration_name=integration_name,
        storage_folder_name=storage_folder_name,
        dataset_title=dataset_title,
        domain=domain,
        limit=limit,
    )


if __name__ == "__main__":
    typer.run(main)
