# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "pyyaml",
#     "typer",
#     "wandb>=0.18.0",
# ]
# ///
"""Export an Encord data-group dataset to a W&B dataset artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import UUID

import typer
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WANDB_CONFIG = SCRIPT_DIR.parent / "wandb_config.yaml"
EXPORT_ROOT = REPO_ROOT / "exports/encord-dataset-export"
CHUNK_SIZE = 1000
CAMERA_ORDER = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CAMERA_TO_DROID_KEY = {
    "cam_high": "exterior_image_1_left",
    "cam_left_wrist": "wrist_image_left",
    "cam_right_wrist": "wrist_image_right",
}


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"{label} does not exist: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"{label} must contain a YAML object")
    return loaded


def required(config: dict[str, Any], key: str, label: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise typer.BadParameter(f"{label} is missing required key: {key}")
    return value


def create_client():
    from encord.user_client import EncordUserClient

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def make_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = EXPORT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def item_metadata(item: Any) -> dict[str, Any]:
    return getattr(item, "client_metadata", None) or {}


def source_uri(item: Any) -> str:
    metadata = item_metadata(item)
    uri = metadata.get("source_uri") or metadata.get("s3_uri") or metadata.get("source_s3_uri")
    if uri:
        return str(uri)
    source_key = metadata.get("source_key")
    if source_key:
        return f"s3://ego-data-collection-encord/{source_key}"
    raise ValueError(f"No S3 source URI found for item {item.uuid} ({item.name})")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        return bucket, unquote(parsed.path.lstrip("/"))
    raise ValueError(f"Unsupported S3 URI format: {uri}")


def s3_client(unsigned: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if unsigned:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def group_children(item: Any, client: Any) -> list[Any]:
    children_by_uuid = {str(child.uuid): child for child in item.get_child_items()}

    try:
        summary = item.get_summary()
    except Exception:
        return list(children_by_uuid.values())

    if summary.data_group is not None:
        child_uuids = [
            child.uuid
            for child in summary.data_group.layout_contents.values()
            if str(child.uuid) not in children_by_uuid
        ]
        if child_uuids:
            for child in client.get_storage_items(child_uuids):
                children_by_uuid[str(child.uuid)] = child

    return list(children_by_uuid.values())


def video_children_by_camera(group_item: Any, client: Any) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    videos = {}
    for child in group_children(group_item, client):
        if child.item_type != StorageItemType.VIDEO:
            continue
        camera_name = item_metadata(child).get("camera_name")
        if camera_name:
            videos[str(camera_name)] = child
    return videos


def lerobot_video_path(episode_index: int, camera_name: str) -> Path:
    chunk = episode_index // CHUNK_SIZE
    video_key = f"observation.images.{CAMERA_TO_DROID_KEY[camera_name]}"
    return Path("dataset") / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def download_video(client_s3: Any, uri: str, destination: Path) -> None:
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client_s3.download_file(bucket, key, str(destination))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))


def export_dataset(
    *,
    client: Any,
    dataset_hash: UUID,
    output_dir: Path,
    limit: int | None,
    unsigned_s3: bool,
) -> dict[str, Any]:
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    if limit is not None:
        data_rows = data_rows[:limit]
    typer.echo(f"Exporting {len(data_rows)} Encord data groups from dataset {dataset_hash}...")

    backing_ids = [row.backing_item_uuid for row in data_rows]
    group_items_by_uuid = {str(item.uuid): item for item in client.get_storage_items(backing_ids)}
    client_s3 = s3_client(unsigned_s3)

    episodes = []
    source_items = []
    video_keys = [f"observation.images.{CAMERA_TO_DROID_KEY[camera]}" for camera in CAMERA_ORDER]

    for episode_index, row in enumerate(data_rows):
        group_item = group_items_by_uuid.get(str(row.backing_item_uuid))
        if group_item is None:
            raise ValueError(f"Could not resolve backing storage item for data row {row.uid}")

        videos = video_children_by_camera(group_item, client)
        missing = [camera for camera in CAMERA_ORDER if camera not in videos]
        if missing:
            raise ValueError(f"Group {group_item.uuid} is missing camera videos: {missing}")

        typer.echo(f"[{episode_index + 1}/{len(data_rows)}] {group_item.name}")
        for camera_name in CAMERA_ORDER:
            item = videos[camera_name]
            uri = source_uri(item)
            relative_path = lerobot_video_path(episode_index, camera_name)
            local_path = output_dir / relative_path
            typer.echo(f"  downloading {camera_name}: {uri}")
            download_video(client_s3, uri, local_path)
            source_items.append({
                "episode_index": episode_index,
                "data_hash": row.uid,
                "data_group_uuid": str(group_item.uuid),
                "video_storage_item_uuid": str(item.uuid),
                "camera_name": camera_name,
                "video_key": str(Path(relative_path).parent.relative_to(Path("dataset") / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}")),
                "artifact_path": str(relative_path),
                "source_uri": uri,
                "client_metadata": item_metadata(item),
            })

        episodes.append({
            "episode_index": episode_index,
            "tasks": [],
            "length": None,
            "success": None,
            "encord_data_hash": str(row.uid),
            "encord_data_group_uuid": str(group_item.uuid),
            "encord_data_title": row.title,
        })

    meta_dir = output_dir / "dataset" / "meta"
    write_jsonl(meta_dir / "episodes.jsonl", episodes)
    write_json(meta_dir / "source_dataset_items.json", source_items)
    write_json(meta_dir / "source_dataset_manifest.json", {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "encord_dataset_hash": str(dataset_hash),
        "encord_dataset_title": dataset.title,
        "episode_count": len(episodes),
        "camera_order": CAMERA_ORDER,
        "video_keys": video_keys,
        "lerobot_root": "dataset",
    })
    write_json(meta_dir / "info.json", {
        "codebase_version": "v2.0",
        "robot_type": "droid",
        "total_episodes": len(episodes),
        "total_frames": None,
        "total_tasks": 0,
        "total_videos": len(video_keys),
        "total_chunks": (len(episodes) // CHUNK_SIZE) + (1 if len(episodes) % CHUNK_SIZE else 0),
        "chunks_size": CHUNK_SIZE,
        "fps": None,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            video_key: {"dtype": "video", "shape": [None, None, 3], "names": ["height", "width", "channel"]}
            for video_key in video_keys
        },
    })

    return {
        "encord_dataset_hash": str(dataset_hash),
        "encord_dataset_title": dataset.title,
        "episode_count": len(episodes),
        "video_count": len(source_items),
        "local_dataset_dir": str(output_dir / "dataset"),
    }


def log_to_wandb(
    *,
    wandb_config: dict[str, Any],
    output_dir: Path,
    summary: dict[str, Any],
    aliases: list[str],
) -> dict[str, str]:
    import wandb

    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    artifact_name = required(wandb_config, "source_artifact_name", "W&B config")
    run_name = (
        f"encord-dataset-{artifact_name}-"
        f"{str(summary['encord_dataset_hash'])[:8]}-{summary['episode_count']}eps"
    )

    with wandb.init(entity=entity, project=project, job_type="encord-dataset-export", name=run_name) as run:
        artifact = wandb.Artifact(
            artifact_name,
            type="dataset",
            metadata=summary,
            description=f"Encord dataset export {summary['encord_dataset_hash']}",
        )
        artifact.add_dir(str(output_dir / "dataset"), name="dataset")
        logged = run.log_artifact(artifact, aliases=aliases)
        logged.wait()
        artifact_ref = f"{artifact_name}:{logged.version}"
        return {"dataset_artifact": artifact_ref, "run_url": run.url}


def main(
    dataset_hash: Annotated[UUID, typer.Option(help="Encord dataset hash to export.")],
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
    limit: Annotated[int | None, typer.Option(help="Optional max number of data groups to export.")] = None,
    alias: Annotated[list[str] | None, typer.Option("--alias", help="W&B artifact alias. Repeatable.")] = None,
    unsigned_s3: Annotated[bool, typer.Option(help="Use unsigned S3 requests for public buckets.")] = False,
) -> None:
    wandb_settings = load_yaml(wandb_config, "W&B config")
    output_dir = make_output_dir()
    typer.echo(f"Writing local export to {output_dir}")

    client = create_client()
    summary = export_dataset(
        client=client,
        dataset_hash=dataset_hash,
        output_dir=output_dir,
        limit=limit,
        unsigned_s3=unsigned_s3,
    )
    lineage = log_to_wandb(
        wandb_config=wandb_settings,
        output_dir=output_dir,
        summary=summary,
        aliases=alias or ["latest"],
    )
    write_json(output_dir / "wandb_lineage.json", lineage)

    typer.echo(f"dataset artifact: {lineage['dataset_artifact']}")
    typer.echo(f"run: {lineage['run_url']}")
    typer.echo(f"local files: {output_dir}")


if __name__ == "__main__":
    typer.run(main)
