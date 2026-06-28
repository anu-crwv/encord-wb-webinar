# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "botocore",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "pyyaml",
#     "tqdm",
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
import shutil
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

import typer
import yaml
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WANDB_CONFIG = SCRIPT_DIR.parent / "wandb_config.yaml"
DEFAULT_EXPORT_CONFIG = SCRIPT_DIR / "dataset_export_config.yaml"
EXPORT_ROOT = REPO_ROOT / "exports/encord-dataset-export"
S3_CACHE_ROOT = EXPORT_ROOT / "_cache" / "s3"
CHUNK_SIZE = 1000
CAMERA_ORDER = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CAMERA_TO_DROID_KEY = {
    "cam_high": "exterior_image_1_left",
    "cam_left_wrist": "wrist_image_left",
    "cam_right_wrist": "wrist_image_right",
}
META_ENTRY_PATHS = [
    "dataset/meta/episodes.jsonl",
    "dataset/meta/source_dataset_items.json",
    "dataset/meta/source_dataset_manifest.json",
    "dataset/meta/info.json",
]


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


def configured_tags(config: dict[str, Any]) -> list[str]:
    tags = config.get("tags") or []
    if isinstance(tags, str):
        return [tags]
    if not isinstance(tags, list):
        raise typer.BadParameter("Dataset export config tags must be a list or string.")
    return [str(tag) for tag in tags]


def configured_description(config: dict[str, Any], summary: dict[str, Any]) -> str:
    return str(config.get("description") or f"Encord dataset export {summary['encord_dataset_hash']}")


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


def metadata_value(metadata: Any, key: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def coerce_fps(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    if fps <= 0:
        return None
    return fps


def item_fps(item: Any) -> float | None:
    client_meta = item_metadata(item)
    for key in ("fps", "collection_fps"):
        fps = coerce_fps(client_meta.get(key))
        if fps is not None:
            return fps
    return coerce_fps(metadata_value(getattr(item, "metadata", None), "fps"))


def shared_fps(fps_values: list[float]) -> float | None:
    if not fps_values:
        typer.echo("Warning: no FPS found in video item metadata; writing fps=null.", err=True)
        return None

    distinct: dict[float, float] = {}
    for fps in fps_values:
        distinct.setdefault(round(fps, 6), fps)
    if len(distinct) > 1:
        values = ", ".join(str(value) for value in sorted(distinct.values()))
        raise ValueError(f"Exported videos have multiple FPS values: {values}")
    return next(iter(distinct.values()))


def combined_fps(base_info: dict[str, Any] | None, fps_values: list[float]) -> float | None:
    base_fps = coerce_fps((base_info or {}).get("fps"))
    new_fps = shared_fps(fps_values) if fps_values else None
    if base_fps is not None and new_fps is not None and round(base_fps, 6) != round(new_fps, 6):
        raise ValueError(f"New videos have FPS {new_fps}, but base artifact has FPS {base_fps}")
    if base_fps is not None:
        return base_fps
    if new_fps is not None:
        return new_fps
    typer.echo("Warning: no FPS found in base artifact or new video metadata; writing fps=null.", err=True)
    return None


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


def s3_cache_path(bucket: str, key: str) -> Path:
    key_parts = key.split("/")
    if not bucket or not key or any(part == ".." for part in key_parts):
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    cache_parts = [part for part in key_parts if part not in {"", "."}]
    if not cache_parts:
        raise ValueError(f"Unsafe S3 cache path for s3://{bucket}/{key}")
    return S3_CACHE_ROOT / bucket / Path(*cache_parts)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def download_video(client_s3: Any, uri: str, destination: Path) -> bool:
    bucket, key = parse_s3_uri(uri)
    cache_path = s3_cache_path(bucket, key)
    downloaded = False

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            client_s3.download_file(bucket, key, str(tmp_path))
            os.replace(tmp_path, cache_path)
            downloaded = True
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    link_or_copy(cache_path, destination)
    return downloaded


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def qualified_artifact_ref(wandb_config: dict[str, Any], artifact_ref: str) -> str:
    entity = required(wandb_config, "entity", "W&B config")
    project = required(wandb_config, "project", "W&B config")
    if "/" not in artifact_ref.split(":", 1)[0]:
        return f"{entity}/{project}/{artifact_ref}"
    return artifact_ref


def artifact_aliases(artifact: Any) -> list[str]:
    aliases = getattr(artifact, "aliases", None) or []
    return [str(alias) for alias in aliases]


def artifact_attr(artifact: Any, name: str) -> Any:
    value = getattr(artifact, name, None)
    return value() if callable(value) else value


def artifact_version(artifact: Any) -> str:
    version = artifact_attr(artifact, "version")
    if version not in (None, ""):
        return str(version)

    name = str(artifact_attr(artifact, "name") or "")
    if ":" in name:
        return name.rsplit(":", 1)[1]

    raise ValueError("Could not resolve base dataset artifact to an immutable W&B version.")


def base_artifact_fields(base_artifact: dict[str, Any] | None) -> dict[str, Any]:
    if base_artifact is None:
        return {}
    return {
        "base_dataset_artifact": base_artifact["resolved_ref"],
        "base_dataset_artifact_requested": base_artifact["requested_ref"],
        "base_dataset_artifact_version": base_artifact["version"],
        "base_dataset_artifact_digest": base_artifact.get("digest"),
        "base_dataset_artifact_url": base_artifact.get("url"),
        "base_dataset_artifact_aliases": base_artifact.get("aliases", []),
    }


def download_artifact_entry(artifact: Any, name: str, root: Path) -> Path:
    return Path(artifact.get_entry(name).download(root=str(root)))


def validate_base_metadata(episodes: list[dict[str, Any]], source_items: list[dict[str, Any]]) -> None:
    indices = sorted(int(row["episode_index"]) for row in episodes)
    if len(indices) != len(set(indices)):
        raise ValueError("Base artifact has duplicate episode_index values")
    expected = list(range(indices[-1] + 1)) if indices else []
    if indices != expected:
        raise ValueError(f"Base artifact episode indices are not contiguous from 0: {indices[:5]}...{indices[-5:]}")

    episode_set = set(indices)
    items_by_episode: dict[int, list[dict[str, Any]]] = {index: [] for index in indices}
    for item in source_items:
        episode_index = int(item["episode_index"])
        if episode_index not in episode_set:
            raise ValueError(f"Base artifact source item references missing episode_index {episode_index}")
        items_by_episode[episode_index].append(item)

    for episode_index in indices:
        cameras = {str(item.get("camera_name")) for item in items_by_episode.get(episode_index, [])}
        missing = [camera for camera in CAMERA_ORDER if camera not in cameras]
        if missing:
            raise ValueError(f"Base artifact episode {episode_index} is missing source items for cameras: {missing}")


def load_base_artifact_metadata(
    *,
    wandb_config: dict[str, Any],
    base_artifact_ref: str,
    output_dir: Path,
) -> dict[str, Any]:
    import wandb

    artifact_ref = qualified_artifact_ref(wandb_config, base_artifact_ref)
    typer.echo(f"Loading base dataset artifact metadata from {artifact_ref}...")
    artifact = wandb.Api().artifact(artifact_ref)
    version = artifact_version(artifact)
    resolved_ref = f"{artifact_ref.split(':', 1)[0]}:{version}"
    if resolved_ref != artifact_ref:
        typer.echo(f"Resolved base dataset artifact to {resolved_ref}.")

    artifact_dir = output_dir / "base_artifact_metadata"
    episodes_path = download_artifact_entry(artifact, "dataset/meta/episodes.jsonl", artifact_dir)
    items_path = download_artifact_entry(artifact, "dataset/meta/source_dataset_items.json", artifact_dir)
    manifest_path = download_artifact_entry(artifact, "dataset/meta/source_dataset_manifest.json", artifact_dir)
    info_path = download_artifact_entry(artifact, "dataset/meta/info.json", artifact_dir)

    episodes = read_jsonl(episodes_path)
    source_items = json.loads(items_path.read_text())
    if not isinstance(source_items, list):
        raise ValueError("Base artifact source_dataset_items.json must contain a JSON list")
    validate_base_metadata(episodes, source_items)

    return {
        "requested_ref": base_artifact_ref,
        "qualified_requested_ref": artifact_ref,
        "resolved_ref": resolved_ref,
        "version": version,
        "digest": artifact_attr(artifact, "digest"),
        "url": artifact_attr(artifact, "url"),
        "aliases": artifact_aliases(artifact),
        "episodes": episodes,
        "source_items": source_items,
        "manifest": json.loads(manifest_path.read_text()),
        "info": json.loads(info_path.read_text()),
    }


def base_identity_sets(base_artifact: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    data_hashes: set[str] = set()
    group_uuids: set[str] = set()
    if base_artifact is None:
        return data_hashes, group_uuids

    for episode in base_artifact["episodes"]:
        if episode.get("encord_data_hash"):
            data_hashes.add(str(episode["encord_data_hash"]))
        if episode.get("encord_data_group_uuid"):
            group_uuids.add(str(episode["encord_data_group_uuid"]))

    for item in base_artifact["source_items"]:
        if item.get("data_hash"):
            data_hashes.add(str(item["data_hash"]))
        if item.get("data_group_uuid"):
            group_uuids.add(str(item["data_group_uuid"]))

    return data_hashes, group_uuids


def next_episode_index(base_episodes: list[dict[str, Any]]) -> int:
    if not base_episodes:
        return 0
    return max(int(row["episode_index"]) for row in base_episodes) + 1


def total_chunks(total_episodes: int) -> int:
    return (total_episodes // CHUNK_SIZE) + (1 if total_episodes % CHUNK_SIZE else 0)


def source_item_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    camera_name = str(item.get("camera_name") or "")
    camera_index = CAMERA_ORDER.index(camera_name) if camera_name in CAMERA_ORDER else len(CAMERA_ORDER)
    return int(item.get("episode_index", -1)), camera_index, str(item.get("artifact_path") or "")


def build_info(
    *,
    base_info: dict[str, Any] | None,
    total_episodes: int,
    dataset_fps: float | None,
    video_keys: list[str],
) -> dict[str, Any]:
    info = dict(base_info or {})
    features = dict(info.get("features") or {})
    if not features:
        features = {
            video_key: {
                "dtype": "video",
                "shape": [None, None, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": dataset_fps},
            }
            for video_key in video_keys
        }

    info.update({
        "codebase_version": info.get("codebase_version", "v2.0"),
        "robot_type": info.get("robot_type", "droid"),
        "total_episodes": total_episodes,
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks", 0),
        "total_videos": len(video_keys),
        "total_chunks": total_chunks(total_episodes),
        "chunks_size": CHUNK_SIZE,
        "fps": dataset_fps,
        "splits": info.get("splits", {"train": "0:100"}),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    })
    return info


def write_dataset_metadata(
    *,
    output_dir: Path,
    dataset_hash: UUID,
    dataset_title: str,
    episodes: list[dict[str, Any]],
    source_items: list[dict[str, Any]],
    dataset_fps: float | None,
    base_artifact: dict[str, Any] | None,
    new_episode_count: int,
    new_video_count: int,
    skipped_incomplete_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    video_keys = [f"observation.images.{CAMERA_TO_DROID_KEY[camera]}" for camera in CAMERA_ORDER]
    sorted_episodes = sorted(episodes, key=lambda row: int(row["episode_index"]))
    sorted_source_items = sorted(source_items, key=source_item_sort_key)
    preserved_episode_count = len(base_artifact["episodes"]) if base_artifact else 0
    preserved_video_count = len(base_artifact["source_items"]) if base_artifact else 0
    skipped_incomplete_groups = skipped_incomplete_groups or []

    meta_dir = output_dir / "dataset" / "meta"
    write_jsonl(meta_dir / "episodes.jsonl", sorted_episodes)
    write_json(meta_dir / "source_dataset_items.json", sorted_source_items)
    write_json(meta_dir / "source_dataset_manifest.json", {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "encord_dataset_hash": str(dataset_hash),
        "encord_source_dataset_hash": str(dataset_hash),
        "encord_dataset_title": dataset_title,
        "episode_count": len(sorted_episodes),
        "camera_order": CAMERA_ORDER,
        "video_keys": video_keys,
        "lerobot_root": "dataset",
        "preserved_episode_count": preserved_episode_count,
        "new_episode_count": new_episode_count,
        "skipped_incomplete_group_count": len(skipped_incomplete_groups),
        "skipped_incomplete_groups": skipped_incomplete_groups,
        **base_artifact_fields(base_artifact),
    })
    write_json(meta_dir / "info.json", build_info(
        base_info=(base_artifact or {}).get("info"),
        total_episodes=len(sorted_episodes),
        dataset_fps=dataset_fps,
        video_keys=video_keys,
    ))

    return {
        "encord_dataset_hash": str(dataset_hash),
        "encord_source_dataset_hash": str(dataset_hash),
        "encord_dataset_title": dataset_title,
        "episode_count": len(sorted_episodes),
        "video_count": len(sorted_source_items),
        "preserved_episode_count": preserved_episode_count,
        "preserved_video_count": preserved_video_count,
        "new_episode_count": new_episode_count,
        "new_video_count": new_video_count,
        "skipped_incomplete_group_count": len(skipped_incomplete_groups),
        "skipped_incomplete_groups": skipped_incomplete_groups,
        "local_dataset_dir": str(output_dir / "dataset"),
        **base_artifact_fields(base_artifact),
    }


def incomplete_group_record(row: Any, group_item: Any, missing_cameras: list[str]) -> dict[str, Any]:
    return {
        "data_hash": str(row.uid),
        "data_title": row.title,
        "data_group_uuid": str(group_item.uuid),
        "data_group_name": getattr(group_item, "name", None),
        "missing_cameras": missing_cameras,
    }


def export_dataset(
    *,
    client: Any,
    dataset_hash: UUID,
    output_dir: Path,
    limit: int | None,
    unsigned_s3: bool,
    base_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    typer.echo(f"Found {len(data_rows)} Encord data groups in dataset {dataset_hash}.")

    existing_data_hashes, existing_group_uuids = base_identity_sets(base_artifact)
    if base_artifact is None and limit is not None:
        data_rows = data_rows[:limit]

    all_candidate_rows = [row for row in data_rows if str(row.uid) not in existing_data_hashes]
    skipped_by_data_hash = len(data_rows) - len(all_candidate_rows)
    candidate_rows = all_candidate_rows
    if base_artifact is not None and limit is not None:
        candidate_rows = candidate_rows[:limit]

    backing_ids = [row.backing_item_uuid for row in candidate_rows]
    group_items_by_uuid = {
        str(item.uuid): item for item in client.get_storage_items(backing_ids)
    } if backing_ids else {}

    export_rows = []
    skipped_existing_by_group = 0
    for row in candidate_rows:
        group_item = group_items_by_uuid.get(str(row.backing_item_uuid))
        if group_item is None:
            raise ValueError(f"Could not resolve backing storage item for data row {row.uid}")
        if str(group_item.uuid) in existing_group_uuids:
            skipped_existing_by_group += 1
            continue
        export_rows.append((row, group_item))

    if base_artifact is not None:
        typer.echo(
            f"Base artifact has {len(base_artifact['episodes'])} episodes; "
            f"skipping {skipped_by_data_hash + skipped_existing_by_group} existing rows."
        )
    typer.echo(f"Exporting {len(export_rows)} new Encord data groups...")

    client_s3 = s3_client(unsigned_s3)
    base_episodes = list((base_artifact or {}).get("episodes") or [])
    base_source_items = list((base_artifact or {}).get("source_items") or [])
    new_episodes = []
    new_source_items = []
    skipped_incomplete_groups = []
    fps_values = []
    first_episode_index = next_episode_index(base_episodes)
    cache_hits = 0
    cache_downloads = 0

    total_video_files = len(export_rows) * len(CAMERA_ORDER)
    typer.echo(f"Using shared S3 video cache at {S3_CACHE_ROOT}")
    typer.echo("Checking each data group's cameras as it is exported; incomplete groups are skipped.")
    with tqdm(
        total=total_video_files,
        desc="Exporting video slots",
        unit="file",
        dynamic_ncols=True,
        mininterval=5.0,
    ) as progress:
        for row, group_item in export_rows:
            videos = video_children_by_camera(group_item, client)
            missing = [camera for camera in CAMERA_ORDER if camera not in videos]
            if missing:
                skipped_incomplete_groups.append(incomplete_group_record(row, group_item, missing))
                progress.write(
                    f"Skipping incomplete data group {row.title} ({row.uid}); "
                    f"missing cameras: {', '.join(missing)}"
                )
                progress.set_postfix(
                    {
                        "cached": cache_hits,
                        "downloaded": cache_downloads,
                        "skipped_groups": len(skipped_incomplete_groups),
                    },
                    refresh=False,
                )
                progress.update(len(CAMERA_ORDER))
                continue

            episode_index = first_episode_index + len(new_episodes)
            for camera_name in CAMERA_ORDER:
                item = videos[camera_name]
                uri = source_uri(item)
                fps = item_fps(item)
                if fps is not None:
                    fps_values.append(fps)
                relative_path = lerobot_video_path(episode_index, camera_name)
                local_path = output_dir / relative_path
                downloaded = download_video(client_s3, uri, local_path)
                if downloaded:
                    cache_downloads += 1
                else:
                    cache_hits += 1
                progress.set_postfix(
                    {
                        "cached": cache_hits,
                        "downloaded": cache_downloads,
                        "skipped_groups": len(skipped_incomplete_groups),
                    },
                    refresh=False,
                )
                progress.update()
                new_source_items.append({
                    "episode_index": episode_index,
                    "data_hash": row.uid,
                    "data_group_uuid": str(group_item.uuid),
                    "video_storage_item_uuid": str(item.uuid),
                    "camera_name": camera_name,
                    "video_key": str(Path(relative_path).parent.relative_to(
                        Path("dataset") / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
                    )),
                    "artifact_path": str(relative_path),
                    "source_uri": uri,
                    "fps": fps,
                    "client_metadata": item_metadata(item),
                })

            new_episodes.append({
                "episode_index": episode_index,
                "tasks": [],
                "length": None,
                "success": None,
                "encord_data_hash": str(row.uid),
                "encord_data_group_uuid": str(group_item.uuid),
                "encord_data_title": row.title,
            })

    if skipped_incomplete_groups:
        typer.echo(f"Skipped {len(skipped_incomplete_groups)} incomplete data groups.")

    dataset_fps = combined_fps((base_artifact or {}).get("info"), fps_values)
    return write_dataset_metadata(
        output_dir=output_dir,
        dataset_hash=dataset_hash,
        dataset_title=dataset.title,
        episodes=base_episodes + new_episodes,
        source_items=base_source_items + new_source_items,
        dataset_fps=dataset_fps,
        base_artifact=base_artifact,
        new_episode_count=len(new_episodes),
        new_video_count=len(new_source_items),
        skipped_incomplete_groups=skipped_incomplete_groups,
    )


def local_artifact_files(output_dir: Path) -> list[Path]:
    dataset_dir = output_dir / "dataset"
    if not dataset_dir.exists():
        return []
    return sorted(path for path in dataset_dir.rglob("*") if path.is_file())


def log_to_wandb(
    *,
    wandb_config: dict[str, Any],
    output_dir: Path,
    summary: dict[str, Any],
    tags: list[str],
    description: str,
    base_artifact: dict[str, Any] | None,
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
        if base_artifact is None:
            artifact = wandb.Artifact(
                artifact_name,
                type="dataset",
                metadata=summary,
                description=description,
            )
            artifact.add_dir(str(output_dir / "dataset"), name="dataset")
            logged = run.log_artifact(artifact, aliases=["latest"], tags=tags)
        else:
            typer.echo(f"Using base dataset artifact {base_artifact['resolved_ref']}.")
            saved = run.use_artifact(base_artifact["resolved_ref"])
            draft = saved.new_draft()
            draft.metadata.update(summary)
            draft.description = description

            for entry_name in META_ENTRY_PATHS:
                draft.remove(entry_name)
                draft.add_file(str(output_dir / entry_name), name=entry_name)

            new_files = [path for path in local_artifact_files(output_dir) if "dataset/meta" not in path.as_posix()]
            typer.echo(f"Adding {len(new_files)} new artifact files to incremental draft...")
            for path in new_files:
                draft.add_file(str(path), name=path.relative_to(output_dir).as_posix())
            logged = run.log_artifact(draft, aliases=["latest"], tags=tags)

        logged.wait()
        artifact_ref = f"{artifact_name}:{logged.version}"
        return {"dataset_artifact": artifact_ref, "run_url": run.url}


def main(
    dataset_hash: Annotated[UUID, typer.Option(help="Encord dataset hash to export.")],
    wandb_config: Annotated[Path, typer.Option(help="W&B config YAML.")] = DEFAULT_WANDB_CONFIG,
    export_config: Annotated[Path, typer.Option(help="Dataset export config YAML.")] = DEFAULT_EXPORT_CONFIG,
    limit: Annotated[int | None, typer.Option(help="Optional max number of data groups to export.")] = None,
    unsigned_s3: Annotated[bool, typer.Option(help="Use unsigned S3 requests for public buckets.")] = False,
    base_artifact_ref: Annotated[
        str | None,
        typer.Option(help="Existing W&B dataset artifact version to append to incrementally."),
    ] = None,
) -> None:
    wandb_settings = load_yaml(wandb_config, "W&B config")
    export_settings = load_yaml(export_config, "Dataset export config")
    output_dir = make_output_dir()
    typer.echo(f"Writing local export to {output_dir}")

    base_artifact = None
    if base_artifact_ref:
        base_artifact = load_base_artifact_metadata(
            wandb_config=wandb_settings,
            base_artifact_ref=base_artifact_ref,
            output_dir=output_dir,
        )

    client = create_client()
    summary = export_dataset(
        client=client,
        dataset_hash=dataset_hash,
        output_dir=output_dir,
        limit=limit,
        unsigned_s3=unsigned_s3,
        base_artifact=base_artifact,
    )
    write_json(output_dir / "local_export_summary.json", summary)

    if base_artifact is not None and summary["new_episode_count"] == 0:
        typer.echo("No new Encord data groups found; not logging a new W&B artifact version.")
        typer.echo(f"local files: {output_dir}")
        return

    lineage = log_to_wandb(
        wandb_config=wandb_settings,
        output_dir=output_dir,
        summary=summary,
        tags=configured_tags(export_settings),
        description=configured_description(export_settings, summary),
        base_artifact=base_artifact,
    )
    write_json(output_dir / "wandb_lineage.json", lineage)

    typer.echo(f"dataset artifact: {lineage['dataset_artifact']}")
    typer.echo(f"run: {lineage['run_url']}")
    typer.echo(f"local files: {output_dir}")


if __name__ == "__main__":
    typer.run(main)
