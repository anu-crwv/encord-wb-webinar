# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "encord==0.1.199",
#     "typer",
# ]
# ///
"""Build validated three-camera Encord data groups from an existing raw folder."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import UUID

import typer

CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EPISODE_DIR_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CHUNK_RE = re.compile(r"^chunk-\d+$")


@dataclass(frozen=True)
class EpisodeGroupPlan:
    episode_path: str
    videos: tuple[Any, ...]
    metadata_items: tuple[Any, ...]
    task_name: str | None
    collection_datetime: str | None


def create_client(ssh_key_file: Path, domain: str | None = None) -> Any:
    from encord import EncordUserClient

    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"SSH key file does not exist: {ssh_key_file}")
    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def item_metadata(item: Any) -> dict[str, Any]:
    value = getattr(item, "client_metadata", None) or {}
    return dict(value) if isinstance(value, dict) else {}


def normalize_source_path(value: Any) -> str:
    text = str(value or "")
    if text.startswith(("s3://", "http://", "https://")):
        return urlparse(text).path.lstrip("/")
    return text.lstrip("/")


def source_path(item: Any) -> str:
    metadata = item_metadata(item)
    return normalize_source_path(
        metadata.get("source_key")
        or metadata.get("source_uri")
        or metadata.get("s3_uri")
        or getattr(item, "name", "")
    )


def derive_episode_path(value: Any) -> str | None:
    parts = [part for part in normalize_source_path(value).split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_DIR_RE.fullmatch(part):
            return "/".join(parts[: index + 1]) + "/"
    return None


def episode_path_from_item(item: Any, *, inspect_children: bool = False) -> str | None:
    metadata = item_metadata(item)
    if metadata.get("episode_path"):
        value = str(metadata["episode_path"]).strip("/") + "/"
        if derive_episode_path(value):
            return value
    for key in (
        "source_key",
        "source_uri",
        "s3_uri",
        "source_s3_uri",
        "objectUrl",
        "object_url",
    ):
        value = derive_episode_path(metadata.get(key))
        if value:
            return value
    value = derive_episode_path(getattr(item, "name", None))
    if value or not inspect_children:
        return value
    for child in item.get_child_items():
        child_path = episode_path_from_item(child)
        if child_path:
            return child_path
    return None


def path_parts(value: str) -> list[str]:
    return [part for part in value.strip("/").split("/") if part]


def camera_name_from_item(item: Any) -> str | None:
    metadata = item_metadata(item)
    direct = metadata.get("camera_name")
    if direct:
        return str(direct)

    sensor_key = str(metadata.get("sensor_key") or "")
    if sensor_key:
        candidate = sensor_key.rsplit(".", 1)[-1]
        if candidate in CAMERA_ORDER:
            return candidate

    for part in path_parts(source_path(item)):
        candidate = part.rsplit(".", 1)[-1]
        if candidate in CAMERA_ORDER:
            return candidate
    return None


def metadata_role(item: Any) -> str:
    metadata = item_metadata(item)
    role = str(metadata.get("metadata_file_role") or "")
    if role and role != "none":
        return role
    name = Path(source_path(item)).name
    known = {
        "info.json": "info_json",
        "tasks.jsonl": "tasks_jsonl",
        "episodes.jsonl": "episodes_jsonl",
        "episodes_stats.jsonl": "episodes_stats_jsonl",
    }
    return known.get(name, sanitize_layout_key(name))


def is_metadata_item(item: Any) -> bool:
    metadata = item_metadata(item)
    extension = str(metadata.get("file_ext") or Path(source_path(item)).suffix).lower()
    role = str(metadata.get("metadata_file_role") or "none")
    return extension in {".json", ".jsonl"} or role != "none"


def load_items_by_episode(folder: Any) -> dict[str, dict[str, list[Any]]]:
    from encord.orm.storage import StorageItemType

    by_episode: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"videos": [], "metadata": []}
    )
    scanned = 0
    skipped = 0
    for item in folder.list_items(
        page_size=1000,
        item_types=[StorageItemType.VIDEO, StorageItemType.PLAIN_TEXT],
    ):
        scanned += 1
        episode_path = episode_path_from_item(item)
        if not episode_path:
            skipped += 1
            continue
        if item.item_type == StorageItemType.VIDEO:
            by_episode[episode_path]["videos"].append(item)
        elif is_metadata_item(item):
            by_episode[episode_path]["metadata"].append(item)

    typer.echo(f"Scanned raw items: {scanned}")
    typer.echo(f"Items without an episode path: {skipped}")
    typer.echo(f"Discovered episodes: {len(by_episode)}")
    return by_episode


def one_consistent_value(items: list[Any], key: str) -> str | None:
    values = {
        str(item_metadata(item)[key])
        for item in items
        if item_metadata(item).get(key) not in (None, "")
    }
    if len(values) > 1:
        raise ValueError(f"conflicting {key} values: {sorted(values)}")
    return next(iter(values), None)


def validate_episode_groups(
    by_episode: dict[str, dict[str, list[Any]]],
) -> list[EpisodeGroupPlan]:
    plans: list[EpisodeGroupPlan] = []
    errors: list[str] = []
    for episode_path, items in sorted(by_episode.items()):
        videos_by_camera: dict[str, list[Any]] = defaultdict(list)
        for item in items["videos"]:
            camera = camera_name_from_item(item)
            if camera is None:
                errors.append(
                    f"{episode_path}: could not resolve camera for {getattr(item, 'name', item)}"
                )
                continue
            videos_by_camera[camera].append(item)

        missing = [camera for camera in CAMERA_ORDER if not videos_by_camera[camera]]
        duplicates = [
            camera for camera in CAMERA_ORDER if len(videos_by_camera[camera]) > 1
        ]
        unexpected = sorted(set(videos_by_camera) - set(CAMERA_ORDER))
        if missing:
            errors.append(f"{episode_path}: missing cameras {missing}")
        if duplicates:
            errors.append(f"{episode_path}: duplicate cameras {duplicates}")
        if unexpected:
            errors.append(f"{episode_path}: unexpected cameras {unexpected}")
        if not items["metadata"]:
            errors.append(f"{episode_path}: no JSON/JSONL metadata item")
        if missing or duplicates or unexpected or not items["metadata"]:
            continue

        all_items = items["videos"] + items["metadata"]
        try:
            task_name = one_consistent_value(all_items, "task_name")
            collection_datetime = one_consistent_value(
                all_items,
                "collection_datetime",
            )
        except ValueError as exc:
            errors.append(f"{episode_path}: {exc}")
            continue

        plans.append(
            EpisodeGroupPlan(
                episode_path=episode_path,
                videos=tuple(videos_by_camera[camera][0] for camera in CAMERA_ORDER),
                metadata_items=tuple(
                    sorted(items["metadata"], key=lambda item: metadata_role(item))
                ),
                task_name=task_name,
                collection_datetime=collection_datetime,
            )
        )

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise typer.BadParameter(
            f"Raw folder failed data-group validation:\n{preview}{suffix}"
        )
    if not plans:
        raise typer.BadParameter("No complete three-camera episodes were found.")
    return plans


def clean_name_part(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("/", "-")).strip()


def first_path_match(episode_path: str, pattern: re.Pattern[str]) -> str | None:
    return next(
        (part for part in path_parts(episode_path) if pattern.fullmatch(part)),
        None,
    )


def group_name(plan: EpisodeGroupPlan) -> str:
    episode = path_parts(plan.episode_path)[-1]
    date = (plan.collection_datetime or first_path_match(plan.episode_path, DATE_RE) or "")[:10]
    chunk = next(
        (
            part
            for item in plan.videos
            for part in path_parts(source_path(item))
            if CHUNK_RE.fullmatch(part)
        ),
        None,
    )
    values = [plan.task_name, date or None, episode, chunk]
    name = " | ".join(clean_name_part(value) for value in values if value)
    return name[:120]


def sanitize_layout_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_") or "item"


def unique_key(base: str, used: set[str]) -> str:
    key = base
    suffix = 2
    while key in used:
        key = f"{base}_{suffix}"
        suffix += 1
    used.add(key)
    return key


def build_custom_group(plan: EpisodeGroupPlan, source_folder_id: UUID) -> Any:
    from encord.orm.group_layout import (
        DataUnitCarouselTile,
        DataUnitTile,
        LayoutGrid,
    )
    from encord.orm.storage import DataGroupCustom

    used: set[str] = set()
    layout_contents: dict[str, UUID] = {}
    camera_keys: dict[str, str] = {}
    for camera, item in zip(CAMERA_ORDER, plan.videos, strict=True):
        key = unique_key(f"camera_{camera}", used)
        camera_keys[camera] = key
        layout_contents[key] = item.uuid

    metadata_keys = []
    for item in plan.metadata_items:
        key = unique_key(f"metadata_{sanitize_layout_key(metadata_role(item))}", used)
        metadata_keys.append(key)
        layout_contents[key] = item.uuid

    wrist_grid = LayoutGrid(
        direction="row",
        splitPercentage=50,
        first=DataUnitTile(key=camera_keys["cam_left_wrist"]),
        second=DataUnitTile(key=camera_keys["cam_right_wrist"]),
    )
    metadata_carousel = DataUnitCarouselTile(
        keys=metadata_keys,
        carouselPosition="bottom",
        carouselSize=10,
    )
    layout = LayoutGrid(
        direction="row",
        splitPercentage=50,
        first=DataUnitTile(key=camera_keys["cam_high"]),
        second=LayoutGrid(
            direction="column",
            splitPercentage=50,
            first=wrist_grid,
            second=metadata_carousel,
        ),
    )
    metadata = {
        "episode_path": plan.episode_path,
        "source_folder_id": str(source_folder_id),
        "video_uuids": [str(item.uuid) for item in plan.videos],
        "metadata_uuids": [str(item.uuid) for item in plan.metadata_items],
    }
    if plan.task_name:
        metadata["task_name"] = plan.task_name
    if plan.collection_datetime:
        metadata["collection_datetime"] = plan.collection_datetime

    return DataGroupCustom(
        name=group_name(plan),
        layoutContents=layout_contents,
        layout=layout,
        clientMetadata=metadata,
    )


def existing_groups_by_episode(folder: Any) -> dict[str, UUID]:
    from encord.orm.storage import StorageItemType

    groups: dict[str, UUID] = {}
    for item in folder.list_items(
        page_size=1000,
        item_types=[StorageItemType.GROUP],
    ):
        episode_path = episode_path_from_item(item, inspect_children=True)
        if not episode_path:
            continue
        if episode_path in groups:
            raise typer.BadParameter(
                f"Output folder contains duplicate data groups for {episode_path}"
            )
        groups[episode_path] = item.uuid
    return groups


def create_group_dataset(client: Any, title: str, group_uuids: list[UUID]) -> str:
    from encord.orm.dataset import StorageLocation

    response = client.create_dataset(
        dataset_title=title,
        dataset_type=StorageLocation.CORD_STORAGE,
        create_backing_folder=False,
    )
    dataset = client.get_dataset(str(response.dataset_hash))
    dataset.link_items(group_uuids)
    return str(response.dataset_hash)


def create_data_groups(
    client: Any,
    source_folder_id: UUID,
    *,
    output_folder_id: UUID | None,
    output_folder_name: str | None,
    dataset_hash: str | None,
    output_dataset_title: str | None,
    limit: int | None,
) -> str:
    if output_folder_id is not None and output_folder_name:
        raise typer.BadParameter(
            "Pass either --output-folder-id or --output-folder-name, not both."
        )
    if dataset_hash and output_dataset_title:
        raise typer.BadParameter(
            "Pass either --dataset-hash or --output-dataset-title, not both."
        )
    if limit is not None and limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero.")

    source_folder = client.get_storage_folder(source_folder_id)
    plans = validate_episode_groups(load_items_by_episode(source_folder))
    output_folder = (
        client.get_storage_folder(output_folder_id)
        if output_folder_id is not None
        else None
    )
    existing = existing_groups_by_episode(output_folder) if output_folder else {}
    selected = plans[:limit] if limit is not None else plans
    pending = [plan for plan in selected if plan.episode_path not in existing]
    group_inputs = [build_custom_group(plan, source_folder_id) for plan in pending]

    typer.echo(f"Validated episodes: {len(plans)}")
    typer.echo(
        "Reusing existing groups: "
        f"{sum(plan.episode_path in existing for plan in selected)}"
    )
    typer.echo(f"Groups selected for the dataset: {len(selected)}")
    typer.echo(f"New groups to create: {len(pending)}")
    for plan in selected[:10]:
        typer.echo(f"  {group_name(plan)}")
    if len(selected) > 10:
        typer.echo(f"  ... and {len(selected) - 10} more")

    if output_folder is None and pending:
        title = output_folder_name or f"{source_folder.name} - data groups"
        output_folder = client.create_storage_folder(
            name=title,
            description="Three-camera episode groups with metadata sidecars.",
            client_metadata={"source_folder_id": str(source_folder_id)},
        )
    if output_folder is None:
        raise RuntimeError("No output folder is available for the selected groups.")
    typer.echo(f"Output folder: {output_folder.uuid} | {output_folder.name}")

    created: list[UUID] = []
    for index, group_input in enumerate(group_inputs, start=1):
        try:
            created.append(output_folder.create_data_group(group_input))
        except Exception as exc:
            raise RuntimeError(
                f"Data-group creation failed after {len(created)} successful groups: {exc}"
            ) from exc
        typer.echo(f"Created {index}/{len(group_inputs)}")

    created_by_episode = {
        plan.episode_path: group_uuid
        for plan, group_uuid in zip(pending, created, strict=True)
    }
    selected_group_uuids = [
        existing.get(plan.episode_path) or created_by_episode[plan.episode_path]
        for plan in selected
    ]

    if dataset_hash:
        dataset = client.get_dataset(dataset_hash)
        dataset.link_items(selected_group_uuids)
        result_dataset_hash = dataset_hash
        typer.echo(f"Linked {len(selected_group_uuids)} groups to dataset {dataset_hash}.")
    else:
        title = output_dataset_title or f"{source_folder.name} - grouped dataset"
        result_dataset_hash = create_group_dataset(
            client,
            title,
            selected_group_uuids,
        )
        typer.echo(
            f"Created dataset {result_dataset_hash} with {len(selected_group_uuids)} groups."
        )

    typer.echo(f"Created data groups: {len(created)}")
    return result_dataset_hash


def main(
    ssh_key_file: Annotated[
        Path,
        typer.Option(help="Path to the Encord SSH private-key file."),
    ],
    source_folder_id: Annotated[
        UUID,
        typer.Option(help="Folder containing raw videos and JSON/JSONL metadata items."),
    ],
    output_folder_id: Annotated[
        UUID | None,
        typer.Option(help="Existing folder in which to create missing data groups."),
    ] = None,
    output_folder_name: Annotated[
        str | None,
        typer.Option(help="Name for a new data-group folder."),
    ] = None,
    dataset_hash: Annotated[
        str | None,
        typer.Option(help="Optional existing dataset to link the selected groups into."),
    ] = None,
    output_dataset_title: Annotated[
        str | None,
        typer.Option(help="Title for the new grouped dataset; defaults from the source folder."),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(help="Optional Encord API domain, for example https://api.us.encord.com."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of validated groups to create."),
    ] = None,
) -> None:
    client = create_client(ssh_key_file.expanduser(), domain)
    create_data_groups(
        client,
        source_folder_id,
        output_folder_id=output_folder_id,
        output_folder_name=output_folder_name,
        dataset_hash=dataset_hash,
        output_dataset_title=output_dataset_title,
        limit=limit,
    )


if __name__ == "__main__":
    typer.run(main)
