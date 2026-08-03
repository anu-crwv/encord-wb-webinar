# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Rebuild Encord projects with video-only groups while preserving metadata and labels.

The command is dry-run by default. A full migration is intentionally explicit:

    uv run --script scripts/encord/data-registration/rebuild_projects_from_existing_videos.py \
      --apply --detach-after-validation

The source datasets stay attached until every selected project has passed the
group, dataset, metadata, and label validation gates.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

import typer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_JSON = SCRIPT_DIR / "encord_project_video_recovery_state.json"
DEFAULT_REPORT_JSON = SCRIPT_DIR / "encord_project_video_recovery_report.json"
DEFAULT_MASTER_FOLDER_HASH = "019fa354-8eb5-7ea1-badd-87c7f31db011"
DEFAULT_PROJECT_HASHES = [
    "317f9b62-a069-431b-934f-cd454b12f637",
    "23f30792-4d41-4d02-8c39-128dbfe3262f",
    "08411c84-7e66-4ad9-a63d-d948f9e821a1",
    "f288281b-14f3-4aa7-b27e-be2473219900",
    "cc0e3584-b47f-403d-944f-eec760a0f632",
    "f09ef7ae-2590-4ece-919a-bff7aa58789e",
]
SOURCE_DATASET_BY_PROJECT = {
    "317f9b62-a069-431b-934f-cd454b12f637": "c4d02044-55d8-47dc-b2d3-fa30d0bbfc38",
    "23f30792-4d41-4d02-8c39-128dbfe3262f": "c4d02044-55d8-47dc-b2d3-fa30d0bbfc38",
    "08411c84-7e66-4ad9-a63d-d948f9e821a1": "eb0d82b3-3312-4b75-86a4-dfbb05c5fa5b",
    "f288281b-14f3-4aa7-b27e-be2473219900": "0005fbe1-8401-4f04-a948-9937d4b94d89",
    "cc0e3584-b47f-403d-944f-eec760a0f632": "0005fbe1-8401-4f04-a948-9937d4b94d89",
    "f09ef7ae-2590-4ece-919a-bff7aa58789e": "c0426355-68fb-485a-875c-df34a6f50e7e",
}
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EPISODE_RE = re.compile(r"^episode_(\d+)(?:_[A-Za-z0-9-]+)?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SCHEMA_VERSION = 1
STATE_MIGRATION_ID = "six-project-video-only-recovery-v1"
VIDEO_ROOT_LABEL_TRANSFORM = "video-majority-to-group-root-v1"
METADATA_COMPLETENESS_KEYS = (
    "has_info_json",
    "has_tasks_jsonl",
    "has_episodes_jsonl",
    "has_episodes_stats_jsonl",
    "has_parquet",
)


class MigrationError(RuntimeError):
    """A validation failure that must stop the migration."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path | UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    if hasattr(value, "value"):
        return json_safe(value.value)
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)


def load_state(
    path: Path, master_folder_hash: str, project_hashes: list[str]
) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text())
        if state.get("schema_version") != SCHEMA_VERSION:
            raise MigrationError(
                f"Unsupported state schema {state.get('schema_version')}; expected {SCHEMA_VERSION}."
            )
        if state.get("migration_id") != STATE_MIGRATION_ID:
            raise MigrationError(f"State belongs to another migration: {path}")
        if state.get("master_folder_hash") != master_folder_hash:
            raise MigrationError(
                "State master folder does not match --master-folder-hash. "
                "Use a different --state-json for a different source folder."
            )
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "migration_id": STATE_MIGRATION_ID,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "master_folder_hash": master_folder_hash,
            "master_videos": {},
            "projects": {},
        }

    for project_hash in project_hashes:
        source_dataset_hash = SOURCE_DATASET_BY_PROJECT.get(project_hash)
        if source_dataset_hash is None:
            raise MigrationError(
                f"No audited source dataset is configured for project {project_hash}. "
                "Add it to SOURCE_DATASET_BY_PROJECT before applying changes."
            )
        project_state = state["projects"].setdefault(
            project_hash,
            {
                "source_dataset_hash": source_dataset_hash,
                "groups": {},
                "labels": {},
                "old_dataset_detached": False,
            },
        )
        if project_state.get("source_dataset_hash") != source_dataset_hash:
            raise MigrationError(
                f"Source dataset changed in state for project {project_hash}."
            )
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json_atomic(path, state)


def chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def create_client(ssh_key_file: Path, domain: str) -> Any:
    from encord.user_client import EncordUserClient

    key_path = ssh_key_file.expanduser().resolve()
    if not key_path.is_file():
        raise MigrationError(f"Encord SSH key file does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(
        ssh_private_key_path=key_path,
        domain=domain.rstrip("/"),
    )


def normalize_source_path(value: Any) -> str:
    path = unquote(str(value or "")).strip()
    if path.startswith(("s3://", "r2://", "http://", "https://")):
        path = urlparse(path).path
    parts = [part for part in path.strip("/").split("/") if part]
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ["raw-feed", "trossen-data"]:
            return "/".join(parts[index:])
    return "/".join(parts)


def episode_path_from_value(value: Any) -> str | None:
    parts = [part for part in normalize_source_path(value).split("/") if part]
    for index, part in enumerate(parts):
        if EPISODE_RE.fullmatch(part):
            if index < 2 or parts[:2] != ["raw-feed", "trossen-data"]:
                return None
            return "/".join(parts[: index + 1]) + "/"
    return None


def item_metadata(item: Any) -> dict[str, Any]:
    return copy.deepcopy(getattr(item, "client_metadata", None) or {})


def episode_path_from_item(item: Any) -> str | None:
    metadata = item_metadata(item)
    episode_path = episode_path_from_value(metadata.get("episode_path"))
    if episode_path:
        return episode_path
    for key in ("source_key", "source_uri", "objectUrl", "object_url"):
        episode_path = episode_path_from_value(metadata.get(key))
        if episode_path:
            return episode_path
    return episode_path_from_value(getattr(item, "name", ""))


def camera_from_values(metadata: dict[str, Any], name: str) -> str | None:
    declared = str(metadata.get("camera_name") or "")
    if declared in CAMERAS:
        return declared

    sensor_key = str(metadata.get("sensor_key") or "")
    for camera in CAMERAS:
        if sensor_key.endswith(camera):
            return camera

    path = normalize_source_path(
        metadata.get("source_key") or metadata.get("source_uri") or name
    )
    for part in PurePosixPath(path).parts:
        for camera in CAMERAS:
            if part == camera or part.endswith(f".{camera}"):
                return camera
    return None


def item_snapshot(item: Any) -> dict[str, Any]:
    return {
        "uuid": str(item.uuid),
        "name": str(getattr(item, "name", "")),
        "item_type": str(
            getattr(
                getattr(item, "item_type", None),
                "value",
                getattr(item, "item_type", ""),
            )
        ),
        "client_metadata": item_metadata(item),
    }


def snapshot_score(
    snapshot: dict[str, Any], episode_path: str
) -> tuple[int, int, int, str]:
    metadata = snapshot.get("client_metadata") or {}
    return (
        int(normalize_source_path(snapshot.get("name")).startswith(episode_path)),
        int(bool(metadata.get("source_uri"))),
        len(metadata),
        snapshot["uuid"],
    )


def get_storage_items_batched(
    client: Any, item_ids: Iterable[str], batch_size: int = 500
) -> dict[str, Any]:
    unique = list(dict.fromkeys(str(item_id) for item_id in item_ids if item_id))
    found: dict[str, Any] = {}
    for batch in chunks(unique, batch_size):
        for item in client.get_storage_items(batch):
            found[str(item.uuid)] = item
    return found


def master_index_key(episode_path: str, camera: str) -> str:
    return f"{episode_path}|{camera}"


def load_master_index(
    client: Any,
    folder_hash: str,
    state: dict[str, Any],
    state_path: Path,
    refresh: bool,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    from encord.orm.storage import StorageItemType

    cached = state.get("master_videos") or {}
    if cached and not refresh:
        typer.echo(f"Reusing {len(cached):,} cached master-video index entries.")
        index = {}
        for value in cached.values():
            index[(value["episode_path"], value["camera"])] = value["item"]
        return index, dict(state.get("master_index_summary") or {})

    folder = client.get_storage_folder(folder_hash)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    listed = 0
    typer.echo(f"Indexing videos in master folder {folder_hash}...")
    for item in folder.list_items(page_size=1000, item_types=[StorageItemType.VIDEO]):
        listed += 1
        episode_path = episode_path_from_item(item)
        camera = camera_from_values(item_metadata(item), str(getattr(item, "name", "")))
        if not episode_path or not camera:
            if len(skipped) < 100:
                skipped.append(
                    {
                        "uuid": str(item.uuid),
                        "name": str(getattr(item, "name", "")),
                        "episode_path": episode_path,
                        "camera": camera,
                    }
                )
            continue
        snapshot = item_snapshot(item)
        key = (episode_path, camera)
        existing = index.get(key)
        if existing is None:
            index[key] = snapshot
            continue
        selected = max(
            (existing, snapshot), key=lambda value: snapshot_score(value, episode_path)
        )
        rejected = snapshot if selected is existing else existing
        index[key] = selected
        if len(duplicates) < 100:
            duplicates.append(
                {
                    "episode_path": episode_path,
                    "camera": camera,
                    "selected_uuid": selected["uuid"],
                    "rejected_uuid": rejected["uuid"],
                }
            )

    state["master_videos"] = {
        master_index_key(episode_path, camera): {
            "episode_path": episode_path,
            "camera": camera,
            "item": snapshot,
        }
        for (episode_path, camera), snapshot in sorted(index.items())
    }
    summary = {
        "folder_hash": folder_hash,
        "listed_video_count": listed,
        "indexed_video_count": len(index),
        "indexed_episode_count": len({episode for episode, _camera in index}),
        "duplicate_slot_count": len(duplicates),
        "duplicate_samples": duplicates,
        "skipped_count": listed - len(index) - len(duplicates),
        "skipped_samples": skipped,
        "indexed_at": now_iso(),
    }
    state["master_index_summary"] = summary
    save_state(state_path, state)
    typer.echo(
        f"Master index: {summary['indexed_video_count']:,} videos across "
        f"{summary['indexed_episode_count']:,} episodes."
    )
    return index, summary


def group_child_ids(item: Any) -> list[str]:
    metadata = item_metadata(item)
    child_ids = [
        str(value)
        for key in ("video_uuids", "json_uuids")
        for value in (metadata.get(key) or [])
        if value
    ]
    if child_ids:
        return list(dict.fromkeys(child_ids))

    try:
        summary = item.get_summary()
        data_group = getattr(summary, "data_group", None)
        if data_group is None:
            return []
        return list(
            dict.fromkeys(
                str(child.uuid) for child in data_group.layout_contents.values()
            )
        )
    except Exception as exc:
        raise MigrationError(
            f"Could not resolve child items for source group {item.uuid}: {exc}"
        ) from exc


def load_source_dataset(client: Any, dataset_hash: str) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    dataset = client.get_dataset(dataset_hash)
    rows = list(dataset.data_rows)
    backing = get_storage_items_batched(
        client,
        [str(row.backing_item_uuid) for row in rows],
    )
    missing_backing = [
        str(row.backing_item_uuid)
        for row in rows
        if str(row.backing_item_uuid) not in backing
    ]
    if missing_backing:
        raise MigrationError(
            f"Dataset {dataset_hash} has {len(missing_backing)} inaccessible backing items."
        )

    all_child_ids: list[str] = []
    group_children: dict[str, list[str]] = {}
    for item in backing.values():
        if item.item_type != StorageItemType.GROUP:
            continue
        ids = group_child_ids(item)
        group_children[str(item.uuid)] = ids
        all_child_ids.extend(ids)
    children = get_storage_items_batched(client, all_child_ids)

    records: list[dict[str, Any]] = []
    source_video_snapshots: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    collisions: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        item = backing[str(row.backing_item_uuid)]
        child_items = [
            children[child_id]
            for child_id in group_children.get(str(item.uuid), [])
            if child_id in children
        ]
        episode_path = episode_path_from_item(item)
        if not episode_path:
            episode_path = next(
                (
                    episode_path_from_item(child)
                    for child in child_items
                    if episode_path_from_item(child)
                ),
                None,
            )
        if not episode_path:
            raise MigrationError(
                f"Could not derive an episode path for dataset row {row.uid} ({row.title})."
            )

        source_kind = "group" if item.item_type == StorageItemType.GROUP else "video"
        records.append(
            {
                "episode_path": episode_path,
                "source_data_hash": str(row.uid),
                "source_data_title": str(row.title),
                "source_item_uuid": str(item.uuid),
                "source_item_name": str(getattr(item, "name", row.title)),
                "source_item_metadata": item_metadata(item),
                "source_kind": source_kind,
            }
        )

        candidate_items = child_items if source_kind == "group" else [item]
        for child in candidate_items:
            camera = camera_from_values(
                item_metadata(child), str(getattr(child, "name", ""))
            )
            child_episode = episode_path_from_item(child) or episode_path
            if camera and child_episode == episode_path:
                source_video_snapshots[(episode_path, camera)].append(
                    item_snapshot(child)
                )

    by_episode: dict[str, dict[str, Any]] = {}
    for record in records:
        episode_path = record["episode_path"]
        if episode_path in by_episode:
            collisions[episode_path].extend(
                [
                    by_episode[episode_path]["source_data_hash"],
                    record["source_data_hash"],
                ]
            )
        else:
            by_episode[episode_path] = record
    if collisions:
        sample = {
            key: sorted(set(value)) for key, value in list(collisions.items())[:10]
        }
        raise MigrationError(
            f"Dataset {dataset_hash} has duplicate episode rows: {sample}"
        )

    return {
        "dataset_hash": dataset_hash,
        "dataset_title": str(dataset.title),
        "dataset_description": str(getattr(dataset, "description", "") or ""),
        "records": sorted(records, key=lambda record: record["episode_path"]),
        "records_by_episode": by_episode,
        "source_video_snapshots": source_video_snapshots,
        "source_kind_counts": dict(
            Counter(record["source_kind"] for record in records)
        ),
    }


def attached_dataset_hashes(project: Any) -> set[str]:
    return {str(dataset.dataset_hash) for dataset in project.list_datasets()}


def audit_projects(
    client: Any,
    project_hashes: list[str],
    master_index: dict[tuple[str, str], dict[str, Any]],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    dataset_cache: dict[str, dict[str, Any]] = {}
    project_audits: dict[str, Any] = {}
    candidates_by_target_uuid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for project_hash in project_hashes:
        project = client.get_project(project_hash)
        project_state = state["projects"][project_hash]
        source_dataset_hash = project_state["source_dataset_hash"]
        source = dataset_cache.get(source_dataset_hash)
        if source is None:
            source = load_source_dataset(client, source_dataset_hash)
            dataset_cache[source_dataset_hash] = source
        attached = attached_dataset_hashes(project)
        if source_dataset_hash not in attached and not project_state.get(
            "old_dataset_detached"
        ):
            raise MigrationError(
                f"Source dataset {source_dataset_hash} is not attached to project {project_hash}."
            )

        missing: list[dict[str, str]] = []
        target_slots: dict[str, dict[str, str]] = {}
        for record in source["records"]:
            episode_path = record["episode_path"]
            camera_map: dict[str, str] = {}
            for camera in CAMERAS:
                master = master_index.get((episode_path, camera))
                if master is None:
                    missing.append({"episode_path": episode_path, "camera": camera})
                    continue
                camera_map[camera] = master["uuid"]
                for candidate in source["source_video_snapshots"].get(
                    (episode_path, camera), []
                ):
                    candidates_by_target_uuid[master["uuid"]].append(candidate)
            target_slots[episode_path] = camera_map

        label_rows = list(project.list_label_rows_v2())
        source_hashes = {record["source_data_hash"] for record in source["records"]}
        source_label_rows = [
            row for row in label_rows if str(row.data_hash) in source_hashes
        ]
        source_labeled = sum(1 for row in source_label_rows if row.label_hash)
        if not project_state.get("old_dataset_detached") and len(
            source_label_rows
        ) != len(source["records"]):
            raise MigrationError(
                f"Project {project_hash} exposes {len(source_label_rows)} source label rows "
                f"for {len(source['records'])} source data rows."
            )

        project_state.update(
            {
                "project_title": project.title,
                "source_dataset_title": source["dataset_title"],
                "source_record_count": len(source["records"]),
            }
        )
        project_audits[project_hash] = {
            "project_hash": project_hash,
            "project_title": project.title,
            "ontology_hash": project.ontology_hash,
            "source_dataset_hash": source_dataset_hash,
            "source_dataset_title": source["dataset_title"],
            "source_record_count": len(source["records"]),
            "source_kind_counts": source["source_kind_counts"],
            "source_label_row_count": len(source_label_rows),
            "source_labeled_row_count": source_labeled,
            "attached_dataset_hashes": sorted(attached),
            "target_slots": target_slots,
            "missing_master_video_count": len(missing),
            "missing_master_videos": missing,
        }
        typer.echo(
            f"Audit {project.title}: rows={len(source['records']):,}, "
            f"labeled={source_labeled:,}, missing master videos={len(missing):,}."
        )

    return project_audits, dataset_cache, candidates_by_target_uuid


def episode_metadata(episode_path: str) -> dict[str, Any]:
    parts = PurePosixPath(episode_path.strip("/")).parts
    episode_id = parts[-1]
    match = EPISODE_RE.fullmatch(episode_id)
    if not match:
        raise MigrationError(f"Invalid episode path: {episode_path}")
    metadata: dict[str, Any] = {
        "source_family": parts[1],
        "episode_id": episode_id,
        "episode_index": int(match.group(1)),
        "episode_path": episode_path,
    }
    if len(parts) >= 7 and parts[:2] == ("raw-feed", "trossen-data"):
        metadata["task_name"] = parts[2]
        metadata["environment"] = parts[3]
        if DATE_RE.fullmatch(parts[5]):
            metadata["collection_datetime"] = parts[5]
    return metadata


def nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def desired_video_metadata(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    episode_path: str,
    camera: str,
) -> dict[str, Any]:
    target_metadata = copy.deepcopy(target.get("client_metadata") or {})
    merged: dict[str, Any] = {}
    for candidate in sorted(
        candidates,
        key=lambda value: len(value.get("client_metadata") or {}),
        reverse=True,
    ):
        for key, value in (candidate.get("client_metadata") or {}).items():
            if nonempty(value):
                merged.setdefault(key, copy.deepcopy(value))
    for key, value in target_metadata.items():
        if nonempty(value):
            merged[key] = copy.deepcopy(value)

    source_key = normalize_source_path(
        target_metadata.get("source_key")
        or target.get("name")
        or next(
            (
                candidate.get("client_metadata", {}).get("source_key")
                for candidate in candidates
                if candidate.get("client_metadata", {}).get("source_key")
            ),
            "",
        )
    )
    if not source_key.startswith(episode_path):
        episode_id = PurePosixPath(episode_path.rstrip("/")).name
        source_key = f"{episode_path}videos/chunk-000/observation.images.{camera}/{episode_id}.mp4"

    merged.update(episode_metadata(episode_path))
    merged.update(
        {
            "source_key": source_key,
            "file_ext": ".mp4",
            "metadata_file_role": "none",
            "camera_name": camera,
            "sensor_key": f"observation.images.{camera}",
            "Tag": merged.get("Tag") or "A",
            "Data Type": merged.get("Data Type") or "video",
            "Extension": merged.get("Extension") or ".mp4",
        }
    )
    for key in METADATA_COMPLETENESS_KEYS:
        merged[key] = bool(merged.get(key, True))

    if not merged.get("source_uri"):
        for candidate in candidates:
            source_uri = candidate.get("client_metadata", {}).get("source_uri")
            if source_uri:
                merged["source_uri"] = source_uri
                break
    return merged


def apply_master_metadata(
    client: Any,
    project_audits: dict[str, Any],
    master_index: dict[tuple[str, str], dict[str, Any]],
    candidates_by_target_uuid: dict[str, list[dict[str, Any]]],
    state: dict[str, Any],
    state_path: Path,
    apply: bool,
    bundle_size: int,
) -> dict[str, Any]:
    from encord.http.bundle import Bundle

    needed: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for audit in project_audits.values():
        for episode_path, camera_map in audit["target_slots"].items():
            for camera, item_uuid in camera_map.items():
                needed[item_uuid] = (
                    episode_path,
                    camera,
                    master_index[(episode_path, camera)],
                )

    live_items = get_storage_items_batched(client, needed)
    changes: list[tuple[Any, dict[str, Any], str, str]] = []
    missing_live = sorted(set(needed) - set(live_items))
    if missing_live:
        raise MigrationError(
            f"{len(missing_live)} selected master videos are no longer accessible."
        )

    for item_uuid, (episode_path, camera, snapshot) in needed.items():
        item = live_items[item_uuid]
        desired = desired_video_metadata(
            item_snapshot(item),
            candidates_by_target_uuid.get(item_uuid, []),
            episode_path,
            camera,
        )
        existing = item_metadata(item)
        if any(existing.get(key) != value for key, value in desired.items()):
            changes.append((item, desired, episode_path, camera))
        snapshot["client_metadata"] = desired

    summary = {
        "selected_video_count": len(needed),
        "metadata_change_count": len(changes),
        "applied": apply,
    }
    if not apply:
        return summary

    typer.echo(f"Updating client metadata on {len(changes):,} master videos...")
    for batch_index, batch in enumerate(chunks(changes, bundle_size), start=1):
        bundle = Bundle(bundle_size=bundle_size)
        for item, desired, _episode_path, _camera in batch:
            item.update(client_metadata=desired, bundle=bundle)
        bundle.execute()
        state["master_metadata_batches_applied"] = (
            state.get("master_metadata_batches_applied", 0) + 1
        )
        save_state(state_path, state)
        typer.echo(
            f"  metadata batch {batch_index}: {min(batch_index * bundle_size, len(changes)):,}/"
            f"{len(changes):,}"
        )
    state["master_metadata_validated_at"] = now_iso()
    save_state(state_path, state)
    return summary


def repair_missing_master_videos(
    client: Any,
    master_folder_hash: str,
    missing_slots: set[tuple[str, str]],
    file_map_path: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    resolved_map = file_map_path.expanduser().resolve()
    if not resolved_map.is_file():
        raise MigrationError(f"Missing-video file map does not exist: {resolved_map}")
    file_map = json.loads(resolved_map.read_text())
    files = file_map.get("files")
    if not isinstance(files, dict):
        raise MigrationError(f"Recovery file map has no files object: {resolved_map}")

    candidates: dict[tuple[str, str], tuple[Path, str, dict[str, Any]]] = {}
    for relative_path, entry in files.items():
        if not isinstance(entry, dict) or entry.get("data_type") != "video":
            continue
        title = str(entry.get("title") or relative_path)
        metadata = copy.deepcopy(entry.get("client_metadata") or {})
        episode_path = episode_path_from_value(metadata.get("episode_path") or title)
        camera = camera_from_values(metadata, title)
        key = (episode_path, camera) if episode_path and camera else None
        if key not in missing_slots:
            continue
        path = resolved_map.parent / relative_path
        if key in candidates:
            raise MigrationError(
                f"Recovery file map contains duplicate candidates for {episode_path} | {camera}."
            )
        candidates[key] = (path, title, metadata)

    absent = sorted(missing_slots - set(candidates))
    if absent:
        raise MigrationError(
            f"Recovery file map is missing {len(absent)} required video entries: {absent}"
        )
    missing_files = [
        str(path)
        for path, _title, _metadata in candidates.values()
        if not path.is_file()
    ]
    if missing_files:
        raise MigrationError(
            f"Recovered video files are missing locally: {missing_files}"
        )

    folder = client.get_storage_folder(master_folder_hash)
    results = []
    typer.echo(
        f"Uploading {len(candidates):,} recovered videos to the master folder..."
    )
    for episode_path, camera in sorted(candidates):
        path, title, metadata = candidates[(episode_path, camera)]
        item_uuid = folder.upload_video(path, title, metadata)
        result = {
            "episode_path": episode_path,
            "camera": camera,
            "path": str(path),
            "title": title,
            "item_uuid": str(item_uuid),
            "uploaded_at": now_iso(),
        }
        results.append(result)
        state.setdefault("master_video_repairs", []).append(result)
        save_state(state_path, state)
        typer.echo(f"  uploaded {camera}: {item_uuid}")
    return {
        "file_map": str(resolved_map),
        "uploaded_count": len(results),
        "uploads": results,
    }


def clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("/", " - ")).strip()


def derived_group_name(record: dict[str, Any]) -> str:
    if record["source_kind"] == "group" and record.get("source_item_name"):
        return clean_name(record["source_item_name"])[:160]
    metadata = episode_metadata(record["episode_path"])
    parts = [
        metadata.get("task_name"),
        str(metadata.get("collection_datetime") or "")[:10],
        metadata["episode_id"],
    ]
    return " | ".join(clean_name(str(part)) for part in parts if part)[:160]


def build_video_group(
    *,
    project_hash: str,
    source_dataset_hash: str,
    master_folder_hash: str,
    record: dict[str, Any],
    camera_map: dict[str, str],
) -> Any:
    from encord.orm.group_layout import DataUnitTile, LayoutGrid
    from encord.orm.storage import DataGroupCustom

    layout_contents = {
        f"camera_{camera}": UUID(camera_map[camera]) for camera in CAMERAS
    }
    right_side = LayoutGrid(
        direction="column",
        split_percentage=50,
        first=DataUnitTile(key="camera_cam_left_wrist"),
        second=DataUnitTile(key="camera_cam_right_wrist"),
    )
    client_metadata = copy.deepcopy(record.get("source_item_metadata") or {})
    old_source_folder = client_metadata.get("source_folder_id")
    if old_source_folder:
        client_metadata["recovery_original_source_folder_id"] = old_source_folder
    client_metadata.update(
        {
            "probe": "recovered-video-only-group",
            "migration_id": STATE_MIGRATION_ID,
            "episode_path": record["episode_path"],
            "source_project_hash": project_hash,
            "source_dataset_hash": source_dataset_hash,
            "source_data_hash": record["source_data_hash"],
            "source_item_uuid": record["source_item_uuid"],
            "source_group_uuid": (
                record["source_item_uuid"] if record["source_kind"] == "group" else None
            ),
            "source_folder_id": master_folder_hash,
            "video_uuids": [camera_map[camera] for camera in CAMERAS],
            "camera_uuid_map": {camera: camera_map[camera] for camera in CAMERAS},
            "json_uuids": [],
        }
    )
    return DataGroupCustom(
        name=derived_group_name(record),
        layout_contents=layout_contents,
        layout=LayoutGrid(
            direction="row",
            split_percentage=50,
            first=DataUnitTile(key="camera_cam_high"),
            second=right_side,
        ),
        client_metadata=client_metadata,
    )


def output_folder_name(project_title: str, project_hash: str) -> str:
    return f"[Recovered video groups] {project_title} ({project_hash[:8]})"[:200]


def ensure_project_folder(
    client: Any,
    project_hash: str,
    project_title: str,
    source_dataset_hash: str,
    project_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
) -> Any:
    folder_hash = project_state.get("folder_hash")
    if folder_hash:
        return client.get_storage_folder(folder_hash)

    folder = client.create_storage_folder(
        name=output_folder_name(project_title, project_hash),
        description=(
            "Recovered three-camera, video-only data groups. "
            f"Source project: {project_hash}; source dataset: {source_dataset_hash}."
        ),
        client_metadata={
            "probe": "recovered-video-only-project-folder",
            "migration_id": STATE_MIGRATION_ID,
            "source_project_hash": project_hash,
            "source_dataset_hash": source_dataset_hash,
        },
    )
    project_state["folder_hash"] = str(folder.uuid)
    save_state(state_path, state)
    typer.echo(f"Created project folder {folder.uuid}: {folder.name}")
    return folder


def adopt_existing_groups(
    folder: Any,
    project_hash: str,
    project_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    from encord.orm.storage import StorageItemType

    discovered: dict[str, dict[str, Any]] = {}
    for item in folder.list_items(page_size=1000, item_types=[StorageItemType.GROUP]):
        metadata = item_metadata(item)
        if metadata.get("migration_id") != STATE_MIGRATION_ID:
            continue
        if metadata.get("source_project_hash") != project_hash:
            continue
        episode_path = episode_path_from_value(metadata.get("episode_path"))
        if not episode_path:
            raise MigrationError(
                f"Migrated group {item.uuid} has no valid episode_path."
            )
        if episode_path in discovered:
            raise MigrationError(
                f"Output folder {folder.uuid} contains duplicate groups for {episode_path}."
            )
        discovered[episode_path] = {
            "group_uuid": str(item.uuid),
            "source_data_hash": str(metadata.get("source_data_hash") or ""),
        }

    if discovered != project_state.get("groups", {}):
        project_state["groups"] = discovered
        save_state(state_path, state)


def create_project_groups(
    client: Any,
    project_hash: str,
    audit: dict[str, Any],
    source: dict[str, Any],
    master_folder_hash: str,
    project_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    batch_size: int,
) -> Any:
    folder = ensure_project_folder(
        client,
        project_hash,
        audit["project_title"],
        audit["source_dataset_hash"],
        project_state,
        state,
        state_path,
    )
    adopt_existing_groups(
        folder,
        project_hash,
        project_state,
        state,
        state_path,
    )
    existing = project_state["groups"]
    pending = [
        record for record in source["records"] if record["episode_path"] not in existing
    ]
    typer.echo(
        f"{audit['project_title']}: {len(existing):,} existing groups, "
        f"{len(pending):,} to create."
    )

    for batch_index, batch in enumerate(chunks(pending, batch_size), start=1):
        params = [
            build_video_group(
                project_hash=project_hash,
                source_dataset_hash=audit["source_dataset_hash"],
                master_folder_hash=master_folder_hash,
                record=record,
                camera_map=audit["target_slots"][record["episode_path"]],
            )
            for record in batch
        ]
        created = folder.create_data_groups(params)
        if len(created) != len(batch):
            raise MigrationError(
                f"Created {len(created)} groups from a batch of {len(batch)} for {project_hash}."
            )
        for record, group_uuid in zip(batch, created, strict=True):
            project_state["groups"][record["episode_path"]] = {
                "group_uuid": str(group_uuid),
                "source_data_hash": record["source_data_hash"],
            }
        save_state(state_path, state)
        typer.echo(
            f"  group batch {batch_index}: {len(project_state['groups']):,}/"
            f"{len(source['records']):,}"
        )
    return folder


def replacement_dataset_title(project_title: str, project_hash: str) -> str:
    return f"[Recovered video groups] {project_title} ({project_hash[:8]})"[:200]


def dataset_hash_from_listing(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("dataset_hash") or value.get("dataset_uuid") or value.get("uuid")
        )
    return str(
        getattr(value, "dataset_hash", None)
        or getattr(value, "dataset_uuid", None)
        or getattr(value, "uuid", None)
    )


def ensure_replacement_dataset(
    client: Any,
    project: Any,
    project_hash: str,
    audit: dict[str, Any],
    source: dict[str, Any],
    project_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    link_batch_size: int,
    dataset_row_timeout_seconds: int,
) -> Any:
    from encord.orm.dataset import StorageLocation

    dataset_hash = project_state.get("replacement_dataset_hash")
    title = replacement_dataset_title(audit["project_title"], project_hash)
    if not dataset_hash:
        matches = client.get_datasets(title_eq=title)
        if len(matches) > 1:
            raise MigrationError(
                f"Multiple datasets have the recovery title {title!r}."
            )
        if matches:
            dataset_hash = dataset_hash_from_listing(matches[0])
        else:
            response = client.create_dataset(
                dataset_title=title,
                dataset_description=(
                    "Video-only recovery dataset. "
                    f"Source dataset: {audit['source_dataset_hash']}; "
                    f"source project: {project_hash}; migration: {STATE_MIGRATION_ID}."
                ),
                dataset_type=StorageLocation.CORD_STORAGE,
                create_backing_folder=False,
            )
            dataset_hash = str(response.dataset_hash)
        project_state["replacement_dataset_hash"] = dataset_hash
        save_state(state_path, state)
        typer.echo(f"Replacement dataset: {dataset_hash}")

    dataset = client.get_dataset(dataset_hash)
    group_uuids = [
        UUID(project_state["groups"][record["episode_path"]]["group_uuid"])
        for record in source["records"]
    ]
    for batch in chunks(group_uuids, link_batch_size):
        dataset.link_items(batch)

    expected_group_ids = {str(value) for value in group_uuids}
    deadline = time.monotonic() + dataset_row_timeout_seconds
    while True:
        dataset = client.get_dataset(dataset_hash)
        rows = list(dataset.data_rows)
        actual_group_ids = {str(row.backing_item_uuid) for row in rows}
        unexpected = actual_group_ids - expected_group_ids
        if unexpected:
            raise MigrationError(
                f"Replacement dataset {dataset_hash} contains unexpected backing groups: "
                f"{sorted(unexpected)[:20]}"
            )
        if actual_group_ids == expected_group_ids:
            break
        if time.monotonic() >= deadline:
            raise MigrationError(
                f"Replacement dataset {dataset_hash} exposed {len(actual_group_ids)} backing "
                f"groups within {dataset_row_timeout_seconds}s; expected {len(expected_group_ids)}."
            )
        typer.echo(
            f"Waiting for replacement dataset rows: {len(actual_group_ids):,}/"
            f"{len(expected_group_ids):,}..."
        )
        time.sleep(5)
    project_state["replacement_dataset_linked_at"] = now_iso()
    save_state(state_path, state)

    attached = attached_dataset_hashes(project)
    if dataset_hash not in attached:
        project.add_datasets([dataset_hash])
        project_state["replacement_dataset_attached_at"] = now_iso()
        save_state(state_path, state)
        typer.echo(f"Attached replacement dataset {dataset_hash} to {project_hash}.")
    return dataset


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def classification_signature(answers: dict[str, Any]) -> list[dict[str, Any]]:
    signature = [
        {
            "featureHash": answer.get("featureHash"),
            "classifications": answer.get("classifications") or [],
        }
        for answer in answers.values()
        if answer.get("classifications")
    ]
    return sorted(
        signature,
        key=lambda value: (
            str(value.get("featureHash") or ""),
            canonical_json(value.get("classifications") or []),
        ),
    )


def covered_frame_count(answer: dict[str, Any]) -> int:
    total = 0
    for frame_range in answer.get("range") or []:
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or not all(isinstance(value, int) for value in frame_range)
        ):
            raise MigrationError(f"Invalid source classification range: {frame_range}")
        start, end = frame_range
        if end < start:
            raise MigrationError(f"Invalid descending source range: {frame_range}")
        total += end - start + 1
    return total


def select_visible_video_answers(
    source_answers: dict[str, Any],
    episode_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_feature: dict[str, list[tuple[str, dict[str, Any], int]]] = defaultdict(list)
    for answer_hash, answer in source_answers.items():
        if not answer.get("classifications"):
            continue
        feature_hash = str(answer.get("featureHash") or "")
        if not feature_hash:
            raise MigrationError(
                f"{episode_path} has a classification answer without featureHash."
            )
        by_feature[feature_hash].append(
            (answer_hash, answer, covered_frame_count(answer))
        )

    selected: dict[str, Any] = {}
    selection_audit: list[dict[str, Any]] = []
    for feature_hash, variants in sorted(by_feature.items()):
        max_coverage = max(coverage for _hash, _answer, coverage in variants)
        winners = [
            (answer_hash, answer, coverage)
            for answer_hash, answer, coverage in variants
            if coverage == max_coverage
        ]
        winner_values = {
            canonical_json(answer.get("classifications") or [])
            for _answer_hash, answer, _coverage in winners
        }
        if len(winner_values) > 1:
            raise MigrationError(
                f"{episode_path} has tied, conflicting classification answers for "
                f"feature {feature_hash}; refusing to choose one."
            )
        answer_hash, answer, coverage = min(winners, key=lambda value: value[0])
        visible_answer = copy.deepcopy(answer)
        visible_answer["classificationHash"] = (
            visible_answer.get("classificationHash") or answer_hash
        )
        visible_answer["spaces"] = {}
        visible_answer["range"] = [[0, 0]]
        selected[answer_hash] = visible_answer
        selection_audit.append(
            {
                "feature_hash": feature_hash,
                "selected_classification_hash": answer_hash,
                "selected_covered_frames": coverage,
                "variant_count": len(variants),
                "variants": [
                    {
                        "classification_hash": variant_hash,
                        "covered_frames": variant_coverage,
                        "range": variant_answer.get("range") or [],
                        "classifications": variant_answer.get("classifications") or [],
                    }
                    for variant_hash, variant_answer, variant_coverage in variants
                ],
            }
        )
    return selected, selection_audit


def contains_nonempty_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value and nonempty(value[key]):
            return True
        return any(contains_nonempty_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(contains_nonempty_key(child, key) for child in value)
    return False


def validate_source_label_payload(
    source_payload: dict[str, Any],
    source_kind: str,
    episode_path: str,
) -> None:
    if source_payload.get("object_answers"):
        raise MigrationError(
            f"{episode_path} contains object answers; refusing lossy migration."
        )
    if source_payload.get("object_actions"):
        raise MigrationError(
            f"{episode_path} contains object actions; refusing lossy migration."
        )
    if contains_nonempty_key(source_payload.get("data_units", {}), "objects"):
        raise MigrationError(
            f"{episode_path} contains frame objects; refusing lossy migration."
        )
    for answer in (source_payload.get("classification_answers") or {}).values():
        if answer.get("spaces"):
            raise MigrationError(
                f"{episode_path} has space-specific classifications; refusing lossy migration."
            )
    if source_kind == "group" and contains_nonempty_key(
        source_payload.get("spaces", {}), "objects"
    ):
        raise MigrationError(
            f"{episode_path} has group-space objects; refusing lossy migration."
        )


def clear_target_labels(target_payload: dict[str, Any]) -> None:
    target_payload["object_answers"] = {}
    target_payload["object_actions"] = {}
    target_payload["classification_answers"] = {}
    for data_unit in target_payload.get("data_units", {}).values():
        data_unit["labels"] = {"objects": [], "classifications": []}
    for space in target_payload.get("spaces", {}).values():
        space["labels"] = {}


def transform_source_labels(
    source_payload: dict[str, Any],
    target_payload: dict[str, Any],
    source_kind: str,
    episode_path: str,
    target_high_space_id: str,
) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    validate_source_label_payload(source_payload, source_kind, episode_path)
    clear_target_labels(target_payload)
    source_answers = copy.deepcopy(source_payload.get("classification_answers") or {})

    if source_kind == "group":
        for answer in source_answers.values():
            answer["spaces"] = {}
            answer["range"] = []
        target_payload["classification_answers"] = source_answers
        return (
            classification_signature(source_answers),
            "group-root-global-v1",
            None,
            {},
        )

    visible_answers, selection_audit = select_visible_video_answers(
        source_answers,
        episode_path,
    )
    target_payload["classification_answers"] = visible_answers
    return (
        classification_signature(visible_answers),
        VIDEO_ROOT_LABEL_TRANSFORM,
        None,
        {
            "source_video_classification_answers": source_answers,
            "source_video_visible_label_selection": selection_audit,
            "source_video_high_space_uuid": target_high_space_id,
        },
    )


def label_entry_is_current(
    record: dict[str, Any], entry: dict[str, Any] | None
) -> bool:
    if entry is None:
        return False
    if record["source_kind"] == "video":
        return entry.get("signature_mode") == VIDEO_ROOT_LABEL_TRANSFORM
    return True


def initialize_rows(project: Any, rows: list[Any], bundle_size: int) -> None:
    if not rows:
        return
    with project.create_bundle(bundle_size=min(bundle_size, len(rows))) as bundle:
        for row in rows:
            row.initialise_labels(bundle=bundle)


def wait_for_target_label_rows(
    project: Any,
    target_hashes: set[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        rows = {
            str(row.data_hash): row
            for row in project.list_label_rows_v2()
            if str(row.data_hash) in target_hashes
        }
        if len(rows) == len(target_hashes):
            return rows
        if time.monotonic() >= deadline:
            raise MigrationError(
                f"Only {len(rows)}/{len(target_hashes)} replacement label rows appeared "
                f"within {timeout_seconds}s."
            )
        typer.echo(
            f"Waiting for replacement workflow rows: {len(rows):,}/{len(target_hashes):,}..."
        )
        time.sleep(10)


def copy_project_labels(
    client: Any,
    project_hash: str,
    audit: dict[str, Any],
    source: dict[str, Any],
    target_dataset: Any,
    project_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    video_batch_size: int,
    group_batch_size: int,
    label_row_timeout_seconds: int,
) -> dict[str, Any]:
    project = client.get_project(project_hash)
    target_rows = list(target_dataset.data_rows)
    target_hash_by_group_uuid = {
        str(row.backing_item_uuid): str(row.uid) for row in target_rows
    }
    target_hash_by_episode = {
        episode_path: target_hash_by_group_uuid[group["group_uuid"]]
        for episode_path, group in project_state["groups"].items()
    }
    video_records = [
        record for record in source["records"] if record["source_kind"] == "video"
    ]
    video_group_items_by_episode: dict[str, Any] = {}
    if video_records:
        video_group_items = get_storage_items_batched(
            client,
            [
                project_state["groups"][record["episode_path"]]["group_uuid"]
                for record in video_records
            ],
        )
        video_group_items_by_episode = {
            record["episode_path"]: video_group_items[
                project_state["groups"][record["episode_path"]]["group_uuid"]
            ]
            for record in video_records
        }
    target_hashes = set(target_hash_by_episode.values())
    project_rows = list(project.list_label_rows_v2())
    rows_by_hash = {str(row.data_hash): row for row in project_rows}
    missing_source = [
        record["source_data_hash"]
        for record in source["records"]
        if record["source_data_hash"] not in rows_by_hash
        and not label_entry_is_current(
            record,
            project_state["labels"].get(record["episode_path"]),
        )
    ]
    if missing_source:
        raise MigrationError(
            f"Project {project_hash} is missing {len(missing_source)} uncopied source label rows."
        )

    target_rows_by_hash = wait_for_target_label_rows(
        project,
        target_hashes,
        label_row_timeout_seconds,
    )
    pending = [
        record
        for record in source["records"]
        if not label_entry_is_current(
            record,
            project_state["labels"].get(record["episode_path"]),
        )
    ]
    current_count = len(source["records"]) - len(pending)
    typer.echo(
        f"{audit['project_title']}: {current_count:,} label rows current, "
        f"{len(pending):,} to copy or confirm empty."
    )

    cursor = 0
    while cursor < len(pending):
        source_kind = pending[cursor]["source_kind"]
        batch_size = video_batch_size if source_kind == "video" else group_batch_size
        batch: list[dict[str, Any]] = []
        while cursor < len(pending) and len(batch) < batch_size:
            if pending[cursor]["source_kind"] != source_kind:
                break
            batch.append(pending[cursor])
            cursor += 1

        labeled_records = [
            record
            for record in batch
            if rows_by_hash[record["source_data_hash"]].label_hash is not None
        ]
        source_rows = [
            rows_by_hash[record["source_data_hash"]] for record in labeled_records
        ]
        selected_target_rows = [
            target_rows_by_hash[target_hash_by_episode[record["episode_path"]]]
            for record in labeled_records
        ]
        initialize_rows(project, source_rows, batch_size)
        initialize_rows(project, selected_target_rows, batch_size)

        payloads_to_save: list[Any] = []
        video_metadata_details: dict[str, dict[str, Any]] = {}
        for record, source_row, target_row in zip(
            labeled_records,
            source_rows,
            selected_target_rows,
            strict=True,
        ):
            source_payload = source_row.to_encord_dict()
            target_payload = target_row.to_encord_dict()
            target_high_space_id = audit["target_slots"][record["episode_path"]][
                "cam_high"
            ]
            signature, signature_mode, target_space_id, transform_details = (
                transform_source_labels(
                    source_payload,
                    target_payload,
                    record["source_kind"],
                    record["episode_path"],
                    target_high_space_id,
                )
            )
            try:
                target_row.from_labels_dict(target_payload)
            except Exception as exc:
                raise MigrationError(
                    f"Could not transform labels for {record['episode_path']}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                transformed_payload = target_row.to_encord_dict()
                transformed_answers = transformed_payload["classification_answers"]
                transformed_signature = classification_signature(transformed_answers)
            except Exception as exc:
                raise MigrationError(
                    f"Could not serialize transformed labels for {record['episode_path']}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if transformed_signature != signature:
                raise MigrationError(
                    f"In-memory label round trip changed classifications for {record['episode_path']}."
                )
            payloads_to_save.append(target_row)
            project_state["labels"][record["episode_path"]] = {
                "source_labeled": True,
                "source_label_hash": str(source_row.label_hash),
                "target_data_hash": target_hash_by_episode[record["episode_path"]],
                "classification_signature": signature,
                "signature_mode": signature_mode,
                "target_space_id": target_space_id,
                "source_video_visible_label_selection": transform_details.get(
                    "source_video_visible_label_selection"
                ),
            }
            if record["source_kind"] == "video":
                video_metadata_details[record["episode_path"]] = transform_details

        labeled_paths = {record["episode_path"] for record in labeled_records}
        for record in batch:
            if record["episode_path"] in labeled_paths:
                continue
            project_state["labels"][record["episode_path"]] = {
                "source_labeled": False,
                "source_label_hash": None,
                "target_data_hash": target_hash_by_episode[record["episode_path"]],
                "classification_signature": [],
                "signature_mode": (
                    VIDEO_ROOT_LABEL_TRANSFORM
                    if record["source_kind"] == "video"
                    else "group-root-global-v1"
                ),
                "target_space_id": None,
                "source_video_visible_label_selection": (
                    [] if record["source_kind"] == "video" else None
                ),
            }
            if record["source_kind"] == "video":
                video_metadata_details[record["episode_path"]] = {
                    "source_video_classification_answers": {},
                    "source_video_visible_label_selection": [],
                    "source_video_high_space_uuid": audit["target_slots"][
                        record["episode_path"]
                    ]["cam_high"],
                }

        try:
            with project.create_bundle(bundle_size=batch_size) as bundle:
                for target_row in payloads_to_save:
                    target_row.save(bundle=bundle, validate_before_saving=True)
                for episode_path, details in video_metadata_details.items():
                    group_item = video_group_items_by_episode[episode_path]
                    metadata = item_metadata(group_item)
                    metadata.update(
                        {
                            "source_label_transform": VIDEO_ROOT_LABEL_TRANSFORM,
                            **details,
                        }
                    )
                    group_item.update(client_metadata=metadata, bundle=bundle)
        except Exception as exc:
            error = str(exc)
            if len(error) > 2000:
                error = error[:2000] + "... [truncated]"
            raise MigrationError(
                f"Could not save label/metadata batch beginning at "
                f"{batch[0]['episode_path']}: {type(exc).__name__}: {error}"
            ) from exc

        save_state(state_path, state)
        current_after = sum(
            label_entry_is_current(
                record,
                project_state["labels"].get(record["episode_path"]),
            )
            for record in source["records"]
        )
        typer.echo(f"  labels current: {current_after:,}/{len(source['records']):,}")

    return validate_project_labels(
        client,
        project_hash,
        target_hash_by_episode,
        project_state,
        group_batch_size,
    )


def validate_project_labels(
    client: Any,
    project_hash: str,
    target_hash_by_episode: dict[str, str],
    project_state: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    project = client.get_project(project_hash)
    target_hashes = set(target_hash_by_episode.values())
    rows = {
        str(row.data_hash): row
        for row in project.list_label_rows_v2()
        if str(row.data_hash) in target_hashes
    }
    failures: list[dict[str, Any]] = []
    labeled_entries = [
        (episode_path, label_state)
        for episode_path, label_state in project_state["labels"].items()
        if label_state["source_labeled"]
    ]
    for batch in chunks(labeled_entries, batch_size):
        batch_rows = [rows[entry["target_data_hash"]] for _episode, entry in batch]
        initialize_rows(project, batch_rows, batch_size)
        for (episode_path, expected), row in zip(batch, batch_rows, strict=True):
            answers = row.to_encord_dict().get("classification_answers") or {}
            actual = classification_signature(answers)
            if actual != expected["classification_signature"]:
                failures.append(
                    {
                        "episode_path": episode_path,
                        "target_data_hash": expected["target_data_hash"],
                        "expected": expected["classification_signature"],
                        "actual": actual,
                    }
                )

    result = {
        "target_label_row_count": len(rows),
        "expected_target_label_row_count": len(target_hashes),
        "source_labeled_count": len(labeled_entries),
        "verified_labeled_count": len(labeled_entries) - len(failures),
        "failure_count": len(failures),
        "failures": failures[:100],
        "passed": len(rows) == len(target_hashes) and not failures,
    }
    if not result["passed"]:
        raise MigrationError(
            f"Label validation failed for project {project_hash}: {result}"
        )
    return result


def validate_project_migration(
    client: Any,
    project_hash: str,
    audit: dict[str, Any],
    source: dict[str, Any],
    project_state: dict[str, Any],
    label_validation: dict[str, Any],
) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    project = client.get_project(project_hash)
    folder = client.get_storage_folder(project_state["folder_hash"])
    all_group_items = list(
        folder.list_items(page_size=1000, item_types=[StorageItemType.GROUP])
    )
    foreign_group_uuids = [
        str(item.uuid)
        for item in all_group_items
        if item_metadata(item).get("migration_id") != STATE_MIGRATION_ID
    ]
    group_items = [
        item
        for item in all_group_items
        if item_metadata(item).get("migration_id") == STATE_MIGRATION_ID
        and item_metadata(item).get("source_project_hash") == project_hash
    ]
    source_record_by_episode = {
        record["episode_path"]: record for record in source["records"]
    }
    groups_by_episode = {}
    group_failures: list[dict[str, Any]] = []
    for group in group_items:
        metadata = item_metadata(group)
        episode_path = episode_path_from_value(metadata.get("episode_path"))
        if not episode_path:
            group_failures.append(
                {"group_uuid": str(group.uuid), "reason": "missing episode_path"}
            )
            continue
        expected_camera_map = audit["target_slots"].get(episode_path)
        actual_camera_map = metadata.get("camera_uuid_map")
        source_record = source_record_by_episode.get(episode_path)
        video_label_metadata_missing = (
            source_record is not None
            and source_record["source_kind"] == "video"
            and (
                metadata.get("source_label_transform") != VIDEO_ROOT_LABEL_TRANSFORM
                or "source_video_classification_answers" not in metadata
                or "source_video_visible_label_selection" not in metadata
            )
        )
        if (
            expected_camera_map is None
            or actual_camera_map != expected_camera_map
            or metadata.get("json_uuids") != []
            or metadata.get("video_uuids")
            != [expected_camera_map[camera] for camera in CAMERAS]
            or video_label_metadata_missing
        ):
            group_failures.append(
                {
                    "group_uuid": str(group.uuid),
                    "episode_path": episode_path,
                    "expected_camera_map": expected_camera_map,
                    "actual_camera_map": actual_camera_map,
                    "json_uuids": metadata.get("json_uuids"),
                    "video_label_metadata_missing": video_label_metadata_missing,
                }
            )
        groups_by_episode[episode_path] = str(group.uuid)

    target_dataset = client.get_dataset(project_state["replacement_dataset_hash"])
    target_rows = list(target_dataset.data_rows)
    expected_group_uuids = {
        value["group_uuid"] for value in project_state["groups"].values()
    }
    actual_group_uuids = {str(row.backing_item_uuid) for row in target_rows}
    attached = attached_dataset_hashes(project)
    source_attached = audit["source_dataset_hash"] in attached
    target_attached = project_state["replacement_dataset_hash"] in attached
    source_attachment_expected = not project_state.get("old_dataset_detached", False)
    source_attachment_valid = source_attached == source_attachment_expected
    expected_episodes = {record["episode_path"] for record in source["records"]}

    result = {
        "project_hash": project_hash,
        "expected_episode_count": len(expected_episodes),
        "folder_group_count": len(group_items),
        "folder_total_group_count": len(all_group_items),
        "foreign_group_count": len(foreign_group_uuids),
        "foreign_group_uuids": foreign_group_uuids[:100],
        "dataset_row_count": len(target_rows),
        "label_validation": label_validation,
        "group_failure_count": len(group_failures),
        "group_failures": group_failures[:100],
        "source_dataset_attached": source_attached,
        "source_dataset_attachment_expected": source_attachment_expected,
        "source_dataset_attachment_valid": source_attachment_valid,
        "replacement_dataset_attached": target_attached,
        "passed": (
            set(groups_by_episode) == expected_episodes
            and len(group_items) == len(expected_episodes)
            and not group_failures
            and actual_group_uuids == expected_group_uuids
            and len(target_rows) == len(expected_episodes)
            and label_validation["passed"]
            and source_attachment_valid
            and target_attached
        ),
        "validated_at": now_iso(),
    }
    return result


def detach_source_datasets(
    client: Any,
    project_hashes: list[str],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    failed = [
        project_hash
        for project_hash in project_hashes
        if not state["projects"][project_hash].get("validation", {}).get("passed")
    ]
    if failed:
        raise MigrationError(
            "Global detachment gate failed; source datasets remain attached. "
            f"Unvalidated projects: {failed}"
        )

    typer.echo("All selected projects passed. Detaching source datasets...")
    for project_hash in project_hashes:
        project_state = state["projects"][project_hash]
        project = client.get_project(project_hash)
        source_dataset_hash = project_state["source_dataset_hash"]
        replacement_hash = project_state["replacement_dataset_hash"]
        attached = attached_dataset_hashes(project)
        if replacement_hash not in attached:
            raise MigrationError(
                f"Replacement dataset {replacement_hash} disappeared from {project_hash}; aborting."
            )
        if source_dataset_hash in attached:
            project.remove_datasets([source_dataset_hash])
        attached_after = attached_dataset_hashes(project)
        if (
            source_dataset_hash in attached_after
            or replacement_hash not in attached_after
        ):
            raise MigrationError(
                f"Post-detachment dataset validation failed for project {project_hash}."
            )
        project_state["old_dataset_detached"] = True
        project_state["old_dataset_detached_at"] = now_iso()
        project_state["final_attached_dataset_hashes"] = sorted(attached_after)
        save_state(state_path, state)
        typer.echo(
            f"  {project_hash}: detached {source_dataset_hash}; retained {replacement_hash}."
        )


def build_report(
    state: dict[str, Any],
    project_audits: dict[str, Any],
    metadata_summary: dict[str, Any],
    apply: bool,
    detach_after_validation: bool,
) -> dict[str, Any]:
    return {
        "migration_id": STATE_MIGRATION_ID,
        "generated_at": now_iso(),
        "apply": apply,
        "detach_after_validation": detach_after_validation,
        "master_folder_hash": state["master_folder_hash"],
        "master_index_summary": state.get("master_index_summary"),
        "master_metadata": metadata_summary,
        "projects": {
            project_hash: {
                "audit": audit,
                "state": state["projects"][project_hash],
            }
            for project_hash, audit in project_audits.items()
        },
    }


def validate_cli(
    project_hashes: list[str],
    apply: bool,
    detach_after_validation: bool,
    group_batch_size: int,
    link_batch_size: int,
    metadata_bundle_size: int,
    video_label_batch_size: int,
    group_label_batch_size: int,
) -> None:
    if not project_hashes:
        raise typer.BadParameter("Select at least one --project-hash.")
    unknown = sorted(set(project_hashes) - set(SOURCE_DATASET_BY_PROJECT))
    if unknown:
        raise typer.BadParameter(
            f"Projects have no audited source dataset mapping: {unknown}"
        )
    if detach_after_validation and not apply:
        raise typer.BadParameter("--detach-after-validation requires --apply.")
    for name, value in (
        ("--group-batch-size", group_batch_size),
        ("--link-batch-size", link_batch_size),
        ("--metadata-bundle-size", metadata_bundle_size),
        ("--video-label-batch-size", video_label_batch_size),
        ("--group-label-batch-size", group_label_batch_size),
    ):
        if value < 1:
            raise typer.BadParameter(f"{name} must be at least 1.")


def main(
    project_hashes: Annotated[
        list[str] | None,
        typer.Option(
            "--project-hash",
            help="Project to migrate. Repeat to select a subset; defaults to the audited six.",
        ),
    ] = None,
    master_folder_hash: Annotated[
        str,
        typer.Option(help="Folder containing the canonical ungrouped Trossen videos."),
    ] = DEFAULT_MASTER_FOLDER_HASH,
    ssh_key_file: Annotated[
        Path | None,
        typer.Option(
            "--ssh-key-file",
            "-k",
            envvar="ENCORD_SSH_KEY_FILE",
            help="Path to the Encord SSH private key.",
        ),
    ] = None,
    encord_domain: Annotated[
        str,
        typer.Option(help="Encord API domain."),
    ] = "https://api.encord.com",
    state_json: Annotated[
        Path,
        typer.Option(
            help="Atomic resume state. Keep this file until migration is complete."
        ),
    ] = DEFAULT_STATE_JSON,
    report_json: Annotated[
        Path,
        typer.Option(help="Dry-run or final migration report."),
    ] = DEFAULT_REPORT_JSON,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply", help="Create/update Encord resources. Default is audit-only."
        ),
    ] = False,
    detach_after_validation: Annotated[
        bool,
        typer.Option(
            "--detach-after-validation",
            help="Detach old datasets only after every selected project passes all gates.",
        ),
    ] = False,
    refresh_master_index: Annotated[
        bool,
        typer.Option(
            "--refresh-master-index",
            help="Relist the master folder instead of reusing the state cache.",
        ),
    ] = False,
    missing_video_file_map: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Recovery file_map.json used to upload only master-video slots missing "
                "from the live audit. Requires --apply."
            )
        ),
    ] = None,
    group_batch_size: Annotated[
        int,
        typer.Option(help="Data groups created per API batch."),
    ] = 50,
    link_batch_size: Annotated[
        int,
        typer.Option(help="Groups linked to a replacement dataset per API call."),
    ] = 500,
    metadata_bundle_size: Annotated[
        int,
        typer.Option(help="Storage item metadata updates per bundle."),
    ] = 500,
    video_label_batch_size: Annotated[
        int,
        typer.Option(
            help="Large single-video source label rows initialized per batch."
        ),
    ] = 5,
    group_label_batch_size: Annotated[
        int,
        typer.Option(
            help="Group source/target label rows initialized or saved per batch."
        ),
    ] = 50,
    label_row_timeout_seconds: Annotated[
        int,
        typer.Option(help="Seconds to wait for replacement workflow label rows."),
    ] = 900,
    dataset_row_timeout_seconds: Annotated[
        int,
        typer.Option(
            help="Seconds to wait for replacement dataset rows after linking."
        ),
    ] = 900,
) -> None:
    selected_projects = list(dict.fromkeys(project_hashes or DEFAULT_PROJECT_HASHES))
    validate_cli(
        selected_projects,
        apply,
        detach_after_validation,
        group_batch_size,
        link_batch_size,
        metadata_bundle_size,
        video_label_batch_size,
        group_label_batch_size,
    )
    if ssh_key_file is None:
        raise typer.BadParameter("Pass --ssh-key-file or set ENCORD_SSH_KEY_FILE.")

    state_path = state_json.expanduser().resolve()
    report_path = report_json.expanduser().resolve()
    state = load_state(state_path, master_folder_hash, selected_projects)
    save_state(state_path, state)
    client = create_client(ssh_key_file, encord_domain)

    try:
        master_index, _master_summary = load_master_index(
            client,
            master_folder_hash,
            state,
            state_path,
            refresh_master_index,
        )
        project_audits, dataset_cache, candidates = audit_projects(
            client,
            selected_projects,
            master_index,
            state,
        )
        save_state(state_path, state)

        missing = [
            {
                "project_hash": project_hash,
                **entry,
            }
            for project_hash, audit in project_audits.items()
            for entry in audit["missing_master_videos"]
        ]
        repair_summary = None
        if missing and missing_video_file_map is not None:
            if not apply:
                raise MigrationError("--missing-video-file-map requires --apply.")
            unique_missing = {
                (entry["episode_path"], entry["camera"]) for entry in missing
            }
            repair_summary = repair_missing_master_videos(
                client,
                master_folder_hash,
                unique_missing,
                missing_video_file_map,
                state,
                state_path,
            )
            master_index, _master_summary = load_master_index(
                client,
                master_folder_hash,
                state,
                state_path,
                refresh=True,
            )
            project_audits, dataset_cache, candidates = audit_projects(
                client,
                selected_projects,
                master_index,
                state,
            )
            missing = [
                {
                    "project_hash": project_hash,
                    **entry,
                }
                for project_hash, audit in project_audits.items()
                for entry in audit["missing_master_videos"]
            ]
        metadata_summary = apply_master_metadata(
            client,
            project_audits,
            master_index,
            candidates,
            state,
            state_path,
            apply=apply and not missing,
            bundle_size=metadata_bundle_size,
        )
        report = build_report(
            state,
            project_audits,
            metadata_summary,
            apply,
            detach_after_validation,
        )
        report["master_video_repair"] = repair_summary
        write_json_atomic(report_path, report)
        if missing:
            unique_missing = {
                (entry["episode_path"], entry["camera"]) for entry in missing
            }
            typer.echo(
                f"Missing {len(unique_missing):,} unique master video slots. "
                "No Encord mutations were made.",
                err=True,
            )
            for episode_path, camera in sorted(unique_missing)[:30]:
                typer.echo(f"  {episode_path} | {camera}", err=True)
            raise typer.Exit(code=2)

        if not apply:
            typer.echo(f"Audit passed. Report: {report_path}")
            typer.echo(
                "No Encord resources were changed. Re-run with --apply after reviewing the report."
            )
            return

        for project_hash in selected_projects:
            audit = project_audits[project_hash]
            source = dataset_cache[audit["source_dataset_hash"]]
            project_state = state["projects"][project_hash]
            create_project_groups(
                client,
                project_hash,
                audit,
                source,
                master_folder_hash,
                project_state,
                state,
                state_path,
                group_batch_size,
            )
            project = client.get_project(project_hash)
            target_dataset = ensure_replacement_dataset(
                client,
                project,
                project_hash,
                audit,
                source,
                project_state,
                state,
                state_path,
                link_batch_size,
                dataset_row_timeout_seconds,
            )
            label_validation = copy_project_labels(
                client,
                project_hash,
                audit,
                source,
                target_dataset,
                project_state,
                state,
                state_path,
                video_label_batch_size,
                group_label_batch_size,
                label_row_timeout_seconds,
            )
            validation = validate_project_migration(
                client,
                project_hash,
                audit,
                source,
                project_state,
                label_validation,
            )
            project_state["validation"] = validation
            save_state(state_path, state)
            if not validation["passed"]:
                raise MigrationError(
                    f"Validation failed for {project_hash}; no source datasets were detached."
                )
            typer.echo(f"Validated {audit['project_title']}.")

        if detach_after_validation:
            detach_source_datasets(
                client,
                selected_projects,
                state,
                state_path,
            )
        else:
            typer.echo(
                "All migrations validated. Source datasets remain attached because "
                "--detach-after-validation was not passed."
            )

        final_report = build_report(
            state,
            project_audits,
            metadata_summary,
            apply,
            detach_after_validation,
        )
        write_json_atomic(report_path, final_report)
        typer.echo(f"Migration report: {report_path}")
        typer.echo(f"Resume state: {state_path}")
    except MigrationError as exc:
        failure_report = {
            "migration_id": STATE_MIGRATION_ID,
            "generated_at": now_iso(),
            "error": str(exc),
            "state": state,
        }
        write_json_atomic(report_path, failure_report)
        typer.echo(f"Migration stopped safely: {exc}", err=True)
        typer.echo("No automatic source-dataset detachment was performed.", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    typer.run(main)
