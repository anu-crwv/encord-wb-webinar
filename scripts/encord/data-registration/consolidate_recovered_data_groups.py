# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Move recovered Encord data groups into one shared storage folder.

The command is resumable and dry-run by default. It preserves storage item
UUIDs, so replacement dataset rows and their project labels remain linked.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer
from rebuild_projects_from_existing_videos import (
    DEFAULT_PROJECT_HASHES,
    DEFAULT_STATE_JSON,
    STATE_MIGRATION_ID,
    MigrationError,
    attached_dataset_hashes,
    chunks,
    create_client,
    item_metadata,
    now_iso,
    save_state,
    write_json_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_JSON = SCRIPT_DIR / "encord_project_group_consolidation_report.json"
DEFAULT_FOLDER_NAME = "[Recovered video groups] All projects"


def expected_groups_by_project(
    state: dict[str, Any], project_hashes: list[str]
) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    seen: dict[str, str] = {}
    for project_hash in project_hashes:
        project_state = state.get("projects", {}).get(project_hash)
        if project_state is None:
            raise MigrationError(
                f"Project {project_hash} is missing from the state file."
            )
        group_uuids = {
            str(group["group_uuid"])
            for group in project_state.get("groups", {}).values()
        }
        if not group_uuids:
            raise MigrationError(
                f"Project {project_hash} has no recovered groups in state."
            )
        for group_uuid in group_uuids:
            other_project = seen.setdefault(group_uuid, project_hash)
            if other_project != project_hash:
                raise MigrationError(
                    f"Group {group_uuid} is assigned to both {other_project} and {project_hash}."
                )
        expected[project_hash] = group_uuids
    return expected


def list_group_items(folder: Any) -> dict[str, Any]:
    from encord.orm.storage import StorageItemType

    return {
        str(item.uuid): item
        for item in folder.list_items(
            page_size=1000,
            item_types=[StorageItemType.GROUP],
        )
    }


def create_or_load_shared_folder(
    client: Any,
    state: dict[str, Any],
    state_path: Path,
    project_hashes: list[str],
    expected_count: int,
    folder_name: str,
) -> Any:
    folder_hash = state.get("consolidated_folder_hash")
    if folder_hash:
        return client.get_storage_folder(folder_hash)

    folder = client.create_storage_folder(
        name=folder_name[:200],
        description=(
            "Recovered three-camera, video-only data groups shared by the "
            f"{len(project_hashes)} restored projects."
        ),
        client_metadata={
            "probe": "recovered-video-only-consolidated-folder",
            "migration_id": STATE_MIGRATION_ID,
            "source_project_hashes": project_hashes,
            "expected_group_count": expected_count,
        },
    )
    state["consolidated_folder_hash"] = str(folder.uuid)
    state["consolidation"] = {
        "status": "moving",
        "folder_hash": str(folder.uuid),
        "folder_name": folder.name,
        "started_at": now_iso(),
        "expected_group_count": expected_count,
        "projects": {},
    }
    save_state(state_path, state)
    typer.echo(f"Created shared folder {folder.uuid}: {folder.name}")
    return folder


def source_folder_hash(project_state: dict[str, Any], shared_folder_hash: str) -> str:
    original = project_state.get("original_folder_hash")
    if original:
        return str(original)
    current = str(project_state.get("folder_hash") or "")
    if not current:
        raise MigrationError("Recovered project state has no folder_hash.")
    if current == shared_folder_hash:
        raise MigrationError(
            "State points at the shared folder but has no original_folder_hash."
        )
    return current


def move_project_groups(
    *,
    client: Any,
    shared_folder: Any,
    shared_group_uuids: set[str],
    project_hash: str,
    project_state: dict[str, Any],
    expected_uuids: set[str],
    state: dict[str, Any],
    state_path: Path,
    batch_size: int,
) -> int:
    shared_hash = str(shared_folder.uuid)
    missing_from_shared = expected_uuids - shared_group_uuids
    if not missing_from_shared:
        typer.echo(
            f"{project_hash}: all {len(expected_uuids):,} groups already shared."
        )
        return 0

    old_folder_hash = source_folder_hash(project_state, shared_hash)
    old_folder = client.get_storage_folder(old_folder_hash)
    source_items = list_group_items(old_folder)
    movable = sorted(missing_from_shared & set(source_items))
    missing = missing_from_shared - set(movable)
    if missing:
        samples = ", ".join(sorted(missing)[:10])
        raise MigrationError(
            f"{project_hash} has {len(missing):,} groups in neither the source nor "
            f"shared folder. Samples: {samples}"
        )

    moved = 0
    typer.echo(
        f"{project_hash}: moving {len(movable):,}/{len(expected_uuids):,} groups "
        f"from {old_folder_hash}..."
    )
    for batch in chunks(movable, batch_size):
        old_folder.move_items_to_folder(
            target_folder=shared_folder,
            items_to_move=[UUID(group_uuid) for group_uuid in batch],
        )
        moved += len(batch)
        shared_group_uuids.update(batch)
        progress = (
            state.setdefault("consolidation", {})
            .setdefault("projects", {})
            .setdefault(project_hash, {})
        )
        progress.update(
            {
                "source_folder_hash": old_folder_hash,
                "expected_group_count": len(expected_uuids),
                "observed_moved_count": len(expected_uuids - missing_from_shared)
                + moved,
                "updated_at": now_iso(),
            }
        )
        save_state(state_path, state)
        typer.echo(f"  moved {moved:,}/{len(movable):,}")
    return moved


def validate_shared_groups(
    shared_items: dict[str, Any],
    expected_by_project: dict[str, set[str]],
) -> dict[str, Any]:
    expected_all = set().union(*expected_by_project.values())
    scoped_items = {
        group_uuid: item
        for group_uuid, item in shared_items.items()
        if group_uuid in expected_all
        or item_metadata(item).get("migration_id") == STATE_MIGRATION_ID
    }
    actual_all = set(scoped_items)
    unexpected = actual_all - expected_all
    missing = expected_all - actual_all
    metadata_failures: list[dict[str, str]] = []
    observed_projects: Counter[str] = Counter()

    for group_uuid, item in scoped_items.items():
        metadata = item_metadata(item)
        project_hash = str(metadata.get("source_project_hash") or "")
        observed_projects[project_hash] += 1
        expected_for_project = expected_by_project.get(project_hash, set())
        reasons = []
        if metadata.get("migration_id") != STATE_MIGRATION_ID:
            reasons.append("wrong migration_id")
        if group_uuid not in expected_for_project:
            reasons.append("wrong source_project_hash")
        if metadata.get("json_uuids") != []:
            reasons.append("json_uuids is not empty")
        video_uuids = metadata.get("video_uuids")
        if not isinstance(video_uuids, list) or len(video_uuids) != 3:
            reasons.append("video_uuids does not contain three videos")
        if reasons:
            metadata_failures.append(
                {"group_uuid": group_uuid, "reason": ", ".join(reasons)}
            )

    result = {
        "expected_group_count": len(expected_all),
        "actual_group_count": len(actual_all),
        "folder_total_group_count": len(shared_items),
        "ignored_unrelated_group_count": len(shared_items) - len(scoped_items),
        "missing_group_count": len(missing),
        "missing_group_uuids": sorted(missing)[:100],
        "unexpected_group_count": len(unexpected),
        "unexpected_group_uuids": sorted(unexpected)[:100],
        "metadata_failure_count": len(metadata_failures),
        "metadata_failures": metadata_failures[:100],
        "expected_project_counts": {
            key: len(value) for key, value in expected_by_project.items()
        },
        "observed_project_counts": dict(sorted(observed_projects.items())),
    }
    result["passed"] = not missing and not unexpected and not metadata_failures
    return result


def validate_project_links(
    client: Any,
    project_hash: str,
    project_state: dict[str, Any],
    expected_group_uuids: set[str],
) -> dict[str, Any]:
    replacement_hash = str(project_state["replacement_dataset_hash"])
    source_hash = str(project_state["source_dataset_hash"])
    dataset = client.get_dataset(replacement_hash)
    dataset_rows = list(dataset.data_rows)
    actual_group_uuids = {str(row.backing_item_uuid) for row in dataset_rows}
    target_data_hashes = {str(row.uid) for row in dataset_rows}

    project = client.get_project(project_hash)
    attached = attached_dataset_hashes(project)
    target_label_rows = {
        str(row.data_hash): row
        for row in project.list_label_rows_v2()
        if str(row.data_hash) in target_data_hashes
    }
    expected_labeled = sum(
        bool(label.get("source_labeled"))
        for label in project_state.get("labels", {}).values()
    )
    actual_labeled = sum(
        row.label_hash is not None for row in target_label_rows.values()
    )

    result = {
        "project_hash": project_hash,
        "replacement_dataset_hash": replacement_hash,
        "expected_group_count": len(expected_group_uuids),
        "dataset_row_count": len(dataset_rows),
        "dataset_group_uuid_match": actual_group_uuids == expected_group_uuids,
        "target_label_row_count": len(target_label_rows),
        "expected_labeled_row_count": expected_labeled,
        "actual_labeled_row_count": actual_labeled,
        "attached_dataset_hashes": sorted(attached),
        "replacement_dataset_attached": replacement_hash in attached,
        "source_dataset_detached": source_hash not in attached,
    }
    result["passed"] = (
        result["dataset_group_uuid_match"]
        and len(dataset_rows) == len(expected_group_uuids)
        and len(target_label_rows) == len(target_data_hashes)
        and actual_labeled == expected_labeled
        and result["replacement_dataset_attached"]
        and result["source_dataset_detached"]
    )
    return result


def validate_old_folders_are_empty(
    client: Any,
    state: dict[str, Any],
    project_hashes: list[str],
    shared_folder_hash: str,
) -> dict[str, Any]:
    folders: dict[str, dict[str, Any]] = {}
    for project_hash in project_hashes:
        project_state = state["projects"][project_hash]
        old_folder_hash = source_folder_hash(project_state, shared_folder_hash)
        old_items = list_group_items(client.get_storage_folder(old_folder_hash))
        folders[project_hash] = {
            "folder_hash": old_folder_hash,
            "group_count": len(old_items),
            "group_uuids": sorted(old_items)[:100],
            "passed": not old_items,
        }
    return {
        "folders": folders,
        "passed": all(result["passed"] for result in folders.values()),
    }


def main(
    state_json: Annotated[
        Path,
        typer.Option(
            help="Recovery migration state containing the existing group UUIDs."
        ),
    ] = DEFAULT_STATE_JSON,
    report_json: Annotated[
        Path,
        typer.Option(help="Consolidation verification report."),
    ] = DEFAULT_REPORT_JSON,
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
    folder_name: Annotated[
        str,
        typer.Option(help="Name for the shared storage folder."),
    ] = DEFAULT_FOLDER_NAME,
    batch_size: Annotated[
        int,
        typer.Option(help="Groups moved per API request."),
    ] = 250,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Create the shared folder and move groups."),
    ] = False,
) -> None:
    if batch_size < 1 or batch_size > 1000:
        raise typer.BadParameter("--batch-size must be between 1 and 1000.")
    if ssh_key_file is None:
        raise typer.BadParameter("Pass --ssh-key-file or set ENCORD_SSH_KEY_FILE.")

    state_path = state_json.expanduser().resolve()
    report_path = report_json.expanduser().resolve()
    if not state_path.is_file():
        raise typer.BadParameter(f"State file does not exist: {state_path}")
    state = json.loads(state_path.read_text())
    if state.get("migration_id") != STATE_MIGRATION_ID:
        raise typer.BadParameter(
            f"State belongs to another migration: {state.get('migration_id')}"
        )

    project_hashes = [
        project_hash
        for project_hash in DEFAULT_PROJECT_HASHES
        if project_hash in state.get("projects", {})
    ]
    expected_by_project = expected_groups_by_project(state, project_hashes)
    expected_total = sum(len(value) for value in expected_by_project.values())
    client = create_client(ssh_key_file, encord_domain)

    if not apply:
        report = {
            "migration_id": STATE_MIGRATION_ID,
            "generated_at": now_iso(),
            "apply": False,
            "expected_group_count": expected_total,
            "expected_project_counts": {
                key: len(value) for key, value in expected_by_project.items()
            },
            "existing_shared_folder_hash": state.get("consolidated_folder_hash"),
        }
        write_json_atomic(report_path, report)
        typer.echo(
            f"Dry run: {expected_total:,} groups from {len(project_hashes)} projects "
            f"will move into one folder. Report: {report_path}"
        )
        return

    shared_folder = create_or_load_shared_folder(
        client,
        state,
        state_path,
        project_hashes,
        expected_total,
        folder_name,
    )
    shared_items = list_group_items(shared_folder)
    shared_group_uuids = set(shared_items)
    moved_total = 0
    for project_hash in project_hashes:
        moved_total += move_project_groups(
            client=client,
            shared_folder=shared_folder,
            shared_group_uuids=shared_group_uuids,
            project_hash=project_hash,
            project_state=state["projects"][project_hash],
            expected_uuids=expected_by_project[project_hash],
            state=state,
            state_path=state_path,
            batch_size=batch_size,
        )

    shared_items = list_group_items(shared_folder)
    shared_validation = validate_shared_groups(shared_items, expected_by_project)
    project_validations = {
        project_hash: validate_project_links(
            client,
            project_hash,
            state["projects"][project_hash],
            expected_by_project[project_hash],
        )
        for project_hash in project_hashes
    }
    old_folder_validation = validate_old_folders_are_empty(
        client,
        state,
        project_hashes,
        str(shared_folder.uuid),
    )
    passed = (
        shared_validation["passed"]
        and old_folder_validation["passed"]
        and all(validation["passed"] for validation in project_validations.values())
    )
    report = {
        "migration_id": STATE_MIGRATION_ID,
        "generated_at": now_iso(),
        "apply": True,
        "shared_folder_hash": str(shared_folder.uuid),
        "shared_folder_name": shared_folder.name,
        "moved_this_run": moved_total,
        "shared_validation": shared_validation,
        "old_folder_validation": old_folder_validation,
        "projects": project_validations,
        "passed": passed,
    }
    write_json_atomic(report_path, report)
    if not passed:
        raise MigrationError(
            f"Post-move validation failed. State folder pointers were not changed; see {report_path}."
        )

    completed_at = now_iso()
    for project_hash in project_hashes:
        project_state = state["projects"][project_hash]
        old_folder_hash = source_folder_hash(
            project_state,
            str(shared_folder.uuid),
        )
        project_state.setdefault("original_folder_hash", old_folder_hash)
        project_state["folder_hash"] = str(shared_folder.uuid)
        project_state["consolidated_at"] = completed_at
        project_state["validation"] = {
            **project_state.get("validation", {}),
            "folder_group_count": len(expected_by_project[project_hash]),
            "folder_total_group_count": len(shared_items),
            "foreign_group_count": 0,
            "foreign_group_uuids": [],
            "validated_at": completed_at,
            "passed": True,
        }
    state["consolidation"] = {
        "status": "complete",
        "folder_hash": str(shared_folder.uuid),
        "folder_name": shared_folder.name,
        "expected_group_count": expected_total,
        "actual_group_count": len(shared_items),
        "completed_at": completed_at,
        "project_counts": {
            key: len(value) for key, value in expected_by_project.items()
        },
        "report_json": str(report_path),
    }
    save_state(state_path, state)
    typer.echo(
        f"Consolidated {expected_total:,} groups in {shared_folder.uuid}; "
        "dataset and label links passed verification."
    )
    typer.echo(f"Report: {report_path}")
    typer.echo(f"State: {state_path}")


if __name__ == "__main__":
    typer.run(main)
