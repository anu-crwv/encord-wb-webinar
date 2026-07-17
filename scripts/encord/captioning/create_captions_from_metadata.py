# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "numpy",
#     "typer",
# ]
# ///
"""Create Language Instruction 1/2/3 captions from Encord task metadata."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from encord import EncordUserClient
from encord.constants.enums import DataType
from encord.objects import Classification
from encord.objects.frames import Range

from helper.captioning_v1 import CLASSIFICATION_TITLES, TASK_CAPTIONS, caption_variants_for_task


DEFAULT_ARM_PHRASE = "the robot arm"
BUNDLE_SIZE = 100


def log_progress(action: str, index: int, total: int) -> None:
    if index == total or index % 50 == 0:
        typer.echo(f"{action} {index}/{total}...")


def row_chunks(rows: list[Any]) -> list[list[Any]]:
    return [rows[index : index + BUNDLE_SIZE] for index in range(0, len(rows), BUNDLE_SIZE)]


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def client_from_env() -> EncordUserClient:
    key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def data_type_name(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def is_video(label_row: Any) -> bool:
    data_type = getattr(label_row, "data_type", None)
    normalized = data_type_name(data_type).lower()
    return data_type == DataType.VIDEO or normalized in {"video", "video/mp4"}


def is_group(label_row: Any) -> bool:
    return data_type_name(getattr(label_row, "data_type", None)).lower() == "group"


def is_supported_label_row(label_row: Any) -> bool:
    return is_video(label_row) or is_group(label_row)


def full_range(label_row: Any) -> Range:
    frames = int(getattr(label_row, "number_of_frames", 0) or 0)
    return Range(start=0, end=max(frames - 1, 0))


def language_instruction_instances(label_row: Any) -> list[Any]:
    instances = []
    for instance in label_row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        if getattr(ontology_item, "title", None) in CLASSIFICATION_TITLES:
            instances.append(instance)
    return instances


def has_language_instruction(label_row: Any) -> bool:
    return bool(language_instruction_instances(label_row))


def item_metadata(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "client_metadata", None) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def task_name_from_episode_path(episode_path: Any) -> str | None:
    parts = [part for part in str(episode_path or "").strip("/").split("/") if part]
    if "raw-feed" not in parts:
        return None
    index = parts.index("raw-feed")
    if index + 2 >= len(parts):
        return None
    family = parts[index + 1]
    if family not in {"trossen-data", "trossen-data-stationary"}:
        return None
    return parts[index + 2]


def task_name_from_title(title: Any) -> str | None:
    task_name = str(title or "").split(" | ", 1)[0].strip()
    return task_name or None


def task_name_from_child_metadata(item: Any) -> str | None:
    try:
        children = list(item.get_child_items())
    except Exception as exc:
        typer.echo(f"Warning: could not inspect child metadata for {getattr(item, 'name', item)}: {exc}", err=True)
        return None

    for child in children:
        task_name = item_metadata(child).get("task_name")
        if task_name:
            return str(task_name)
    return None


def resolve_task_name(row: Any, item: Any) -> tuple[str | None, str]:
    metadata = item_metadata(item)
    if metadata.get("task_name"):
        return str(metadata["task_name"]), "client_metadata.task_name"

    task_name = task_name_from_episode_path(metadata.get("episode_path"))
    if task_name:
        return task_name, "client_metadata.episode_path"

    task_name = task_name_from_title(getattr(row, "title", None))
    if task_name:
        return task_name, "data_row.title"

    task_name = task_name_from_child_metadata(item)
    if task_name:
        return task_name, "child_client_metadata.task_name"

    return None, "missing"


def task_by_data_hash(project: Any, client: EncordUserClient) -> dict[str, str]:
    typer.echo("Loading attached dataset metadata...")
    datasets = list(project.list_datasets())
    if len(datasets) != 1:
        raise typer.BadParameter(f"Expected exactly one attached dataset, found {len(datasets)}.")

    dataset = client.get_dataset(str(datasets[0].dataset_hash))
    data_rows = list(dataset.data_rows)
    typer.echo(f"Found {len(data_rows)} dataset rows.")
    dataset_type_counts = Counter(data_type_name(getattr(row, "data_type", None)) for row in data_rows)
    typer.echo(f"Dataset row data types: {format_counter(dataset_type_counts)}.")

    backing_ids = [row.backing_item_uuid for row in data_rows if getattr(row, "backing_item_uuid", None)]
    storage_items = (
        {str(item.uuid): item for item in client.get_storage_items(backing_ids)}
        if backing_ids
        else {}
    )
    storage_type_counts = Counter(data_type_name(getattr(item, "item_type", None)) for item in storage_items.values())
    typer.echo(
        "Backing storage item types: "
        f"{format_counter(storage_type_counts)}."
    )

    tasks: dict[str, str] = {}
    task_sources: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    missing_storage_items = 0
    for row in data_rows:
        item = storage_items.get(str(row.backing_item_uuid))
        if item is None:
            missing_storage_items += 1
            continue
        task_name, source = resolve_task_name(row, item)
        task_sources[source] += 1
        if task_name:
            tasks[str(row.uid)] = task_name
            task_counts[task_name] += 1

    typer.echo(f"Resolved task metadata for {len(tasks)} rows.")
    typer.echo(f"Task metadata sources: {format_counter(task_sources)}.")
    typer.echo(f"Resolved task counts: {format_counter(task_counts)}.")
    if missing_storage_items:
        typer.echo(f"Warning: {missing_storage_items} dataset rows were missing backing storage items.", err=True)
    return tasks


def validate_caption_classifications(project: Any) -> None:
    typer.echo("Validating caption ontology classifications...")
    missing = []
    for title in CLASSIFICATION_TITLES:
        try:
            project.ontology_structure.get_child_by_title(title=title, type_=Classification)
        except Exception:
            missing.append(title)
    if missing:
        raise typer.BadParameter(
            "Project ontology is missing required caption classifications: "
            f"{', '.join(missing)}"
        )


def captions_for_task(task_name: str) -> tuple[str, str, str]:
    return caption_variants_for_task(task_name, DEFAULT_ARM_PHRASE)


def add_language_instructions(row: Any, captions: tuple[str, str, str], overwrite: bool) -> None:
    row_range = full_range(row) if is_video(row) else None
    for title, caption in zip(CLASSIFICATION_TITLES, captions, strict=True):
        classification = row.ontology_structure.get_child_by_title(
            title=title,
            type_=Classification,
        )
        if classification is None:
            raise RuntimeError(f"Classification not found in ontology: {title}")

        instance = classification.create_instance()
        instance.set_answer(answer=caption)
        if row_range is not None:
            instance.set_for_frames(frames=row_range, overwrite=overwrite)
        row.add_classification_instance(instance, force=overwrite)


def main(
    project_hash: Annotated[str, typer.Argument(help="Encord project hash.")],
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing Language Instruction 1/2/3 captions.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Resolve and report candidate captions without initializing or saving labels."),
    ] = False,
) -> None:
    typer.echo("Connecting to Encord...")
    client = client_from_env()
    project = client.get_project(project_hash)
    validate_caption_classifications(project)
    tasks = task_by_data_hash(project, client)

    typer.echo("Listing label rows...")
    all_rows = list(project.list_label_rows_v2())
    typer.echo(f"Found {len(all_rows)} label rows.")
    typer.echo(
        "Label row data types: "
        f"{format_counter(Counter(data_type_name(getattr(row, 'data_type', None)) for row in all_rows))}."
    )
    rows = [row for row in all_rows if is_supported_label_row(row)]
    ignored_row_type_count = len(all_rows) - len(rows)
    typer.echo(f"Found {len(rows)} video/group label rows.")

    captions_by_hash = {
        str(row.data_hash): captions_for_task(tasks[str(row.data_hash)])
        for row in rows
        if str(row.data_hash) in tasks and tasks[str(row.data_hash)] in TASK_CAPTIONS
    }
    missing_task_count = sum(1 for row in rows if str(row.data_hash) not in tasks)
    unsupported_tasks = Counter(
        tasks[str(row.data_hash)]
        for row in rows
        if str(row.data_hash) in tasks and tasks[str(row.data_hash)] not in TASK_CAPTIONS
    )
    unsupported_task_count = sum(unsupported_tasks.values())
    skipped_before_existing = ignored_row_type_count + missing_task_count + unsupported_task_count
    if unsupported_tasks:
        typer.echo(f"Unsupported task names: {format_counter(unsupported_tasks)}.")
    if missing_task_count:
        typer.echo(f"Rows missing resolved task metadata: {missing_task_count}.")
    if ignored_row_type_count:
        typer.echo(f"Ignored non-video/non-group label rows: {ignored_row_type_count}.")

    rows = [row for row in rows if str(row.data_hash) in captions_by_hash]
    typer.echo(
        f"Caption candidates: {len(rows)}. "
        f"Skipped before existing-label checks: {skipped_before_existing}."
    )
    if not rows:
        typer.echo("No rows matched the task-to-caption mapping.")
        return
    if dry_run:
        typer.echo("Dry run enabled; not initializing, adding, or saving labels.")
        return

    typer.echo(f"Preparing captions for {len(rows)} mapped rows.")
    if overwrite:
        typer.echo("Overwrite enabled: clearing existing labels with empty include hash sets...")
        cleared = 0
        for chunk in row_chunks(rows):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.initialise_labels(
                        include_object_feature_hashes=set(),
                        include_classification_feature_hashes=set(),
                        overwrite=True,
                        bundle=bundle,
                    )
            cleared += len(chunk)
            log_progress("Cleared", cleared, len(rows))
    else:
        typer.echo("Initializing existing labels...")
        initialized = 0
        for chunk in row_chunks(rows):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.initialise_labels(bundle=bundle)
            initialized += len(chunk)
            log_progress("Initialized", initialized, len(rows))

    typer.echo("Preparing caption updates...")
    touched = []
    skipped_existing = 0
    for index, row in enumerate(rows, start=1):
        captions = captions_by_hash[str(row.data_hash)]
        if has_language_instruction(row) and not overwrite:
            skipped_existing += 1
            continue

        add_language_instructions(row, captions, overwrite=overwrite)
        touched.append(row)
        log_progress("Prepared", index, len(rows))

    typer.echo(f"Prepared {len(touched)} rows for update; skipped {skipped_existing} existing caption rows.")
    if touched:
        typer.echo(f"Saving {len(touched)} updated rows...")
        saved = 0
        for chunk in row_chunks(touched):
            with project.create_bundle(bundle_size=len(chunk)) as bundle:
                for row in chunk:
                    row.save(bundle=bundle)
            saved += len(chunk)
            log_progress("Saved", saved, len(touched))

    typer.echo(
        f"Updated {len(touched)} label rows; "
        f"skipped {skipped_before_existing + skipped_existing}."
    )


if __name__ == "__main__":
    typer.run(main)
