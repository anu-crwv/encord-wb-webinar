# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "boto3",
#     "encord==0.1.199",
#     "numpy",
#     "pyarrow",
#     "pyyaml",
#     "typer",
# ]
# ///
"""Create the three Language Instruction labels from Encord task metadata."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

import typer
from encord.constants.enums import DataType
from encord.objects import Classification
from encord.objects.frames import Range
from helper.captioning_v1 import (
    CLASSIFICATION_TITLES,
    DEFAULT_TASK_CAPTIONS_PATH,
    SOURCE_PARQUET_COLUMNS,
    TASK_CAPTIONS,
    TaskCaptionTemplate,
    caption_variants_for_task,
    infer_arm_phrase_from_table,
    load_task_captions,
)

BUNDLE_SIZE = 100


@dataclass(frozen=True)
class CaptionPlan:
    row: Any
    arm_phrase: str
    captions: tuple[str, str, str]


@dataclass(frozen=True)
class TaskSource:
    task_name: str
    source_parquet_uri: str | None


def create_client(ssh_key_file: Path, domain: str | None = None) -> Any:
    from encord import EncordUserClient

    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"SSH key file does not exist: {ssh_key_file}")
    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def chunks(values: list[Any], size: int = BUNDLE_SIZE) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def data_type_name(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return text.rsplit(".", 1)[-1].upper()


def is_supported_label_row(label_row: Any) -> bool:
    value = getattr(label_row, "data_type", None)
    return value == DataType.VIDEO or data_type_name(value) in {"VIDEO", "GROUP"}


def item_metadata(item: Any) -> dict[str, Any]:
    value = getattr(item, "client_metadata", None) or {}
    return dict(value) if isinstance(value, dict) else {}


def task_name_from_episode_path(value: Any) -> str | None:
    parts = [part for part in str(value or "").strip("/").split("/") if part]
    if "raw-feed" not in parts:
        return None
    index = parts.index("raw-feed")
    if index + 2 >= len(parts):
        return None
    if parts[index + 1] not in {"trossen-data", "trossen-data-stationary"}:
        return None
    return parts[index + 2]


def task_name_from_title(value: Any) -> str | None:
    task_name = str(value or "").split(" | ", 1)[0].strip()
    return task_name or None


def task_name_from_child_metadata(item: Any) -> str | None:
    for child in item.get_child_items():
        task_name = item_metadata(child).get("task_name")
        if task_name:
            return str(task_name)
    return None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3" and parsed.netloc and parsed.path.lstrip("/"):
        return parsed.netloc, unquote(parsed.path.lstrip("/"))
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        bucket = parsed.netloc.split(".s3.", 1)[0]
        key = unquote(parsed.path.lstrip("/"))
        if bucket and key:
            return bucket, key
    raise typer.BadParameter(f"Expected an S3 object URI, got: {uri}")


def canonical_s3_uri(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"s3://{bucket}/{key}"


def episode_path_from_value(value: Any) -> str | None:
    text = str(value or "")
    if "://" in text:
        text = unquote(urlparse(text).path.lstrip("/"))
    parts = [part for part in text.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if part.startswith("episode_"):
            return "/".join(parts[: index + 1]) + "/"
    return None


def group_children(item: Any, client: Any) -> list[Any]:
    children = list(item.get_child_items())
    by_uuid = {str(child.uuid): child for child in children}
    try:
        summary = item.get_summary()
    except (AttributeError, TypeError):
        return children
    layout = getattr(getattr(summary, "data_group", None), "layout_contents", {}) or {}
    missing = [child.uuid for child in layout.values() if str(child.uuid) not in by_uuid]
    if missing:
        for child in client.get_storage_items(missing):
            by_uuid[str(child.uuid)] = child
    return list(by_uuid.values())


def source_parquet_uri_for_item(item: Any, client: Any) -> str | None:
    candidates = [item, *group_children(item, client)]
    for candidate in candidates:
        metadata = item_metadata(candidate)
        for key in ("source_parquet_uri", "parquet_uri"):
            if metadata.get(key):
                return canonical_s3_uri(str(metadata[key]))

    episode_path = None
    video_uri = None
    for candidate in candidates:
        metadata = item_metadata(candidate)
        episode_path = episode_path or episode_path_from_value(
            metadata.get("episode_path")
        )
        for key in ("source_uri", "s3_uri", "source_s3_uri"):
            if metadata.get(key):
                video_uri = video_uri or str(metadata[key])
                episode_path = episode_path or episode_path_from_value(metadata[key])
    if not episode_path or not video_uri:
        return None
    bucket, _ = parse_s3_uri(video_uri)
    episode_id = episode_path.rstrip("/").rsplit("/", 1)[-1]
    return (
        f"s3://{bucket}/{episode_path.rstrip('/')}/data/chunk-000/"
        f"{episode_id}.parquet"
    )


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


def task_source_by_data_hash(project: Any, client: Any) -> dict[str, TaskSource]:
    attached = list(project.list_datasets())
    if len(attached) != 1:
        raise typer.BadParameter(
            f"Expected exactly one dataset attached to the project, found {len(attached)}."
        )

    dataset = client.get_dataset(str(attached[0].dataset_hash))
    data_rows = list(dataset.data_rows)
    backing_ids = [
        row.backing_item_uuid
        for row in data_rows
        if getattr(row, "backing_item_uuid", None) is not None
    ]
    storage_items = {
        str(item.uuid): item for item in client.get_storage_items(backing_ids)
    }

    tasks: dict[str, TaskSource] = {}
    sources: Counter[str] = Counter()
    for row in data_rows:
        data_hash = str(row.uid)
        if data_hash in tasks:
            raise typer.BadParameter(f"Duplicate dataset data hash: {data_hash}")
        item = storage_items.get(str(getattr(row, "backing_item_uuid", "")))
        if item is None:
            sources["missing_storage_item"] += 1
            continue
        task_name, source = resolve_task_name(row, item)
        sources[source] += 1
        if task_name:
            tasks[data_hash] = TaskSource(
                task_name=task_name,
                source_parquet_uri=source_parquet_uri_for_item(item, client),
            )

    typer.echo(f"Task metadata sources: {format_counter(sources)}")
    return tasks


def read_source_parquet(s3: Any, uri: str) -> Any:
    import pyarrow.parquet as pq

    bucket, key = parse_s3_uri(uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        payload = body.read()
    finally:
        close = getattr(body, "close", None)
        if close:
            close()
    return pq.read_table(BytesIO(payload), columns=list(SOURCE_PARQUET_COLUMNS))


def create_s3_client(profile: str | None) -> Any:
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3")


def validate_caption_ontology(project: Any) -> None:
    missing = []
    for title in CLASSIFICATION_TITLES:
        try:
            project.ontology_structure.get_child_by_title(
                title=title,
                type_=Classification,
            )
        except Exception:
            missing.append(title)
    if missing:
        raise typer.BadParameter(
            "Project ontology is missing caption classifications: "
            + ", ".join(missing)
        )


def build_caption_plans(
    project: Any,
    client: Any,
    task_captions: dict[str, TaskCaptionTemplate] = TASK_CAPTIONS,
    s3: Any | None = None,
) -> list[CaptionPlan]:
    validate_caption_ontology(project)
    tasks = task_source_by_data_hash(project, client)
    all_rows = list(project.list_label_rows_v2())
    rows = [row for row in all_rows if is_supported_label_row(row)]

    seen_hashes: set[str] = set()
    missing_tasks: list[str] = []
    unsupported: Counter[str] = Counter()
    plans: list[CaptionPlan] = []
    for row in rows:
        data_hash = str(row.data_hash)
        if data_hash in seen_hashes:
            raise typer.BadParameter(f"Duplicate project label-row data hash: {data_hash}")
        seen_hashes.add(data_hash)

        task_source = tasks.get(data_hash)
        if task_source is None:
            missing_tasks.append(data_hash)
            continue
        task_name = task_source.task_name
        if task_name not in task_captions:
            unsupported[task_name] += 1
            continue
        arm_phrase = "the robot arm"
        if s3 is not None:
            if not task_source.source_parquet_uri:
                raise typer.BadParameter(
                    f"Dataset row {data_hash} has no resolvable source Parquet URI."
                )
            try:
                table = read_source_parquet(s3, task_source.source_parquet_uri)
                arm_phrase = infer_arm_phrase_from_table(table)
            except typer.BadParameter:
                raise
            except Exception as exc:
                raise typer.BadParameter(
                    f"Could not infer active arm for row {data_hash} from "
                    f"{task_source.source_parquet_uri}: {exc}"
                ) from exc
        plans.append(
            CaptionPlan(
                row=row,
                arm_phrase=arm_phrase,
                captions=caption_variants_for_task(
                    task_name,
                    arm_phrase,
                    task_captions=task_captions,
                ),
            )
        )

    typer.echo(f"Project label rows: {len(all_rows)}")
    typer.echo(f"Video/data-group rows: {len(rows)}")
    typer.echo(f"Mapped caption rows: {len(plans)}")
    if missing_tasks or unsupported:
        details = []
        if missing_tasks:
            preview = ", ".join(missing_tasks[:5])
            details.append(f"{len(missing_tasks)} rows missing task metadata ({preview})")
        if unsupported:
            details.append(f"unsupported tasks: {format_counter(unsupported)}")
        raise typer.BadParameter("; ".join(details))
    if not plans:
        raise typer.BadParameter("No video or data-group rows matched the caption mapping.")
    if s3 is not None:
        typer.echo(
            "Active-arm captions: "
            + format_counter(Counter(plan.arm_phrase for plan in plans))
        )
    return plans


def initialize_rows(project: Any, rows: list[Any]) -> None:
    for batch in chunks(rows):
        with project.create_bundle(bundle_size=len(batch)) as bundle:
            for row in batch:
                row.initialise_labels(bundle=bundle)


def caption_instances(label_row: Any) -> list[Any]:
    return [
        instance
        for instance in label_row.get_classification_instances()
        if getattr(getattr(instance, "ontology_item", None), "title", None)
        in CLASSIFICATION_TITLES
    ]


def existing_caption_titles(label_row: Any) -> set[str]:
    return {
        str(instance.ontology_item.title)
        for instance in caption_instances(label_row)
    }


def replace_language_instructions(
    row: Any,
    captions: tuple[str, str, str],
) -> None:
    for instance in caption_instances(row):
        row.remove_classification(instance)

    is_video = data_type_name(getattr(row, "data_type", None)) == "VIDEO"
    row_range: Range | None = None
    if is_video:
        frame_count = int(getattr(row, "number_of_frames", 0) or 0)
        if frame_count <= 0:
            raise typer.BadParameter(
                f"Video label row {row.data_hash} reports no frames."
            )
        row_range = Range(start=0, end=frame_count - 1)

    for title, caption in zip(CLASSIFICATION_TITLES, captions, strict=True):
        classification = row.ontology_structure.get_child_by_title(
            title=title,
            type_=Classification,
        )
        instance = classification.create_instance()
        instance.set_answer(answer=caption)
        if row_range is not None:
            instance.set_for_frames(frames=row_range, overwrite=True)
        row.add_classification_instance(instance)


def save_rows(project: Any, rows: list[Any]) -> None:
    for batch in chunks(rows):
        with project.create_bundle(bundle_size=len(batch)) as bundle:
            for row in batch:
                row.save(bundle=bundle)


def create_captions(
    client: Any,
    project_hash: str,
    *,
    caption_map: Path = DEFAULT_TASK_CAPTIONS_PATH,
    overwrite: bool,
    s3: Any | None = None,
) -> None:
    try:
        task_captions = load_task_captions(caption_map)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Loaded {len(task_captions)} task caption templates from {caption_map}.")
    project = client.get_project(project_hash)
    plans = build_caption_plans(project, client, task_captions, s3=s3)
    initialize_rows(project, [plan.row for plan in plans])

    complete = 0
    partial: list[str] = []
    pending: list[CaptionPlan] = []
    expected = set(CLASSIFICATION_TITLES)
    for plan in plans:
        present = existing_caption_titles(plan.row)
        if present == expected and not overwrite:
            complete += 1
        elif present and present != expected and not overwrite:
            partial.append(str(plan.row.data_hash))
        else:
            pending.append(plan)

    if partial:
        preview = ", ".join(partial[:5])
        raise typer.BadParameter(
            f"{len(partial)} rows have partial Language Instruction labels ({preview}); "
            "rerun with --overwrite to replace only those caption fields."
        )

    typer.echo(f"Already complete: {complete}")
    typer.echo(f"Caption rows to update: {len(pending)}")
    for plan in pending:
        replace_language_instructions(plan.row, plan.captions)
    save_rows(project, [plan.row for plan in pending])
    typer.echo(f"Updated {len(pending)} label rows.")


def main(
    ssh_key_file: Annotated[
        Path,
        typer.Option(help="Path to the Encord SSH private-key file."),
    ],
    project_hash: Annotated[str, typer.Option(help="Caption project hash.")],
    caption_map: Annotated[
        Path | None,
        typer.Option(help="Optional YAML task-to-caption mapping."),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(help="Optional Encord API domain, for example https://api.us.encord.com."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace only existing Language Instruction 1/2/3 labels."),
    ] = False,
    infer_active_arm: Annotated[
        bool,
        typer.Option(
            help="Read each source Parquet into memory and describe left, right, or both arms in Language Instruction 3."
        ),
    ] = False,
    aws_profile: Annotated[
        str | None,
        typer.Option(help="Optional AWS profile used only with --infer-active-arm."),
    ] = None,
) -> None:
    if aws_profile and not infer_active_arm:
        raise typer.BadParameter("--aws-profile requires --infer-active-arm.")
    client = create_client(ssh_key_file.expanduser(), domain)
    create_captions(
        client,
        project_hash,
        caption_map=(caption_map or DEFAULT_TASK_CAPTIONS_PATH).expanduser(),
        overwrite=overwrite,
        s3=create_s3_client(aws_profile) if infer_active_arm else None,
    )


if __name__ == "__main__":
    typer.run(main)
