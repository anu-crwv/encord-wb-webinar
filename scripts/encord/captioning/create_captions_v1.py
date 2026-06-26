# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "numpy",
#     "pyarrow",
#     "typer",
# ]
# ///
"""Create an Encord V1 caption project from a dataset hash."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
from typing import Annotated, Any
from urllib.parse import unquote, urlparse
from uuid import UUID

import typer

from encord import EncordUserClient
from encord.objects import Classification
from encord.objects.frames import Range
from encord.orm.project import ManualReviewWorkflowSettings

from helper.captioning_v1 import (
    CLASSIFICATION_TITLES,
    DEFAULT_ONTOLOGY_HASH,
    SOURCE_PARQUET_COLUMNS,
    TASK_CAPTIONS,
    caption_variants_for_task,
    infer_arm_phrase_from_table,
    read_cached_parquet_table,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SOURCE_PARQUET_CACHE_DIR = REPO_ROOT / "exports" / "encord-source-parquet-cache"
BUNDLE_SIZE = 100
EPISODE_DIR_RE = re.compile(r"^episode_\d+(?:_[A-Za-z0-9]+)?$")


@dataclass(frozen=True)
class CaptionPlanRow:
    data_hash: str
    data_title: str
    storage_item_uuid: str
    task_name: str
    episode_path: str
    source_parquet_uri: str
    arm_phrase: str
    captions: tuple[str, str, str]


def log_progress(action: str, index: int, total: int) -> None:
    if index == total or index % 50 == 0:
        typer.echo(f"{action} {index}/{total}...")


def row_chunks(rows: list[Any]) -> list[list[Any]]:
    return [rows[index : index + BUNDLE_SIZE] for index in range(0, len(rows), BUNDLE_SIZE)]


def create_client() -> EncordUserClient:
    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE to your Encord SSH private key file path.")
    key_path = Path(ssh_key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"ENCORD_SSH_KEY_FILE does not exist: {key_path}")
    typer.echo("Connecting to Encord...")
    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def s3_client() -> Any:
    import boto3

    return boto3.client("s3")


def item_metadata(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "client_metadata", None) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def normalize_source_path(value: Any) -> str:
    path = str(value or "")
    if path.startswith("s3://"):
        return unquote(urlparse(path).path.lstrip("/"))
    if path.startswith("http://") or path.startswith("https://"):
        return unquote(urlparse(path).path.lstrip("/"))
    return path.lstrip("/")


def path_parts(value: Any) -> tuple[str, ...]:
    return PurePosixPath(normalize_source_path(value)).parts


def episode_path_from_source(value: Any) -> str | None:
    parts = path_parts(value)
    for index, part in enumerate(parts):
        if EPISODE_DIR_RE.fullmatch(part):
            return "/".join(parts[: index + 1]) + "/"
    return None


def episode_path_from_metadata(metadata: dict[str, Any], fallback_name: Any = None) -> str | None:
    if metadata.get("episode_path"):
        return str(metadata["episode_path"])
    for key in ("source_key", "source_uri", "s3_uri", "source_s3_uri", "objectUrl", "object_url"):
        derived = episode_path_from_source(metadata.get(key))
        if derived:
            return derived
    return episode_path_from_source(fallback_name)


def episode_id_from_path(episode_path: str | None) -> str | None:
    if not episode_path:
        return None
    for part in reversed(path_parts(episode_path)):
        if EPISODE_DIR_RE.fullmatch(part):
            return part
    return None


def task_name_from_episode_path(episode_path: str | None) -> str | None:
    if not episode_path:
        return None
    parts = path_parts(episode_path)
    if "raw-feed" in parts:
        index = parts.index("raw-feed")
        family = parts[index + 1] if index + 1 < len(parts) else None
        if family in {"trossen-data", "trossen-data-stationary"} and index + 2 < len(parts):
            return parts[index + 2]
    return None


def source_s3_uri(metadata: dict[str, Any]) -> str | None:
    for key in ("source_uri", "s3_uri", "source_s3_uri", "objectUrl", "object_url"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def source_bucket(metadata: dict[str, Any]) -> str | None:
    uri = source_s3_uri(metadata)
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc
    if parsed.scheme in {"http", "https"} and ".s3." in parsed.netloc:
        return parsed.netloc.split(".s3.", 1)[0]
    return None


def source_parquet_uri(metadata: dict[str, Any], fallback_title: Any = None) -> str | None:
    for key in ("parquet_uri", "source_parquet_uri"):
        if metadata.get(key):
            return str(metadata[key])

    episode_path = episode_path_from_metadata(metadata, fallback_title)
    episode_id = str(metadata.get("episode_id") or episode_id_from_path(episode_path) or "")
    bucket = source_bucket(metadata)
    if not episode_path or not episode_id or not bucket:
        return None
    return f"s3://{bucket}/{episode_path.rstrip('/')}/data/chunk-000/{episode_id}.parquet"


def group_children(item: Any, client: EncordUserClient) -> list[Any]:
    children_by_uuid: dict[str, Any] = {}
    try:
        for child in item.get_child_items():
            children_by_uuid[str(child.uuid)] = child
    except Exception:
        pass

    try:
        summary = item.get_summary()
    except Exception:
        return list(children_by_uuid.values())

    data_group = getattr(summary, "data_group", None)
    layout_contents = getattr(data_group, "layout_contents", None)
    if layout_contents:
        child_uuids = [
            child.uuid
            for child in layout_contents.values()
            if str(child.uuid) not in children_by_uuid
        ]
        if child_uuids:
            for child in client.get_storage_items(child_uuids):
                children_by_uuid[str(child.uuid)] = child
    return list(children_by_uuid.values())


def merged_row_metadata(item: Any, client: EncordUserClient, fallback_title: Any = None) -> dict[str, Any]:
    metadata = item_metadata(item)
    need_children = not source_s3_uri(metadata) or not metadata.get("task_name")
    children = group_children(item, client) if need_children else []

    for child in children:
        child_metadata = item_metadata(child)
        for key in (
            "task_name",
            "episode_path",
            "episode_id",
            "source_uri",
            "s3_uri",
            "source_s3_uri",
            "objectUrl",
            "object_url",
            "parquet_uri",
            "source_parquet_uri",
        ):
            if not metadata.get(key) and child_metadata.get(key):
                metadata[key] = child_metadata[key]

    episode_path = episode_path_from_metadata(metadata, fallback_title)
    if episode_path and not metadata.get("episode_path"):
        metadata["episode_path"] = episode_path
    if episode_path and not metadata.get("episode_id"):
        metadata["episode_id"] = episode_id_from_path(episode_path)
    if episode_path and not metadata.get("task_name"):
        metadata["task_name"] = task_name_from_episode_path(episode_path)
    return metadata


def is_failed_episode(metadata: dict[str, Any], fallback_title: Any = None) -> bool:
    values = [
        metadata.get("episode_path"),
        metadata.get("source_uri"),
        metadata.get("source_key"),
        fallback_title,
    ]
    return any("failed-videos" in normalize_source_path(value).split("/") for value in values if value)


def validate_ontology(client: EncordUserClient, ontology_hash: str) -> None:
    typer.echo(f"Validating ontology {ontology_hash}...")
    ontology = client.get_ontology(ontology_hash)
    missing = []
    for title in CLASSIFICATION_TITLES:
        try:
            ontology.structure.get_child_by_title(title=title, type_=Classification)
        except Exception:
            missing.append(title)
    if missing:
        raise typer.BadParameter(
            f"Ontology {ontology_hash} is missing required classifications: {', '.join(missing)}"
        )


def resolve_dataset_rows(
    *,
    client: EncordUserClient,
    dataset_hash: str,
    limit: int | None,
) -> tuple[Any, list[Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    typer.echo(f"Loading dataset {dataset_hash}...")
    dataset = client.get_dataset(dataset_hash)
    data_rows = list(dataset.data_rows)
    if limit is not None:
        data_rows = data_rows[:limit]
    typer.echo(f"Preparing {len(data_rows)} dataset rows.")

    backing_ids = [row.backing_item_uuid for row in data_rows if getattr(row, "backing_item_uuid", None)]
    storage_items = {str(item.uuid): item for item in client.get_storage_items(backing_ids)} if backing_ids else {}

    metadata_by_hash: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    unsupported_tasks: dict[str, int] = {}
    missing_parquets: list[str] = []

    for row in data_rows:
        data_hash = str(row.uid)
        item = storage_items.get(str(getattr(row, "backing_item_uuid", "")))
        if item is None:
            skipped.append({"data_hash": data_hash, "data_title": row.title, "reason": "missing_storage_item"})
            continue

        metadata = merged_row_metadata(item, client, row.title)
        metadata_by_hash[data_hash] = metadata

        if is_failed_episode(metadata, row.title):
            skipped.append({"data_hash": data_hash, "data_title": row.title, "reason": "failed_episode_prefix"})
            continue

        task_name = metadata.get("task_name")
        if task_name not in TASK_CAPTIONS:
            unsupported_tasks[str(task_name or "")] = unsupported_tasks.get(str(task_name or ""), 0) + 1
            continue

        if not source_parquet_uri(metadata, row.title):
            missing_parquets.append(str(row.title))

    if unsupported_tasks:
        details = ", ".join(f"{task or '<missing>'}: {count}" for task, count in sorted(unsupported_tasks.items()))
        raise typer.BadParameter(f"Unsupported or missing task_name values: {details}")
    if missing_parquets:
        sample = "; ".join(missing_parquets[:5])
        raise typer.BadParameter(f"Could not resolve source parquet URI for {len(missing_parquets)} rows. Examples: {sample}")

    return dataset, data_rows, metadata_by_hash, skipped


def build_caption_plan(
    *,
    client_s3: Any,
    data_rows: list[Any],
    metadata_by_hash: dict[str, dict[str, Any]],
    source_parquet_cache_dir: Path,
    skipped: list[dict[str, Any]],
) -> list[CaptionPlanRow]:
    plan: list[CaptionPlanRow] = []
    failures: list[dict[str, Any]] = []
    typer.echo("Inferring arm-aware captions from source parquets...")

    for index, row in enumerate(data_rows, start=1):
        data_hash = str(row.uid)
        metadata = metadata_by_hash.get(data_hash) or {}
        if is_failed_episode(metadata, row.title):
            log_progress("Prepared", index, len(data_rows))
            continue

        task_name = str(metadata.get("task_name") or "")
        parquet_uri = source_parquet_uri(metadata, row.title)
        if task_name not in TASK_CAPTIONS or not parquet_uri:
            log_progress("Prepared", index, len(data_rows))
            continue

        try:
            table = read_cached_parquet_table(
                client_s3,
                parquet_uri,
                source_parquet_cache_dir,
                columns=SOURCE_PARQUET_COLUMNS,
            )
            arm_phrase = infer_arm_phrase_from_table(table)
        except Exception as exc:
            failures.append({
                "data_hash": data_hash,
                "data_title": row.title,
                "source_parquet_uri": parquet_uri,
                "error": str(exc),
            })
            log_progress("Prepared", index, len(data_rows))
            continue

        plan.append(CaptionPlanRow(
            data_hash=data_hash,
            data_title=str(row.title),
            storage_item_uuid=str(getattr(row, "backing_item_uuid", "")),
            task_name=task_name,
            episode_path=str(episode_path_from_metadata(metadata, row.title) or ""),
            source_parquet_uri=parquet_uri,
            arm_phrase=arm_phrase,
            captions=caption_variants_for_task(task_name, arm_phrase),
        ))
        log_progress("Prepared", index, len(data_rows))

    if failures:
        skipped.extend({"reason": "arm_inference_failed", **failure} for failure in failures)
        sample = "; ".join(f"{failure['data_title']}: {failure['error']}" for failure in failures[:3])
        raise typer.BadParameter(f"Arm inference failed for {len(failures)} rows. Examples: {sample}")
    if not plan:
        raise typer.BadParameter("No dataset rows produced valid caption plans.")
    return plan


def frame_range(label_row: Any) -> Range | None:
    frames = int(getattr(label_row, "number_of_frames", 0) or 0)
    if frames <= 0:
        return None
    return Range(start=0, end=frames - 1)


def existing_classification_titles(label_row: Any) -> set[str]:
    titles = set()
    for instance in label_row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        title = getattr(ontology_item, "title", None)
        if title:
            titles.add(str(title))
    return titles


def add_caption_instances(
    label_row: Any,
    captions: tuple[str, str, str],
    *,
    overwrite: bool,
) -> None:
    row_range = frame_range(label_row)
    for title, caption in zip(CLASSIFICATION_TITLES, captions, strict=True):
        classification = label_row.ontology_structure.get_child_by_title(
            title=title,
            type_=Classification,
        )
        instance = classification.create_instance()
        instance.set_answer(answer=caption)
        if row_range is not None:
            instance.set_for_frames(frames=row_range, overwrite=overwrite)
        label_row.add_classification_instance(instance, force=overwrite)


def create_project_and_write_labels(
    *,
    client: EncordUserClient,
    dataset_hash: str,
    project_title: str,
    ontology_hash: str,
    plan: list[CaptionPlanRow],
    overwrite: bool,
) -> str:
    typer.echo(f"Creating Encord project {project_title!r}...")
    project_hash = client.create_project(
        project_title=project_title,
        dataset_hashes=[dataset_hash],
        ontology_hash=ontology_hash,
        workflow_settings=ManualReviewWorkflowSettings(),
    )
    project = client.get_project(project_hash)
    typer.echo(f"Created project {project_hash}.")

    rows_by_hash = {str(row.data_hash): row for row in project.list_label_rows_v2()}
    missing = [row.data_hash for row in plan if row.data_hash not in rows_by_hash]
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"Project is missing {len(missing)} label rows for dataset rows. Examples: {sample}")

    plan_by_hash = {row.data_hash: row for row in plan}
    label_rows = [rows_by_hash[data_hash] for data_hash in plan_by_hash]
    typer.echo(f"Initializing {len(label_rows)} label rows...")
    initialized = 0
    for chunk in row_chunks(label_rows):
        with project.create_bundle(bundle_size=len(chunk)) as bundle:
            for label_row in chunk:
                label_row.initialise_labels(bundle=bundle)
        initialized += len(chunk)
        log_progress("Initialized", initialized, len(label_rows))

    touched = []
    skipped_existing = 0
    for label_row in label_rows:
        current_titles = existing_classification_titles(label_row)
        if current_titles.intersection(CLASSIFICATION_TITLES) and not overwrite:
            skipped_existing += 1
            continue
        add_caption_instances(label_row, plan_by_hash[str(label_row.data_hash)].captions, overwrite=overwrite)
        touched.append(label_row)

    typer.echo(f"Saving {len(touched)} label rows...")
    saved = 0
    for chunk in row_chunks(touched):
        with project.create_bundle(bundle_size=len(chunk)) as bundle:
            for label_row in chunk:
                label_row.save(bundle=bundle)
        saved += len(chunk)
        log_progress("Saved", saved, len(touched))

    typer.echo(f"Updated {len(touched)} label rows; skipped {skipped_existing} existing caption rows.")
    return str(project_hash)


def main(
    dataset_hash: Annotated[UUID, typer.Argument(help="Encord dataset hash.")],
    project_title: Annotated[str, typer.Argument(help="Title for the new Encord project.")],
    ontology_hash: Annotated[
        str,
        typer.Option(help="Ontology hash containing Language Instruction 1/2/3 classifications."),
    ] = DEFAULT_ONTOLOGY_HASH,
    source_parquet_cache_dir: Annotated[
        Path,
        typer.Option(help="Local cache root for source episode parquets."),
    ] = DEFAULT_SOURCE_PARQUET_CACHE_DIR,
    limit: Annotated[int | None, typer.Option(help="Limit dataset rows for smoke runs.")] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite existing caption classifications if present."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Preflight and generate captions without creating the project."),
    ] = False,
) -> None:
    client = create_client()
    dataset_hash_str = str(dataset_hash)
    validate_ontology(client, ontology_hash)
    dataset, data_rows, metadata_by_hash, skipped = resolve_dataset_rows(
        client=client,
        dataset_hash=dataset_hash_str,
        limit=limit,
    )

    plan = build_caption_plan(
        client_s3=s3_client(),
        data_rows=data_rows,
        metadata_by_hash=metadata_by_hash,
        source_parquet_cache_dir=source_parquet_cache_dir,
        skipped=skipped,
    )

    typer.echo(f"Caption plan ready for {len(plan)} rows from dataset {dataset_hash_str} ({dataset.title}).")
    if skipped:
        typer.echo(f"Skipped {len(skipped)} rows.")

    if dry_run:
        typer.echo("Dry run enabled; not creating an Encord project.")
        for row in plan[:5]:
            typer.echo(
                f"{row.data_hash}: {row.task_name} | "
                f"{CLASSIFICATION_TITLES[0]}={row.captions[0]!r}; "
                f"{CLASSIFICATION_TITLES[1]}={row.captions[1]!r}; "
                f"{CLASSIFICATION_TITLES[2]}={row.captions[2]!r}"
            )
        return

    project_hash = create_project_and_write_labels(
        client=client,
        dataset_hash=dataset_hash_str,
        project_title=project_title,
        ontology_hash=ontology_hash,
        plan=plan,
        overwrite=overwrite,
    )
    typer.echo(f"project_hash: {project_hash}")
    typer.echo(f"source parquet cache: {source_parquet_cache_dir}")


if __name__ == "__main__":
    typer.run(main)
