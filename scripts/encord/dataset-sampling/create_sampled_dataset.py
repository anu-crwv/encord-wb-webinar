# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create an Encord dataset sampled uniformly across metadata distributions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Annotated, Any
from uuid import UUID

import typer


METADATA_KEYS = ["task_name", "collection_datetime"]
ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY_FILE"


@dataclass(frozen=True)
class SampleRow:
    backing_item_uuid: Any
    metadata_values: dict[str, str]


def create_client(ssh_key_file: Path | None) -> Any:
    from encord.user_client import EncordUserClient

    key_file = ssh_key_file or os.environ.get(ENCORD_SSH_KEY_ENV)
    if not key_file:
        raise typer.BadParameter(f"Pass --ssh-key-file or set {ENCORD_SSH_KEY_ENV}.")

    key_path = Path(key_file).expanduser()
    if not key_path.exists():
        raise typer.BadParameter(f"SSH key file does not exist: {key_path}")

    return EncordUserClient.create_with_ssh_private_key(key_path.read_text())


def row_metadata(row: Any) -> dict[str, Any]:
    metadata = row.client_metadata
    return dict(metadata) if metadata is not None else {}


def metadata_values(metadata: dict[str, Any]) -> dict[str, str] | None:
    if any(key not in metadata for key in METADATA_KEYS):
        return None
    return {key: str(metadata[key]) for key in METADATA_KEYS}


def load_rows(dataset: Any) -> tuple[list[SampleRow], int, int]:
    data_rows = list(dataset.data_rows)

    rows = []
    excluded = 0
    for row in data_rows:
        backing_item_uuid = getattr(row, "backing_item_uuid", None)
        if backing_item_uuid is None:
            excluded += 1
            continue

        values = metadata_values(row_metadata(row))
        if values is None:
            excluded += 1
            continue

        rows.append(
            SampleRow(
                backing_item_uuid=backing_item_uuid,
                metadata_values=values,
            )
        )

    return rows, len(data_rows), excluded


def print_distribution(title: str, rows: list[SampleRow]) -> None:
    typer.echo(title)
    for key in METADATA_KEYS:
        counts = Counter(row.metadata_values[key] for row in rows)
        typer.echo(f"{key}: {len(counts)} unique")
        for value, count in counts.most_common():
            typer.echo(f"  {value}: {count}")


def choose_most_common_buckets(
    buckets: dict[str, list[SampleRow]],
    needed: int,
    rng: random.Random,
) -> dict[str, int]:
    selected: dict[str, int] = {}
    counts_to_values: dict[int, list[str]] = defaultdict(list)
    for value, rows in buckets.items():
        counts_to_values[len(rows)].append(value)

    for count in sorted(counts_to_values, reverse=True):
        values = counts_to_values[count]
        remaining = needed - len(selected)
        if remaining <= 0:
            break
        if len(values) <= remaining:
            for value in values:
                selected[value] = 1
            continue
        for value in rng.sample(values, remaining):
            selected[value] = 1
        break

    return selected


def allocate_evenly(
    buckets: dict[str, list[SampleRow]],
    needed: int,
    rng: random.Random,
) -> dict[str, int]:
    total_available = sum(len(rows) for rows in buckets.values())
    remaining = min(needed, total_available)
    allocations = {value: 0 for value in buckets}

    while remaining > 0:
        active = [value for value, rows in buckets.items() if allocations[value] < len(rows)]
        if not active:
            break

        if remaining < len(active):
            for value in rng.sample(active, remaining):
                allocations[value] += 1
            remaining = 0
            break

        share = remaining // len(active)
        for value in active:
            capacity = len(buckets[value]) - allocations[value]
            take = min(share, capacity)
            allocations[value] += take
            remaining -= take

    return {value: count for value, count in allocations.items() if count > 0}


def bucket_allocations(
    buckets: dict[str, list[SampleRow]],
    needed: int,
    rng: random.Random,
) -> dict[str, int]:
    capped_needed = min(needed, sum(len(rows) for rows in buckets.values()))
    if capped_needed <= 0:
        return {}
    if len(buckets) > capped_needed:
        return choose_most_common_buckets(buckets, capped_needed, rng)
    return allocate_evenly(buckets, capped_needed, rng)


def sample_level(
    rows: list[SampleRow],
    needed: int,
    key_index: int,
    rng: random.Random,
) -> list[SampleRow]:
    capped_needed = min(needed, len(rows))
    if capped_needed <= 0:
        return []
    if key_index >= len(METADATA_KEYS):
        return rng.sample(rows, capped_needed)

    key = METADATA_KEYS[key_index]
    buckets: dict[str, list[SampleRow]] = defaultdict(list)
    for row in rows:
        buckets[row.metadata_values[key]].append(row)

    selected: list[SampleRow] = []
    for value, count in bucket_allocations(buckets, capped_needed, rng).items():
        selected.extend(sample_level(buckets[value], count, key_index + 1, rng))
    return selected


def create_sampled_dataset(client: Any, title: str, rows: list[SampleRow]) -> str:
    from encord.orm.dataset import StorageLocation

    response = client.create_dataset(
        dataset_title=title,
        dataset_type=StorageLocation.CORD_STORAGE,
        create_backing_folder=False,
    )
    output_dataset = client.get_dataset(str(response.dataset_hash))
    output_dataset.link_items([row.backing_item_uuid for row in rows])
    return str(response.dataset_hash)


def main(
    dataset_hash: Annotated[UUID, typer.Option(help="Source Encord dataset hash.")],
    target_dataset_size: Annotated[int, typer.Option(help="Target sampled dataset size.")],
    output_dataset_title: Annotated[
        str | None,
        typer.Option(help="Title for the created Encord dataset."),
    ] = None,
    ssh_key_file: Annotated[
        Path | None,
        typer.Option(help=f"Encord SSH private key file. Falls back to {ENCORD_SSH_KEY_ENV}."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed for reproducible sampling.")] = None,
) -> None:
    if target_dataset_size <= 0:
        raise typer.BadParameter("--target-dataset-size must be greater than 0.")

    rng = random.Random(seed)
    client = create_client(ssh_key_file)
    source_dataset = client.get_dataset(dataset_hash)
    from encord.client import DatasetAccessSettings

    source_dataset.set_access_settings(DatasetAccessSettings(fetch_client_metadata=True))

    rows, total_rows, excluded_rows = load_rows(source_dataset)
    typer.echo(f"Source dataset: {source_dataset.title} ({dataset_hash})")
    typer.echo(f"Data rows: {total_rows}")
    typer.echo(f"Eligible rows: {len(rows)}")
    typer.echo(f"Excluded rows: {excluded_rows}")

    if not rows:
        raise typer.BadParameter("No rows have all configured metadata keys.")

    print_distribution("Source distribution:", rows)

    sampled_rows = sample_level(rows, target_dataset_size, 0, rng)
    if not sampled_rows:
        raise typer.BadParameter("No rows selected.")

    print_distribution("Sample distribution:", sampled_rows)

    title = output_dataset_title or f"{source_dataset.title}_sampled"
    output_dataset_hash = create_sampled_dataset(client, title, sampled_rows)

    typer.echo(f"Created dataset: {output_dataset_hash}")
    typer.echo(f"Linked files: {len(sampled_rows)}")


if __name__ == "__main__":
    typer.run(main)
