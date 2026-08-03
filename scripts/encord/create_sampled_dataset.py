# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "encord==0.1.199",
#     "typer",
# ]
# ///
"""Create a reproducible Encord dataset balanced across metadata fields."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

DEFAULT_STRATIFY_KEYS = ("task_name", "collection_datetime")


@dataclass(frozen=True)
class SampleRow:
    backing_item_uuid: Any
    metadata_values: dict[str, str]


def create_client(ssh_key_file: Path, domain: str | None = None) -> Any:
    from encord import EncordUserClient

    if not ssh_key_file.is_file():
        raise typer.BadParameter(f"SSH key file does not exist: {ssh_key_file}")
    kwargs: dict[str, Any] = {"ssh_private_key_path": ssh_key_file}
    if domain:
        kwargs["domain"] = domain
    return EncordUserClient.create_with_ssh_private_key(**kwargs)


def row_metadata(row: Any) -> dict[str, Any]:
    value = getattr(row, "client_metadata", None) or {}
    return dict(value) if isinstance(value, dict) else {}


def load_rows(
    dataset: Any,
    metadata_keys: tuple[str, ...],
) -> tuple[list[SampleRow], int, Counter[str]]:
    data_rows = list(dataset.data_rows)
    excluded: Counter[str] = Counter()
    rows: list[SampleRow] = []
    seen_backing_items: set[str] = set()

    for row in data_rows:
        backing_item_uuid = getattr(row, "backing_item_uuid", None)
        if backing_item_uuid is None:
            excluded["missing backing item"] += 1
            continue
        backing_key = str(backing_item_uuid)
        if backing_key in seen_backing_items:
            raise typer.BadParameter(
                f"Source dataset contains duplicate backing item {backing_key}."
            )
        seen_backing_items.add(backing_key)

        metadata = row_metadata(row)
        missing = [
            key
            for key in metadata_keys
            if metadata.get(key) in (None, "")
        ]
        if missing:
            excluded["missing " + ", ".join(missing)] += 1
            continue

        rows.append(
            SampleRow(
                backing_item_uuid=backing_item_uuid,
                metadata_values={key: str(metadata[key]) for key in metadata_keys},
            )
        )
    return rows, len(data_rows), excluded


def print_distribution(
    title: str,
    rows: list[SampleRow],
    metadata_keys: tuple[str, ...],
) -> None:
    typer.echo(title)
    for key in metadata_keys:
        counts = Counter(row.metadata_values[key] for row in rows)
        typer.echo(f"  {key}: {len(counts)} values")
        for value, count in sorted(counts.items()):
            typer.echo(f"    {value}: {count}")


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
        values = sorted(counts_to_values[count])
        remaining = needed - len(selected)
        if remaining <= 0:
            break
        if len(values) <= remaining:
            selected.update({value: 1 for value in values})
        else:
            selected.update({value: 1 for value in rng.sample(values, remaining)})
            break
    return selected


def allocate_evenly(
    buckets: dict[str, list[SampleRow]],
    needed: int,
    rng: random.Random,
) -> dict[str, int]:
    remaining = min(needed, sum(len(rows) for rows in buckets.values()))
    allocations = {value: 0 for value in buckets}
    while remaining > 0:
        active = sorted(
            value
            for value, rows in buckets.items()
            if allocations[value] < len(rows)
        )
        if not active:
            break
        if remaining < len(active):
            for value in rng.sample(active, remaining):
                allocations[value] += 1
            break

        share = remaining // len(active)
        for value in active:
            capacity = len(buckets[value]) - allocations[value]
            take = min(share, capacity)
            allocations[value] += take
            remaining -= take
    return {value: count for value, count in allocations.items() if count}


def bucket_allocations(
    buckets: dict[str, list[SampleRow]],
    needed: int,
    rng: random.Random,
) -> dict[str, int]:
    capped = min(needed, sum(len(rows) for rows in buckets.values()))
    if capped <= 0:
        return {}
    if len(buckets) > capped:
        return choose_most_common_buckets(buckets, capped, rng)
    return allocate_evenly(buckets, capped, rng)


def sample_level(
    rows: list[SampleRow],
    needed: int,
    metadata_keys: tuple[str, ...],
    key_index: int,
    rng: random.Random,
) -> list[SampleRow]:
    capped = min(needed, len(rows))
    if capped <= 0:
        return []
    if key_index >= len(metadata_keys):
        return rng.sample(rows, capped)

    key = metadata_keys[key_index]
    buckets: dict[str, list[SampleRow]] = defaultdict(list)
    for row in rows:
        buckets[row.metadata_values[key]].append(row)

    selected: list[SampleRow] = []
    allocations = bucket_allocations(buckets, capped, rng)
    for value in sorted(allocations):
        selected.extend(
            sample_level(
                buckets[value],
                allocations[value],
                metadata_keys,
                key_index + 1,
                rng,
            )
        )
    return selected


def create_output_dataset(client: Any, title: str, rows: list[SampleRow]) -> str:
    from encord.orm.dataset import StorageLocation

    response = client.create_dataset(
        dataset_title=title,
        dataset_type=StorageLocation.CORD_STORAGE,
        create_backing_folder=False,
    )
    output = client.get_dataset(str(response.dataset_hash))
    output.link_items([row.backing_item_uuid for row in rows])
    return str(response.dataset_hash)


def sample_dataset(
    client: Any,
    dataset_hash: str,
    target_dataset_size: int,
    *,
    output_dataset_title: str | None,
    metadata_keys: tuple[str, ...],
    seed: int,
) -> str:
    if target_dataset_size <= 0:
        raise typer.BadParameter("--target-dataset-size must be greater than zero.")
    if not metadata_keys:
        raise typer.BadParameter("Pass at least one --stratify-by field.")
    if len(set(metadata_keys)) != len(metadata_keys):
        raise typer.BadParameter("--stratify-by fields must be unique.")

    source = client.get_dataset(dataset_hash)
    from encord.client import DatasetAccessSettings

    source.set_access_settings(DatasetAccessSettings(fetch_client_metadata=True))
    rows, total, excluded = load_rows(source, metadata_keys)
    typer.echo(f"Source dataset: {source.title} ({dataset_hash})")
    typer.echo(f"Data rows: {total}")
    typer.echo(f"Eligible rows: {len(rows)}")
    typer.echo(f"Excluded rows: {sum(excluded.values())}")
    for reason, count in sorted(excluded.items()):
        typer.echo(f"  {reason}: {count}")

    if not rows:
        raise typer.BadParameter(
            "No rows contain every requested metadata field: "
            + ", ".join(metadata_keys)
        )
    if target_dataset_size > len(rows):
        raise typer.BadParameter(
            f"Requested {target_dataset_size} rows but only {len(rows)} are eligible."
        )

    print_distribution("Source distribution:", rows, metadata_keys)
    selected = sample_level(
        rows,
        target_dataset_size,
        metadata_keys,
        0,
        random.Random(seed),
    )
    if len(selected) != target_dataset_size:
        raise RuntimeError(
            f"Sampler selected {len(selected)} rows, expected {target_dataset_size}."
        )
    if len({str(row.backing_item_uuid) for row in selected}) != len(selected):
        raise RuntimeError("Sampler selected a backing item more than once.")
    print_distribution("Sample distribution:", selected, metadata_keys)

    title = output_dataset_title or f"{source.title} - balanced sample"
    output_hash = create_output_dataset(client, title, selected)
    typer.echo(f"Created dataset: {output_hash}")
    typer.echo(f"Linked rows: {len(selected)}")
    return output_hash


def main(
    ssh_key_file: Annotated[
        Path,
        typer.Option(help="Path to the Encord SSH private-key file."),
    ],
    dataset_hash: Annotated[str, typer.Option(help="Source Encord dataset hash.")],
    target_dataset_size: Annotated[
        int,
        typer.Option(min=1, help="Number of rows in the balanced sample."),
    ],
    output_dataset_title: Annotated[
        str | None,
        typer.Option(help="Title for the new sampled dataset."),
    ] = None,
    stratify_by: Annotated[
        list[str] | None,
        typer.Option(
            "--stratify-by",
            help="Metadata field to balance; repeat to define nested balancing order.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for reproducible selection."),
    ] = 0,
    domain: Annotated[
        str | None,
        typer.Option(help="Optional Encord API domain, for example https://api.us.encord.com."),
    ] = None,
) -> None:
    metadata_keys = tuple(stratify_by or DEFAULT_STRATIFY_KEYS)
    client = create_client(ssh_key_file.expanduser(), domain)
    sample_dataset(
        client,
        dataset_hash,
        target_dataset_size,
        output_dataset_title=output_dataset_title,
        metadata_keys=metadata_keys,
        seed=seed,
    )


if __name__ == "__main__":
    typer.run(main)
