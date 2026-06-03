# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "typer",
# ]
# ///
"""Explore an S3 prefix before building an Encord registration JSON.

Run:
    uv run --script scripts/encord/explore_s3_prefix.py s3://bucket/path/ --profile default
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated

import boto3
import typer

EXTENSION_TO_DATA_TYPE = {
    ".jpeg": "image",
    ".jpg": "image",
    ".png": "image",
    ".webp": "image",
    ".avif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".webm": "video",
    ".mkv": "video",
    ".m4v": "video",
    ".wmv": "video",
    ".3gp": "video",
    ".3g2": "video",
    ".mj2": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".eac3": "audio",
    ".m4a": "audio",
    ".mpeg": "audio",
    ".x-wav": "audio",
    ".pdf": "pdf",
    ".txt": "text",
    ".html": "text",
    ".md": "text",
    ".xml": "text",
    ".json": "metadata/text",
    ".jsonl": "metadata/text",
    ".csv": "metadata/tabular",
    ".tsv": "metadata/tabular",
    ".parquet": "metadata/tabular",
    ".yaml": "metadata/config",
    ".yml": "metadata/config",
    ".bag": "robotics log",
    ".mcap": "robotics log",
    ".db3": "robotics log",
    ".pcd": "point cloud",
    ".ply": "point cloud",
    ".las": "point cloud",
    ".laz": "point cloud",
    ".e57": "point cloud",
    ".npy": "array/tensor",
    ".npz": "array/tensor",
    ".h5": "array/tensor",
    ".hdf5": "array/tensor",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
}


@dataclass
class PrefixSummary:
    prefix: str
    objects: list[dict]
    hit_limit: bool


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise typer.BadParameter("Use an S3 URI like s3://bucket/prefix/")

    without_scheme = uri.removeprefix("s3://")
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise typer.BadParameter("S3 URI must include a bucket name")

    return bucket, prefix


def get_bucket_region(s3, bucket: str) -> str:
    response = s3.head_bucket(Bucket=bucket)
    headers = response["ResponseMetadata"]["HTTPHeaders"]
    return headers.get("x-amz-bucket-region", "us-east-1")


def list_top_level(s3, bucket: str, prefix: str) -> tuple[list[dict], list[str]]:
    paginator = s3.get_paginator("list_objects_v2")
    files: list[dict] = []
    folders: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        files.extend(obj for obj in page.get("Contents", []) if not obj["Key"].endswith("/"))
        folders.extend(item["Prefix"] for item in page.get("CommonPrefixes", []))

    return files, folders


def list_one_level(
    s3,
    bucket: str,
    prefix: str,
    max_folders: int,
    max_files: int,
) -> tuple[list[dict], list[str], bool]:
    paginator = s3.get_paginator("list_objects_v2")
    files: list[dict] = []
    folders: list[str] = []
    truncated = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/") and len(files) < max_files:
                files.append(obj)
        for item in page.get("CommonPrefixes", []):
            if len(folders) < max_folders:
                folders.append(item["Prefix"])
        if len(files) >= max_files and len(folders) >= max_folders:
            truncated = True
            break
        truncated = truncated or page.get("IsTruncated", False)

    return files, folders, truncated


def iter_recursive_objects(s3, bucket: str, prefix: str, max_objects: int):
    paginator = s3.get_paginator("list_objects_v2")
    seen = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue

            yield obj
            seen += 1
            if seen >= max_objects:
                return


def summarize_prefix(s3, bucket: str, prefix: str, max_objects: int) -> PrefixSummary:
    objects = list(iter_recursive_objects(s3, bucket, prefix, max_objects))
    return PrefixSummary(prefix=prefix, objects=objects, hit_limit=len(objects) == max_objects)


def extension_for_key(key: str) -> str:
    suffix = PurePosixPath(key).suffix.lower()
    return suffix or "[no extension]"


def data_type_for_key(key: str) -> str:
    return EXTENSION_TO_DATA_TYPE.get(extension_for_key(key), "unknown/unsupported")


def print_top_level(files: list[dict], folders: list[str]) -> None:
    typer.echo("\nTop-level folders:")
    if folders:
        for folder in folders[:50]:
            typer.echo(f"  DIR   {folder}")
        if len(folders) > 50:
            typer.echo(f"  ... {len(folders) - 50} more folders")
    else:
        typer.echo("  None")

    typer.echo("\nTop-level files:")
    if files:
        for obj in files[:50]:
            typer.echo(f"  FILE  {obj['Key']}  ({obj['Size']} bytes)")
        if len(files) > 50:
            typer.echo(f"  ... {len(files) - 50} more files")
    else:
        typer.echo("  None")


def print_prefix_tree(
    s3,
    bucket: str,
    root_prefix: str,
    max_depth: int,
    max_folders_per_prefix: int,
    max_files_per_prefix: int,
) -> None:
    typer.echo(f"\nPrefix tree sample, depth {max_depth}:")
    queue = deque([(root_prefix, 0)])
    seen = {root_prefix}

    while queue:
        prefix, depth = queue.popleft()
        indent = "  " * depth
        files, folders, truncated = list_one_level(
            s3,
            bucket,
            prefix,
            max_folders=max_folders_per_prefix,
            max_files=max_files_per_prefix,
        )
        label = prefix or "[bucket root]"
        suffix = " ..." if truncated else ""
        typer.echo(f"{indent}DIR   {label}{suffix}")
        for obj in files:
            typer.echo(f"{indent}  FILE  {obj['Key']}  ({obj['Size']} bytes)")
        if depth >= max_depth:
            continue
        for folder in folders:
            if folder not in seen:
                seen.add(folder)
                queue.append((folder, depth + 1))


def print_summary(title: str, objects: list[dict], hit_limit: bool, max_objects: int) -> None:
    ext_counts = Counter(extension_for_key(obj["Key"]) for obj in objects)
    type_counts = Counter(data_type_for_key(obj["Key"]) for obj in objects)
    total_size = sum(obj["Size"] for obj in objects)

    typer.echo(f"\n{title}: {len(objects)} objects, {total_size:,} bytes")
    if hit_limit:
        typer.echo(f"Stopped at limit {max_objects}; prefix may contain more.")

    typer.echo("Data types:")
    for data_type, count in type_counts.most_common():
        typer.echo(f"  {data_type}: {count}")

    typer.echo("Extensions:")
    for ext, count in ext_counts.most_common():
        typer.echo(f"  {ext}: {count}")

    typer.echo("Sample objects:")
    for obj in objects[:15]:
        typer.echo(f"  {obj['Key']}  ({obj['Size']} bytes)")


def print_balanced_recursive_summary(
    s3,
    bucket: str,
    root_prefix: str,
    top_folders: list[str],
    max_objects: int,
    sample_per_folder: int,
) -> None:
    prefixes = top_folders or [root_prefix]
    summaries: list[PrefixSummary] = []
    remaining = max_objects

    for prefix in prefixes:
        if remaining <= 0:
            break
        limit = min(sample_per_folder, remaining)
        summary = summarize_prefix(s3, bucket, prefix, limit)
        summaries.append(summary)
        remaining -= len(summary.objects)
        print_summary(
            f"Recursive sample for {prefix}",
            summary.objects,
            summary.hit_limit,
            limit,
        )

    combined = [obj for summary in summaries for obj in summary.objects]
    print_summary(
        "Combined balanced recursive sample",
        combined,
        len(combined) >= max_objects,
        max_objects,
    )


def main(
    s3_uri: Annotated[str, typer.Argument(help="S3 prefix to inspect, e.g. s3://bucket/path/")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="AWS profile name. Omit to use default AWS resolution."),
    ] = None,
    max_objects: Annotated[
        int,
        typer.Option("--max-objects", help="Maximum recursive objects to sample across all top-level folders."),
    ] = 2_000,
    sample_per_folder: Annotated[
        int,
        typer.Option("--sample-per-folder", help="Maximum recursive objects to sample from each top-level folder."),
    ] = 300,
    tree_depth: Annotated[
        int,
        typer.Option("--tree-depth", help="Depth for the delimiter-based prefix tree sample."),
    ] = 2,
    tree_folders: Annotated[
        int,
        typer.Option("--tree-folders", help="Maximum child folders to show per prefix in the tree sample."),
    ] = 25,
    tree_files: Annotated[
        int,
        typer.Option("--tree-files", help="Maximum direct files to show per prefix in the tree sample."),
    ] = 5,
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--top-level-only", help="Also sample objects recursively by top-level folder."),
    ] = True,
) -> None:
    bucket, prefix = parse_s3_uri(s3_uri)
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    region = get_bucket_region(s3, bucket)
    typer.echo(f"Bucket: {bucket}")
    typer.echo(f"Prefix: {prefix or '[bucket root]'}")
    typer.echo(f"Region: {region}")

    files, folders = list_top_level(s3, bucket, prefix)
    print_top_level(files, folders)
    print_prefix_tree(
        s3,
        bucket,
        prefix,
        max_depth=tree_depth,
        max_folders_per_prefix=tree_folders,
        max_files_per_prefix=tree_files,
    )

    if recursive:
        print_balanced_recursive_summary(
            s3,
            bucket,
            prefix,
            top_folders=folders,
            max_objects=max_objects,
            sample_per_folder=sample_per_folder,
        )


if __name__ == "__main__":
    typer.run(main)
