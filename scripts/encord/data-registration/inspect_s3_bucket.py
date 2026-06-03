# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "typer",
# ]
# ///
"""Explore an S3 prefix before building an Encord registration JSON.

Run:
    uv run --script scripts/encord/data-registration/inspect_s3_bucket.py s3://bucket/path/ --profile default
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

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


def serialize_object(obj: dict) -> dict[str, Any]:
    last_modified = obj.get("LastModified")
    return {
        "key": obj["Key"],
        "size_bytes": obj["Size"],
        "extension": extension_for_key(obj["Key"]),
        "data_type": data_type_for_key(obj["Key"]),
        "last_modified": last_modified.isoformat() if last_modified else None,
    }


def summarize_objects(objects: list[dict], hit_limit: bool, max_objects: int) -> dict[str, Any]:
    ext_counts = Counter(extension_for_key(obj["Key"]) for obj in objects)
    type_counts = Counter(data_type_for_key(obj["Key"]) for obj in objects)

    return {
        "sampled_object_count": len(objects),
        "sampled_total_size_bytes": sum(obj["Size"] for obj in objects),
        "hit_limit": hit_limit,
        "limit": max_objects,
        "data_type_counts": dict(type_counts.most_common()),
        "extension_counts": dict(ext_counts.most_common()),
        "sample_objects": [serialize_object(obj) for obj in objects[:15]],
    }


def build_prefix_tree(
    s3,
    bucket: str,
    root_prefix: str,
    max_depth: int,
    max_folders_per_prefix: int,
    max_files_per_prefix: int,
) -> dict[str, Any]:
    def build_node(prefix: str, depth: int) -> dict[str, Any]:
        files, folders, truncated = list_one_level(
            s3,
            bucket,
            prefix,
            max_folders=max_folders_per_prefix,
            max_files=max_files_per_prefix,
        )
        node = {
            "prefix": prefix,
            "depth": depth,
            "truncated": truncated,
            "sampled_direct_files": [serialize_object(obj) for obj in files],
            "child_prefixes_sampled": folders,
            "children": [],
        }
        if depth < max_depth:
            node["children"] = [build_node(folder, depth + 1) for folder in folders]
        return node

    return build_node(root_prefix, 0)


def build_balanced_recursive_summaries(
    s3,
    bucket: str,
    root_prefix: str,
    top_folders: list[str],
    max_objects: int,
    sample_per_folder: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefixes = top_folders or [root_prefix]
    summaries: list[dict[str, Any]] = []
    combined_objects: list[dict] = []
    remaining = max_objects

    for prefix in prefixes:
        if remaining <= 0:
            break
        limit = min(sample_per_folder, remaining)
        summary = summarize_prefix(s3, bucket, prefix, limit)
        combined_objects.extend(summary.objects)
        remaining -= len(summary.objects)
        summaries.append(
            {
                "prefix": prefix,
                **summarize_objects(summary.objects, summary.hit_limit, limit),
            }
        )

    combined_summary = summarize_objects(
        combined_objects,
        len(combined_objects) >= max_objects,
        max_objects,
    )
    return summaries, combined_summary


def build_inspection_json(
    s3,
    bucket: str,
    prefix: str,
    region: str,
    files: list[dict],
    folders: list[str],
    max_objects: int,
    sample_per_folder: int,
    tree_depth: int,
    tree_folders: int,
    tree_files: int,
    recursive: bool,
) -> dict[str, Any]:
    recursive_summaries: list[dict[str, Any]] = []
    combined_summary: dict[str, Any] | None = None
    if recursive:
        recursive_summaries, combined_summary = build_balanced_recursive_summaries(
            s3,
            bucket,
            prefix,
            folders,
            max_objects=max_objects,
            sample_per_folder=sample_per_folder,
        )

    return {
        "s3_uri": f"s3://{bucket}/{prefix}",
        "bucket": bucket,
        "prefix": prefix,
        "region": region,
        "limits": {
            "max_objects": max_objects,
            "sample_per_folder": sample_per_folder,
            "tree_depth": tree_depth,
            "tree_folders_per_prefix": tree_folders,
            "tree_files_per_prefix": tree_files,
            "recursive": recursive,
        },
        "top_level": {
            "folders": folders,
            "files": [serialize_object(obj) for obj in files],
        },
        "directory_tree_sample": build_prefix_tree(
            s3,
            bucket,
            prefix,
            max_depth=tree_depth,
            max_folders_per_prefix=tree_folders,
            max_files_per_prefix=tree_files,
        ),
        "recursive_samples": recursive_summaries,
        "combined_recursive_sample": combined_summary,
        "notes": [
            "The tree uses delimiter-based S3 prefix listings.",
            "Nodes marked truncated contain additional files or folders beyond the configured sample limits.",
            "Recursive samples are balanced by top-level prefix to avoid committing repetitive full-bucket listings.",
        ],
    }


def write_inspection_json(output: Path, inspection: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inspection, indent=2) + "\n")


def default_output_path(bucket: str, prefix: str) -> Path:
    name = prefix.rstrip("/") or bucket
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return Path(__file__).with_name(f"{slug}_s3_structure.json")


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
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the sampled bucket structure to this JSON file."),
    ] = None,
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

    output_path = output or default_output_path(bucket, prefix)
    inspection = build_inspection_json(
        s3,
        bucket,
        prefix,
        region,
        files,
        folders,
        max_objects=max_objects,
        sample_per_folder=sample_per_folder,
        tree_depth=tree_depth,
        tree_folders=tree_folders,
        tree_files=tree_files,
        recursive=recursive,
    )
    write_inspection_json(output_path, inspection)
    typer.echo(f"\nWrote JSON: {output_path}")


if __name__ == "__main__":
    typer.run(main)
