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

from collections import Counter
from pathlib import PurePosixPath
from typing import Annotated

import boto3
import typer


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
        files.extend(page.get("Contents", []))
        folders.extend(item["Prefix"] for item in page.get("CommonPrefixes", []))

    return files, folders


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


def extension_for_key(key: str) -> str:
    suffix = PurePosixPath(key).suffix.lower()
    return suffix or "[no extension]"


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


def print_recursive_summary(objects: list[dict], max_objects: int) -> None:
    ext_counts = Counter(extension_for_key(obj["Key"]) for obj in objects)
    total_size = sum(obj["Size"] for obj in objects)

    typer.echo(f"\nRecursive sample: {len(objects)} objects, {total_size:,} bytes")
    if len(objects) == max_objects:
        typer.echo(f"Stopped at --max-objects={max_objects}; prefix may contain more.")

    typer.echo("\nExtensions:")
    for ext, count in ext_counts.most_common():
        typer.echo(f"  {ext}: {count}")

    typer.echo("\nSample objects:")
    for obj in objects[:30]:
        typer.echo(f"  {obj['Key']}  ({obj['Size']} bytes)")


def main(
    s3_uri: Annotated[str, typer.Argument(help="S3 prefix to inspect, e.g. s3://bucket/path/")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="AWS profile name. Omit to use default AWS resolution."),
    ] = None,
    max_objects: Annotated[
        int,
        typer.Option("--max-objects", help="Maximum recursive objects to sample."),
    ] = 2_000,
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--top-level-only", help="Also sample objects recursively."),
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

    if recursive:
        objects = list(iter_recursive_objects(s3, bucket, prefix, max_objects))
        print_recursive_summary(objects, max_objects)


if __name__ == "__main__":
    typer.run(main)
