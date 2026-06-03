# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "typer",
# ]
# ///
"""Create the compact Encord client metadata schema for raw-feed registration.

Run:
    uv run --script scripts/encord/data-registration/create_metadata_schema.py \
      s3://ego-data-collection-encord/raw-feed/ \
      --profile encord-robotics \
      --ssh-key-file /path/to/encord_key \
      --dry-run
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import boto3
import typer
from encord.metadata_schema import MetadataSchemaError
from encord.user_client import EncordUserClient

MAX_ENUM_VALUES = 255

ENUM_FIELDS = {
    "source_family",
    "task_name",
    "environment",
    "file_ext",
    "metadata_file_role",
    "camera_name",
    "sensor_key",
    "robot_type",
    "codebase_version",
    "trossen_subversion",
    "video_codec",
}

SCALAR_FIELDS: dict[str, str] = {
    "collection_datetime": "datetime",
    "has_info_json": "boolean",
    "has_tasks_jsonl": "boolean",
    "has_episodes_jsonl": "boolean",
    "has_episodes_stats_jsonl": "boolean",
    "has_parquet": "boolean",
    "video_has_audio": "boolean",
    "video_width": "number",
    "video_height": "number",
    "collection_fps": "number",
    "state_dim": "number",
    "action_dim": "number",
    "episode_index": "number",
    "source_key": "text",
    "source_uri": "text",
    "episode_path": "text",
    "episode_id": "varchar",
}

STATIC_ENUM_VALUES = {
    "metadata_file_role": {
        "none",
        "info",
        "tasks",
        "episodes",
        "episodes_stats",
        "dataset_metadata",
        "metadata",
    },
    "file_ext": {
        ".avi",
        ".csv",
        ".html",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".m4a",
        ".md",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".txt",
        ".wav",
        ".webm",
        ".xml",
        ".yaml",
        ".yml",
    },
}


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise typer.BadParameter("Use an S3 URI like s3://bucket/prefix/")
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    if not bucket:
        raise typer.BadParameter("S3 URI must include a bucket name")
    return bucket, prefix


def iter_objects(s3, bucket: str, prefix: str, max_objects: int) -> tuple[list[dict], bool]:
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict] = []
    hit_limit = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or PurePosixPath(key).name == ".DS_Store":
                continue
            objects.append(obj)
            if len(objects) >= max_objects:
                hit_limit = True
                return objects, hit_limit
    return objects, hit_limit


def extension_for_key(key: str) -> str:
    return PurePosixPath(key).suffix.lower()


def metadata_file_role(key: str) -> str:
    name = PurePosixPath(key).name.lower()
    if name == "info.json":
        return "info"
    if name == "tasks.jsonl":
        return "tasks"
    if name == "episodes.jsonl":
        return "episodes"
    if name == "episodes_stats.jsonl":
        return "episodes_stats"
    if name == "dataset_metadata.json":
        return "dataset_metadata"
    if name in {"metadata.json", "metadata.yaml", "metadata.yml"}:
        return "metadata"
    return "none"


def source_family_for_key(key: str) -> str | None:
    parts = PurePosixPath(key).parts
    if "raw-feed" in parts:
        idx = parts.index("raw-feed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[0] if parts else None


def parse_path_metadata(key: str) -> dict[str, Any]:
    parts = PurePosixPath(key).parts
    out: dict[str, Any] = {}
    if "raw-feed" not in parts:
        return out
    idx = parts.index("raw-feed")
    if idx + 2 >= len(parts):
        return out

    family = parts[idx + 1]
    if family in {"trossen-data", "trossen-data-stationary"} and idx + 6 < len(parts):
        out["source_family"] = family
        out["task_name"] = parts[idx + 2]
        out["environment"] = parts[idx + 3]
        dt = parse_datetime_token(parts[idx + 5])
        if dt:
            out["collection_datetime"] = dt
    elif family == "egocentric" and idx + 3 < len(parts) and parts[idx + 2] == "Meta-POC":
        out["source_family"] = family
        out["environment"] = parts[idx + 2]
        out["task_name"] = parts[idx + 3]

    for i, part in enumerate(parts[idx + 2 :], start=idx + 2):
        if re.fullmatch(r"episode_\d{6}", part):
            out["episode_id"] = part
            out["episode_index"] = int(part.rsplit("_", 1)[1])
            out["episode_path"] = "/".join(parts[: i + 1]) + "/"
            break

    for part in parts:
        if part.startswith("observation.images."):
            out["sensor_key"] = part
            out["camera_name"] = part.removeprefix("observation.images.")
            break

    return out


def parse_datetime_token(token: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    match = re.search(r"(20\d{12})", token)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            return None
    return None


def read_small_json(s3, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read(2_000_000)
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        typer.echo(f"Warning: could not read {key}: {exc}", err=True)
        return None


def discover_enum_values(s3, bucket: str, objects: list[dict]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for field, field_values in STATIC_ENUM_VALUES.items():
        values[field].update(field_values)

    for obj in objects:
        key = obj["Key"]
        ext = extension_for_key(key)
        if ext:
            values["file_ext"].add(ext)
        role = metadata_file_role(key)
        values["metadata_file_role"].add(role)

        family = source_family_for_key(key)
        if family:
            values["source_family"].add(family)

        path_meta = parse_path_metadata(key)
        for field in ("source_family", "task_name", "environment", "camera_name", "sensor_key"):
            if path_meta.get(field):
                values[field].add(str(path_meta[field]))

        if role == "info":
            info = read_small_json(s3, bucket, key)
            if not info:
                continue
            for field in ("robot_type", "codebase_version", "trossen_subversion"):
                if info.get(field):
                    values[field].add(str(info[field]))
            features = info.get("features") or {}
            for feature_key, feature in features.items():
                if isinstance(feature, dict) and feature.get("dtype") == "video":
                    values["sensor_key"].add(feature_key)
                    values["camera_name"].add(feature_key.removeprefix("observation.images."))
                    codec = (feature.get("info") or {}).get("video.codec")
                    if codec:
                        values["video_codec"].add(str(codec))

    return values


def connect_client(ssh_key_file: str) -> EncordUserClient:
    if not ssh_key_file:
        raise typer.BadParameter("--ssh-key-file is required")
    return EncordUserClient.create_with_ssh_private_key(Path(ssh_key_file).read_text())


def apply_schema(client: EncordUserClient, enum_values: dict[str, set[str]], dry_run: bool) -> None:
    schema = client.metadata_schema()
    changes: list[str] = []

    for field in sorted(ENUM_FIELDS):
        values = sorted(v for v in enum_values.get(field, set()) if v)
        if not values:
            changes.append(f"SKIP enum {field}: no values discovered")
            continue
        if len(values) > MAX_ENUM_VALUES:
            raise typer.BadParameter(
                f"Enum field {field} has {len(values)} values, exceeding Encord limit {MAX_ENUM_VALUES}"
            )

        existing_type = schema.get_key_type(field)
        if existing_type is None:
            changes.append(f"ADD enum {field}: {len(values)} values")
            if not dry_run:
                schema.add_enum(field, values=values)
        elif existing_type != "enum":
            raise MetadataSchemaError(f"{field} exists as {existing_type}, expected enum")
        else:
            existing_values = set(schema.get_enum_options(field))
            missing = sorted(set(values) - existing_values)
            if missing:
                changes.append(f"ADD enum values {field}: {missing}")
                if not dry_run:
                    schema.add_enum_options(field, values=missing)

    for field, data_type in sorted(SCALAR_FIELDS.items()):
        existing_type = schema.get_key_type(field)
        expected_type = "varchar" if data_type == "string" else data_type
        if existing_type is None:
            changes.append(f"ADD scalar {field}: {data_type}")
            if not dry_run:
                schema.add_scalar(field, data_type=data_type)
        elif existing_type != expected_type:
            raise MetadataSchemaError(f"{field} exists as {existing_type}, expected {expected_type}")

    for change in changes:
        typer.echo(change)
    if not changes:
        typer.echo("Schema already up to date.")
    if not dry_run:
        schema.save()
        typer.echo("Saved metadata schema.")


def main(
    s3_uri: Annotated[str, typer.Argument(help="S3 prefix to inspect for enum values.")],
    profile: Annotated[str | None, typer.Option("--profile", "-p", help="AWS profile name.")] = None,
    ssh_key_file: Annotated[str, typer.Option("--ssh-key-file", help="Path to Encord SSH private key.")] = "",
    max_objects: Annotated[int, typer.Option("--max-objects", help="Maximum S3 objects to inspect.")] = 50_000,
    dry_run: Annotated[bool, typer.Option("--dry-run/--apply", help="Print schema changes without saving.")] = True,
) -> None:
    bucket, prefix = parse_s3_uri(s3_uri)
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    objects, hit_limit = iter_objects(s3, bucket, prefix, max_objects)
    if hit_limit:
        typer.echo(f"Warning: stopped at --max-objects={max_objects}; enum discovery may be incomplete.", err=True)

    enum_values = discover_enum_values(s3, bucket, objects)
    typer.echo(f"Inspected {len(objects)} S3 objects.")
    client = connect_client(ssh_key_file)
    apply_schema(client, enum_values, dry_run=dry_run)


if __name__ == "__main__":
    typer.run(main)
