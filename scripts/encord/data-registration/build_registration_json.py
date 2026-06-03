# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "pyarrow",
#     "typer",
# ]
# ///
"""Build an Encord registration JSON for an S3 raw-feed prefix.

Run:
    uv run --script scripts/encord/data-registration/build_registration_json.py \
      s3://ego-data-collection-encord/raw-feed/trossen-data/ \
      --profile encord-robotics \
      --output scripts/encord/data-registration/upload_json.dry-run.json \
      --dry-run
"""

from __future__ import annotations

import json
import mimetypes
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import PurePosixPath
from typing import Annotated, Any
from urllib.parse import quote

import boto3
import pyarrow.parquet as pq
import typer

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".3gp", ".3g2", ".mj2", ".avi"}
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".avif", ".bmp", ".tiff", ".tif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".eac3", ".m4a", ".mpeg", ".x-wav"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".html", ".md", ".xml", ".json", ".jsonl", ".yaml", ".yml", ".csv"}
SKIP_EXTENSIONS = {".parquet"}

CATEGORY_BY_EXT = {
    **{ext: "videos" for ext in VIDEO_EXTENSIONS},
    **{ext: "images" for ext in IMAGE_EXTENSIONS},
    **{ext: "audio" for ext in AUDIO_EXTENSIONS},
    **{ext: "pdfs" for ext in PDF_EXTENSIONS},
    **{ext: "text" for ext in TEXT_EXTENSIONS},
}

UPLOAD_KEYS = ["images", "videos", "audio", "text", "pdfs", "image_groups", "scenes", "data_groups"]


@dataclass
class EpisodeContext:
    episode_path: str | None
    has_info_json: bool = False
    has_tasks_jsonl: bool = False
    has_episodes_jsonl: bool = False
    has_episodes_stats_jsonl: bool = False
    has_parquet: bool = False
    info: dict[str, Any] | None = None
    state_dim: int | None = None
    action_dim: int | None = None
    parquet_checked: bool = False
    keys: list[str] = field(default_factory=list)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise typer.BadParameter("Use an S3 URI like s3://bucket/prefix/")
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    if not bucket:
        raise typer.BadParameter("S3 URI must include a bucket name")
    return bucket, prefix


def get_bucket_region(s3, bucket: str) -> str:
    response = s3.head_bucket(Bucket=bucket)
    return response["ResponseMetadata"]["HTTPHeaders"].get("x-amz-bucket-region", "us-east-1")


def extension_for_key(key: str) -> str:
    return PurePosixPath(key).suffix.lower()


def is_folder_marker_or_system_file(key: str) -> bool:
    return key.endswith("/") or PurePosixPath(key).name == ".DS_Store"


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

    family = source_family_for_key(key)
    if family:
        out["source_family"] = family

    if "raw-feed" in parts:
        idx = parts.index("raw-feed")
        family = parts[idx + 1] if idx + 1 < len(parts) else None
        if idx + 6 < len(parts) and family in {"trossen-data", "trossen-data-stationary"}:
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

    for i, part in enumerate(parts):
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

    if "collection_datetime" not in out:
        for part in reversed(parts):
            dt = parse_datetime_token(part)
            if dt:
                out["collection_datetime"] = dt
                break

    return out


def episode_path_for_key(key: str) -> str:
    path_meta = parse_path_metadata(key)
    if path_meta.get("episode_path"):
        return str(path_meta["episode_path"])
    parent = str(PurePosixPath(key).parent)
    return parent + ("/" if parent and parent != "." else "")


def list_all_objects(s3, bucket: str, prefix: str) -> list[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not is_folder_marker_or_system_file(obj["Key"]):
                objects.append(obj)
    return objects


def list_one_level(s3, bucket: str, prefix: str) -> tuple[list[dict], list[str]]:
    paginator = s3.get_paginator("list_objects_v2")
    files: list[dict] = []
    folders: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        files.extend(obj for obj in page.get("Contents", []) if not is_folder_marker_or_system_file(obj["Key"]))
        folders.extend(item["Prefix"] for item in page.get("CommonPrefixes", []))
    return files, folders


def is_episode_prefix(prefix: str) -> bool:
    last = PurePosixPath(prefix.rstrip("/")).name
    return bool(re.fullmatch(r"episode_\d{6}", last))


def dry_run_objects(
    s3,
    bucket: str,
    prefix: str,
    max_episodes: int,
    max_prefixes: int,
) -> list[dict]:
    root_files, root_folders = list_one_level(s3, bucket, prefix)
    seed_prefixes = root_folders or [prefix]
    direct_root_files = [
        obj
        for obj in root_files
        if category_for_key(obj["Key"]) and extension_for_key(obj["Key"]) not in SKIP_EXTENSIONS
    ]
    visited = 1
    sampled_episode_paths: set[str] = set()
    sampled_tasks: set[str] = set()
    selected: list[dict] = []

    def add_leaf_sample(sample_prefix: str, files: list[dict]) -> bool:
        registerable = [
            obj
            for obj in files
            if category_for_key(obj["Key"]) and extension_for_key(obj["Key"]) not in SKIP_EXTENSIONS
        ]
        if not registerable:
            return False
        path_meta = parse_path_metadata(registerable[0]["Key"])
        task_key = str(path_meta.get("task_name") or sample_prefix)
        if task_key in sampled_tasks:
            return False
        selected.extend(registerable[:3])
        sampled_tasks.add(task_key)
        sampled_episode_paths.add(sample_prefix)
        return True

    for seed in seed_prefixes:
        if len(sampled_episode_paths) >= max_episodes or visited >= max_prefixes:
            break
        queue = deque([seed])
        while queue and len(sampled_episode_paths) < max_episodes and visited < max_prefixes:
            current = queue.popleft()
            visited += 1
            files, folders = list_one_level(s3, bucket, current)

            if is_episode_prefix(current):
                path_meta = parse_path_metadata(current.rstrip("/"))
                task_name = str(path_meta.get("task_name") or current)
                if task_name not in sampled_tasks:
                    episode_objects = list_all_objects(s3, bucket, current)
                    if episode_objects:
                        selected.extend(episode_objects)
                        sampled_tasks.add(task_name)
                        sampled_episode_paths.add(current)
                break

            if files and add_leaf_sample(current, files):
                break

            queue.extend(folders)

    if not selected:
        selected = direct_root_files[: max_episodes * 3] or list_all_objects(s3, bucket, prefix)[: max_episodes * 10]

    typer.echo(
        f"Dry run selected {len(selected)} objects from {len(sampled_episode_paths)} sampled prefixes "
        f"after visiting {visited} prefixes.",
        err=True,
    )
    return selected


def category_for_key(key: str) -> str | None:
    return CATEGORY_BY_EXT.get(extension_for_key(key))


def object_url(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{quote(key, safe='/-_.~')}"


def read_small_text(s3, bucket: str, key: str, max_bytes: int = 2_000_000) -> str | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read(max_bytes)
        return body.decode("utf-8")
    except Exception as exc:
        typer.echo(f"Warning: could not read {key}: {exc}", err=True)
        return None


def read_info_json(s3, bucket: str, key: str) -> dict[str, Any] | None:
    text = read_small_text(s3, bucket, key)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        typer.echo(f"Warning: invalid JSON in {key}: {exc}", err=True)
        return None


def read_parquet_dims(s3, bucket: str, key: str) -> tuple[int | None, int | None]:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        table = pq.read_table(BytesIO(body), columns=["action", "observation.state"])
        data = table.to_pydict()
        action = data.get("action") or []
        state = data.get("observation.state") or []
        action_dim = len(action[0]) if action else None
        state_dim = len(state[0]) if state else None
        return action_dim, state_dim
    except Exception as exc:
        typer.echo(f"Warning: could not inspect parquet {key}: {exc}", err=True)
        return None, None


def build_contexts(s3, bucket: str, objects: list[dict]) -> dict[str, EpisodeContext]:
    contexts: dict[str, EpisodeContext] = {}
    for obj in objects:
        key = obj["Key"]
        episode_path = episode_path_for_key(key)
        ctx = contexts.setdefault(episode_path, EpisodeContext(episode_path=episode_path))
        ctx.keys.append(key)
        role = metadata_file_role(key)
        ext = extension_for_key(key)
        ctx.has_info_json = ctx.has_info_json or role == "info"
        ctx.has_tasks_jsonl = ctx.has_tasks_jsonl or role == "tasks"
        ctx.has_episodes_jsonl = ctx.has_episodes_jsonl or role == "episodes"
        ctx.has_episodes_stats_jsonl = ctx.has_episodes_stats_jsonl or role == "episodes_stats"
        ctx.has_parquet = ctx.has_parquet or ext == ".parquet"

    for ctx in contexts.values():
        info_key = next((key for key in ctx.keys if metadata_file_role(key) == "info"), None)
        if info_key:
            ctx.info = read_info_json(s3, bucket, info_key)
        parquet_key = next((key for key in ctx.keys if extension_for_key(key) == ".parquet"), None)
        if parquet_key:
            action_dim, state_dim = read_parquet_dims(s3, bucket, parquet_key)
            ctx.action_dim = action_dim
            ctx.state_dim = state_dim
            ctx.parquet_checked = True
    return contexts


def add_if_present(metadata: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    metadata[key] = value


def metadata_for_object(bucket: str, key: str, size: int, ctx: EpisodeContext) -> dict[str, Any]:
    path_meta = parse_path_metadata(key)
    ext = extension_for_key(key)
    role = metadata_file_role(key)
    metadata: dict[str, Any] = {
        "source_key": key,
        "source_uri": f"s3://{bucket}/{key}",
        "file_ext": ext,
        "metadata_file_role": role,
        "has_info_json": ctx.has_info_json,
        "has_tasks_jsonl": ctx.has_tasks_jsonl,
        "has_episodes_jsonl": ctx.has_episodes_jsonl,
        "has_episodes_stats_jsonl": ctx.has_episodes_stats_jsonl,
        "has_parquet": ctx.has_parquet,
    }

    for field in (
        "source_family",
        "task_name",
        "environment",
        "collection_datetime",
        "episode_id",
        "episode_index",
        "episode_path",
        "camera_name",
        "sensor_key",
    ):
        add_if_present(metadata, field, path_meta.get(field))

    info = ctx.info or {}
    add_if_present(metadata, "robot_type", info.get("robot_type"))
    add_if_present(metadata, "codebase_version", info.get("codebase_version"))
    add_if_present(metadata, "trossen_subversion", info.get("trossen_subversion"))
    add_if_present(metadata, "collection_fps", info.get("fps"))
    add_if_present(metadata, "state_dim", ctx.state_dim)
    add_if_present(metadata, "action_dim", ctx.action_dim)

    sensor_key = path_meta.get("sensor_key")
    if sensor_key:
        feature = (info.get("features") or {}).get(sensor_key) or {}
        video_info = feature.get("info") or {}
        add_if_present(metadata, "video_width", video_info.get("video.width"))
        add_if_present(metadata, "video_height", video_info.get("video.height"))
        add_if_present(metadata, "video_codec", video_info.get("video.codec"))
        if "has_audio" in video_info:
            metadata["video_has_audio"] = bool(video_info["has_audio"])

    return metadata


def build_item(bucket: str, region: str, obj: dict, ctx: EpisodeContext) -> tuple[str, dict] | None:
    key = obj["Key"]
    category = category_for_key(key)
    if not category or extension_for_key(key) in SKIP_EXTENSIONS:
        return None

    item = {
        "objectUrl": object_url(bucket, region, key),
        "title": key,
        "clientMetadata": metadata_for_object(bucket, key, obj["Size"], ctx),
    }
    if category == "text":
        mime_type = mimetypes.guess_type(key)[0]
        if mime_type:
            item["textMetadata"] = {"fileSize": obj["Size"], "mime_type": mime_type}
    return category, item


def build_upload_json(
    bucket: str,
    region: str,
    objects: list[dict],
    contexts: dict[str, EpisodeContext],
) -> tuple[dict, dict[str, int]]:
    upload_json: dict[str, Any] = {key: [] for key in UPLOAD_KEYS}
    skipped: defaultdict[str, int] = defaultdict(int)

    for obj in sorted(objects, key=lambda x: x["Key"]):
        key = obj["Key"]
        ext = extension_for_key(key)
        if ext in SKIP_EXTENSIONS:
            skipped[ext] += 1
            continue
        ctx = contexts[episode_path_for_key(key)]
        built = build_item(bucket, region, obj, ctx)
        if built is None:
            skipped[ext or "[no extension]"] += 1
            continue
        category, item = built
        upload_json[category].append(item)

    upload_json["skip_duplicate_urls"] = True
    return upload_json, dict(sorted(skipped.items()))


def main(
    s3_uri: Annotated[str, typer.Argument(help="S3 prefix to register.")],
    profile: Annotated[str | None, typer.Option("--profile", "-p", help="AWS profile name.")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output upload JSON path.")] = "./registration.json",
    dry_run: Annotated[bool, typer.Option("--dry-run/--full", help="Generate a small representative JSON.")] = False,
    dry_run_max_episodes: Annotated[
        int,
        typer.Option("--dry-run-max-episodes", help="Maximum sampled episode prefixes in dry-run mode."),
    ] = 12,
    dry_run_max_prefixes: Annotated[
        int,
        typer.Option("--dry-run-max-prefixes", help="Maximum prefixes to visit while finding dry-run samples."),
    ] = 2_000,
) -> None:
    bucket, prefix = parse_s3_uri(s3_uri)
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")
    region = get_bucket_region(s3, bucket)

    objects = (
        dry_run_objects(s3, bucket, prefix, dry_run_max_episodes, dry_run_max_prefixes)
        if dry_run
        else list_all_objects(s3, bucket, prefix)
    )
    contexts = build_contexts(s3, bucket, objects)
    upload_json, skipped_counts = build_upload_json(bucket, region, objects, contexts)

    output_path = PurePosixPath(output)
    with open(output_path, "w") as f:
        json.dump(upload_json, f, indent=2)

    counts = {key: len(upload_json[key]) for key in UPLOAD_KEYS if upload_json[key]}
    typer.echo(f"Wrote {output}")
    typer.echo(f"Registered item counts: {counts}")
    if skipped_counts:
        typer.echo(f"Skipped counts: {skipped_counts}")


if __name__ == "__main__":
    typer.run(main)
