# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "pyarrow",
#     "typer",
# ]
# ///
"""Export Encord caption labels into a DROID-shaped LeRobot dataset.

The script never mutates the source dataset. Output metadata files are written
only when missing; existing metadata files are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Annotated, Any

import pyarrow as pa
import pyarrow.parquet as pq
import typer


LANG_KEYS = [
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
]

DEFAULT_CAMERA_MAP = {
    "observation.images.cam_high": "observation.images.exterior_image_1_left",
    "observation.images.cam_left_wrist": "observation.images.wrist_image_left",
    "observation.images.cam_right_wrist": "observation.images.exterior_image_2_left",
}


@dataclass(frozen=True)
class SourceEpisode:
    dataset_root: Path
    source_episode_index: int
    source_episode_chunk: int
    parquet_path: Path
    info: dict[str, Any]
    episode_meta: dict[str, Any]
    video_keys: list[str]

    @property
    def candidates(self) -> set[str]:
        stem = self.parquet_path.stem
        idx = self.source_episode_index
        values = {
            self.dataset_root.name,
            stem,
            f"episode_{idx:06d}",
            f"{idx:06d}",
            str(idx),
            self.parquet_path.name,
        }
        for task in self.episode_meta.get("tasks") or []:
            if task:
                values.add(str(task))
        return {v for v in values if v}


@dataclass
class LabelRecord:
    match_value: str | None
    captions: list[str]
    raw: dict[str, Any]


def parse_json_object(raw: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return default
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise typer.BadParameter("Expected a JSON object")
    return value


def parse_path_list(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def iter_strings(data: Any) -> list[str]:
    values: list[str] = []
    if isinstance(data, str):
        if data.strip():
            values.append(data.strip())
    elif isinstance(data, dict):
        for value in data.values():
            values.extend(iter_strings(value))
    elif isinstance(data, list):
        for item in data:
            values.extend(iter_strings(item))
    return values


def find_first_string_by_key(data: Any, key_names: set[str]) -> str | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in key_names:
                strings = iter_strings(value)
                if strings:
                    return strings[0]
        for value in data.values():
            found = find_first_string_by_key(value, key_names)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_first_string_by_key(item, key_names)
            if found:
                return found
    return None


def load_label_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("labels", "label_rows", "data_units"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [item for item in value.values() if isinstance(item, dict)]
        return [data]
    raise typer.BadParameter(f"Unsupported label JSON shape in {path}")


def export_encord_labels(ssh_key_file: Path, project_hash: str) -> list[dict[str, Any]]:
    from encord.user_client import EncordUserClient

    client = EncordUserClient.create_with_ssh_private_key(ssh_key_file.read_text())
    project = client.get_project(project_hash)
    label_rows = list(project.list_label_rows_v2())
    if label_rows:
        with project.create_bundle(bundle_size=min(100, len(label_rows))) as bundle:
            for label_row in label_rows:
                label_row.initialise_labels(bundle=bundle)
    return [label_row.to_encord_dict() for label_row in label_rows]


def build_label_records(
    rows: list[dict[str, Any]],
    caption_paths: list[str],
    match_path: str | None,
    caption_key_names: set[str],
    match_key_names: set[str],
) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    for row in rows:
        captions: list[str] = []
        for path in caption_paths:
            value = get_path(row, path)
            strings = iter_strings(value)
            if strings:
                captions.append(strings[0])
        if not captions:
            found = find_first_string_by_key(row, caption_key_names)
            if found:
                captions.append(found)

        match_value = None
        if match_path:
            value = get_path(row, match_path)
            strings = iter_strings(value)
            match_value = strings[0] if strings else None
        if match_value is None:
            match_value = find_first_string_by_key(row, match_key_names)

        records.append(LabelRecord(match_value=match_value, captions=captions, raw=row))
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def discover_dataset_roots(source_root: Path) -> list[Path]:
    if (source_root / "meta" / "info.json").exists():
        return [source_root]
    roots = []
    for info_path in sorted(source_root.glob("**/meta/info.json")):
        roots.append(info_path.parent.parent)
    return roots


def discover_source_episodes(source_root: Path) -> list[SourceEpisode]:
    episodes: list[SourceEpisode] = []
    for root in discover_dataset_roots(source_root):
        info = json.loads((root / "meta" / "info.json").read_text())
        chunks_size = int(info.get("chunks_size", 1000))
        data_pattern = info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        video_keys = [
            key for key, feature in (info.get("features") or {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]
        episode_rows = read_jsonl(root / "meta" / "episodes.jsonl")
        if episode_rows:
            candidate_indices = [int(row["episode_index"]) for row in episode_rows]
            meta_by_index = {int(row["episode_index"]): row for row in episode_rows}
        else:
            candidate_indices = []
            meta_by_index = {}
            for parquet in sorted(root.glob("data/chunk-*/episode_*.parquet")):
                candidate_indices.append(int(parquet.stem.replace("episode_", "")))

        for ep_idx in candidate_indices:
            chunk = ep_idx // chunks_size
            parquet_path = root / data_pattern.format(
                episode_chunk=chunk, episode_index=ep_idx
            )
            if not parquet_path.exists():
                continue
            table = pq.read_table(parquet_path)
            meta = meta_by_index.get(ep_idx, {
                "episode_index": ep_idx,
                "tasks": [],
                "length": table.num_rows,
            })
            episodes.append(SourceEpisode(
                dataset_root=root,
                source_episode_index=ep_idx,
                source_episode_chunk=chunk,
                parquet_path=parquet_path,
                info=info,
                episode_meta=meta,
                video_keys=video_keys,
            ))
    return sorted(episodes, key=lambda ep: (ep.dataset_root.name, ep.source_episode_index))


def match_label(episode: SourceEpisode, records: list[LabelRecord]) -> LabelRecord | None:
    candidates = episode.candidates
    for record in records:
        if not record.match_value:
            continue
        match = record.match_value.strip()
        if match in candidates:
            return record
        if any(candidate and candidate in match for candidate in candidates if len(candidate) >= 6):
            return record
        if any(match and match in candidate for candidate in candidates if len(match) >= 6):
            return record
    return None


def set_or_append_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        index = table.schema.get_field_index(name)
        return table.set_column(index, name, array)
    return table.append_column(name, array)


def rewrite_episode_parquet(
    source: SourceEpisode,
    output_path: Path,
    new_episode_index: int,
    global_start_index: int,
    task_id: int,
) -> int:
    table = pq.read_table(source.parquet_path)
    rows = table.num_rows
    table = set_or_append_column(
        table, "episode_index", pa.array([new_episode_index] * rows, type=pa.int64())
    )
    table = set_or_append_column(
        table, "frame_index", pa.array(list(range(rows)), type=pa.int64())
    )
    table = set_or_append_column(
        table, "index", pa.array(list(range(global_start_index, global_start_index + rows)), type=pa.int64())
    )
    table = set_or_append_column(
        table, "task_index", pa.array([task_id] * rows, type=pa.int64())
    )
    for key in LANG_KEYS:
        table = set_or_append_column(table, key, pa.array([task_id] * rows, type=pa.int64()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return rows


def copy_episode_videos(
    source: SourceEpisode,
    output_root: Path,
    new_episode_index: int,
    camera_map: dict[str, str],
    overwrite_data: bool,
) -> list[tuple[str, str]]:
    video_pattern = source.info.get(
        "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    output_video_keys: list[tuple[str, str]] = []
    for source_key in source.video_keys:
        dest_key = camera_map.get(source_key, source_key)
        source_video = source.dataset_root / video_pattern.format(
            episode_chunk=source.source_episode_chunk,
            video_key=source_key,
            episode_index=source.source_episode_index,
        )
        if not source_video.exists():
            continue
        dest_video = output_root / video_pattern.format(
            episode_chunk=new_episode_index // 1000,
            video_key=dest_key,
            episode_index=new_episode_index,
        )
        if dest_video.exists() and not overwrite_data:
            raise FileExistsError(f"Refusing to overwrite video: {dest_video}")
        dest_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_video, dest_video)
        output_video_keys.append((source_key, dest_key))
    return output_video_keys


def feature_for_column(source_info: dict[str, Any], column: str) -> dict[str, Any] | None:
    features = source_info.get("features") or {}
    feature = features.get(column)
    return feature if isinstance(feature, dict) else None


def infer_vector_width(table: pa.Table, column: str) -> int:
    if column not in table.column_names or table.num_rows == 0:
        return 0
    value = table[column][0].as_py()
    return len(value) if isinstance(value, list) else 1


def build_default_split(table: pa.Table, column: str, name: str) -> dict[str, list[int]]:
    width = infer_vector_width(table, column)
    return {name: [0, width]} if width else {}


def write_text_if_missing(path: Path, text: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def build_modality_json(
    first_table: pa.Table,
    source_info: dict[str, Any],
    video_keys: list[str],
    state_split: dict[str, list[int]],
    action_split: dict[str, list[int]],
) -> dict[str, Any]:
    state_feature = feature_for_column(source_info, "observation.state") or {}
    action_feature = feature_for_column(source_info, "action") or {}
    state_dtype = state_feature.get("dtype", "float32")
    action_dtype = action_feature.get("dtype", "float32")

    if not state_split:
        state_split = build_default_split(first_table, "observation.state", "state")
    if not action_split:
        action_split = build_default_split(first_table, "action", "action")

    return {
        "state": {
            key: {
                "original_key": "observation.state",
                "start": bounds[0],
                "end": bounds[1],
                "rotation_type": None,
                "absolute": True,
                "dtype": state_dtype,
                "range": None,
            }
            for key, bounds in state_split.items()
        },
        "action": {
            key: {
                "original_key": "action",
                "start": bounds[0],
                "end": bounds[1],
                "rotation_type": None,
                "absolute": True,
                "dtype": action_dtype,
                "range": None,
            }
            for key, bounds in action_split.items()
        },
        "video": {
            key.replace("observation.images.", ""): {"original_key": key}
            for key in sorted(set(video_keys))
        },
        "annotation": {
            "language.language_instruction": {"original_key": "annotation.language.language_instruction"},
            "language.language_instruction_2": {"original_key": "annotation.language.language_instruction_2"},
            "language.language_instruction_3": {"original_key": "annotation.language.language_instruction_3"},
        },
    }


def build_info_json(
    source_info: dict[str, Any],
    output_episodes: list[dict[str, Any]],
    video_features: dict[str, dict[str, Any]],
    first_table: pa.Table,
) -> dict[str, Any]:
    features = dict(source_info.get("features") or {})
    for key in list(features):
        if key.startswith("observation.images."):
            features.pop(key)
    for key, feature in sorted(video_features.items()):
        features[key] = feature
    for key in LANG_KEYS:
        features[key] = {"dtype": "int64", "shape": [1]}
    if "task_index" in first_table.column_names:
        features["task_index"] = {"dtype": "int64", "shape": [1]}
    if "episode_index" in first_table.column_names:
        features["episode_index"] = {"dtype": "int64", "shape": [1]}
    if "frame_index" in first_table.column_names:
        features["frame_index"] = {"dtype": "int64", "shape": [1]}
    if "index" in first_table.column_names:
        features["index"] = {"dtype": "int64", "shape": [1]}

    info = dict(source_info)
    info["codebase_version"] = source_info.get("codebase_version", "v2.0")
    info["total_episodes"] = len(output_episodes)
    info["total_frames"] = int(sum(ep["length"] for ep in output_episodes))
    info["total_tasks"] = len({task for ep in output_episodes for task in ep.get("tasks", [])})
    info["total_videos"] = len(video_features)
    info["total_chunks"] = (len(output_episodes) // 1000) + (1 if len(output_episodes) % 1000 else 0)
    info["chunks_size"] = 1000
    info["splits"] = {"train": f"0:{len(output_episodes)}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["features"] = features
    return info


def write_metadata_if_missing(
    output_root: Path,
    source_info: dict[str, Any],
    output_episodes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    video_features: dict[str, dict[str, Any]],
    first_table: pa.Table,
    state_split: dict[str, list[int]],
    action_split: dict[str, list[int]],
) -> list[str]:
    written: list[str] = []
    meta_dir = output_root / "meta"

    tasks_text = "".join(json.dumps(row) + "\n" for row in tasks)
    if write_text_if_missing(meta_dir / "tasks.jsonl", tasks_text):
        written.append("meta/tasks.jsonl")

    episodes_text = "".join(json.dumps(row) + "\n" for row in output_episodes)
    if write_text_if_missing(meta_dir / "episodes.jsonl", episodes_text):
        written.append("meta/episodes.jsonl")

    info = build_info_json(source_info, output_episodes, video_features, first_table)
    if write_text_if_missing(meta_dir / "info.json", json.dumps(info, indent=4) + "\n"):
        written.append("meta/info.json")

    modality = build_modality_json(first_table, source_info, sorted(video_features), state_split, action_split)
    if write_text_if_missing(meta_dir / "modality.json", json.dumps(modality, indent=4) + "\n"):
        written.append("meta/modality.json")

    manifest = {
        "source": "encord_caption_export",
        "metadata_policy": "source metadata untouched; existing output metadata not overwritten",
        "episodes": output_episodes,
    }
    if write_text_if_missing(output_root / "encord_droid_export_manifest.json", json.dumps(manifest, indent=2) + "\n"):
        written.append("encord_droid_export_manifest.json")

    return written


def main(
    source_root: Annotated[Path, typer.Argument(help="LeRobot-like source dataset root or parent directory.")],
    output_dir: Annotated[Path, typer.Argument(help="Output DROID-shaped LeRobot dataset directory.")],
    labels_json: Annotated[Path | None, typer.Option(help="Local Encord label-row JSON for offline export.")] = None,
    ssh_key_file: Annotated[Path | None, typer.Option(help="Encord SSH private key file.")] = None,
    project_hash: Annotated[str | None, typer.Option(help="Encord project hash.")] = None,
    caption_paths: Annotated[str | None, typer.Option(help="Comma-separated dot paths for caption fields.")] = None,
    caption_key_names: Annotated[str, typer.Option(help="Fallback comma-separated field names for captions.")] = "video_description,Video Description,caption,description",
    match_path: Annotated[str | None, typer.Option(help="Dot path for the Encord value used to match source episodes.")] = None,
    match_key_names: Annotated[str, typer.Option(help="Fallback comma-separated field names for matching.")] = "data_title,title,name,episode_id,storage_item_name",
    camera_map_json: Annotated[str | None, typer.Option(help="JSON object mapping source video keys to output video keys.")] = None,
    state_split_json: Annotated[str | None, typer.Option(help='JSON object like {"joint_position":[0,14],"gripper_position":[14,16]}.')] = None,
    action_split_json: Annotated[str | None, typer.Option(help='JSON object like {"joint_position":[0,14],"gripper_position":[14,16]}.')] = None,
    generate_metadata: Annotated[bool, typer.Option(help="Generate missing output metadata files without overwriting existing files.")] = True,
    overwrite_data: Annotated[bool, typer.Option(help="Allow overwriting output parquet/video files.")] = False,
) -> None:
    if labels_json is None and (ssh_key_file is None or project_hash is None):
        raise typer.BadParameter("Pass either --labels-json or both --ssh-key-file and --project-hash")

    camera_map = parse_json_object(camera_map_json, DEFAULT_CAMERA_MAP)
    state_split = parse_json_object(state_split_json, {})
    action_split = parse_json_object(action_split_json, {})
    caption_path_list = parse_path_list(caption_paths)

    label_rows = load_label_json(labels_json) if labels_json else export_encord_labels(ssh_key_file, project_hash)  # type: ignore[arg-type]
    records = build_label_records(
        rows=label_rows,
        caption_paths=caption_path_list,
        match_path=match_path,
        caption_key_names={name.lower() for name in parse_path_list(caption_key_names)},
        match_key_names={name.lower() for name in parse_path_list(match_key_names)},
    )

    source_episodes = discover_source_episodes(source_root)
    if not source_episodes:
        raise typer.BadParameter(f"No source episodes found under {source_root}")

    task_to_id: dict[str, int] = {"not provided": 0}
    output_episodes: list[dict[str, Any]] = []
    output_video_features: dict[str, dict[str, Any]] = {}
    total_rows = 0
    first_output_table: pa.Table | None = None

    for new_idx, source in enumerate(source_episodes):
        record = match_label(source, records)
        caption = record.captions[0] if record and record.captions else "not provided"
        if caption not in task_to_id:
            task_to_id[caption] = len(task_to_id)
        task_id = task_to_id[caption]

        chunk = new_idx // 1000
        output_parquet = output_dir / f"data/chunk-{chunk:03d}/episode_{new_idx:06d}.parquet"
        if output_parquet.exists() and not overwrite_data:
            raise FileExistsError(f"Refusing to overwrite parquet: {output_parquet}")
        length = rewrite_episode_parquet(source, output_parquet, new_idx, total_rows, task_id)
        total_rows += length
        if first_output_table is None:
            first_output_table = pq.read_table(output_parquet)

        copied_videos = copy_episode_videos(source, output_dir, new_idx, camera_map, overwrite_data)
        for source_key, dest_key in copied_videos:
            source_feature = feature_for_column(source.info, source_key) or {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": source.info.get("fps"),
                    "video.codec": "unknown",
                    "video.pix_fmt": "unknown",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            output_video_features.setdefault(dest_key, dict(source_feature))
        output_episodes.append({
            "episode_index": new_idx,
            "tasks": [] if caption == "not provided" else [caption],
            "length": length,
            "source_dataset": source.dataset_root.name,
            "source_episode_index": source.source_episode_index,
            "caption_matched": bool(record and record.captions),
        })

    tasks = [
        {"task_index": task_id, "task": task}
        for task, task_id in sorted(task_to_id.items(), key=lambda item: item[1])
    ]

    written_metadata: list[str] = []
    if generate_metadata:
        assert first_output_table is not None
        written_metadata = write_metadata_if_missing(
            output_root=output_dir,
            source_info=source_episodes[0].info,
            output_episodes=output_episodes,
            tasks=tasks,
            video_features=output_video_features,
            first_table=first_output_table,
            state_split=state_split,
            action_split=action_split,
        )

    typer.echo(f"Exported {len(output_episodes)} episodes to {output_dir}")
    typer.echo(f"Tasks: {len(tasks)}")
    typer.echo(f"Metadata written: {', '.join(written_metadata) if written_metadata else 'none'}")


if __name__ == "__main__":
    typer.run(main)
