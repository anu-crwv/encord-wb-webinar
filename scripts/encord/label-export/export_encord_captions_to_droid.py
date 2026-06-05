# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord @ git+ssh://git@github.com/encord-team/encord-client-python-private.git@b1edece2",
#     "pyarrow",
#     "typer",
# ]
# ///
"""Export Encord video captions into a DROID-shaped LeRobot dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
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

DROID_VIDEO_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]

SOURCE_TO_DROID = {
    "observation.images.cam_high": "observation.images.exterior_image_1_left",
    "observation.images.cam_right_wrist": "observation.images.exterior_image_2_left",
    "observation.images.cam_left_wrist": "observation.images.wrist_image_left",
}

CAPTION_KEYS = {
    "language instruction",
    "language_instruction",
    "video description",
    "video_description",
    "caption",
    "description",
}

MATCH_KEYS = {
    "data_title",
    "data title",
    "title",
    "name",
    "episode_id",
    "episode id",
    "storage_item_name",
}


@dataclass(frozen=True)
class Episode:
    root: Path
    index: int
    parquet_path: Path
    info: dict[str, Any]
    meta: dict[str, Any]
    video_keys: list[str]

    @property
    def chunk(self) -> int:
        return self.index // int(self.info.get("chunks_size", 1000))

    @property
    def match_candidates(self) -> set[str]:
        pattern = self.info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        values = {
            self.root.name,
            self.parquet_path.name,
            self.parquet_path.stem,
            f"episode_{self.index:06d}",
            f"{self.index:06d}",
            str(self.index),
        }
        for video_key in self.video_keys:
            rel = Path(pattern.format(
                episode_chunk=self.chunk,
                video_key=video_key,
                episode_index=self.index,
            ))
            values.update({video_key, rel.name, rel.stem, rel.as_posix(), (self.root / rel).as_posix()})
        return {value for value in values if value}


@dataclass(frozen=True)
class Caption:
    match_value: str | None
    text: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strings_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(strings_from(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(strings_from(item))
        return strings
    return []


def find_string_by_key(value: Any, keys: set[str]) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace("_", " ") in keys or key.lower() in keys:
                strings = strings_from(item)
                if strings:
                    return strings[0]
        for item in value.values():
            found = find_string_by_key(item, keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_string_by_key(item, keys)
            if found:
                return found
    return None


def load_label_rows(labels_json: Path | None, project_hash: str | None) -> list[dict[str, Any]]:
    if labels_json:
        data = json.loads(labels_json.read_text())
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("label_rows", "labels", "data_units"):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
                if isinstance(rows, dict):
                    return [row for row in rows.values() if isinstance(row, dict)]
            return [data]
        raise typer.BadParameter(f"Unsupported label JSON shape: {labels_json}")

    if not project_hash:
        raise typer.BadParameter("Pass --project-hash or --labels-json")

    ssh_key_file = os.environ.get("ENCORD_SSH_KEY_FILE")
    if not ssh_key_file:
        raise typer.BadParameter("Set ENCORD_SSH_KEY_FILE or pass --labels-json")

    from encord.user_client import EncordUserClient

    client = EncordUserClient.create_with_ssh_private_key(Path(ssh_key_file).expanduser().read_text())
    project = client.get_project(project_hash)
    rows = list(project.list_label_rows_v2())
    if rows:
        with project.create_bundle(bundle_size=min(100, len(rows))) as bundle:
            for row in rows:
                row.initialise_labels(bundle=bundle)
    return [row.to_encord_dict() for row in rows]


def captions_from_rows(rows: list[dict[str, Any]]) -> list[Caption]:
    captions: list[Caption] = []
    for row in rows:
        text = find_string_by_key(row, CAPTION_KEYS)
        if not text:
            continue
        captions.append(Caption(
            match_value=find_string_by_key(row, MATCH_KEYS),
            text=text,
        ))
    return captions


def discover_episodes(source_root: Path) -> list[Episode]:
    roots = [source_root] if (source_root / "meta/info.json").exists() else [
        path.parent.parent for path in sorted(source_root.glob("**/meta/info.json"))
    ]
    episodes: list[Episode] = []
    for root in roots:
        info = json.loads((root / "meta/info.json").read_text())
        data_pattern = info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        video_keys = [
            key for key, feature in (info.get("features") or {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]
        metas = {int(row["episode_index"]): row for row in read_jsonl(root / "meta/episodes.jsonl")}
        indices = sorted(metas) or [
            int(path.stem.replace("episode_", "")) for path in sorted(root.glob("data/chunk-*/episode_*.parquet"))
        ]
        for index in indices:
            chunk = index // int(info.get("chunks_size", 1000))
            parquet_path = root / data_pattern.format(episode_chunk=chunk, episode_index=index)
            if parquet_path.exists():
                episodes.append(Episode(root, index, parquet_path, info, metas.get(index, {}), video_keys))
    return episodes


def match_caption(episode: Episode, captions: list[Caption]) -> Caption | None:
    candidates = episode.match_candidates
    for caption in captions:
        if not caption.match_value:
            continue
        match = caption.match_value.strip()
        if match in candidates:
            return caption
        if any(candidate in match for candidate in candidates if len(candidate) >= 6):
            return caption
        if any(match in candidate for candidate in candidates if len(match) >= 6):
            return caption
    return None


def set_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def rewrite_parquet(source: Episode, output_path: Path, new_index: int, global_start: int, task_id: int) -> int:
    table = pq.read_table(source.parquet_path)
    rows = table.num_rows
    table = set_column(table, "episode_index", pa.array([new_index] * rows, type=pa.int64()))
    table = set_column(table, "frame_index", pa.array(list(range(rows)), type=pa.int64()))
    table = set_column(table, "index", pa.array(list(range(global_start, global_start + rows)), type=pa.int64()))
    table = set_column(table, "task_index", pa.array([task_id] * rows, type=pa.int64()))
    for key in LANG_KEYS:
        table = set_column(table, key, pa.array([task_id] * rows, type=pa.int64()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return rows


def source_video_path(episode: Episode, video_key: str) -> Path:
    pattern = episode.info.get(
        "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    return episode.root / pattern.format(
        episode_chunk=episode.chunk,
        video_key=video_key,
        episode_index=episode.index,
    )


def missing_droid_views(video_sources: dict[str, str]) -> list[str]:
    return [key for key in DROID_VIDEO_KEYS if key not in video_sources]


def choose_video_sources(episode: Episode) -> dict[str, str]:
    available = [key for key in episode.video_keys if source_video_path(episode, key).exists()]
    mapped = {
        SOURCE_TO_DROID.get(key, key): key
        for key in available
    }
    ordered = {key: mapped[key] for key in DROID_VIDEO_KEYS if key in mapped}
    ordered.update({key: value for key, value in mapped.items() if key not in ordered})
    return ordered


def copy_videos(
    episode: Episode,
    output_root: Path,
    new_index: int,
    overwrite: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    features = episode.info.get("features") or {}
    output_features: dict[str, dict[str, Any]] = {}
    copy_plan: list[dict[str, Any]] = []
    for dest_key, source_key in choose_video_sources(episode).items():
        source_path = source_video_path(episode, source_key)
        dest_path = output_root / f"videos/chunk-{new_index // 1000:03d}/{dest_key}/episode_{new_index:06d}.mp4"
        if dest_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite video: {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        output_features[dest_key] = dict(features.get(source_key, {"dtype": "video", "shape": [480, 640, 3]}))
        copy_plan.append({
            "source_key": source_key,
            "output_key": dest_key,
        })
    return output_features, copy_plan


def split_for(table: pa.Table, column: str) -> dict[str, list[int]]:
    if column not in table.column_names or table.num_rows == 0:
        return {}
    value = table[column][0].as_py()
    width = len(value) if isinstance(value, list) else 1
    if width == 16:
        return {"joint_position": [0, 14], "gripper_position": [14, 16]}
    return {column.split(".")[-1]: [0, width]}


def write_metadata(
    output_root: Path,
    source_info: dict[str, Any],
    episodes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    video_features: dict[str, dict[str, Any]],
    first_table: pa.Table,
    copy_plan: list[dict[str, Any]],
    training_ready: bool,
    missing_views: list[dict[str, Any]],
) -> list[str]:
    written: list[str] = []
    meta = output_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    files = {
        meta / "tasks.jsonl": "".join(json.dumps(row) + "\n" for row in tasks),
        meta / "episodes.jsonl": "".join(json.dumps(row) + "\n" for row in episodes),
        output_root / "encord_droid_export_manifest.json": json.dumps({
            "training_ready": training_ready,
            "caption_export_only": not training_ready,
            "required_droid_video_keys": DROID_VIDEO_KEYS,
            "missing_train_time_views": missing_views,
            "video_copy_plan": copy_plan,
            "training_view_policy": "Trainable exports require all three DROID video views. Partial-view exports are for caption QA/versioning only.",
        }, indent=2) + "\n",
    }
    if not training_ready:
        files[output_root / "NOT_TRAINABLE.md"] = (
            "# Not Trainable\n\n"
            "This export is for caption QA/versioning only because at least one episode is missing required DROID video views.\n"
        )

    info = dict(source_info)
    info.update({
        "total_episodes": len(episodes),
        "total_frames": int(sum(row["length"] for row in episodes)),
        "total_tasks": len(tasks),
        "total_videos": len(copy_plan),
        "total_chunks": (len(episodes) // 1000) + (1 if len(episodes) % 1000 else 0),
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"} if training_ready else {"caption_qc": f"0:{len(episodes)}"},
        "training_ready": training_ready,
        "caption_export_only": not training_ready,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    })
    features = {
        key: value for key, value in (source_info.get("features") or {}).items()
        if not key.startswith("observation.images.")
    }
    features.update(video_features)
    for key in LANG_KEYS:
        features[key] = {"dtype": "int64", "shape": [1]}
    info["features"] = features
    files[meta / "info.json"] = json.dumps(info, indent=4) + "\n"

    modality = {
        "state": {
            key: {"original_key": "observation.state", "start": bounds[0], "end": bounds[1], "rotation_type": None, "absolute": True}
            for key, bounds in split_for(first_table, "observation.state").items()
        },
        "action": {
            key: {"original_key": "action", "start": bounds[0], "end": bounds[1], "rotation_type": None, "absolute": True}
            for key, bounds in split_for(first_table, "action").items()
        },
        "video": {
            key.replace("observation.images.", ""): {"original_key": key}
            for key in sorted(video_features)
        },
        "annotation": {
            key.replace("annotation.", ""): {"original_key": key}
            for key in LANG_KEYS
        },
    }
    files[meta / "modality.json"] = json.dumps(modality, indent=4) + "\n"

    for path, text in files.items():
        if path.exists():
            continue
        path.write_text(text)
        written.append(path.relative_to(output_root).as_posix())
    return written


def main(
    source_root: Annotated[Path, typer.Argument(help="LeRobot-style source dataset root or parent directory.")],
    output_dir: Annotated[Path, typer.Argument(help="Output dataset directory.")],
    project_hash: Annotated[str | None, typer.Option(help="Encord project hash. Uses ENCORD_SSH_KEY_FILE.")] = None,
    labels_json: Annotated[Path | None, typer.Option(help="Local label JSON for testing/offline export.")] = None,
    include_unmatched: Annotated[bool, typer.Option(help="Also export episodes with no matched caption as 'not provided'.")] = False,
    overwrite: Annotated[bool, typer.Option(help="Overwrite output parquet/video files.")] = False,
) -> None:
    captions = captions_from_rows(load_label_rows(labels_json, project_hash))
    episodes = discover_episodes(source_root)
    if not episodes:
        raise typer.BadParameter(f"No source episodes found under {source_root}")

    task_to_id = {"not provided": 0}
    output_episode_rows: list[dict[str, Any]] = []
    output_video_features: dict[str, dict[str, Any]] = {}
    video_copy_plan: list[dict[str, Any]] = []
    missing_views: list[dict[str, Any]] = []
    video_sources_by_episode: dict[tuple[str, int], dict[str, str]] = {}
    first_table: pa.Table | None = None
    total_frames = 0

    selected = []
    for episode in episodes:
        caption = match_caption(episode, captions)
        if caption or include_unmatched:
            selected.append((episode, caption))
    if not selected:
        raise typer.BadParameter("No source episodes matched caption rows.")
    for episode, _ in selected:
        video_sources = choose_video_sources(episode)
        video_sources_by_episode[(episode.root.as_posix(), episode.index)] = video_sources
        missing = missing_droid_views(video_sources)
        if missing:
            missing_views.append({
                "source_dataset": episode.root.name,
                "source_episode_index": episode.index,
                "missing_video_keys": missing,
            })
    training_ready = not missing_views

    for new_index, (episode, caption) in enumerate(selected):
        text = caption.text if caption else "not provided"
        task_to_id.setdefault(text, len(task_to_id))
        task_id = task_to_id[text]

        parquet_path = output_dir / f"data/chunk-{new_index // 1000:03d}/episode_{new_index:06d}.parquet"
        if parquet_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite parquet: {parquet_path}")
        length = rewrite_parquet(episode, parquet_path, new_index, total_frames, task_id)
        total_frames += length
        if first_table is None:
            first_table = pq.read_table(parquet_path)

        video_features, copy_plan = copy_videos(episode, output_dir, new_index, overwrite)
        output_video_features.update(video_features)
        video_copy_plan.extend({"episode_index": new_index, **row} for row in copy_plan)
        missing = missing_droid_views(video_sources_by_episode[(episode.root.as_posix(), episode.index)])
        output_episode_rows.append({
            "episode_index": new_index,
            "tasks": [] if text == "not provided" else [text],
            "length": length,
            "source_dataset": episode.root.name,
            "source_episode_index": episode.index,
            "caption_matched": caption is not None,
            "training_ready": not missing,
            "missing_train_time_views": missing,
        })

    tasks = [
        {"task_index": task_id, "task": task}
        for task, task_id in sorted(task_to_id.items(), key=lambda item: item[1])
    ]
    assert first_table is not None
    written = write_metadata(
        output_dir,
        episodes[0].info,
        output_episode_rows,
        tasks,
        output_video_features,
        first_table,
        video_copy_plan,
        training_ready,
        missing_views,
    )
    typer.echo(f"Exported {len(output_episode_rows)} episodes to {output_dir}")
    typer.echo(f"Training ready: {'yes' if training_ready else 'no (caption QA/versioning export only)'}")
    typer.echo(f"Metadata written: {', '.join(written) if written else 'none'}")


if __name__ == "__main__":
    typer.run(main)
