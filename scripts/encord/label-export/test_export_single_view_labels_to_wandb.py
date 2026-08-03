# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml",
#     "typer",
# ]
# ///
"""Helper checks for Encord label overlay source URI resolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("export_single_view_labels_to_wandb.py")


def load_exporter_module():
    spec = importlib.util.spec_from_file_location("export_single_view_labels_to_wandb", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_row_resolves_parquet_from_source_artifact() -> None:
    exporter = load_exporter_module()
    episode_path = "raw-feed/trossen-data/Batteries/Industrial/Sofia_Infante/2026-05-08/episode_000000/"
    source_uri = (
        "s3://ego-data-collection-encord/"
        f"{episode_path}videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    )
    source_episode = {
        "episode_index": 0,
        "source_video_items": [
            {
                "camera_name": "cam_high",
                "source_uri": source_uri,
                "client_metadata": {
                    "episode_path": episode_path,
                    "episode_id": "episode_000000",
                    "camera_name": "cam_high",
                },
            }
        ],
    }
    row = {"episode_path": episode_path}

    parquet_uri, resolution = exporter.resolve_source_parquet_uri(row, source_episode)
    assert parquet_uri == (
        "s3://ego-data-collection-encord/"
        f"{episode_path}data/chunk-000/episode_000000.parquet"
    )
    assert resolution == "source_artifact_derived"

    expected_info_uri = "s3://ego-data-collection-encord/" f"{episode_path}meta/info.json"
    info_uri = exporter.resolve_source_info_uri(row, source_episode)
    assert info_uri == expected_info_uri


def test_label_metadata_parquet_takes_priority() -> None:
    exporter = load_exporter_module()
    row = {
        "episode_path": "raw-feed/trossen-data/Batteries/Industrial/Sofia_Infante/2026-05-08/episode_000000/",
        "source_parquet_uri": "s3://example-bucket/custom/path/episode_000000.parquet",
    }
    source_episode = {
        "episode_index": 0,
        "source_video_items": [
            {
                "source_uri": (
                    "s3://ego-data-collection-encord/raw-feed/trossen-data/Batteries/Industrial/"
                    "Sofia_Infante/2026-05-08/episode_000000/videos/chunk-000/"
                    "observation.images.cam_high/episode_000000.mp4"
                ),
                "client_metadata": {
                    "episode_path": row["episode_path"],
                    "episode_id": "episode_000000",
                },
            }
        ],
    }

    parquet_uri, resolution = exporter.resolve_source_parquet_uri(row, source_episode)
    assert parquet_uri == row["source_parquet_uri"]
    assert resolution == "label_metadata"


if __name__ == "__main__":
    test_group_row_resolves_parquet_from_source_artifact()
    test_label_metadata_parquet_takes_priority()
    print("label export helper checks passed")
