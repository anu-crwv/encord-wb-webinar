from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock
from uuid import UUID

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from expand_encord_folder_from_r2_mcaps import (
    CAMERA_ORDER,
    ExtractedEpisode,
    GroupInventory,
    SelectionMode,
    VideoInventory,
    canonical_episode_path,
    order_candidates,
    output_specs,
    reconcile_state_with_live_items,
    recovery_episode_from_r2,
    should_schedule,
    upload_extracted_episode,
)


def make_episode(
    *,
    task: str = "Task A",
    environment: str = "Industrial",
    index: int = 1,
    ingestion_id: str = "ingestion-1",
):
    key = (
        f"trossen-data-mobile/{task}/{environment}/Operator/2026-07-01/"
        f"episode_{index:06}.mcap"
    )
    episode = recovery_episode_from_r2(
        bucket="trossen-robotics-data",
        prefix="trossen-data-mobile",
        key=key,
        size=100 + index,
        ingestion_id=ingestion_id,
    )
    assert episode is not None
    return episode


def test_r2_key_builds_canonical_three_camera_episode() -> None:
    episode = make_episode(index=42)

    assert episode.episode_path.endswith("/episode_000042/")
    assert tuple(episode.videos) == CAMERA_ORDER
    assert {
        video.client_metadata["camera_name"] for video in episode.videos.values()
    } == set(CAMERA_ORDER)
    assert all(
        video.client_metadata["r2_expansion_ingestion_id"] == "ingestion-1"
        for video in episode.videos.values()
    )


def test_invalid_r2_key_is_ignored() -> None:
    episode = recovery_episode_from_r2(
        bucket="bucket",
        prefix="trossen-data-mobile",
        key="trossen-data-mobile/not-an-episode.mcap",
        size=10,
        ingestion_id="ingestion-1",
    )

    assert episode is None


def test_balanced_selection_round_robins_task_environment() -> None:
    episodes = [
        make_episode(task="Task A", environment="Industrial", index=1),
        make_episode(task="Task A", environment="Industrial", index=2),
        make_episode(task="Task A", environment="Industrial", index=3),
        make_episode(task="Task A", environment="Kitchen", index=4),
        make_episode(task="Task B", environment="Industrial", index=5),
    ]

    ordered = order_candidates(episodes, SelectionMode.BALANCED)
    first_round = {
        (
            episode.task_name,
            Path(episode.episode_path.rstrip("/")).parts[-4],
        )
        for episode in ordered[:3]
    }

    assert len(first_round) == 3
    assert [episode.episode_index for episode in ordered[3:]] == [2, 3]


def test_canonical_episode_path_ignores_noncanonical_values() -> None:
    assert (
        canonical_episode_path(
            "raw-feed/trossen-data/Task/Industrial/A/2026-07-01/"
            "episode_000123/videos/chunk-000/file.mp4"
        )
        == "raw-feed/trossen-data/Task/Industrial/A/2026-07-01/episode_000123/"
    )
    assert canonical_episode_path("unrelated/episode_000123/file.mp4") is None


def test_scheduler_replaces_failures_without_overfilling() -> None:
    assert should_schedule(successful=100, active=23, target=2250, depth=24)
    assert not should_schedule(successful=100, active=24, target=2250, depth=24)
    assert not should_schedule(successful=2249, active=1, target=2250, depth=24)
    assert should_schedule(successful=2249, active=0, target=2250, depth=24)


def test_live_reconciliation_adopts_and_stales_groups() -> None:
    episode = make_episode()
    episode_path = episode.episode_path
    camera_map = {
        camera: f"00000000-0000-0000-0000-00000000000{index}"
        for index, camera in enumerate(CAMERA_ORDER, start=1)
    }
    videos = VideoInventory(
        title_to_uuid={},
        camera_uuids_by_episode={episode_path: camera_map},
        ingestion_ids_by_episode={episode_path: {"ingestion-1"}},
        duplicate_slots=[],
        unclassified_video_count=0,
    )
    groups = GroupInventory(
        group_uuids_by_ingestion_episode={("ingestion-1", episode_path): "group-uuid"},
        group_counts_by_ingestion_episode={("ingestion-1", episode_path): 1},
        camera_maps_by_group_uuid={"group-uuid": camera_map},
        validation_failures_by_group_uuid={},
        episode_counts={episode_path: 1},
    )
    state = {
        "ingestion_id": "ingestion-1",
        "episodes": {},
    }

    assert reconcile_state_with_live_items(
        state=state,
        video_inventory=videos,
        group_inventory=groups,
    )
    assert state["episodes"][episode_path]["status"] == "grouped"

    empty_groups = GroupInventory(
        group_uuids_by_ingestion_episode={},
        group_counts_by_ingestion_episode={},
        camera_maps_by_group_uuid={},
        validation_failures_by_group_uuid={},
        episode_counts={},
    )
    assert reconcile_state_with_live_items(
        state=state,
        video_inventory=videos,
        group_inventory=empty_groups,
    )
    assert state["episodes"][episode_path]["status"] == "stale"


class FailingUploadFolder:
    def __init__(self) -> None:
        self.upload_calls = 0
        self.deleted: list[UUID] = []

    def upload_video(self, _path: Path, _title: str, _metadata: dict) -> UUID:
        self.upload_calls += 1
        if self.upload_calls == 2:
            raise RuntimeError("simulated camera upload failure")
        return UUID("00000000-0000-0000-0000-000000000001")

    def delete_storage_items(self, uuids: list[UUID]) -> None:
        self.deleted.extend(uuids)


def test_partial_episode_upload_is_rolled_back(tmp_path: Path) -> None:
    episode = make_episode()
    output_root = tmp_path / "output"
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    for spec in output_specs(episode, output_root):
        if spec["data_type"] != "video":
            continue
        path = output_root / spec["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    extracted = ExtractedEpisode(
        episode=episode,
        cache_path=str(cache_root / "episode.mcap"),
        download_action="downloaded",
        episode_dir=str(output_root / episode.episode_path),
        details={"video_probes": {}, "frame_counts": {}},
    )
    folder = FailingUploadFolder()

    uploaded, terminal = upload_extracted_episode(
        extracted=extracted,
        folder=folder,
        known_titles={},
        known_lock=Lock(),
        output_root=output_root,
        cache_root=cache_root,
        video_folder_hash="video-folder",
        cleanup_failed=False,
    )

    assert uploaded is None
    assert terminal is not None
    assert terminal.stage == "upload"
    assert "simulated camera upload failure" in str(terminal.error)
    assert folder.deleted == [UUID("00000000-0000-0000-0000-000000000001")]
