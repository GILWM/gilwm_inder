"""End-to-end paired-video + WorldArena HDF5 dataloader probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset
from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import read_action_sequence


def _write_action_hdf5(path: Path, length: int) -> np.ndarray:
    left_arm = np.arange(length * 6, dtype=np.float32).reshape(length, 6)
    left_gripper = np.arange(length, dtype=np.float32)
    right_arm = left_arm + 1000
    right_gripper = left_gripper + 2000
    vector = np.concatenate([left_arm, left_gripper[:, None], right_arm, right_gripper[:, None]], axis=1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("joint_action/left_arm", data=left_arm)
        handle.create_dataset("joint_action/left_gripper", data=left_gripper)
        handle.create_dataset("joint_action/right_arm", data=right_arm)
        handle.create_dataset("joint_action/right_gripper", data=right_gripper)
        handle.create_dataset("joint_action/vector", data=vector)
    return vector


def test_multiple_filtered_manifests_resolve_video_caption_and_action_paths():
    with tempfile.TemporaryDirectory(prefix="sam3d-multi-manifest-") as directory:
        root = Path(directory)
        manifests = []
        for batch in ("core15k", "robotwin2"):
            filtered = root / f"{batch}_filter_75_455"
            sample = filtered / f"{batch}_00000001"
            sample.mkdir(parents=True)
            (sample / "video.mp4").write_bytes(b"placeholder")
            _write_action_hdf5(sample / "actions.hdf5", 121)
            (sample / "instruction.json").write_text(json.dumps({"instruction": f"test {batch}"}), encoding="utf-8")
            record = {
                "batch": batch,
                "video": f"{batch}_00000001/video.mp4",
                "instruction": f"{batch}_00000001/instruction.json",
                "actions": f"{batch}_00000001/actions.hdf5",
                "frame_count": 121,
            }
            manifest = filtered / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifests.append(str(manifest))

        dataset = SAM3DVideoDataset(
            dataset_dir=str(root),
            manifest_paths=manifests,
            included_batches=["core15k", "robotwin2"],
            sam3d_cache_dir=str(root / "missing-sam-cache"),
            sam3d_required=False,
            action_required=True,
            num_frames=93,
            video_size=(64, 64),
            sampling_mode="uniform",
            filter_short_videos=True,
        )
        assert len(dataset) == 2
        assert all(Path(path).is_file() for path in dataset.video_paths)
        assert all(Path(path).is_file() for path in dataset.caption_paths)
        assert all(path is not None and Path(path).is_file() for path in dataset.action_hdf5_paths)
        assert dataset.action_hdf5_root is None
        assert dataset.action_enabled is True

        dataset._get_frames_with_frame_ids = lambda _: (
            torch.zeros(3, 93, 64, 64),
            24.0,
            np.rint(np.linspace(0, 120, 93)).astype(np.int64),
            121,
        )
        item = dataset[0]
        assert item["actions_B_T_D"].shape == (93, 14)
        assert item["action_valid_B"].item() is True


def run_probe(action_root: Path, source_video: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sam3d-action-dataset-") as directory:
        root = Path(directory)
        sample = root / "robotwin" / "episode1"
        sample.mkdir(parents=True)
        os.symlink(source_video, sample / "video.mp4")
        (sample / "instruction.json").write_text(
            json.dumps({"instruction": "test the paired action loader"}) + "\n",
            encoding="utf-8",
        )
        record = {
            "batch": "robotwin",
            "video": "robotwin/episode1/video.mp4",
            "instruction": "robotwin/episode1/instruction.json",
            "trajectory_hdf5": "data/fixed_scene_task/episode1.hdf5",
        }
        (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

        dataset = SAM3DVideoDataset(
            dataset_dir=str(root),
            sam3d_cache_dir=str(root / "missing-sam-cache"),
            sam3d_required=False,
            action_hdf5_root=str(action_root),
            action_required=True,
            num_frames=93,
            video_size=(64, 64),
            sampling_mode="uniform",
            filter_short_videos=True,
        )
        item = dataset[0]
        assert item["video"].shape == (3, 93, 64, 64)
        assert item["actions_B_T_D"].shape == (93, 14)
        assert item["action_valid_B"].item() is True
        assert item["ai_caption"] == "test the paired action loader"
        source_actions = read_action_sequence(action_root / "data" / "fixed_scene_task" / "episode1.hdf5")
        np.testing.assert_allclose(item["actions_B_T_D"][0].numpy(), source_actions[0])
        np.testing.assert_allclose(item["actions_B_T_D"][-1].numpy(), source_actions[-1])
        assert torch.isfinite(item["actions_B_T_D"]).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action_root", type=Path)
    parser.add_argument("source_video", type=Path)
    args = parser.parse_args()
    run_probe(args.action_root, args.source_video)
    print("SAM3D paired action dataset probe passed")
