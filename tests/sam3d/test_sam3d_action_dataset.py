"""End-to-end paired-video + WorldArena HDF5 dataloader probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset
from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import read_action_sequence


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
        source_actions = read_action_sequence(
            action_root / "data" / "fixed_scene_task" / "episode1.hdf5"
        )
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
