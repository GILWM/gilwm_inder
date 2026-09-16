"""End-to-end paired-video + WorldArena HDF5 dataloader probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
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


def _alignment_only_dataset() -> SAM3DVideoDataset:
    dataset = object.__new__(SAM3DVideoDataset)
    dataset.sampling_mode = "continuous"
    dataset.sequence_length = 93
    dataset.continuous_condition_policy = "restrict"
    return dataset


def test_legacy_frame_zero_cache_restricts_continuous_crop_to_frame_zero():
    dataset = _alignment_only_dataset()
    cache = {
        "sam_masks": torch.ones(8, 4, 4),
        "sam_mask_meta": torch.ones(8, 8),
        "sam3d_geometry": torch.ones(8, 4, 4),
        "sam3d_shape_latents": torch.ones(8, 2, 8),
        "sam3d_object_pose": torch.ones(8, 10),
    }
    np.testing.assert_array_equal(dataset._condition_start_frames(cache), np.asarray([0]))
    np.testing.assert_array_equal(dataset._sample_frame_ids(140, np.asarray([0])), np.arange(93))


def test_drop_unaligned_policy_never_attaches_frame_zero_cache_to_later_crop():
    dataset = _alignment_only_dataset()
    dataset.continuous_condition_policy = "drop_unaligned"
    cache = {
        "sam_masks": torch.ones(8, 4, 4),
        "sam_mask_meta": torch.ones(8, 8),
        "sam3d_geometry": torch.ones(8, 4, 4),
        "sam3d_shape_latents": torch.ones(8, 2, 8),
        "sam3d_object_pose": torch.ones(8, 10),
    }
    assert dataset._condition_cache_for_sampled_frame(cache, 0) is cache
    assert dataset._condition_cache_for_sampled_frame(cache, 17) is None


def test_uniform_93_sampling_keeps_endpoints_and_frame_zero_condition():
    dataset = _alignment_only_dataset()
    dataset.sampling_mode = "uniform"
    dataset.continuous_condition_policy = "drop_unaligned"
    cache = {
        "sam_masks": torch.ones(8, 4, 4),
        "sam_mask_meta": torch.ones(8, 8),
        "sam3d_geometry": torch.ones(8, 4, 4),
        "sam3d_shape_latents": torch.ones(8, 2, 8),
        "sam3d_object_pose": torch.ones(8, 10),
    }

    expected = np.rint(np.linspace(0, 139, 93)).astype(np.int64)
    sampled = dataset._sample_frame_ids(140, np.asarray([0]))

    np.testing.assert_array_equal(sampled, expected)
    assert sampled[0] == 0
    assert sampled[-1] == 139
    assert dataset._condition_cache_for_sampled_frame(cache, int(sampled[0])) is cache


def test_complete_object_overlay_and_disabled_teacher_tokens_use_network_shapes():
    with tempfile.TemporaryDirectory(prefix="sam3d-complete-overlay-") as directory:
        root = Path(directory)
        condition_path = root / "condition.pt"
        object_path = root / "sam3d_objects.pt"
        torch.save(
            {
                "sam3d_tokens": torch.randn(2, 3, 768),
                "sam_masks": torch.ones(2, 4, 5),
                "sam_mask_meta": torch.ones(2, 8),
                "sam3d_geometry": torch.zeros(8, 4, 5),
                "sam3d_shape_latents": torch.zeros(2, 3, 8),
                "sam3d_object_pose": torch.zeros(2, 10),
            },
            condition_path,
        )
        torch.save(
            {
                "sam3d_geometry": torch.ones(8, 4, 5),
                "sam3d_shape_latents": torch.ones(2, 3, 8),
                "sam3d_object_pose": torch.ones(2, 10),
            },
            object_path,
        )

        dataset = _alignment_only_dataset()
        dataset.sam3d_required = True
        dataset.sam3d_require_complete_conditions = True
        dataset.sam3d_load_teacher_tokens = False
        dataset.sam3d_condition_paths = [str(condition_path)]
        dataset.sam3d_object_paths = [str(object_path)]
        dataset.sam3d_teacher_frames = 8
        dataset.sam3d_num_tokens = 256
        dataset.sam3d_token_dim = 768
        dataset.sam_mask_instances = 2
        dataset.sam_mask_size = (4, 5)
        dataset.sam_mask_meta_dim = 8
        dataset.sam3d_geometry_channels = 8
        dataset.sam3d_geometry_size = (4, 5)
        dataset.sam3d_shape_instances = 2
        dataset.sam3d_shape_tokens = 3
        dataset.sam3d_shape_dim = 8
        dataset.sam3d_pose_dim = 10

        raw = dataset._load_raw_condition(0)
        assert raw is not None
        condition = dataset._load_condition(0, source_frame_index=0, cache=raw)

        assert condition["sam3d_tokens_B_F_N_D"].shape == (1, 1, 768)
        assert not torch.count_nonzero(condition["sam3d_tokens_B_F_N_D"])
        assert torch.count_nonzero(condition["sam3d_geometry_B_C_H_W"])
        assert torch.count_nonzero(condition["sam3d_shape_latents_B_K_N_D"])
        assert torch.count_nonzero(condition["sam3d_object_pose_B_K_D"])

        object_path.unlink()
        with pytest.raises(FileNotFoundError, match="Required SAM 3D object sidecar"):
            dataset._load_raw_condition(0)


def test_schema_v3_selects_the_exact_sampled_crop_start_condition():
    dataset = _alignment_only_dataset()
    anchors = torch.tensor([0, 17, 37])
    masks = torch.stack([torch.full((8, 4, 4), float(index)) for index in anchors])
    meta = torch.stack([torch.full((8, 8), float(index)) for index in anchors])
    cache = {
        "condition_frame_indices": anchors,
        "sam_masks": masks,
        "sam_mask_meta": meta,
        "sam3d_geometry": torch.zeros(8, 4, 4),
        "sam3d_shape_latents": torch.zeros(8, 2, 8),
        "sam3d_object_pose": torch.zeros(8, 10),
    }
    np.testing.assert_array_equal(dataset._condition_start_frames(cache), anchors.numpy())
    selected = dataset._select_condition_frame(cache, "sam_masks", masks, 3, source_frame_index=17)
    assert torch.equal(selected, torch.full((8, 4, 4), 17.0))


def test_active_native_sam3d_timeline_is_intersected_with_sam_mask_timeline():
    dataset = _alignment_only_dataset()
    cache = {
        "condition_frame_indices": torch.tensor([0, 17, 37]),
        "sam_masks": torch.ones(3, 8, 4, 4),
        "sam_mask_meta": torch.ones(3, 8, 8),
        "sam3d_geometry": torch.ones(2, 8, 4, 4),
        "sam3d_geometry_frame_indices": torch.tensor([0, 37]),
        "sam3d_shape_latents": torch.ones(2, 8, 2, 8),
        "sam3d_shape_latents_frame_indices": torch.tensor([0, 37]),
        "sam3d_object_pose": torch.ones(2, 8, 10),
        "sam3d_object_pose_frame_indices": torch.tensor([0, 37]),
    }
    np.testing.assert_array_equal(dataset._condition_start_frames(cache), np.asarray([0, 37]))


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

        with pytest.raises(RuntimeError, match="expected 3 usable samples, found 2"):
            SAM3DVideoDataset(
                dataset_dir=str(root),
                manifest_paths=manifests,
                included_batches=["core15k", "robotwin2"],
                sam3d_cache_dir=str(root / "missing-sam-cache"),
                sam3d_required=False,
                action_required=True,
                expected_num_samples=3,
                num_frames=93,
                video_size=(64, 64),
                sampling_mode="uniform",
                filter_short_videos=True,
            )

        dataset._get_frames_with_frame_ids = lambda _, **__: (
            torch.zeros(3, 93, 64, 64),
            24.0,
            np.rint(np.linspace(0, 120, 93)).astype(np.int64),
            121,
        )
        item = dataset[0]
        assert item["actions_B_T_D"].shape == (93, 14)
        assert item["action_valid_B"].item() is True
        source_indices = np.rint(np.linspace(0, 120, 93)).astype(np.int64)
        source_actions = read_action_sequence(Path(dataset.action_hdf5_paths[0]))
        np.testing.assert_allclose(item["actions_B_T_D"].numpy(), source_actions[source_indices])


def test_dataset_loading_retries_are_bounded_and_fail_loudly():
    dataset = object.__new__(SAM3DVideoDataset)
    dataset.video_paths = ["bad-a.mp4", "bad-b.mp4"]
    dataset.max_load_retries = 2
    dataset.num_failed_loads = 0
    calls = []

    def fail(index: int):
        calls.append(index)
        raise OSError("broken sample")

    dataset._get_item = fail
    with pytest.raises(RuntimeError, match="bounded retry budget"):
        dataset[0]
    assert len(calls) == 3
    assert dataset.num_failed_loads == 3


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
