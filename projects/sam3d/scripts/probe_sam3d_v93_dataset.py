#!/usr/bin/env python3
from __future__ import annotations

import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset


def main() -> None:
    dataset = SAM3DVideoDataset(
        dataset_dir="/datahdd/mccxadmin/train_data",
        sam3d_cache_dir="/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93-small",
        sam3d_required=True,
        sam3d_native_index="/datahdd/mccxadmin/cosmos-sam3d-native-index/core15k-legacy4k-v93-small.txt",
        repeat_factor=32,
        included_batches=["core15k", "legacy4k"],
        num_frames=93,
        video_size=(480, 640),
        sampling_mode="uniform",
        filter_short_videos=True,
        sam3d_teacher_frames=8,
        sam3d_num_tokens=256,
        sam3d_token_dim=768,
        sam_mask_instances=8,
        sam_mask_size=(120, 160),
        sam_mask_meta_dim=8,
        sam3d_geometry_channels=8,
        sam3d_geometry_size=(120, 160),
        sam3d_shape_instances=8,
        sam3d_shape_tokens=256,
        sam3d_shape_dim=8,
        sam3d_pose_dim=10,
    )
    assert len(dataset) == 128, len(dataset)
    sample = dataset[0]
    assert sample["video"].shape == (3, 93, 480, 640), sample["video"].shape
    nonzero = {
        key: int(torch.count_nonzero(sample[key]))
        for key in (
            "sam3d_tokens_B_F_N_D",
            "sam_mask_B_K_H_W",
            "sam_mask_meta_B_K_D",
            "sam3d_geometry_B_C_H_W",
            "sam3d_shape_latents_B_K_N_D",
            "sam3d_object_pose_B_K_D",
        )
    }
    assert all(value > 0 for value in nonzero.values()), nonzero
    print(
        "SAM3D_V93_DATASET_OK",
        f"logical_samples={len(dataset)}",
        f"video_shape={tuple(sample['video'].shape)}",
        f"fps={sample['fps']}",
        f"nonzero={nonzero}",
    )


if __name__ == "__main__":
    main()
