#!/usr/bin/env python3
from __future__ import annotations

import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset


def main() -> None:
    dataset = SAM3DVideoDataset(
        dataset_dir="/datahdd/mccxadmin/train_data",
        sam3d_cache_dir="/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93",
        sam3d_objects_root="/datahdd/mccxadmin/cosmos-sam3d-sidecars/core15k-legacy4k-v93-small",
        sam3d_native_index="/datahdd/mccxadmin/cosmos-sam3d-native-index/core15k-legacy4k-v93-small.txt",
        sam3d_required=True,
        included_batches=["core15k", "legacy4k"],
        num_frames=93,
        video_size=(480, 640),
        sampling_mode="uniform",
        filter_short_videos=True,
    )
    assert len(dataset) == 4, len(dataset)
    for index in range(len(dataset)):
        condition = dataset._load_condition(index)
        assert torch.count_nonzero(condition["sam3d_shape_latents_B_K_N_D"]), index
        assert torch.count_nonzero(condition["sam3d_geometry_B_C_H_W"]), index
        assert torch.count_nonzero(condition["sam3d_object_pose_B_K_D"]), index
    print("SAM3D_SIDECAR_OVERLAY_OK", f"samples={len(dataset)}")


if __name__ == "__main__":
    main()
