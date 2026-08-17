#!/usr/bin/env python3
from __future__ import annotations

import os

import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset


def main() -> None:
    dataset = SAM3DVideoDataset(
        dataset_dir="/datahdd/mccxadmin/train_data",
        sam3d_cache_dir="/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93",
        sam3d_objects_root="/datahdd/mccxadmin/cosmos-sam3d-sidecars/core15k-legacy4k-v93",
        sam3d_required=True,
        included_batches=["core15k", "legacy4k"],
        num_frames=93,
        video_size=(480, 640),
        sampling_mode="uniform",
        filter_short_videos=True,
    )
    native = [index for index, path in enumerate(dataset.sam3d_object_paths) if path and os.path.isfile(path)]
    fallback = [index for index, path in enumerate(dataset.sam3d_object_paths) if not path or not os.path.isfile(path)]
    assert len(dataset) == 18_349, len(dataset)
    assert len(native) == 16_874, len(native)
    assert len(fallback) == 1_475, len(fallback)
    for index in native[:16]:
        condition = dataset._load_condition(index)
        assert torch.count_nonzero(condition["sam3d_tokens_B_F_N_D"]), index
        assert torch.count_nonzero(condition["sam3d_shape_latents_B_K_N_D"]), index
        assert torch.count_nonzero(condition["sam3d_geometry_B_C_H_W"]), index
    for index in fallback[:16]:
        condition = dataset._load_condition(index)
        assert torch.count_nonzero(condition["sam3d_tokens_B_F_N_D"]), index
        assert not torch.count_nonzero(condition["sam3d_shape_latents_B_K_N_D"]), index
    print(
        "SAM3D_FORMAL_DATASET_OK",
        f"total={len(dataset)}",
        f"native={len(native)}",
        f"fallback={len(fallback)}",
        "frames=93",
        "sampling=uniform",
    )


if __name__ == "__main__":
    main()
