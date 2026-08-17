#!/usr/bin/env python3
from __future__ import annotations

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset


def main() -> None:
    dataset = SAM3DVideoDataset(
        dataset_dir="/datahdd/mccxadmin/train_data",
        sam3d_cache_dir="/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93",
        sam3d_required=True,
        included_batches=["core15k", "legacy4k"],
        num_frames=93,
        video_size=(480, 640),
        sampling_mode="uniform",
        filter_short_videos=True,
    )
    expected = 18_349
    print(f"SAM3D_V93_COVERAGE samples={len(dataset)} expected={expected} complete={len(dataset) == expected}")
    if len(dataset) != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
