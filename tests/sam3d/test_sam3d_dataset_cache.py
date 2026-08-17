"""Small runtime check for cache normalization on the MUSA PyTorch build."""

from __future__ import annotations

import argparse

import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    masks = SAM3DVideoDataset._resize_nearest_2d(cache["sam_masks"].float(), (120, 160))
    geometry = SAM3DVideoDataset._resize_bilinear_2d(cache["sam3d_geometry"].float(), (120, 160))
    assert masks.shape == (8, 120, 160), masks.shape
    assert geometry.shape == (8, 120, 160), geometry.shape
    print(
        "SAM3D_DATASET_CACHE_OK",
        {key: tuple(value.shape) for key, value in cache.items() if isinstance(value, torch.Tensor)},
    )


if __name__ == "__main__":
    main()
