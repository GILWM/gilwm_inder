#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("episode_*.pt"))
    errors = []
    empty_masks = object_free = 0
    expected_shapes = {
        "sam_masks": (8, 120, 160),
        "sam_mask_meta": (8, 8),
        "sam3d_geometry": (8, 120, 160),
        "sam3d_shape_latents": (8, 256, 8),
        "sam3d_object_pose": (8, 10),
    }
    for path in paths:
        try:
            value = load(path)
            if "sam3d_tokens" in value:
                raise ValueError("teacher tokens present")
            tensors = {}
            for key, shape in expected_shapes.items():
                tensor = torch.as_tensor(value[key])
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"{key} shape={tuple(tensor.shape)} expected={shape}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{key} contains NaN/Inf")
                tensors[key] = tensor
            if not torch.count_nonzero(tensors["sam3d_geometry"]):
                raise ValueError("zero geometry")
            count = int(torch.as_tensor(value["sam3d_object_valid_count"]).item())
            empty_masks += int(not bool(torch.count_nonzero(tensors["sam_masks"])))
            object_free += int(count == 0)
            if count and not torch.count_nonzero(tensors["sam3d_shape_latents"][:count]):
                raise ValueError("zero shape latents for detected objects")
            if count and not torch.count_nonzero(tensors["sam3d_object_pose"][:count]):
                raise ValueError("zero poses for detected objects")
        except Exception as error:
            if len(errors) < 20:
                errors.append(f"{path.name}: {error}")
    print(
        f"INFERENCE_CONDITION_AUDIT root={args.root} files={len(paths)}/{args.expected} "
        f"invalid={len(errors)} empty_masks={empty_masks} object_free={object_free}",
        flush=True,
    )
    if errors:
        print("\n".join(errors), flush=True)
    if len(paths) != args.expected or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
