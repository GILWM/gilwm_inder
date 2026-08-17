#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


def validate_one(path_s: str) -> tuple[str | None, int, float, float]:
    path = Path(path_s)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        shape = torch.as_tensor(value["sam3d_shape_latents"])
        pose = torch.as_tensor(value["sam3d_object_pose"])
        geometry = torch.as_tensor(value["sam3d_geometry"])
        if shape.ndim != 3 or shape.shape[-1] != 8:
            raise ValueError(f"shape={tuple(shape.shape)}")
        if pose.ndim != 2 or pose.shape[-1] != 10:
            raise ValueError(f"pose={tuple(pose.shape)}")
        if tuple(geometry.shape) != (8, 120, 160):
            raise ValueError(f"geometry={tuple(geometry.shape)}")
        for name, tensor in (("shape", shape), ("pose", pose), ("geometry", geometry)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN/Inf")
        objects = int(value.get("object_count", shape.shape[0]))
        if objects < 1 or not torch.count_nonzero(shape) or not torch.count_nonzero(geometry):
            raise ValueError("empty native SAM3D output")
        return None, objects, float(shape.float().abs().mean()), float(geometry.float().abs().mean())
    except Exception as error:
        return f"{path}: {error!r}", 0, 0.0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/*/sam3d_objects.pt"))
    if len(paths) != args.expected:
        raise SystemExit(f"Sidecar count mismatch: {len(paths)} != {args.expected}")
    errors: list[str] = []
    object_count = 0
    shape_mean = geometry_mean = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for error, objects, one_shape_mean, one_geometry_mean in executor.map(
            validate_one, map(str, paths), chunksize=8
        ):
            if error is not None and len(errors) < 20:
                errors.append(error)
            object_count += objects
            shape_mean += one_shape_mean
            geometry_mean += one_geometry_mean
    print(
        f"SAM3D_SIDECARS_VALID files={len(paths)} errors={len(errors)} objects={object_count} "
        f"shape_abs_mean={shape_mean / len(paths):.6f} "
        f"geometry_abs_mean={geometry_mean / len(paths):.6f}"
    )
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
