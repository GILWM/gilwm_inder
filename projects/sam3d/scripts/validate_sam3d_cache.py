#!/usr/bin/env python3
"""Validate Cosmos SAM 3D cache tensors without loading GPU models."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


EXPECTED = {
    "sam3d_tokens": (8, 256, 768),
    "sam_masks": (8, 120, 160),
    "sam_mask_meta": (8, 8),
    "sam3d_geometry": (8, 120, 160),
    "sam3d_shape_latents": (8, 256, 8),
    "sam3d_object_pose": (8, 10),
}


def validate_one(
    payload: tuple[str, bool, bool, bool, bool]
) -> tuple[str | None, int, int, int, float]:
    path_s, require_tokens, require_masks, require_geometry, require_objects = payload
    path = Path(path_s)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        for key, shape in EXPECTED.items():
            tensor = value[key]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{key}: expected {shape}, got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{key}: non-finite values")
        valid_masks = int(value["mask_valid_count"])
        valid_geometry = int(value["geometry_valid"])
        valid_objects = int(value.get("sam3d_object_valid_count", 0))
        if require_tokens and not torch.count_nonzero(value["sam3d_tokens"]):
            raise ValueError("sam3d_tokens are entirely zero")
        if require_masks and valid_masks <= 0:
            raise ValueError("cache has no valid SAM 3 masks")
        if require_geometry and valid_geometry <= 0:
            raise ValueError("cache has no valid SAM 3D geometry")
        if require_objects and valid_objects <= 0:
            raise ValueError("cache has no native SAM 3D Objects stage-1 latents")
        return (
            None,
            valid_masks,
            valid_geometry,
            valid_objects,
            float(value["sam3d_tokens"].float().abs().mean()),
        )
    except Exception as error:
        return f"{path}: {error}", 0, 0, 0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--index", type=Path, help="Validate only sample directories listed one per line")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-nonzero-tokens", action="store_true")
    parser.add_argument("--require-masks", action="store_true")
    parser.add_argument("--require-geometry", action="store_true")
    parser.add_argument("--require-sam3d-objects", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/*/condition.pt"))
    if args.index is not None:
        selected = {
            line.strip().replace("\\", "/")
            for line in args.index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        paths = [path for path in paths if path.relative_to(args.root).parent.as_posix() in selected]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No condition.pt files under {args.root}")
    errors: list[str] = []
    mask_instances = geometry_instances = object_instances = 0
    token_abs_mean = 0.0
    payloads = [
        (
            str(path),
            args.require_nonzero_tokens,
            args.require_masks,
            args.require_geometry,
            args.require_sam3d_objects,
        )
        for path in paths
    ]
    if args.workers == 1:
        results = map(validate_one, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(validate_one, payloads, chunksize=8)
    try:
        for error, masks, geometry, objects, token_mean in results:
            if error is not None:
                errors.append(error)
            mask_instances += masks
            geometry_instances += geometry
            object_instances += objects
            token_abs_mean += token_mean
    finally:
        if args.workers != 1:
            executor.shutdown()
    print(
        f"CACHE_VALIDATION files={len(paths)} errors={len(errors)} "
        f"mask_instances={mask_instances} geometry_files={geometry_instances} "
        f"sam3d_object_instances={object_instances} "
        f"token_abs_mean={token_abs_mean / len(paths):.6f}"
    )
    if errors:
        print("\n".join(errors[:20]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
