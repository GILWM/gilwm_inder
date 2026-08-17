#!/usr/bin/env python3
"""Merge native CUDA SAM 3D Objects sidecars into existing MUSA caches.

This intentionally preserves the exact SAM 3 masks and DINO tokens used to
produce each native sidecar.  It avoids rerunning either frozen model when the
CUDA result is relayed back from DSS.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--shape-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load_tensor_dict(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary: {path}")
    return value


def atomic_save(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".condition.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalized_native_conditions(
    sidecar: dict[str, Any], max_objects: int, shape_tokens: int, source: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    shape = torch.as_tensor(sidecar["sam3d_shape_latents"], dtype=torch.float32)
    pose = torch.as_tensor(sidecar["sam3d_object_pose"], dtype=torch.float32)
    geometry = torch.as_tensor(sidecar["sam3d_geometry"], dtype=torch.float32)
    if shape.ndim == 2:
        shape = shape.unsqueeze(0)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    if shape.ndim != 3 or shape.shape[-1] != 8:
        raise ValueError(f"{source}: shape latents must be [K,N,8], got {tuple(shape.shape)}")
    if pose.ndim != 2 or pose.shape[-1] != 10:
        raise ValueError(f"{source}: object pose must be [K,10], got {tuple(pose.shape)}")
    if geometry.ndim != 3 or geometry.shape[0] != 8:
        raise ValueError(f"{source}: geometry must be [8,H,W], got {tuple(geometry.shape)}")
    for name, tensor in (("shape", shape), ("pose", pose), ("geometry", geometry)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{source}: {name} contains NaN or Inf")

    valid = min(max_objects, shape.shape[0], pose.shape[0], int(sidecar.get("object_count", shape.shape[0])))
    if valid <= 0:
        raise ValueError(f"{source}: no valid native objects")
    shape_out = torch.zeros(max_objects, shape_tokens, 8, dtype=torch.float32)
    pose_out = torch.zeros(max_objects, 10, dtype=torch.float32)
    shape_out[:valid] = F.adaptive_avg_pool1d(shape[:valid].transpose(1, 2), shape_tokens).transpose(1, 2)
    pose_out[:valid] = pose[:valid]
    geometry_out = F.interpolate(
        geometry.unsqueeze(0), (120, 160), mode="bilinear", align_corners=False
    )[0].contiguous()
    if not torch.count_nonzero(shape_out[:valid]):
        raise ValueError(f"{source}: native shape latents are entirely zero")
    if not torch.count_nonzero(geometry_out):
        raise ValueError(f"{source}: native geometry is entirely zero")
    return shape_out.contiguous(), pose_out.contiguous(), geometry_out, valid


def merge_one(
    payload: tuple[str, str, str, str, int, int, bool, bool]
) -> tuple[str, str]:
    (
        base_path_s,
        base_root_s,
        sidecar_root_s,
        output_root_s,
        max_objects,
        shape_tokens,
        allow_missing,
        overwrite,
    ) = payload
    base_path = Path(base_path_s)
    base_root = Path(base_root_s)
    sidecar_root = Path(sidecar_root_s)
    output_root = Path(output_root_s)
    relative = base_path.relative_to(base_root)
    sidecar_path = sidecar_root / relative.parent / "sam3d_objects.pt"
    destination = output_root / relative
    if destination.is_file() and not overwrite:
        return "skipped", str(destination)
    if not sidecar_path.is_file():
        return "missing", str(sidecar_path)
    try:
        base = load_tensor_dict(base_path)
        sidecar = load_tensor_dict(sidecar_path)
        shape, pose, geometry, valid = normalized_native_conditions(
            sidecar, max_objects, shape_tokens, sidecar_path
        )
        base["schema_version"] = max(2, int(base.get("schema_version", 0)))
        base["sam3d_shape_latents"] = shape
        base["sam3d_object_pose"] = pose
        base["sam3d_geometry"] = geometry
        base["sam3d_object_valid_count"] = torch.tensor(valid, dtype=torch.int64)
        base["geometry_valid"] = torch.tensor(True, dtype=torch.bool)
        provenance = dict(base.get("provenance", {}))
        provenance["geometry_source"] = str(sidecar_path)
        provenance["sam3d_objects_source"] = str(sidecar_path)
        base["provenance"] = provenance
        atomic_save(base, destination)
        return "completed", str(destination)
    except Exception as error:
        return "failed", f"{base_path}: {error!r}"


def main() -> None:
    args = parse_args()
    paths = sorted(args.base_cache_root.glob("*/*/condition.pt"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No base cache files under {args.base_cache_root}")

    payloads = [
        (
            str(base_path),
            str(args.base_cache_root),
            str(args.sidecar_root),
            str(args.output_root),
            args.max_objects,
            args.shape_tokens,
            args.allow_missing,
            args.overwrite,
        )
        for base_path in paths
    ]
    counts = {"completed": 0, "skipped": 0, "missing": 0, "failed": 0}
    problems: list[str] = []
    if args.workers == 1:
        results = map(merge_one, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(merge_one, payloads, chunksize=8)
    try:
        for position, (status, detail) in enumerate(results, 1):
            counts[status] += 1
            if status == "failed" or (status == "missing" and not args.allow_missing):
                if len(problems) < 20:
                    problems.append(f"{status.upper()} {detail}")
            if position % 250 == 0 or position == len(payloads):
                print(f"SAM3D_MERGE_PROGRESS position={position}/{len(payloads)} counts={counts}", flush=True)
    finally:
        if args.workers != 1:
            executor.shutdown()

    if problems:
        print("\n".join(problems), flush=True)

    print(
        f"SAM3D_MERGE_DONE completed={counts['completed']} skipped={counts['skipped']} "
        f"missing={counts['missing']} failed={counts['failed']}",
        flush=True,
    )
    if counts["failed"] or (counts["missing"] and not args.allow_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
