#!/usr/bin/env python3
import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


def load_cache(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_one(payload):
    condition_path_s, cache_root_s, sidecar_root_s = payload
    condition_path = Path(condition_path_s)
    cache_root = Path(cache_root_s)
    sidecar_root = Path(sidecar_root_s)
    sample = condition_path.parent.relative_to(cache_root)
    sidecar_path = sidecar_root / sample / "sam3d_objects.pt"
    if not sidecar_path.is_file():
        return sample.as_posix(), "missing", False, False
    try:
        condition = load_cache(condition_path)
        sidecar = load_cache(sidecar_path)

        # Match the actual DataLoader overlay path: SAM 3 masks/meta come from
        # condition.pt, while the newly generated native SAM 3D tensors come
        # from sam3d_objects.pt.  Validating shape/pose from the legacy base
        # cache would incorrectly certify the old coarse conditions.
        expected_base_shapes = {
            "sam_masks": (8, 120, 160),
            "sam_mask_meta": (8, 8),
        }
        base_tensors = {}
        for key, expected_shape in expected_base_shapes.items():
            tensor = torch.as_tensor(condition[key])
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{key} shape {tuple(tensor.shape)} != {expected_shape}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{key} contains NaN or Inf")
            base_tensors[key] = tensor

        geometry = torch.as_tensor(sidecar["sam3d_geometry"])
        shape = torch.as_tensor(sidecar["sam3d_shape_latents"])
        pose = torch.as_tensor(sidecar["sam3d_object_pose"])
        if geometry.ndim != 3 or geometry.shape[0] != 8:
            raise ValueError(f"sam3d_geometry must be [8,H,W], got {tuple(geometry.shape)}")
        if shape.ndim == 2:
            shape = shape.unsqueeze(0)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        if shape.ndim != 3 or shape.shape[-1] != 8:
            raise ValueError(f"sam3d_shape_latents must be [K,N,8], got {tuple(shape.shape)}")
        if pose.ndim != 2 or pose.shape[-1] != 10:
            raise ValueError(f"sam3d_object_pose must be [K,10], got {tuple(pose.shape)}")
        for key, tensor in (
            ("sam3d_geometry", geometry),
            ("sam3d_shape_latents", shape),
            ("sam3d_object_pose", pose),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{key} contains NaN or Inf")
        if not torch.count_nonzero(geometry):
            raise ValueError("geometry is entirely zero")

        object_count = int(torch.as_tensor(sidecar.get("object_count", 0)).item())
        if object_count < 0 or object_count > min(shape.shape[0], pose.shape[0], 8):
            raise ValueError(f"invalid object_count={object_count}")
        if object_count and not torch.count_nonzero(shape[:object_count]):
            raise ValueError("detected objects have zero shape latents")
        if object_count and not torch.count_nonzero(pose[:object_count]):
            raise ValueError("detected objects have zero poses")
        empty_masks = not bool(torch.count_nonzero(base_tensors["sam_masks"]))
        return sample.as_posix(), "complete", empty_masks, object_count == 0
    except Exception as error:
        return sample.as_posix(), f"invalid:{error}", False, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a frozen partial index and succeed when unfinished sidecars are absent.",
    )
    parser.add_argument(
        "--scan-sidecars-only",
        action="store_true",
        help="For a frozen partial run, enumerate completed sidecars and validate their matching base caches.",
    )
    args = parser.parse_args()
    complete = []
    missing = invalid = empty_masks = object_free = 0
    problems = []
    if args.scan_sidecars_only:
        sidecars = sorted(args.sidecar_root.glob("*/*/sam3d_objects.pt"))
        conditions = [
            args.cache_root / sidecar.parent.relative_to(args.sidecar_root) / "condition.pt"
            for sidecar in sidecars
        ]
    else:
        conditions = sorted(args.cache_root.glob("*/*/condition.pt"))
    payloads = [(str(path), str(args.cache_root), str(args.sidecar_root)) for path in conditions]
    if args.workers == 1:
        results = map(validate_one, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(validate_one, payloads, chunksize=16)
    try:
        for position, (sample, status, sample_empty_masks, sample_object_free) in enumerate(results, 1):
            if status == "complete":
                complete.append(sample)
                empty_masks += int(sample_empty_masks)
                object_free += int(sample_object_free)
            elif status == "missing":
                missing += 1
            else:
                invalid += 1
                if len(problems) < 20:
                    problems.append(f"{sample}: {status}")
            if position % 1000 == 0 or position == len(payloads):
                print(
                    f"FULL_CONDITION_AUDIT position={position}/{len(payloads)} complete={len(complete)} "
                    f"missing={missing} invalid={invalid} empty_masks={empty_masks} object_free={object_free}",
                    flush=True,
                )
    finally:
        if args.workers != 1:
            executor.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".native_full_index.", dir=args.output.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text("\n".join(complete) + ("\n" if complete else ""), encoding="utf-8")
        os.replace(temporary, args.output)
        os.chmod(args.output, 0o664)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if problems:
        print("\n".join(problems), flush=True)
    print(
        f"NATIVE_INDEX complete={len(complete)} missing={missing} invalid={invalid} "
        f"empty_masks={empty_masks} object_free={object_free} "
        f"teacher_tokens_required=false output={args.output}"
    )
    if invalid or (missing and not args.allow_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
