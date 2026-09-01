#!/usr/bin/env python3
"""Fill missing training mask caches from materialized first-frame images only.

This intentionally does not compute DINO/REPA teacher tokens or SAM 3D
latents.  The native 25-step SAM 3D Objects pass consumes these masks and
stores geometry, shape and pose in a separate sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


MASK_SIZE = (120, 160)
MAX_OBJECTS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--batch",
        default="core15k",
        help="Manifest batch to process. Use 'all' to process every batch.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=MAX_OBJECTS)
    parser.add_argument("--mask-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_instruction(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("instruction"), str):
            return value["instruction"].strip()
        for nested in value.values():
            if isinstance(nested, str):
                return nested.strip()
            if isinstance(nested, dict):
                for candidate in ("short", "medium", "long", "instruction"):
                    if isinstance(nested.get(candidate), str):
                        return nested[candidate].strip()
    return ""


def compact_valid_masks(
    masks: torch.Tensor, mask_meta: torch.Tensor, image_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    raw_count = min(int(mask_meta[:, 7].sum().item()), masks.shape[0])
    valid_masks: list[torch.Tensor] = []
    valid_meta: list[torch.Tensor] = []
    for mask, meta in zip(masks[:raw_count], mask_meta[:raw_count]):
        full_mask = F.interpolate(mask[None, None], image_size, mode="nearest")[0, 0]
        coordinates = torch.nonzero(full_mask > 0.5, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
        width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
        if height < 3 or width < 3:
            continue
        valid_masks.append(mask)
        valid_meta.append(meta)

    compact_masks = torch.zeros_like(masks)
    compact_meta = torch.zeros_like(mask_meta)
    if valid_masks:
        compact_masks[: len(valid_masks)] = torch.stack(valid_masks)
        compact_meta[: len(valid_meta)] = torch.stack(valid_meta)
    return compact_masks, compact_meta, len(valid_masks)


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


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if not args.sam3_checkpoint.is_file():
        raise FileNotFoundError(args.sam3_checkpoint)
    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401

    from prepare_sam3d_conditions import Sam3MaskEncoder

    sam3 = Sam3MaskEncoder(args.sam3_checkpoint, args.device, args.mask_threshold)
    records = load_records(args.manifest)
    if args.batch.lower() not in {"all", "*"}:
        records = [record for record in records if record.get("batch") == args.batch]
    selected = [
        (index, record)
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    completed = skipped = failed = 0
    started = time.time()
    for position, (batch_index, record) in enumerate(selected, 1):
        sample_dir = Path(record.get("sample_dir", f"{record['batch']}/{record['question_id']}"))
        destination = args.cache_root / sample_dir / "condition.pt"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            image_path = args.dataset_root / record["first_frame"]
            instruction_path = args.dataset_root / record["instruction"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image_size = (image.height, image.width)
                masks, mask_meta = sam3(
                    image,
                    [read_instruction(instruction_path), "robot", "object"],
                    args.max_objects,
                )
            masks, mask_meta, mask_count = compact_valid_masks(masks, mask_meta, image_size)
            atomic_save(
                {
                    "schema_version": 2,
                    "sam_masks": masks.float().contiguous(),
                    "sam_mask_meta": mask_meta.float().contiguous(),
                    "sam3d_geometry": torch.zeros(8, 120, 160, dtype=torch.float32),
                    "sam3d_shape_latents": torch.zeros(args.max_objects, 256, 8, dtype=torch.float32),
                    "sam3d_object_pose": torch.zeros(args.max_objects, 10, dtype=torch.float32),
                    "frame_indices": torch.tensor([0], dtype=torch.int64),
                    "source_frame_count": torch.tensor(1, dtype=torch.int64),
                    "mask_valid_count": torch.tensor(mask_count, dtype=torch.int64),
                    "geometry_valid": torch.tensor(False, dtype=torch.bool),
                    "sam3d_object_valid_count": torch.tensor(0, dtype=torch.int64),
                    "provenance": {
                        "sam3": "facebook/sam3:first_frame_only",
                        "sam3d_token_encoder": "disabled",
                        "geometry_source": "pending_25step_sidecar",
                        "sam3d_objects_source": "pending_25step_sidecar",
                        "source_image": str(image_path),
                    },
                    "manifest_index": batch_index,
                },
                destination,
            )
            completed += 1
            print(
                f"SAM3_MASK completed={completed} id={record['question_id']} "
                f"objects={mask_count} image={image_path}",
                flush=True,
            )
        except Exception as error:
            failed += 1
            print(f"SAM3_MASK_FAIL id={record['question_id']} error={error!r}", flush=True)

        if position % 50 == 0:
            elapsed = max(time.time() - started, 1.0e-6)
            print(
                f"PROGRESS shard={args.shard_index}/{args.num_shards} position={position}/{len(selected)} "
                f"completed={completed} skipped={skipped} failed={failed} "
                f"completed_per_min={completed / elapsed * 60:.2f}",
                flush=True,
            )

    print(
        f"DONE completed={completed} skipped={skipped} failed={failed} "
        f"shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
