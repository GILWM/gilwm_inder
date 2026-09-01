#!/usr/bin/env python3
"""Build no-teacher SAM3 + SAM3D inference conditions from first-frame images."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3d-repo", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--depth-model-path", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def normalize_pointmap(pointmap_H_W_3: torch.Tensor) -> torch.Tensor:
    pointmap = pointmap_H_W_3.float()
    valid = torch.isfinite(pointmap).all(dim=-1)
    clean = torch.nan_to_num(pointmap)
    if valid.any():
        center = clean[valid].median(dim=0).values
        centered = clean - center
        radius = torch.linalg.vector_norm(centered[valid], dim=-1)
        scale = torch.quantile(radius, 0.95).clamp_min(1.0e-6)
        xyz = centered / scale
    else:
        xyz = torch.zeros_like(clean)
    xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
    xyz = xyz.permute(2, 0, 1).unsqueeze(0)
    xyz = F.interpolate(xyz, (120, 160), mode="bilinear", align_corners=False)[0]
    valid_small = F.interpolate(valid[None, None].float(), (120, 160), mode="nearest")[0, 0]
    dx = F.pad(xyz[:, :, 2:] - xyz[:, :, :-2], (1, 1, 0, 0))
    dy = F.pad(xyz[:, 2:, :] - xyz[:, :-2, :], (0, 0, 1, 1))
    normals = torch.cross(dx.permute(1, 2, 0), dy.permute(1, 2, 0), dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1.0e-6).permute(2, 0, 1)
    normals = normals * valid_small.unsqueeze(0)
    return torch.cat([xyz[2:3], xyz, normals, valid_small.unsqueeze(0)], dim=0).contiguous()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from prepare_sam3d_conditions import Sam3MaskEncoder

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    sys.path.insert(0, str(args.sam3d_repo))
    import sam3d_objects  # noqa: F401
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    sam3 = Sam3MaskEncoder(args.sam3_checkpoint, args.device, args.mask_threshold)
    config = OmegaConf.load(args.pipeline_config)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.stage1_runtime_only = True
    config.device = args.device
    config.depth_model.device = args.device
    config.depth_model.model.pretrained_model_name_or_path = str(args.depth_model_path)
    config.workspace_dir = str(args.pipeline_config.parent)
    pipeline = instantiate(config)

    records = load_records(args.manifest)
    selected = [
        (index, record)
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    completed = skipped = failed = 0
    for position, (manifest_index, record) in enumerate(selected, 1):
        sample_id = str(record["sample_id"])
        destination = args.output_root / f"{sample_id}.pt"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            image_path = Path(record["image"])
            with Image.open(image_path) as source:
                pil_image = source.convert("RGB")
                image = np.asarray(pil_image).copy()
                masks, mask_meta = sam3(
                    pil_image,
                    [str(record["prompt"]), "robot", "object"],
                    args.max_objects,
                )
            raw_mask_count = min(int(mask_meta[:, 7].sum().item()), args.max_objects)
            # SAM may occasionally return a one-pixel sliver. SAM3D Objects
            # rejects boxes smaller than 2x2, so compact only geometrically
            # valid masks while preserving their original score order.
            valid_masks = []
            valid_meta = []
            for mask, meta in zip(masks[:raw_mask_count], mask_meta[:raw_mask_count]):
                full_mask = F.interpolate(
                    mask[None, None], image.shape[:2], mode="nearest"
                )[0, 0]
                coordinates = torch.nonzero(full_mask > 0.5, as_tuple=False)
                if coordinates.numel() == 0:
                    continue
                height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
                width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
                # SAM3D computes bbox size as max-min (not pixel count), so
                # it requires an extent of at least three pixels per axis.
                if height < 3 or width < 3:
                    continue
                valid_masks.append(mask)
                valid_meta.append(meta)
            compact_masks = torch.zeros_like(masks)
            compact_meta = torch.zeros_like(mask_meta)
            if valid_masks:
                compact_masks[: len(valid_masks)] = torch.stack(valid_masks)
                compact_meta[: len(valid_meta)] = torch.stack(valid_meta)
            masks = compact_masks
            mask_meta = compact_meta
            mask_count = len(valid_masks)
            masks_for_3d = masks[:mask_count]
            if mask_count:
                masks_for_3d = F.interpolate(
                    masks_for_3d[:, None], image.shape[:2], mode="nearest"
                )[:, 0]

            merged_scene = pipeline.merge_image_and_mask(image, None)
            scene_pointmap = pipeline.compute_pointmap(merged_scene)["pointmap"]
            scene_pointmap = scene_pointmap.permute(1, 2, 0).contiguous()
            geometry = normalize_pointmap(scene_pointmap.detach().float().cpu())
            shape_out = torch.zeros(args.max_objects, 256, 8, dtype=torch.float32)
            pose_out = torch.zeros(args.max_objects, 10, dtype=torch.float32)
            for object_index, mask in enumerate(masks_for_3d):
                output = pipeline.run(
                    image,
                    (mask.numpy() > 0.5).astype(np.uint8) * 255,
                    seed=args.seed + manifest_index * args.max_objects + object_index,
                    stage1_only=True,
                    with_mesh_postprocess=False,
                    with_texture_baking=False,
                    with_layout_postprocess=False,
                    pointmap=scene_pointmap,
                )
                shape = torch.as_tensor(output["shape"]).detach().float().cpu()
                if shape.ndim == 3:
                    shape = shape[0]
                if shape.ndim != 2 or shape.shape[-1] != 8:
                    raise ValueError(f"unexpected_shape_latent:{tuple(shape.shape)}")
                shape = F.adaptive_avg_pool1d(shape.transpose(0, 1).unsqueeze(0), 256)[0].transpose(0, 1)
                translation = torch.as_tensor(output["translation"]).reshape(-1, 3)[0]
                rotation = torch.as_tensor(output["rotation"]).reshape(-1, 4)[0]
                scale = torch.as_tensor(output["scale"]).reshape(-1, 3)[0]
                shape_out[object_index] = torch.nan_to_num(shape)
                pose_out[object_index] = torch.nan_to_num(
                    torch.cat([translation, rotation, scale]).float().cpu()
                )

            atomic_save(
                {
                    "schema_version": 3,
                    "sam_masks": masks.float().contiguous(),
                    "sam_mask_meta": mask_meta.float().contiguous(),
                    "sam3d_geometry": geometry,
                    "sam3d_shape_latents": shape_out,
                    "sam3d_object_pose": pose_out,
                    "mask_valid_count": torch.tensor(mask_count, dtype=torch.int64),
                    "sam3d_object_valid_count": torch.tensor(mask_count, dtype=torch.int64),
                    "geometry_valid": torch.tensor(True, dtype=torch.bool),
                    "source_image": str(image_path),
                    "prompt": str(record["prompt"]),
                    "manifest_index": manifest_index,
                    "teacher_tokens": "disabled",
                },
                destination,
            )
            completed += 1
            print(
                f"INFERENCE_CONDITION position={position}/{len(selected)} id={sample_id} "
                f"objects={mask_count} image={tuple(image.shape)} output={destination}",
                flush=True,
            )
        except Exception as error:
            failed += 1
            print(
                f"INFERENCE_CONDITION_ERROR position={position}/{len(selected)} "
                f"id={sample_id} error={error!r}",
                flush=True,
            )
            traceback.print_exc()

    print(f"DONE completed={completed} skipped={skipped} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
